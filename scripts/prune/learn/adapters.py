"""Frozen BF16 projections with FP32 low-rank state and original-ID transfer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from scripts.prune.score.hooks import iter_video_attention, iter_video_ffn


class ProjectionAdapter(nn.Module):
    def __init__(self, base, rank=16, generator=None):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise ValueError("adapter requires a native linear projection")
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, base.in_features, generator=generator, dtype=torch.float32) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        self.A.data = self.A.data.to(base.weight.device)
        self.B.data = self.B.data.to(base.weight.device)
        self.rank, self.scaling, self.enabled = rank, 1.0, True

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        original = self.base(x)
        if not self.enabled:
            return original
        delta = F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))
        return original + delta * self.scaling

    def merged_weight(self):
        return (self.base.weight.float() + self.scaling * self.B @ self.A).to(self.base.weight.dtype)


def attach(model, seed, rank=16):
    model.requires_grad_(False)
    rng = torch.Generator(device="cpu").manual_seed(seed + 10000)
    result = {}
    for name, attn in iter_video_attention(model):
        if name.endswith(".attn1"):
            for projection in ("v", "out"):
                base = attn.to_v if projection == "v" else attn.to_out[0]
                adapter = ProjectionAdapter(base, rank, rng)
                if projection == "v":
                    attn.to_v = adapter
                else:
                    attn.to_out[0] = adapter
                result[f"{name}.{projection}"] = adapter
    return result


def parameters(adapters):
    return [p for module in adapters.values() for p in (module.A, module.B)]


def state_dict(adapters):
    return {
        f"{name}.{key}": getattr(module, key).detach().cpu().clone()
        for name, module in adapters.items()
        for key in ("A", "B")
    }


def load_state(adapters, state):
    expected = {f"{n}.{k}" for n in adapters for k in ("A", "B")}
    if set(state) != expected:
        raise ValueError("adapter tensor inventory differs")
    for name, module in adapters.items():
        for key in ("A", "B"):
            value = state[f"{name}.{key}"]
            target = getattr(module, key)
            if value.shape != target.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("adapter shape, dtype or values differ")
            target.data.copy_(value)


def channel_indices(heads, head_dim=128):
    if heads != sorted(set(heads)) or not heads or any(h < 0 for h in heads):
        raise ValueError("original heads must be sorted unique identities")
    return (torch.tensor(heads)[:, None] * head_dim + torch.arange(head_dim)[None]).flatten()


def slice_tensor(value, name, head_ids, head_dim=128):
    index = channel_indices(head_ids, head_dim).to(value.device)
    if name.endswith(".v.B"):
        return value.index_select(0, index).clone()
    if name.endswith(".out.A"):
        return value.index_select(1, index).clone()
    return value.clone()


def transfer(state, mask, head_dim=128):
    return {
        name: slice_tensor(value, name, torch.where(mask[int(name.split(".")[0])] == 1)[0].tolist(), head_dim)
        for name, value in state.items()
    }


def transfer_optimizer(optimizer_state, names, mask, head_dim=128):
    import copy

    result = copy.deepcopy(optimizer_state)
    ids = [pid for group in result["param_groups"] for pid in group["params"]]
    if len(ids) != len(names):
        raise ValueError("optimizer parameter inventory differs")
    for pid, name in zip(ids, names, strict=True):
        ids_kept = torch.where(mask[int(name.split(".")[0])] == 1)[0].tolist()
        for key, value in result["state"].get(pid, {}).items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                result["state"][pid][key] = slice_tensor(value, name, ids_kept, head_dim)
    return result


def fresh_source_state(seed, *, blocks=48, width=4096, rank=16):
    """Initialize full-source adapters before any compact slicing."""
    rng = torch.Generator().manual_seed(seed + 10000)
    state = {}
    for block in range(blocks):
        for projection in ("v", "out"):
            prefix = f"{block}.attn1.{projection}"
            state[prefix + ".A"] = torch.randn(rank, width, generator=rng) * 0.01
            state[prefix + ".B"] = torch.zeros(width, rank)
    return state


def attach_ffn(model, seed, rank=16):
    """Use a separate initializer stream without freezing attached head adapters."""
    rng = torch.Generator(device="cpu").manual_seed(seed + 20000)
    result = {}
    for name, ff in iter_video_ffn(model):
        for projection in ("up", "down"):
            base = ff.net[0].proj if projection == "up" else ff.net[2]
            adapter = ProjectionAdapter(base, rank, rng)
            if projection == "up":
                ff.net[0].proj = adapter
            else:
                ff.net[2] = adapter
            result[f"{name}.{projection}"] = adapter
    return result


def slice_ffn_tensor(value, name, indices):
    """Transfer a producing B row or consuming A column at original channel IDs."""
    if indices != sorted(set(indices)) or not indices or indices[0] < 0:
        raise ValueError("original FFN channels must be sorted unique identities")
    index = torch.tensor(indices, device=value.device, dtype=torch.long)
    if name.endswith(".ff.up.B"):
        return value.index_select(0, index).clone()
    if name.endswith(".ff.down.A"):
        return value.index_select(1, index).clone()
    return value.clone()


def transfer_ffn(state, mask):
    return {
        name: slice_ffn_tensor(value, name, torch.where(mask[int(name.split(".")[0])] == 1)[0].tolist())
        for name, value in state.items()
    }
