"""Export sliced weights and split adapters with declared reduction geometry.

The default compacts attention and FFN up projections. It preserves the native
FFN down-projection reduction coordinates to avoid BF16 sum-order changes.
An envelope pins both payloads. See doc/score/export_joint_split.md.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from ltx_core.model.transformer.feed_forward import ShapeFaithfulLinear
from scripts.prune.core.provenance import file_sha256
from scripts.prune.learn import checkpoints
from scripts.prune.score import export_learned_ffn, export_pruned, hooks

FORMAT = "joint_split_bundle_v1"


class SplitAdapter(nn.Module):
    """Restore only missing adapter axes, then execute the original two GEMMs."""

    def __init__(self, base, a, b):
        super().__init__()
        self.base = base.requires_grad_(False)
        if a.dtype != torch.float32 or b.dtype != torch.float32 or a.ndim != 2 or b.ndim != 2:
            raise ValueError("Split adapter masters must be FP32 matrices")
        if b.shape[1] != a.shape[0] or (b.shape[0], a.shape[1]) != tuple(base.weight.shape):
            raise ValueError("Split adapter shape differs from stored base axes")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Nonfinite split adapter")
        self.register_buffer("A", a.to(base.weight.device))
        self.register_buffer("B", b.to(base.weight.device))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        original = self.base(x)
        a, b = self.A, self.B
        if hasattr(self.base, "original_shape"):
            rows, columns = self.base.original_shape
            index = torch.tensor(self.base.retained_indices, device=a.device)
            if self.base.axis == 0:
                b = b.new_zeros(rows, b.shape[1]).index_copy(0, index, b)
            else:
                a = a.new_zeros(a.shape[0], columns).index_copy(1, index, a)
        return original + F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype))


def attach(model, state):
    consumed = set()
    for name, module in hooks.iter_video_attention(model):
        if not name.endswith(".attn1"):
            continue
        for projection, base in (("v", module.to_v), ("out", module.to_out[0])):
            prefix = f"{name}.{projection}"
            if prefix + ".A" not in state:
                continue
            wrapper = SplitAdapter(base, state[prefix + ".A"], state[prefix + ".B"])
            if projection == "v":
                module.to_v = wrapper
            else:
                module.to_out[0] = wrapper
            consumed.update((prefix + ".A", prefix + ".B"))
    for name, module in hooks.iter_video_ffn(model):
        for projection, base in (("up", module.net[0].proj), ("down", module.net[2])):
            prefix = f"{name}.{projection}"
            if prefix + ".A" not in state:
                continue
            wrapper = SplitAdapter(base, state[prefix + ".A"], state[prefix + ".B"])
            if projection == "up":
                module.net[0].proj = wrapper
            else:
                module.net[2] = wrapper
            consumed.update((prefix + ".A", prefix + ".B"))
    if consumed != set(state):
        raise ValueError("Incomplete or unsupported split adapter inventory")


class RestoreInput(nn.Module):
    """Restore original FFN reduction coordinates before a split down projection."""

    def __init__(self, inner, width, indices):
        super().__init__()
        self.inner, self.width, self.indices = inner, width, tuple(indices)

    @property
    def weight(self):
        return self.inner.weight

    def forward(self, x):
        ids = torch.tensor(self.indices, device=x.device)
        expanded = x.new_zeros(*x.shape[:-1], self.width).index_copy(-1, ids, x)
        return self.inner(expanded)


def preserve_ffn_reductions(model, masks):
    for name, ff in hooks.iter_video_ffn(model):
        mask = masks[name]
        ids = [i for i, bit in enumerate(mask) if bit]
        if len(ids) == len(mask):
            continue
        down = ff.net[2]
        if not isinstance(down, SplitAdapter):
            raise ValueError("FFN reduction preservation requires a split down adapter")
        original = down.base
        padded = ShapeFaithfulLinear(
            len(mask), original.weight.shape[0], indices=ids, axis=1, bias=original.bias is not None
        )
        padded.weight = original.weight
        padded.bias = original.bias
        down.base = padded
        ff.net[2] = RestoreInput(down, len(mask), ids)


def bundle(source, masks, state, output, *, provenance=None, mode="compact_preserve_ffn_reduction"):
    if mode not in ("compact", "compact_faithful", "compact_preserve_ffn_reduction"):
        raise ValueError("Unsupported bundle execution mode")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Use a fresh bundle path")
    widths = export_pruned.checkpoint_mask_widths(source)
    if set(masks) != set(widths):
        raise ValueError("Bundle requires complete native mask families")
    head_mask = torch.tensor([masks[f"{b}.attn1"] for b in range(len(widths) // 3)])
    ffn_mask = torch.tensor([masks[f"{b}.ff"] for b in range(len(widths) // 3)])
    # Slice the adapter in the same original coordinates as its base projection.
    from safetensors import safe_open

    with safe_open(source, framework="pt", device="cpu") as handle:
        head_dim = json.loads(handle.metadata()["config"])["transformer"].get("attention_head_dim", 128)
    compact = export_learned_ffn.compact_adapter_state(state, head_mask, ffn_mask, head_dim=head_dim)
    weights = output.with_suffix(".base.safetensors")
    adapter_path = output.with_suffix(".adapters.safetensors")
    export_pruned.export(
        source,
        masks,
        weights,
        model_key="2.5",
        mode="compact" if mode == "compact_preserve_ffn_reduction" else mode,
        provenance_block=dict(family="width_joint_split", qualification="unqualified_until_native_replay"),
    )
    checkpoints.publish(adapter_path, lambda p: save_file(compact, str(p)))
    record = dict(
        format=FORMAT,
        masks=masks,
        source=str(Path(source).resolve()),
        source_sha256=file_sha256(source),
        provenance=provenance or {},
        weights=dict(path=str(weights), sha256=file_sha256(weights)),
        adapters=dict(path=str(adapter_path), sha256=file_sha256(adapter_path)),
        arithmetic="split_adapter_bf16_v1",
        execution_mode=mode,
        qualification="unqualified_until_native_replay",
        compact_execution_qualified=False,
        timing_qualified=False,
    )
    checkpoints.publish_json(output, record)
    return record


def read(path):
    record = json.loads(Path(path).read_text())
    if record.get("format") != FORMAT or record.get("arithmetic") != "split_adapter_bf16_v1":
        raise ValueError("Unsupported split bundle format")
    if record.get("execution_mode") not in (
        "compact",
        "compact_faithful",
        "compact_preserve_ffn_reduction",
    ) or record.get("timing_qualified"):
        raise ValueError("Unsupported bundle execution or unverified timing claim")
    for key in ("weights", "adapters"):
        item = record[key]
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Changed bundle {key}")
    return record


@contextmanager
def open_bundle(current, path):
    record = read(path)
    with current.transformer(Path(record["weights"]["path"])) as model:
        attach(model, load_file(record["adapters"]["path"], device="cpu"))
        if record["execution_mode"] == "compact_preserve_ffn_reduction":
            preserve_ffn_reductions(model, record["masks"])
        yield model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "checkpoint", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    from scripts.prune.data import joint_study
    from scripts.prune.score.export_joint import terminal

    study = joint_study.load(args.study)
    state = terminal(study, args.checkpoint)
    source = study["components"]["transformer"]["path"]
    masks = export_learned_ffn.masks_from_state(state, export_pruned.checkpoint_mask_widths(source))
    bundle(
        source,
        masks,
        state["adapters"],
        args.output,
        provenance=dict(
            study=study["identity"],
            checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=file_sha256(args.checkpoint),
            update=state["update"],
        ),
    )


if __name__ == "__main__":
    main()
