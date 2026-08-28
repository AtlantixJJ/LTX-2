"""Streaming fp32 ridge re-solves for structured pruning reconstruction."""

from __future__ import annotations

from collections.abc import Callable

import torch


def head_index(keep: torch.Tensor, dim_head: int) -> torch.Tensor:
    """Expand retained head numbers into contiguous input-feature indices."""
    keep = keep.to(dtype=torch.long)
    return (keep[:, None] * dim_head + torch.arange(dim_head, device=keep.device)[None]).reshape(-1)


class RidgeAccumulator:
    """Accumulate ``X Xᵀ`` and ``Y Xᵀ`` without retaining calibration activations."""

    def __init__(self, inputs: int, outputs: int, device: torch.device | str) -> None:
        self.gram = torch.zeros((inputs, inputs), dtype=torch.float32, device=device)
        self.cross = torch.zeros((outputs, inputs), dtype=torch.float32, device=device)
        self.samples = 0

    @torch.no_grad()
    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add feature-major ``x=(inputs,N)``, target ``y=(outputs,N)``."""
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
            raise ValueError(f"expected X=(k,N), Y=(d,N); got {tuple(x.shape)}, {tuple(y.shape)}")
        if x.shape[0] != self.gram.shape[0] or y.shape[0] != self.cross.shape[0]:
            raise ValueError("input/output dimensions do not match accumulator")
        xf, yf = x.float(), y.float()
        self.gram.addmm_(xf, xf.T)
        self.cross.addmm_(yf, xf.T)
        self.samples += x.shape[1]

    @torch.no_grad()
    def solve(self, ridge: float = 1e-4) -> torch.Tensor:
        if self.samples == 0:
            raise ValueError("cannot solve an empty calibration accumulator")
        scale = self.gram.diagonal().mean().clamp_min(torch.finfo(torch.float32).eps)
        system = self.gram + torch.eye(self.gram.shape[0], device=self.gram.device, dtype=self.gram.dtype) * (ridge * scale)
        # solve(system.T, cross.T).T is the ridge minimizer W for Y ~= W X.
        return torch.linalg.solve(system, self.cross.T).T


@torch.no_grad()
def attention_accumulator(
    attn, keep: torch.Tensor
) -> tuple[RidgeAccumulator, Callable[[torch.Tensor, torch.Tensor | None], None]]:
    """Return an accumulator and callback for ``to_out[0]`` pre-projection activations."""
    idx = head_index(keep.to(attn.to_out[0].weight.device), attn.dim_head)
    acc = RidgeAccumulator(idx.numel(), attn.to_out[0].weight.shape[0], idx.device)
    full_weight = attn.to_out[0].weight.detach()

    def add(activation: torch.Tensor, token_mask: torch.Tensor | None = None) -> None:
        """Accumulate only task tokens when a ``(B,T,1)`` mask is supplied."""
        if token_mask is not None:
            if token_mask.shape != activation.shape[:2] + (1,):
                raise ValueError(f"token mask {tuple(token_mask.shape)} != activation {tuple(activation.shape)}")
            activation = activation[token_mask[..., 0].bool()]
        x = activation.detach().reshape(-1, activation.shape[-1]).T
        acc.add(x[idx], full_weight.float() @ x.float())

    return acc, add


@torch.no_grad()
def ffn_accumulator(ff, keep: torch.Tensor) -> tuple[RidgeAccumulator, Callable[[torch.Tensor, torch.Tensor | None], None]]:
    """Streaming ridge system for a post-GELU FFN channel subset."""
    keep = keep.to(device=ff.net[2].weight.device, dtype=torch.long)
    weight = ff.net[2].weight.detach()
    acc = RidgeAccumulator(keep.numel(), weight.shape[0], keep.device)

    def add(activation: torch.Tensor, token_mask: torch.Tensor | None = None) -> None:
        if token_mask is not None:
            if token_mask.shape != activation.shape[:2] + (1,):
                raise ValueError(f"token mask {tuple(token_mask.shape)} != activation {tuple(activation.shape)}")
            activation = activation[token_mask[..., 0].bool()]
        x = activation.detach().reshape(-1, activation.shape[-1]).T
        acc.add(x[keep], weight.float() @ x.float())

    return acc, add
