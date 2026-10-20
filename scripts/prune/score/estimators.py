"""Task-independent unit scores and deterministic structural mask allocation.

`rms_projection_scores` is the calibration proxy: sampled post-activation RMS
times the output-projection norm. `projected_energy` is the exact mean squared
norm of each head's projected output on explicit tokens, with FP32 Gram
products and FP64 sums. `fractional_masks` removes the lowest scores per branch.
"""

from __future__ import annotations

import torch


def rms_projection_scores(
    square_mean_sum: torch.Tensor, projection_norm: torch.Tensor, observations: int
) -> torch.Tensor:
    """Sampled post-activation RMS times the matching output-projection norm."""
    if observations < 1 or square_mean_sum.shape != projection_norm.shape:
        raise ValueError("score statistics have incompatible shape or no observations")
    return (square_mean_sum / observations).sqrt() * projection_norm


@torch.no_grad()
def projected_energy(activation, weight, heads, indices, *, token_chunk=256):
    """FP32 head-local Gram products, with FP64 sums over selected query tokens."""
    if (
        activation.ndim != 3
        or heads < 1
        or activation.shape[-1] % heads
        or weight.ndim != 2
        or weight.shape[1] != activation.shape[-1]
    ):
        raise ValueError("incompatible activation, projection or head geometry")
    if (
        indices.ndim != 1
        or indices.dtype != torch.int64
        or indices.numel() == 0
        or token_chunk < 1
        or torch.any(indices < 0)
        or torch.any(indices >= activation.shape[1])
        or indices.unique().numel() != indices.numel()
    ):
        raise ValueError("invalid token indices or chunk size")
    width = activation.shape[-1] // heads
    # Keep each Gram GEMM's original layout and FP32 arithmetic. Batched Gram
    # construction can select another kernel with materially different rounding.
    grams = []
    for head in range(heads):
        w = weight[:, head * width : (head + 1) * width].float()
        grams.append(w.T @ w)
    gram = torch.stack(grams)
    total = torch.zeros(heads, dtype=torch.float64, device=activation.device)
    invalid = torch.zeros((), dtype=torch.bool, device=activation.device)
    for chunk in indices.split(token_chunk):
        a = activation[:, chunk].float().reshape(activation.shape[0], chunk.numel(), heads, width)
        a = a.permute(2, 0, 1, 3).reshape(heads, -1, width).contiguous()
        energy = (torch.bmm(a, gram) * a).sum(-1)
        # Accumulate on the device; synchronizing every head/token chunk makes
        # full-dataset scoring dominated by host waits instead of arithmetic.
        invalid.logical_or_(~torch.isfinite(energy).all() | (energy < 0).any())
        total += energy.double().sum(-1)
    result = total / (activation.shape[0] * indices.numel())
    if invalid.item() or not torch.isfinite(result).all():
        raise ValueError("nonfinite or negative projected energy")
    return result


def fractional_masks(scores: dict[str, torch.Tensor], fraction: float) -> dict[str, list[int]]:
    """Remove the lowest-score units per named branch, keeping at least one."""
    if not 0 <= fraction < 1:
        raise ValueError("pruning fraction must be in [0, 1)")
    result = {}
    for name, values in scores.items():
        if values.ndim != 1 or not values.numel() or not torch.isfinite(values).all():
            raise ValueError(f"{name}: scores must be finite and one-dimensional")
        remove = min(round(values.numel() * fraction), values.numel() - 1)
        mask = torch.ones(values.numel(), dtype=torch.int32)
        if remove:
            mask[torch.argsort(values, stable=True)[:remove]] = 0
        result[name] = mask.tolist()
    return result
