"""Task-independent unit scores and deterministic structural mask allocation."""

from __future__ import annotations

import torch


def rms_projection_scores(square_mean_sum: torch.Tensor, projection_norm: torch.Tensor,
                          observations: int) -> torch.Tensor:
    """Sampled post-activation RMS times the matching output-projection norm."""
    if observations < 1 or square_mean_sum.shape != projection_norm.shape:
        raise ValueError("score statistics have incompatible shape or no observations")
    return (square_mean_sum / observations).sqrt() * projection_norm


def exact_local_head_energy(activation: torch.Tensor, output_weight: torch.Tensor,
                            heads: int, sample_indices: torch.Tensor) -> torch.Tensor:
    """RMS of each head's projected local output on explicit native tokens."""
    if activation.ndim != 3 or heads < 1 or activation.shape[-1] % heads:
        raise ValueError("activation does not fit complete heads")
    if output_weight.ndim != 2 or output_weight.shape[1] != activation.shape[-1]:
        raise ValueError("output projection does not fit activation width")
    if sample_indices.ndim != 1 or not sample_indices.numel():
        raise ValueError("at least one prediction token is required")
    dim = activation.shape[-1] // heads
    sampled = activation[:, sample_indices].float().reshape(-1, heads, dim)
    weight = output_weight.float().reshape(output_weight.shape[0], heads, dim)
    scores = []
    for head in range(heads):
        projected = sampled[:, head] @ weight[:, head].T
        scores.append(projected.square().sum(-1).mean().sqrt())
    return torch.stack(scores)


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
