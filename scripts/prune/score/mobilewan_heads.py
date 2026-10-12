"""Squared projected head energies, equal-view actor statistics and global masks.

Observe native gated activations at to_out[0]. Bias is excluded. This module
never changes activations, applies another gate, or chooses evaluation actors.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch


@torch.no_grad()
def projected_energy(activation, weight, heads, indices, *, token_chunk=256):
    """FP32 head-local Gram products, with FP64 sums over selected query tokens."""
    if (activation.ndim != 3 or heads < 1 or activation.shape[-1] % heads
            or weight.ndim != 2 or weight.shape[1] != activation.shape[-1]):
        raise ValueError("incompatible activation, projection or head geometry")
    if (indices.ndim != 1 or indices.dtype != torch.int64 or indices.numel() == 0
            or token_chunk < 1 or torch.any(indices < 0) or torch.any(indices >= activation.shape[1])
            or indices.unique().numel() != indices.numel()):
        raise ValueError("invalid token indices or chunk size")
    width = activation.shape[-1] // heads
    # Keep each Gram GEMM's original layout and FP32 arithmetic. Batched Gram
    # construction can select another kernel with materially different rounding.
    grams = []
    for head in range(heads):
        w = weight[:, head * width:(head + 1) * width].float()
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


def cohort_statistics(energies, actors, *, replicates=2000, seed=42):
    """Input [view,sigma,head]; pool raw energy before any normalization.

    Call once per block and cohort. Identical actor order and seed produce paired
    draws across blocks. Each draw retains all views and sigmas of its actors.
    """
    values = np.asarray(energies, dtype=np.float64)
    if (values.ndim != 3 or values.shape[0] != len(actors) or min(values.shape) < 1
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("invalid view/sigma/head energies")
    identities = sorted(set(actors))
    if len(identities) < 2 or values.shape[0] < 2 or replicates < 2:
        raise ValueError("statistics require at least two actors, views and replicates")
    # Index zero is pooled; all later entries retain their original sigma order.
    observations = np.concatenate([values.mean(axis=1, keepdims=True), values], axis=1)
    counts = np.array([sum(actor == identity for actor in actors) for identity in identities])
    sums = np.stack([observations[np.array(actors) == identity].sum(axis=0) for identity in identities])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(identities), size=(replicates, len(identities)))
    means = np.stack([sums[draw].sum(axis=0) / counts[draw].sum() for draw in draws])
    denominators = means.mean(axis=-1, keepdims=True)
    mu = observations.mean(axis=0)
    block = mu.mean(axis=-1, keepdims=True)
    if (not np.isfinite(means).all() or not np.isfinite(block).all()
            or not np.isfinite(denominators).all() or np.any(block <= 0) or np.any(denominators <= 0)):
        raise ValueError("nonfinite or zero block mean in cohort or actor bootstrap")
    normalized = means / denominators
    std = observations.std(axis=0, ddof=1)
    layer_views = observations.mean(axis=-1)
    return {
        "view_count": len(actors), "actor_count": len(identities),
        "raw_mean": mu.tolist(), "raw_std": std.tolist(),
        "raw_min": observations.min(axis=0).tolist(), "raw_max": observations.max(axis=0).tolist(),
        "normalized_mean": (mu / block).tolist(), "normalized_std": (std / block).tolist(),
        "raw_ci95": np.percentile(means, [2.5, 97.5], axis=0).transpose(1, 2, 0).tolist(),
        "normalized_ci95": np.percentile(normalized, [2.5, 97.5], axis=0).transpose(1, 2, 0).tolist(),
        "layer_mean": block[:, 0].tolist(), "layer_std": layer_views.std(axis=0, ddof=1).tolist(),
        "layer_ci95": np.percentile(means.mean(axis=-1), [2.5, 97.5], axis=0).T.tolist(),
        "bootstrap": {"replicates": replicates, "seed": seed, "actors": identities,
                      "draws_sha256": hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),
                      "rng": "numpy.default_rng.PCG64", "interval": "pointwise_actor_cluster_percentile",
                      "axis_order": "pooled_then_sigmas"},
    }


def global_masks(scores, fractions=(0.0, 0.10, 0.23, 0.33)):
    """One stable global order; round half up and keep one head per block."""
    order = []
    for block in sorted(scores):
        values = np.asarray(scores[block], dtype=np.float64)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("invalid block scores")
        order.extend((float(value), block, head) for head, value in enumerate(values))
    if not order:
        raise ValueError("no eligible heads")
    order.sort()
    total = len(order)
    feasible = total - len(scores)
    removable, left = [], {block: len(values) for block, values in scores.items()}
    for _, block, head in order:
        if left[block] > 1:
            removable.append((block, head))
            left[block] -= 1
    results = []
    for fraction in fractions:
        if not math.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError("invalid removal fraction")
        count = math.floor(fraction * total + 0.5)
        if count > feasible:
            raise ValueError("infeasible global budget with one head per block")
        masks = {block: [1] * len(values) for block, values in scores.items()}
        for block, head in removable[:count]:
            masks[block][head] = 0
        results.append({"requested_fraction": fraction, "realized_fraction": count / total,
                        "removed_count": count, "eligible_count": total, "masks": masks,
                        "retained": {b: [h for h, keep in enumerate(m) if keep] for b, m in masks.items()},
                        "removed": {b: [h for h, keep in enumerate(m) if not keep] for b, m in masks.items()}})
    return {"global_order": [[block, head] for _, block, head in order], "budgets": results,
            "constraint": "at_least_one_head_per_block"}
