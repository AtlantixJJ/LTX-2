"""Iterative, training-free attention pruning and output-projection re-solving.

This is Phase 2's redundancy correction: prune a small increment, rescore the
survivors under that mask, then optionally ridge-reconstruct each output
projection from the retained head activations.  It intentionally emits masks,
not an exported checkpoint; Phase 4 owns structural checkpoint surgery.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

from scripts.prune import chunk_states, hooks, lstsq


def initial_head_masks(model, device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    return {name: torch.ones(attn.heads, device=device or attn.to_out[0].weight.device) for name, attn in hooks.iter_video_attention(model)}


def iterative_head_masks(model, *, target_sparsity: float, rounds: int,
                         rescore: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]) -> tuple[dict[str, torch.Tensor], list[dict]]:
    """Globally remove low-scoring live heads over rounds, recomputing each time.

    ``rescore`` must score the *currently masked* model (Michel/Gauss--Newton do;
    a contribution sweep is useful for comparison but is not redundancy-aware).
    """
    if not 0 <= target_sparsity < 1:
        raise ValueError("target_sparsity must be in [0, 1)")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    masks = initial_head_masks(model)
    total = sum(v.numel() for v in masks.values())
    target_drop = round(total * target_sparsity)
    history = []
    for round_idx in range(rounds):
        current_drop = sum(int((v == 0).sum()) for v in masks.values())
        remaining = target_drop - current_drop
        if remaining <= 0:
            break
        take = min(remaining, max(1, (target_drop + rounds - 1) // rounds))
        values = rescore(masks)
        candidates = [(float(scores[h]), name, h) for name, scores in values.items() for h in range(scores.numel()) if masks[name][h] != 0]
        if take > len(candidates):
            raise ValueError("requested pruning exceeds live heads")
        candidates.sort(key=lambda item: item[0])
        removed = candidates[:take]
        for _, name, head in removed:
            masks[name][head] = 0
        history.append({"round": round_idx + 1, "removed": [{"name": n, "head": h, "score": s} for s, n, h in removed],
                        "dropped": current_drop + take, "total": total})
    return {name: value.cpu() for name, value in masks.items()}, history


@torch.no_grad()
def reconstruct_attention_projection(model, name: str, keep: torch.Tensor, records: list[Path], denoiser, sigmas: torch.Tensor,
                                     device: torch.device, ridge: float = 1e-4) -> torch.Tensor:
    """Fit one surviving-head ``to_out`` weight with streaming Phase-1 activations.

    The returned matrix has shape ``(model_dim, kept_heads * head_dim)`` and is
    also installed in-memory.  Exporting this heterogeneous width belongs to
    Phase 4, so callers should persist the selected masks/provenance alongside it.
    """
    attn = dict(hooks.iter_video_attention(model))[name]
    keep = keep.to(device=device, dtype=torch.long)
    if keep.numel() == 0:
        raise ValueError(f"{name}: cannot reconstruct an empty attention branch")
    acc, add = lstsq.attention_accumulator(attn, keep)

    token_mask: torch.Tensor | None = None

    def callback(observed_name, activation, _module):
        if observed_name == name:
            if token_mask is None:
                raise RuntimeError("attention reconstruction hook ran without a task-token mask")
            add(activation, token_mask)

    with hooks.collect_activations(model, "head", callback):
        for path in records:
            state, _, meta = chunk_states.load_record(path, device)
            token_mask = chunk_states.chunk_token_mask(state, meta)
            result, _ = denoiser(model, state, None, sigmas, meta.step_index)
            if result is None:
                raise RuntimeError("video denoiser unexpectedly returned no result")
    fitted = acc.solve(ridge).to(dtype=attn.to_out[0].weight.dtype)
    # Keep the loaded module shape until Phase 4, but install the exact dense
    # equivalent of the reduced projection: retained columns are the ridge fit,
    # dropped columns are zero.  This makes T0/T2 evaluation of reconstruction
    # possible now; export later slices precisely ``fitted`` into its smaller
    # checkpoint tensor.
    expanded = torch.zeros_like(attn.to_out[0].weight)
    expanded[:, lstsq.head_index(keep, attn.dim_head)] = fitted
    attn.to_out[0].weight.copy_(expanded)
    return fitted
