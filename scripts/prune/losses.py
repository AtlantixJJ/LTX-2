"""Masked x0-space objectives for refiner calibration and evaluation.

The transformer is an :class:`X0Model`; do not turn its prediction into a
velocity before scoring it.  In particular, doing so at sigma=0 makes an
otherwise benign comparison numerically singular.  ``state.denoise_mask`` is
token-shaped, so these functions work directly on the denoiser output.
"""

from __future__ import annotations

import torch


def _mask_like(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Validate and cast a token mask without silently broadcasting tokens."""
    if mask.shape != value.shape[:2] + (1,):
        raise ValueError(f"mask shape {tuple(mask.shape)} is incompatible with x0 {tuple(value.shape)}")
    return mask.to(device=value.device, dtype=torch.float32)


def _resolve(pred_x0: torch.Tensor, x0_star: torch.Tensor, state, mask: torch.Tensor | None) -> torch.Tensor:
    """Shape-check a prediction/target pair and pick the scoring mask.

    ``mask`` defaults to ``state.denoise_mask``, but callers scoring the AR task
    should pass ``chunk_states.chunk_token_mask(...)`` instead. The two are *not*
    the same set: the calibration geometry leaves latent frame 0 -- the causal
    VAE's standalone single-pixel keyframe -- fresh alongside the ``n_new`` chunk
    frames, so ``denoise_mask`` selects keyframe + chunk. At ``n_new=1`` that
    keyframe is half the fresh-token mass, and it is not what the deployed AR
    refiner predicts. See ``chunk_states.chunk_token_mask``.
    """
    if pred_x0.shape != x0_star.shape:
        raise ValueError(f"prediction {tuple(pred_x0.shape)} != target {tuple(x0_star.shape)}")
    return _mask_like(state.denoise_mask if mask is None else mask, pred_x0)


def x0_loss(pred_x0: torch.Tensor, x0_star: torch.Tensor, state, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean squared x0 error over the scoring mask (default: all fresh tokens)."""
    m = _resolve(pred_x0, x0_star, state, mask)
    return ((pred_x0.float() - x0_star.float()).square() * m).sum() / m.sum().clamp_min(1)


def rel_l2(pred_x0: torch.Tensor, x0_star: torch.Tensor, state, mask: torch.Tensor | None = None) -> torch.Tensor:
    """T0: relative L2 x0 error over the scoring mask (default: all fresh tokens)."""
    m = _resolve(pred_x0, x0_star, state, mask)
    numerator = ((pred_x0.float() - x0_star.float()).square() * m).sum()
    denominator = (x0_star.float().square() * m).sum().add(1e-8)
    return (numerator / denominator).sqrt()
