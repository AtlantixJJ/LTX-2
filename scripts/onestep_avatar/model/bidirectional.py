"""Full-attention segment training and generation without causal cache work.

See doc/model/bidirectional.md. Inputs are tokens on one independent ClipGrid;
callers own data loading, model setup, updates, and saved output records.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from itertools import pairwise

import torch

from scripts.onestep_avatar.model import common
from scripts.onestep_avatar.model.sampling import euler_to, validate_schedule


def plan_samples(
    latent_frames: int,
    *,
    span_latent_frames: int | None = None,
    start_policy: str = "clip_start",
    seed: int = 42,
    step: int = 0,
    rank: int = 0,
    slot: int = 0,
) -> tuple[int, int]:
    """Select one segment, preserving the existing random-window seed key."""
    length = latent_frames if span_latent_frames is None else span_latent_frames
    if not 1 <= length <= latent_frames:
        raise ValueError("segment length must fit the encoded video")
    if start_policy == "clip_start":
        start = 0
    elif start_policy == "random":
        rng = random.Random(f"onestep_avatar.window:{seed}:{step}:{rank}:{slot}")
        start = rng.randrange(latent_frames - length + 1)
    else:
        raise ValueError("start_policy must be clip_start or random")
    return start, start + length


def _check_tokens(grid: common.ClipGrid, source: torch.Tensor, c0: torch.Tensor) -> None:
    expected = grid.latent_frames * grid.tokens_per_latent_frame
    if source.ndim != 3 or source.shape[0] != 1 or source.shape[1] != expected:
        raise ValueError("source must be [1, F*H*W, C] on the segment grid")
    if c0.shape != (1, grid.tokens_per_latent_frame, source.shape[2]):
        raise ValueError("c0 must contain exactly one encoded frame on the segment grid")


def _predict(
    predict_x0: Callable,
    grid: common.ClipGrid,
    tokens: torch.Tensor,
    context: torch.Tensor,
    sigma: float,
    c0: torch.Tensor,
    sigma_dtype: torch.dtype | None,
) -> torch.Tensor:
    conditioned = common.with_clean_prefix(tokens, c0)
    modality = common.block_modality(
        grid,
        conditioned,
        context,
        sigma,
        token_slices=[(0, conditioned.shape[1])],
        clean_prefix_tokens=c0.shape[1],
        sigma_dtype=sigma_dtype,
    )
    prediction = predict_x0(modality)
    if prediction.shape != conditioned.shape:
        raise ValueError("prediction shape must match input tokens")
    return common.with_clean_prefix(prediction, c0)


def train_sample(
    transformer: torch.nn.Module,
    context: torch.Tensor,
    grid: common.ClipGrid,
    capture: torch.Tensor,
    guide: torch.Tensor | None,
    backward: Callable[[torch.Tensor], None],
    *,
    sigma: float,
    seed: int,
    guide_mode: str = "d1",
    accumulation: int = 1,
) -> dict[str, float]:
    """One denoise and immediate backward of fp32 full-frame MSE / A."""
    if accumulation < 1:
        raise ValueError("accumulation must be positive")
    if not 0 < sigma <= 1:
        raise ValueError("sigma must be in (0, 1]")
    source = common.source_for(capture, guide, guide_mode)
    c0 = capture[:, : grid.tokens_per_latent_frame]
    _check_tokens(grid, source, c0)
    prediction = _predict(
        common.denoised_from_velocity_model(transformer),
        grid,
        common.noise_block(source, sigma, seed),
        context,
        sigma,
        c0,
        None,
    )
    mse = common.full_frame_mse(prediction, capture)
    backward(mse / accumulation)
    return {
        "loss": float(mse.detach()),
        "mse": float(mse.detach()),
        "denoise_calls": 1,
        "backward_calls": 1,
        "prime_calls": 0,
        "refresh_calls": 0,
    }


@torch.no_grad()
def sample(
    predict_x0: Callable,
    context: torch.Tensor,
    grid: common.ClipGrid,
    source: torch.Tensor,
    c0: torch.Tensor,
    *,
    schedule: Sequence[float],
    seed: int,
    epsilon: torch.Tensor | None = None,
    sigma_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Generate a segment with fixed initial noise and unchanged first-image data."""

    levels = validate_schedule(list(schedule))
    _check_tokens(grid, source, c0)
    if epsilon is None:
        epsilon = common.epsilon_block(source, seed)
    tokens = common.with_clean_prefix(common.mix_block_noise(source, epsilon, levels[0]), c0)
    for sigma, following in pairwise(levels):
        prediction = _predict(predict_x0, grid, tokens, context, sigma, c0, sigma_dtype)
        tokens = common.with_clean_prefix(euler_to(tokens, prediction, sigma, following), c0)
    return tokens, {"denoise_calls": len(levels) - 1, "prime_calls": 0, "refresh_calls": 0}
