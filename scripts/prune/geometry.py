"""Probed latent geometry -- the one place scripts/prune/* turns pixels into
tokens, so no script re-derives "8" or "32" from the plan's prose.

plans/2026-08-26-refiner-head-ffn-pruning.md §4 decision 2: "Latent geometry is
probed, not assumed. Check the window rules (``F % 8 == 1``, ``(overlap-1) % 8
== 0``) against the *probed* temporal scale factor, not a literal 8."

``SpatioTemporalScaleFactors.from_model_config`` already derives the factors from
a checkpoint's embedded VAE block list, so probing costs one safetensors header
read. On both packs on disk today it returns the (8, 32, 32) default -- 2.3's
monolith derives it from 9 encoder blocks + patch_size 4, and 2.5's diffusion-
decoder VAE file carries a ``vae`` section with no block list so it falls back --
i.e. the literals were right, but they are now *checked* rather than assumed, and
a conv-VAE variant (§3.3) or a 16x16x4 VAE would be picked up instead of silently
producing a wrong token count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from safetensors import safe_open

from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape


@dataclass(frozen=True)
class ProbedScaleFactors:
    factors: SpatioTemporalScaleFactors
    source: str  # which checkpoint the block list came from, or "default"


def _config_of(path: str) -> dict:
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
    return json.loads(meta.get("config", "{}"))


def probe_scale_factors(video_vae_path: str, transformer_path: str) -> ProbedScaleFactors:
    """Derive the video VAE's spatiotemporal scale factors from checkpoint metadata.

    Tries the video VAE checkpoint first (it owns the block list), then the
    transformer (true for the 2.3 monolith, where both live in one file). Reports
    which source actually supplied a block list so a silent fallback to the
    (8, 32, 32) default is visible in the provenance rather than invisible.
    """
    for label, path in (("video_vae", video_vae_path), ("transformer", transformer_path)):
        cfg = _config_of(path)
        vae_cfg = cfg.get("vae", {})
        if vae_cfg.get("encoder_blocks") or vae_cfg.get("decoder_blocks"):
            return ProbedScaleFactors(SpatioTemporalScaleFactors.from_model_config(cfg), label)
    return ProbedScaleFactors(SpatioTemporalScaleFactors.default(), "default")


def pixel_frames_for(latent_frames: int, factors: SpatioTemporalScaleFactors) -> int:
    """Inverse of ``VideoLatentShape.from_pixel_shape``'s temporal rule.

    ``from_pixel_shape`` computes ``(F - 1) // time + 1``; the smallest pixel
    count that lands exactly on ``latent_frames`` is ``time * (n - 1) + 1``,
    which is also the ``F % time == 1`` grid the refine script's windows use.
    """
    if latent_frames < 1:
        raise ValueError(f"latent_frames must be >= 1, got {latent_frames}")
    return factors.time * (latent_frames - 1) + 1


def latent_shape_for(
    latent_frames: int,
    height: int,
    width: int,
    fps: float,
    factors: SpatioTemporalScaleFactors,
    latent_channels: int,
) -> tuple[VideoPixelShape, VideoLatentShape]:
    """Pixel + latent shape for a chunk of ``latent_frames`` at ``height x width``."""
    pixel_shape = VideoPixelShape(
        batch=1, frames=pixel_frames_for(latent_frames, factors), height=height, width=width, fps=fps
    )
    latent = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=latent_channels, scale_factors=factors)
    if latent.frames != latent_frames:
        raise AssertionError(
            f"probed geometry disagrees: asked for {latent_frames} latent frames, "
            f"{pixel_shape.frames} pixel frames round-tripped to {latent.frames} "
            f"(scale_factors={factors})"
        )
    return pixel_shape, latent


def check_window_rules(window_frames: int, overlap_frames: int, factors: SpatioTemporalScaleFactors) -> None:
    """Validate the refine script's window grid against the *probed* temporal factor.

    Raises ``SystemExit`` (not ``AssertionError``) because these are user-supplied
    CLI values, and a misaligned window silently shifts the carried-over latent
    frame against the destination window's grid -- the failure the sliding-window
    script's own docstring warns about.
    """
    t = factors.time
    if window_frames % t != 1:
        raise SystemExit(
            f"--window-frames {window_frames} must satisfy F %% {t} == 1 for the probed VAE "
            f"temporal scale factor {t} (got {window_frames % t})."
        )
    if (overlap_frames - 1) % t != 0:
        raise SystemExit(
            f"--overlap-frames {overlap_frames} must satisfy (overlap - 1) %% {t} == 0 for the probed "
            f"VAE temporal scale factor {t} -- the same grid every window size follows. Each window's "
            "own latent frame 0 is a single-pixel causal keyframe rather than a full temporal block, "
            "so the carried-over span needs that grid to land on whole latent frames of the next "
            f"window's regular latent frames. Nearest valid values: {t * ((overlap_frames - 1) // t) + 1} "
            f"or {t * ((overlap_frames - 1) // t + 1) + 1}."
        )
