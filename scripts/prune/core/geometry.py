"""Probe VAE temporal/spatial factors and convert latent shapes to pixel shapes.

Read factors from the selected encoder/decoder configuration rather than an
entrypoint constant. The aligned frame count is temporal_factor*(latents-1)+1;
this preserves the first image when converting a temporal latent count."""

from __future__ import annotations

import json
from dataclasses import dataclass

from safetensors import safe_open

from ltx_core.types import SpatioTemporalScaleFactors


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
    which is also the ``F % time == 1`` pixel grid of every saved capture.
    """
    if latent_frames < 1:
        raise ValueError(f"latent_frames must be >= 1, got {latent_frames}")
    return factors.time * (latent_frames - 1) + 1
