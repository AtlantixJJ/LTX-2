"""Shared saved-input builders and fixtures; never a collected test module."""

from __future__ import annotations

import torch

from scripts.onestep_avatar.model import causal, common
from scripts.onestep_avatar.tests.support.model import CHANNELS, DEVICE, EDGE, FPS, SCALE, _context


def inputs() -> tuple:
    geometry = causal.CausalGeometry(SCALE, 2, 8)
    grid = common.ClipGrid.build(
        17, EDGE, EDGE, FPS, geometry, device=DEVICE, dtype=torch.float32, latent_channels=CHANNELS
    )
    generator = torch.Generator().manual_seed(72)
    shape = (1, grid.latent_frames * grid.tokens_per_latent_frame, CHANNELS)
    capture, guide, noise = (torch.randn(shape, generator=generator) for _ in range(3))
    return grid, geometry, capture, guide, noise, _context()
