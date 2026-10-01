from __future__ import annotations

from scripts.prune.core import geometry


def test_pixel_latent_round_trip(model):
    for n in (1, 2, 3, 4):
        pixel, latent = geometry.latent_shape_for(n, 512, 512, 30.0, model.scale_factors, model.caps.latent_channels)
        assert pixel.frames == 8 * (n - 1) + 1 and latent.frames == n
