from __future__ import annotations

from scripts.prune.core import geometry, model_registry


def test_pixel_latent_round_trip(model: model_registry.RefinerModel) -> None:
    for n in (1, 2, 3, 4):
        pixel, latent = geometry.latent_shape_for(n, 512, 512, 30.0, model.scale_factors, model.caps.latent_channels)
        assert pixel.frames == 8 * (n - 1) + 1
        assert latent.frames == n
