"""CPU-only assertions over the actual deployed pruning inputs."""

from __future__ import annotations

from scripts.prune.core import model_registry


def test_registry_is_the_real_25_checkpoint(model: model_registry.RefinerModel) -> None:
    assert model.key == "2.5"
    assert model.version[:2] == (2, 5)
    assert model.caps.num_layers == 48
    assert model.caps.num_heads == 32
    assert tuple(model.scale_factors) == (8, 32, 32)
    assert model.sigmas[-3:] == [0.725, 0.421875, 0.0]
