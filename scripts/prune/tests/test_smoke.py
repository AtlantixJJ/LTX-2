"""CPU-only assertions over the actual deployed pruning inputs."""

from __future__ import annotations


def test_registry_is_the_real_25_checkpoint(model):
    assert model.key == "2.5" and model.version[:2] == (2, 5)
    assert model.caps.num_layers == 48 and model.caps.num_heads == 32
    assert tuple(model.scale_factors) == (8, 32, 32)
    assert model.sigmas[-3:] == [0.725, 0.421875, 0.0]


def test_calibration_cache_is_format_2(calibration_index):
    assert calibration_index["format"] == 2
    assert len(calibration_index["records"]) == 528
    assert {record["fps"] for record in calibration_index["records"]} == {24.0, 30.0}
