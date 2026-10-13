import pytest
import torch

from scripts.prune.evaluate.learned_head_media import panels, pixel_metrics


def test_roles_frames_and_fixed_regions():
    rgb = torch.zeros(5, 3, 32, 32, dtype=torch.uint8)
    result = panels(rgb, rgb, rgb, {"energy": rgb, "learned_joint": rgb})
    assert [p.role for p in result] == ["capture_rgb", "capture_vae", "teacher", "energy", "learned_joint"]
    weights = torch.ones(5, 32, 32)
    assert pixel_metrics(rgb, rgb, weights)["mse"] == 0
    with pytest.raises(ValueError, match="coverage"):
        pixel_metrics(rgb, rgb, weights * 0)
