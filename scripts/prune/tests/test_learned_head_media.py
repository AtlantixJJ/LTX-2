import pytest
import torch

from scripts.prune.evaluate import learned_head_media
from scripts.prune.evaluate.learned_head_media import panels, pixel_metrics


def test_roles_frames_and_fixed_regions():
    rgb = torch.zeros(5, 3, 32, 32, dtype=torch.uint8)
    result = panels(rgb, rgb, rgb, {"energy": rgb, "learned_joint": rgb})
    assert [p.role for p in result] == ["capture_rgb", "capture_vae", "teacher", "energy", "learned_joint"]
    weights = torch.ones(5, 32, 32)
    assert pixel_metrics(rgb, rgb, weights)["mse"] == 0
    with pytest.raises(ValueError, match="coverage"):
        pixel_metrics(rgb, rgb, weights * 0)


def test_white_resize_keeps_strict_metric_range(monkeypatch):
    rgb = torch.full((2, 3, 257, 187), 255, dtype=torch.uint8)
    resized = torch.nn.functional.interpolate(
        rgb[1:].float() / 255, (256, 256), mode="bilinear", align_corners=False, antialias=True
    )
    assert resized.max() > 1  # The real CPU kernel reproduces the failure.

    def checked_metric(model, left, right, device, **kwargs):
        for value in (left, right):
            assert torch.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
        return 0.0

    monkeypatch.setattr(learned_head_media, "lpips_distance", checked_metric)
    result = learned_head_media.region_lpips(None, rgb, rgb, torch.ones(2, 257, 187), "cpu")
    assert result["lpips"] == 0 and result["frame_ids"] == [1]
    with pytest.raises(ValueError, match="bounded interpolation roundoff"):
        learned_head_media.region_lpips(None, rgb.float() * 2, rgb, torch.ones(2, 257, 187), "cpu")
