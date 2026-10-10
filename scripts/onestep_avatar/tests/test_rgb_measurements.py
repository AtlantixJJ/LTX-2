"""Reusable RGB measurements validate evidence and weight frames consistently."""

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.onestep_avatar import metrics
from scripts.onestep_avatar.corpus import mask_video


def test_full_rgb_score_averages_error_before_logarithm() -> None:
    reference = torch.zeros(2, 3, 2, 2)
    prediction = reference.clone()
    prediction[1] = 1
    result = metrics.rgb_metrics(prediction, reference)
    assert result["per_frame_mse"] == [0, 1]
    assert result["mse"] == 0.5
    assert result["psnr"] == pytest.approx(3.0102999566)
    assert not result["all_exact_match"]
    exact = metrics.rgb_metrics(reference, reference)
    assert exact["psnr"] is None
    assert exact["all_exact_match"]


def test_subject_mask_preserves_dilation_and_checks_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    path = tmp_path / "mask.mp4"
    assert metrics.subject_mask(path, 1, 12, 16) is None
    path.write_bytes(b"controlled reader fixture")
    raw = np.zeros((1, 6, 8), dtype=np.uint8)
    raw[0, 3, 4] = 255
    monkeypatch.setattr(mask_video, "read_mask_video", lambda _path: raw)
    mask = metrics.subject_mask(path, 1, 12, 16)
    expected = torch.zeros(1, 12, 16, dtype=torch.bool)
    expected[:, 2:12, 4:14] = True
    assert torch.equal(mask, expected)
    assert int(mask.sum()) == 100
    with pytest.raises(ValueError, match="does not cover"):
        metrics.subject_mask(path, 2, 12, 16)
    with pytest.raises(ValueError, match="positive"):
        metrics.subject_mask(path, 0, 12, 16)


def test_subject_measurement_selects_only_supplied_pixels() -> None:
    reference = torch.zeros(2, 3, 2, 3)
    prediction = reference.clone()
    prediction[:, :, :, 0] = 0.5
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    mask[:, :, 0] = True
    result = metrics.subject_rgb_metrics(prediction, reference, mask)
    assert result["mse"] == 0.25
    assert result["psnr"] == pytest.approx(6.020599913)
    assert result["selected_pixels"] == 4
    assert not result["exact_match"]
    mask[:, :, 0] = False
    mask[:, :, 1] = True
    result = metrics.subject_rgb_metrics(prediction, reference, mask)
    assert result["mse"] == 0
    assert result["psnr"] is None
    assert result["exact_match"]


def test_perceptual_scores_weight_frames_and_transform_input_range() -> None:
    prediction = torch.tensor([0, 0.5, 1]).view(3, 1, 1, 1).expand(3, 3, 2, 2)
    reference = torch.zeros_like(prediction)
    sizes = []

    def model(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        sizes.append(len(left))
        assert torch.equal(right, torch.full_like(right, -1))
        assert not torch.is_grad_enabled()
        return (left - right).square().mean(dim=(1, 2, 3), keepdim=True)

    assert metrics.lpips_distance(model, prediction, reference, torch.device("cpu"), batch=2) == pytest.approx(5 / 3)
    assert sizes == [2, 1]


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("empty", "matching nonempty"),
        ("nan", "finite floating pixels"),
        ("range", "finite floating pixels"),
        ("batch", "batch size must be a positive integer"),
    ],
)
def test_perceptual_invalid_inputs_fail_before_model(invalid: str, message: str) -> None:
    pixels = torch.zeros(2, 3, 2, 2)
    if invalid == "empty":
        pixels = pixels[:0]
    elif invalid == "nan":
        pixels[0, 0, 0, 0] = float("nan")
    elif invalid == "range":
        pixels[0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match=message):
        metrics.lpips_distance(
            lambda *_args: pytest.fail("invalid input reached model"),
            pixels,
            pixels,
            torch.device("cpu"),
            batch=0 if invalid == "batch" else 2,
        )
