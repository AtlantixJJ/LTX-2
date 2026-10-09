"""Per-frame perceptual plumbing matches the legacy scorer without loading weights or decoders."""

import pytest
import torch

from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar import metrics


def model(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """An analytic per-frame model checks range conversion and gradient suppression."""
    assert not torch.is_grad_enabled()
    return (left - right).square().mean(dim=(1, 2, 3), keepdim=True)


@pytest.mark.parametrize("frames", [1, 17, 129, 137])
def test_historical_per_frame_lists_and_c0_excluded_means_match_exactly(frames: int) -> None:
    generator = torch.Generator().manual_seed(42)
    prediction = torch.rand(frames, 5, 7, 3, generator=generator)
    capture = torch.rand(frames, 5, 7, 3, generator=generator)
    expected = []
    with torch.no_grad():
        for start in range(0, frames, 16):
            left = prediction[start:start + 16].permute(0, 3, 1, 2) * 2 - 1
            right = capture[start:start + 16].permute(0, 3, 1, 2) * 2 - 1
            expected += model(left, right).flatten().tolist()
    actual = metrics.lpips_frame_scores(
        model, prediction.permute(0, 3, 1, 2), capture.permute(0, 3, 1, 2), torch.device("cpu"))
    assert actual == expected
    assert len(actual) == frames
    if frames > 1:
        assert sum(actual[1:]) / (frames - 1) == sum(expected[1:]) / (frames - 1)


def test_order_short_batch_and_explicit_c0_exclusion() -> None:
    prediction = torch.linspace(0, 1, 19).view(19, 1, 1, 1).expand(19, 3, 2, 2)
    capture = torch.zeros_like(prediction)
    sizes = []

    def checked(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        sizes.append(len(left))
        assert torch.equal(right, torch.full_like(right, -1))
        return model(left, right)

    actual = metrics.lpips_frame_scores(checked, prediction, capture, torch.device("cpu"))
    assert sizes == [16, 3]
    assert actual[0] == 0
    assert actual[-1] == 4
    assert actual == sorted(actual)
    assert sum(actual[1:]) / 18 > sum(actual) / 19


@pytest.mark.parametrize("batch", [0, -1, True, 1.5])
def test_invalid_batch_refuses_before_model(batch: object) -> None:
    pixels = torch.zeros(2, 3, 2, 2)
    with pytest.raises(ValueError, match="positive integer"):
        metrics.lpips_frame_scores(lambda *_a: pytest.fail("invalid batch reached model"),
                                    pixels, pixels, torch.device("cpu"), batch=batch)


@pytest.mark.parametrize("defect", ["empty", "nan", "range", "shape"])
def test_invalid_pixels_refuse_before_model(defect: str) -> None:
    pixels = torch.zeros(2, 3, 2, 2)
    reference = pixels.clone()
    if defect == "empty":
        pixels, reference = pixels[:0], reference[:0]
    elif defect == "nan":
        pixels[0, 0, 0, 0] = float("nan")
    elif defect == "range":
        pixels[0, 0, 0, 0] = 2
    else:
        reference = reference[:1]
    with pytest.raises(ValueError, match=r"RGB|floating|finite|matching|aligned"):
        metrics.lpips_frame_scores(lambda *_a: pytest.fail("invalid pixels reached model"),
                                    pixels, reference, torch.device("cpu"))


@pytest.mark.parametrize("defect", ["count", "nan", "type"])
def test_invalid_model_outputs_refuse(defect: str) -> None:
    pixels = torch.zeros(2, 3, 2, 2)
    bad = torch.ones(3) if defect == "count" else torch.full((2,), float("nan")) if defect == "nan" else [0, 0]
    with pytest.raises(ValueError, match="one finite score per frame"):
        metrics.lpips_frame_scores(lambda *_a: bad, pixels, pixels, torch.device("cpu"))


def test_existing_scalar_preserves_native_batch_sum_rounding() -> None:
    pixels = torch.zeros(19, 3, 2, 2)
    values = torch.linspace(0.00001, 0.988765, 19)
    calls = []

    def fixed(left: torch.Tensor, _right: torch.Tensor) -> torch.Tensor:
        start = sum(calls)
        calls.append(len(left))
        return values[start:start + len(left)]

    expected = (float(values[:8].sum()) + float(values[8:16].sum()) + float(values[16:].sum())) / 19
    assert metrics.lpips_distance(fixed, pixels, pixels, torch.device("cpu")) == expected
    assert calls == [8, 8, 3]
