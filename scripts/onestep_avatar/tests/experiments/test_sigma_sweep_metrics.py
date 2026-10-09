"""Historical boundary measurements preserve offsets, masks and meaningful undefined states."""

import json

import numpy as np
import pytest

from scripts.onestep_avatar import evaluate, metrics


def test_known_ramp_and_seams_have_exact_boundary_and_motion_definitions() -> None:
    frames = np.arange(129, dtype=np.float64)
    jumps = sum(frames >= boundary for boundary in (17, 33, 49, 65, 81, 97, 113))
    capture = np.broadcast_to((0.002 * frames)[:, None, None, None], (129, 2, 3, 3)).copy()
    video = np.broadcast_to((0.001 * frames + 0.01 * jumps)[:, None, None, None], capture.shape).copy()
    result = evaluate.sigma_sweep_boundary_metrics(video, capture, capture / 2, np.ones(capture.shape[:3], dtype=bool))
    assert [row["boundary"] for row in result["per_boundary"]] == [17, 33, 49, 65, 81, 97, 113]
    assert [row["boundary"] for row in result["per_boundary"] if row["post_eviction"]] == [81, 97, 113]
    assert all(row["ratio_to_local_interior"] == pytest.approx(11) for row in result["per_boundary"])
    assert all(row["capture_ratio"] == pytest.approx(1) for row in result["per_boundary"])
    for key in ("mean_boundary_ratio", "mean_boundary_ratio_pre_eviction", "mean_boundary_ratio_post_eviction"):
        assert result[key] == pytest.approx(11)
    assert result["interior_step"] == pytest.approx(0.001)
    assert result["motion_over_capture"] == pytest.approx(0.8125)
    assert result["drift_vs_c0_last"] == pytest.approx(0.198)
    assert len(result["per_frame_err_vs_capture"]) == 129
    assert len(result["per_transition_step"]) == 128
    assert result["per_transition_step"][16] == 0.011
    assert result["per_transition_step"][15] == 0.001


def test_transition_mask_is_union_of_adjacent_frames() -> None:
    video = np.zeros((3, 1, 2, 3), dtype=np.float32)
    video[1, 0, 0], video[1, 0, 1], video[2, 0, 1] = 1, 0.25, 0.75
    mask = np.zeros((3, 1, 2), dtype=bool)
    mask[0, 0, 0], mask[2, 0, 1] = True, True
    assert metrics.masked_rgb_transition_steps(video, mask).tolist() == [1, 0.5]


@pytest.mark.parametrize("empty_mask", [False, True])
def test_no_motion_produces_explicit_null_ratios_and_valid_json(empty_mask: bool) -> None:
    video = np.zeros((129, 1, 2, 3), dtype=np.float32)
    mask = np.full(video.shape[:3], not empty_mask, dtype=bool)
    result = evaluate.sigma_sweep_boundary_metrics(video, video, video, mask)
    assert result["motion_over_capture"] is None
    assert result["motion_ratio_status"] == "undefined_zero_capture_motion"
    assert result["mean_boundary_ratio"] is None
    assert all(
        row["ratio_to_local_interior"] is None and row["capture_ratio"] is None for row in result["per_boundary"]
    )
    assert result["interior_step"] == 0
    json.dumps(result, allow_nan=False)


def test_half_precision_pixel_sums_do_not_overflow() -> None:
    video = np.zeros((2, 256, 256, 3), dtype=np.float16)
    video[1] = 1
    assert metrics.masked_rgb_transition_steps(video, np.ones(video.shape[:3], dtype=bool)).tolist() == [1]


@pytest.mark.parametrize(
    "invalid", ["short", "geometry", "channels", "nan", "range", "integer", "mask_type", "mask_shape"]
)
def test_invalid_measurements_fail_before_scoring(invalid: str) -> None:
    video = np.zeros((129, 1, 2, 3), dtype=np.float32)
    capture = video.copy()
    mask = np.ones(video.shape[:3], dtype=bool)
    if invalid == "short":
        video = video[:128]
    if invalid == "geometry":
        capture = capture[:, :, :1]
    if invalid == "channels":
        video = video[..., :2]
    if invalid == "nan":
        video[0, 0, 0, 0] = np.nan
    if invalid == "range":
        video[0, 0, 0, 0] = 2
    if invalid == "integer":
        video = video.astype(np.uint8)
    if invalid == "mask_type":
        mask = mask.astype(np.uint8)
    if invalid == "mask_shape":
        mask = mask[:128]
    with pytest.raises(ValueError, match="sigma sweep"):
        evaluate.sigma_sweep_boundary_metrics(video, capture, capture, mask)
