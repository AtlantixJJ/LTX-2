from __future__ import annotations

import pytest

from scripts.prune.core import geometry, refine_task


def test_deployed_geometry_is_the_shipped_window(model):
    assert refine_task.deployed_geometry(model.scale_factors).as_dict() == {
        "window_frames": 25, "overlap_frames": 9, "latent_frames": 4,
        "context_latent_frames": 1, "chunk_latent_frames": 2,
        "stride_frames": 16, "scale_factors": [8, 32, 32],
    }


def test_plan_tiles_a_121_frame_clip_with_exact_overlap(model):
    geometry_ = refine_task.deployed_geometry(model.scale_factors)
    assert geometry_.plan(121) == [(0, 25), (16, 41), (32, 57), (48, 73), (64, 89), (80, 105), (96, 121)]
    assert all(end - start == 25 for start, end in geometry_.plan(121))


def test_plan_refuses_a_clip_shorter_than_one_window(model):
    with pytest.raises(ValueError, match="shorter than one"):
        refine_task.deployed_geometry(model.scale_factors).plan(24)


def test_calibration_geometry_round_trips_to_refine_flags(model):
    for n, frames in ((1, 17), (2, 25), (3, 33)):
        geometry_ = refine_task.calibration_geometry(n, model.scale_factors)
        assert (geometry_.window_frames, geometry_.overlap_frames) == (frames, 9)
        assert geometry_.chunk_latent_frames == n and geometry_.context_latent_frames == 1


def test_chunk1_geometry_is_scorable_but_cannot_tile(model):
    with pytest.raises(ValueError, match="to tile a clip"):
        refine_task.calibration_geometry(1, model.scale_factors).plan(121)


@pytest.mark.parametrize(("window", "overlap"), [(24, 9), (25, 10)])
def test_window_rules_reject_off_grid_values(model, window, overlap):
    with pytest.raises(SystemExit):
        geometry.check_window_rules(window, overlap, model.scale_factors)


def test_pixel_latent_round_trip(model):
    for n in (1, 2, 3, 4):
        pixel, latent = geometry.latent_shape_for(n, 512, 512, 30.0, model.scale_factors, model.caps.latent_channels)
        assert pixel.frames == 8 * (n - 1) + 1 and latent.frames == n
