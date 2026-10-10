"""CPU-only, no data/GPU needed. Run with: python -m unittest scripts.onestep_avatar.tests.test_motion"""

from __future__ import annotations

import unittest

import numpy as np
import pytest
import torch

from scripts.onestep_avatar.corpus import motion


def _fake_pose3d(
    n_frames: int, valid: np.ndarray | None = None, raw_size_hw: tuple[float, float] = (4096.0, 3000.0)
) -> dict:
    """A minimal pose3d dict with every key `convert_frame`/`build_motion` touch."""
    if valid is None:
        valid = np.ones(n_frames, dtype=bool)
    h, w = raw_size_hw
    pose3d = {
        "valid": valid,
        "pred_pose_raw": np.zeros((n_frames, 4), dtype=np.float32),
        "shape": np.zeros((n_frames, 4), dtype=np.float32),
        "scale": np.zeros((n_frames, 4), dtype=np.float32),
        "hand": np.zeros((n_frames, 4), dtype=np.float32),
        "face": np.zeros((n_frames, 4), dtype=np.float32),
        "pred_cam": np.zeros((n_frames, 3), dtype=np.float32),
        "pred_keypoints_2d": np.zeros((n_frames, 2, 2), dtype=np.float32),
        "pred_keypoints_3d": np.zeros((n_frames, 2, 3), dtype=np.float32),
        "pred_joint_coords": np.zeros((n_frames, 2, 3), dtype=np.float32),
        "mhr_model_params": np.zeros((n_frames, 4), dtype=np.float32),
        "pred_cam_t": np.zeros((n_frames, 3), dtype=np.float32),
        "cam_rot": np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1)),
        "cam_trans": np.zeros((n_frames, 3), dtype=np.float32),
        "crop_bbox": np.zeros((n_frames, 4), dtype=np.float32),
        "K_raw": np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1)),
        "K_proc": np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1)),
        "bbox_center": np.zeros((n_frames, 2), dtype=np.float32),
        "bbox_scale": np.zeros((n_frames, 2), dtype=np.float32),
        "img_size": np.full((n_frames, 2), 512.0, dtype=np.float32),
        "ori_img_size": np.full((n_frames, 2), (w, h), dtype=np.float32),
        "affine_trans": np.zeros((n_frames, 2, 3), dtype=np.float32),
        "mask_score": np.ones(n_frames, dtype=np.float32),
        # dataset convention: raw_size stored as [W, H], the transpose of ARGAvatar's [H, W]
        "raw_size": np.full((n_frames, 2), (w, h), dtype=np.float32),
    }
    return pose3d


def _fake_bbox(n_frames: int, valid: np.ndarray | None = None, nan_at: tuple[int, ...] = ()) -> dict[str, np.ndarray]:
    if valid is None:
        valid = np.ones(n_frames, dtype=bool)
    xyxy = np.zeros((n_frames, 4), dtype=np.float32)
    for i in nan_at:
        xyxy[i, 0] = np.nan
    return {"xyxy": xyxy, "valid": valid, "score": np.ones(n_frames, dtype=np.float32)}


class TestConvertFrame(unittest.TestCase):
    def test_raw_size_is_swapped_to_h_w(self) -> None:
        pose3d = _fake_pose3d(3, raw_size_hw=(4096.0, 3000.0))  # data stores [W,H] = [3000,4096]
        entry = motion.convert_frame(pose3d, 0)
        torch.testing.assert_close(entry["raw_size"], torch.tensor([4096.0, 3000.0]))

    def test_bg_color_is_white(self) -> None:
        pose3d = _fake_pose3d(1)
        entry = motion.convert_frame(pose3d, 0)
        torch.testing.assert_close(entry["bg_color"], torch.ones(3))

    def test_person_valid_matches_dataset_valid(self) -> None:
        valid = np.array([True, False, True])
        pose3d = _fake_pose3d(3, valid=valid)
        assert motion.convert_frame(pose3d, 1)["person_valid"] is False
        assert motion.convert_frame(pose3d, 0)["person_valid"] is True

    def test_pose_keys_carried_through(self) -> None:
        pose3d = _fake_pose3d(1)
        entry = motion.convert_frame(pose3d, 0)
        for key in motion.POSE_KEEP_KEYS + motion.CACHE_KEYS:
            assert key in entry


class TestAssertRawSizeMatchesFrame(unittest.TestCase):
    def test_passes_when_consistent(self) -> None:
        pose3d = _fake_pose3d(1, raw_size_hw=(4096.0, 3000.0))
        motion.assert_raw_size_matches_frame(pose3d, frame_height=4096)

    def test_raises_on_a_transposed_convention(self) -> None:
        pose3d = _fake_pose3d(1, raw_size_hw=(4096.0, 3000.0))
        with pytest.raises(ValueError, match="raw_size convention has changed"):
            motion.assert_raw_size_matches_frame(pose3d, frame_height=3000)


class TestFillGaps(unittest.TestCase):
    def test_no_gaps_is_identity(self) -> None:
        valid = np.array([True, True, True])
        np.testing.assert_array_equal(motion._fill_gaps(valid, max_gap=3), [0, 1, 2])

    def test_interior_gap_holds_the_frame_before_it(self) -> None:
        valid = np.array([True, False, False, True])
        np.testing.assert_array_equal(motion._fill_gaps(valid, max_gap=3), [0, 0, 0, 3])

    def test_leading_gap_holds_forward_from_first_valid(self) -> None:
        valid = np.array([False, False, True, True])
        np.testing.assert_array_equal(motion._fill_gaps(valid, max_gap=3), [2, 2, 2, 3])

    def test_trailing_gap_holds_backward_from_last_valid(self) -> None:
        valid = np.array([True, True, False, False])
        np.testing.assert_array_equal(motion._fill_gaps(valid, max_gap=3), [0, 1, 1, 1])

    def test_raises_when_gap_exceeds_max(self) -> None:
        valid = np.array([True, False, False, False, False, True])
        with pytest.raises(ValueError, match="exceeds max_gap"):
            motion._fill_gaps(valid, max_gap=3)

    def test_all_invalid_raises(self) -> None:
        valid = np.array([False, False, False])
        with pytest.raises(ValueError, match="every frame"):
            motion._fill_gaps(valid, max_gap=3)


class TestBuildMotion(unittest.TestCase):
    def test_keys_are_zero_padded_and_sorted(self) -> None:
        pose3d = _fake_pose3d(3)
        sam3db = motion.build_motion(pose3d, frame_height=4096, valid=np.ones(3, dtype=bool))
        assert sorted(sam3db.keys()) == ["frames/000000.png", "frames/000001.png", "frames/000002.png"]

    def test_one_entry_per_frame(self) -> None:
        pose3d = _fake_pose3d(5)
        sam3db = motion.build_motion(pose3d, frame_height=4096, valid=np.ones(5, dtype=bool))
        assert len(sam3db) == 5

    def test_raises_when_no_valid_frames(self) -> None:
        pose3d = _fake_pose3d(2)
        with pytest.raises(ValueError, match="no valid frames"):
            motion.build_motion(pose3d, frame_height=4096, valid=np.array([False, False]))


class TestMergeMultiviewRefinement(unittest.TestCase):
    def test_replaces_only_body_trajectory_and_keeps_view_camera(self) -> None:
        view = _fake_pose3d(2)
        refinement = {key: np.asarray(view[key]).copy() for key in motion.REFINED_KEYS}
        refinement["pred_pose_raw"] += 7
        refinement["valid"] = np.array([True, False])

        merged = motion.merge_multiview_refinement(view, refinement)

        np.testing.assert_array_equal(merged["pred_pose_raw"], refinement["pred_pose_raw"])
        np.testing.assert_array_equal(merged["valid"], refinement["valid"])
        assert merged["K_raw"] is view["K_raw"]
        assert merged["cam_rot"] is view["cam_rot"]

    def test_rejects_a_differently_timed_refinement(self) -> None:
        view = _fake_pose3d(2)
        refinement = {key: np.asarray(view[key]).copy() for key in motion.REFINED_KEYS}
        refinement["shape"] = refinement["shape"][:1]

        with pytest.raises(ValueError, match="shape"):
            motion.merge_multiview_refinement(view, refinement)


class TestRepairViewCameraGaps(unittest.TestCase):
    def test_repairs_only_missing_rows_from_the_nearest_valid_view_row(self) -> None:
        pose = _fake_pose3d(4)
        pose["K_raw"][:, 0, 0] = [1, 2, np.nan, 4]
        bbox = _fake_bbox(4, valid=np.array([True, True, False, True]), nan_at=(2,))

        repaired = motion.repair_view_camera_gaps(pose, bbox)

        np.testing.assert_array_equal(repaired["K_raw"][:, 0, 0], [1, 2, 2, 4])
        np.testing.assert_array_equal(repaired["pred_pose_raw"][0], pose["pred_pose_raw"][0])

    def test_multiview_validity_can_drive_motion_after_camera_repair(self) -> None:
        pose = _fake_pose3d(4)
        pose["K_raw"][2] = np.nan
        bbox = _fake_bbox(4, valid=np.array([True, True, False, True]), nan_at=(2,))
        repaired = motion.repair_view_camera_gaps(pose, bbox)

        sam3db = motion.build_motion(repaired, frame_height=4096, valid=np.ones(4, dtype=bool))

        assert len(sam3db) == 4
        assert torch.isfinite(sam3db["frames/000002.png"]["K_raw"]).all()


if __name__ == "__main__":
    unittest.main()
