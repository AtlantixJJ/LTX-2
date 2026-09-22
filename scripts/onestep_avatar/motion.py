"""``pose3d.npy`` (DNARendering's per-view MHR pose+camera export) -> ARGAvatar's ``sam3db``
motion-file format.

Per ``plans/2026-09-05-dnarendering-argavatar-refine-corpus.md`` SS3: ``pose3d.npy``'s keys are
already a complete match for what ARGAvatar's own ``infer_argavatar.build_sam3db_entry`` writes
when it processes a subject image -- ``POSE_KEEP_KEYS`` and ``CACHE_KEYS`` below are copied
verbatim from that function so the two cannot silently diverge. Only three things need fixing
(SS3.1), not re-deriving:

1. ``raw_size`` is stored ``[W, H]`` here (this dataset's convention, same as ``ori_img_size``)
   but ARGAvatar reads it as ``[H, W]`` (``xlib.train.train_helper.build_gs_camera``). Left
   as-is the render would be built for a transposed canvas and ``full_K`` normalisation would
   be wrong with it. Swap it, and assert the result against the video's own decoded height
   rather than trust the source array.
2. ``person_valid`` is this dataset's ``valid``.
3. ``bg_color`` is not in the dataset at all -- emit white, matching
   ``_render_posed_gs_rgba``'s hardcoded white background.

And the pose is fitted under the real calibrated camera (``cameras.npy["K"] ==
pose3d["K_raw"]``, elementwise), so no camera solve is needed to pair a render with its
capture -- ``K_raw`` already is the target video's own camera.
"""

from __future__ import annotations

import numpy as np
import torch

# Copied from ARG-Avatar `scripts/inference/infer_argavatar.py`'s `POSE_KEEP_KEYS` and
# `build_sam3db_entry`'s `cache_map` (minus `raw_size`/`person_valid`/`bg_color`, handled
# separately below because they need a fix rather than a straight copy).
POSE_KEEP_KEYS = (
    "pred_pose_raw", "shape", "scale", "hand", "face", "pred_cam",
    "pred_keypoints_2d", "pred_keypoints_3d", "pred_joint_coords",
    "mhr_model_params", "pred_cam_t", "cam_rot", "cam_trans",
)
CACHE_KEYS = (
    "crop_bbox", "K_raw", "K_proc", "bbox_center", "bbox_scale",
    "img_size", "ori_img_size", "affine_trans", "mask_score",
)

# The clip-level multiview solve deliberately owns only these camera-independent
# parameters.  Camera and image-cache fields remain tied to the driving view.
REFINED_KEYS = ("pred_pose_raw", "shape", "scale", "hand", "face", "valid")


def merge_multiview_refinement(view_pose3d: dict, refinement: dict) -> dict:
    """Overlay a clip's refined body trajectory onto one view's camera record.

    ``refined_pose3d.npy`` cannot itself be rendered: it has no calibrated camera,
    crop, or image-size fields.  Keep those fields from ``view_pose3d`` and replace
    exactly the body parameters produced by the multiview fit.  Shape checks make a
    partial or differently-timed solve fail before ARGAvatar renders a plausible but
    wrong video.
    """
    merged = dict(view_pose3d)
    for key in REFINED_KEYS:
        if key not in refinement:
            raise KeyError(f"refined_pose3d.npy is missing required key {key!r}")
        if key not in view_pose3d:
            raise KeyError(f"view pose3d.npy is missing required key {key!r}")
        if np.asarray(refinement[key]).shape != np.asarray(view_pose3d[key]).shape:
            raise ValueError(
                f"refined_pose3d.npy[{key!r}] has shape {np.asarray(refinement[key]).shape}, "
                f"but the driving view has {np.asarray(view_pose3d[key]).shape}"
            )
        merged[key] = refinement[key]
    return merged


def convert_frame(pose3d: dict, frame_idx: int) -> dict:
    """One ``pose3d.npy`` frame -> one ``sam3db`` entry. Pure, CPU-only, no gating."""
    entry = {}
    for key in POSE_KEEP_KEYS + CACHE_KEYS:
        entry[key] = torch.as_tensor(np.asarray(pose3d[key][frame_idx]), dtype=torch.float32)
    raw_size = np.asarray(pose3d["raw_size"][frame_idx], dtype=np.float32)
    entry["raw_size"] = torch.as_tensor([raw_size[1], raw_size[0]], dtype=torch.float32)  # [W,H]->[H,W]
    entry["person_valid"] = bool(pose3d["valid"][frame_idx])
    entry["bg_color"] = torch.ones(3, dtype=torch.float32)
    return entry


def assert_raw_size_matches_frame(pose3d: dict, frame_height: int, frame_idx: int = 0) -> None:
    """Raise if the swapped ``raw_size`` doesn't match the video's real decoded height.

    A silent transpose bug here corrupts every camera derived from this clip/view (SS3.1) --
    check it against ffprobe/decord's own report of the height, never trust the source array.
    """
    w, h = (float(v) for v in np.asarray(pose3d["raw_size"][frame_idx]))
    if int(round(h)) != int(frame_height):
        raise ValueError(
            f"pose3d['raw_size'][{frame_idx}] = [{w}, {h}] (assumed [W, H]) implies a height "
            f"of {h}, but the video's own decoded height is {frame_height} -- the raw_size "
            f"convention has changed; do not blindly swap it"
        )


def valid_frame_mask(pose3d: dict, bbox: dict) -> np.ndarray:
    """Frames usable for motion conversion: ``pose3d.valid`` AND a valid, finite bbox.

    Neither dataset's validity flag is a superset of the other's, so both are required --
    see SS2 hazard 1 (NaN bboxes can coincide with ``valid == True``... or not).
    """
    valid = np.asarray(pose3d["valid"], dtype=bool) & np.asarray(bbox["valid"], dtype=bool)
    valid &= ~np.isnan(bbox["xyxy"]).any(axis=1)
    return valid


def repair_view_camera_gaps(view_pose3d: dict, bbox: dict) -> dict:
    """Fill missing view-local camera/cache rows without changing valid source rows.

    A multiview refinement supplies the body trajectory for every frame, but its companion
    per-view MHR export can still be absent when that view's detector missed the subject.
    Those misses make *all* camera/cache arrays NaN.  Copy the nearest valid view-local row
    into only those holes; `merge_multiview_refinement` then restores the refined body values
    at their original frame indices.  This is deliberately in-memory: raw corpus evidence is
    never rewritten.
    """
    valid = np.asarray(bbox["valid"], dtype=bool)
    valid &= ~np.isnan(bbox["xyxy"]).any(axis=1)
    if not valid.any():
        raise ValueError("no finite bbox rows available to repair view-local camera fields")
    source_idx = _fill_gaps(valid, max_gap=None)
    repaired = dict(view_pose3d)
    for key, value in view_pose3d.items():
        array = np.asarray(value)
        if array.ndim and array.shape[0] == len(valid):
            copied = array.copy()
            copied[~valid] = array[source_idx[~valid]]
            repaired[key] = copied
    return repaired


def _fill_gaps(valid: np.ndarray, max_gap: int | None) -> np.ndarray:
    """For each frame, the index of the nearest valid frame to source its pose from.

    Holding the nearest valid frame's pose across a short gap is a much smaller error than
    dropping the frame and shifting every later frame's timing, which would desync the motion
    file from the target video it must line up with pixel-for-pixel. Raises if any contiguous
    invalid run exceeds ``max_gap`` -- past that point hold-last stops being a reasonable
    approximation and the clip/view should be excluded instead of silently faked.
    """
    n = len(valid)
    source_idx = np.arange(n)
    run_start = None
    for i in list(range(n)) + [n]:
        frame_valid = valid[i] if i < n else True  # sentinel to flush a trailing run
        if frame_valid:
            if run_start is not None:
                run_len = i - run_start
                if max_gap is not None and run_len > max_gap:
                    raise ValueError(
                        f"invalid-frame run of {run_len} frames at [{run_start}, {i}) exceeds "
                        f"max_gap={max_gap}"
                    )
                hold = run_start - 1 if run_start > 0 else i if i < n else None
                if hold is None:
                    raise ValueError("every frame in this clip/view is invalid")
                source_idx[run_start:i] = hold
                run_start = None
        elif run_start is None:
            run_start = i
    return source_idx


def build_motion(
    pose3d: dict, bbox: dict, frame_height: int, max_gap: int = 3, valid: np.ndarray | None = None,
) -> dict:
    """The full-clip ``sam3db`` dict, keyed ``f"frames/{i:06d}.png"`` (``load_driving_motion``
    consumes ``sorted(sam3db.keys())``, so zero-padding is load-bearing), one entry per frame.
    """
    assert_raw_size_matches_frame(pose3d, frame_height)
    usable = valid_frame_mask(pose3d, bbox) if valid is None else np.asarray(valid, dtype=bool)
    if not usable.any():
        raise ValueError("no valid frames in this clip/view -- nothing to build a motion file from")
    source_idx = _fill_gaps(usable, max_gap)
    return {
        f"frames/{i:06d}.png": convert_frame(pose3d, int(source_idx[i]))
        for i in range(len(usable))
    }
