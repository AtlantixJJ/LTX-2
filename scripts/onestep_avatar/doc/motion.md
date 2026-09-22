# `motion.py` — `pose3d.npy` → ARGAvatar `sam3db`

## Objective

Convert DNARendering's per-view MHR pose+camera export into the motion-file format
ARGAvatar's renderer consumes, **without re-deriving anything**. `POSE_KEEP_KEYS` and
`CACHE_KEYS` are copied verbatim from ARGAvatar's own `infer_argavatar.build_sam3db_entry`
so the two cannot silently diverge.

## Data flow

`pose3d.npy[view]` + `refined_pose3d.npy[clip]` → `merge_multiview_refinement` →
`build_motion(pose3d, bbox, frame_height)` → a `sam3db` dict →
`torch.save` → `motion.pth` → `pipeline.render_motion_window`.

Consumed only by `build_guidance.render_pair`, into a temp file that never outlives the
render.

## Organization logic

Only **three** things need fixing, and the file is organized around naming them explicitly
rather than around a general-purpose converter:

1. **`raw_size` is `[W, H]` here but ARGAvatar reads `[H, W]`.** Left as-is, the render is
   built for a transposed canvas and `full_K` normalisation is wrong with it. The fix asserts
   the swapped result against the video's own decoded height rather than trusting the array.
2. / 3. The other two conversions are documented inline at their call sites.

A conversion that is "three known fixes" should read as three known fixes. Anything more
general would hide which parts are load-bearing.

The default guide path uses the clip-level multiview refinement. It replaces only
`pred_pose_raw`, `shape`, `scale`, `hand`, `face`, and `valid`; per-view camera, crop, and
image-cache fields stay with the driving view. The two records must have identical shapes for
each replaced value. This preserves calibrated camera projection while giving every view the
same refined whole-sequence body motion.

When a view's detector missed frames, its per-view export has NaNs for every camera/cache
field even though the multiview body trajectory is valid. The refined path repairs those
view-local rows in memory by holding the nearest finite row, then applies the refined body
values at every original frame. Its motion validity is the multiview `valid` array; the legacy
per-view path still requires a valid finite bbox. The manifest crop box remains fixed and is
never rebuilt from these repaired rows.

## Gotchas

- **Pose-tracking gaps are a data-quality gate, not an error.** `build_motion` refuses a
  clip whose tracking has holes; `build_guidance` treats that as a per-pair exclusion and
  keeps going. Five attempted pairs were correctly excluded this way in the T2 batch.

## Tests

`tests/test_motion.py`
