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

### Convert one frame

Copy the pose and camera/cache fields named by `POSE_KEEP_KEYS` and `CACHE_KEYS` into CPU float32 tensors.
Keep the calibrated camera fields from the driving view; do not solve a new camera.
Apply these three conversions:

1. Change `raw_size` from dataset `[width,height]` to ARGAvatar `[height,width]`.
   Check the stored height against the actual decoded video height before conversion.
2. Write `person_valid` as the boolean value of the source frame's `valid` field.
3. Write `bg_color=[1,1,1]` in float32, which is the renderer's white background.

The converter does not composite a capture background here.
`build_guidance.py` owns that later pixel operation.

### Refined body and view-local camera

The default guide path uses the clip-level multiview refinement. It replaces only
`pred_pose_raw`, `shape`, `scale`, `hand`, `face`, and `valid`; per-view camera, crop, and
image-cache fields stay with the driving view. The two records must have identical shapes for
each replaced value. This preserves calibrated camera projection while giving every view the
same refined whole-sequence body motion.

For view detector gaps, `repair_view_camera_gaps` first selects rows with a valid bbox and no NaN coordinate.
Copy an earlier valid row over each internal/trailing invalid run.
For a leading run, copy the first following valid row.
Repair only in memory and retain all original frame indices.
Then replace the six refined body fields at their original indices.
This leaves view-local camera/cache values from the repaired view and body motion from the refinement.
No valid source row or raw corpus file is overwritten.
No available camera row is an error.

### Preserve motion timing

For the legacy view-only path, usable frames require both pose and bbox validity, with no NaN bbox coordinate.
For the refined path, use the supplied multiview validity array after camera repair.
`build_motion` uses the same earlier-row/leading-row hold rule for short unusable motion runs.
Reject a run longer than `max_gap` (default three frames), or a clip with no usable frame.
Write one entry per original frame under `frames/{index:06d}.png`.
Zero-padded names sort in playback order. Do not drop bad frames and shift later timing.
Camera repair itself has no maximum-gap limit; motion validity still follows the body record.
The manifest crop remains fixed and is never rebuilt from repaired rows.

## Invariants

- Camera and image-cache fields stay with the driving view.
- Refined body replacement requires equal field shapes and frame timing.
- The conversion preserves frame count and original order.
- Raw size is checked against the video, not inferred from a tensor shape.
- Gap repair changes in-memory rows only.

## Gotchas

- **Pose-tracking gaps are a data-quality gate, not an error.** `build_motion` refuses a
  clip whose tracking has holes; `build_guidance` treats that as a per-pair exclusion and
  keeps going. Five attempted pairs were correctly excluded this way in the T2 batch.

## Tests

`tests/test_motion.py`
Worked conversion check: dataset `raw_size=[1920,1080]` with actual video height 1080
becomes `[1080,1920]`; `valid=True` becomes `person_valid=True`; background is `[1,1,1]`.
Worked gap check: validity `[False,True,False,False,True]` selects source indices `[1,1,1,1,4]`.
The output still contains five entries. A four-frame invalid motion run exceeds the default limit and fails.
