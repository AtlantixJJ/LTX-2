# `motion.py` — `pose3d.npy` → ARGAvatar `sam3db`

## Objective

Convert DNARendering's per-view MHR pose+camera export into the motion-file format
ARGAvatar's renderer consumes, **without re-deriving anything**. `POSE_KEEP_KEYS` and
`CACHE_KEYS` are copied verbatim from ARGAvatar's own `infer_argavatar.build_sam3db_entry`
so the two cannot silently diverge.

## Data flow

`pose3d.npy[view]` → `build_motion(pose3d, bbox, frame_height)` → a `sam3db` dict →
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

## Gotchas

- **Pose-tracking gaps are a data-quality gate, not an error.** `build_motion` refuses a
  clip whose tracking has holes; `build_guidance` treats that as a per-pair exclusion and
  keeps going. Five attempted pairs were correctly excluded this way in the T2 batch.

## Tests

`tests/test_motion.py`
