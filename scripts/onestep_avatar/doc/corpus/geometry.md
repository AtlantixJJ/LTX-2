# `corpus/geometry.py` — the square crop rule

## Objective

Give the crop geometry as one callable rule: **one square box for each (clip,
view), fixed for the whole clip**. The box contains the subject in every frame,
and it is resized to 1024².

A fixed box is not a simplification. A different box for each frame would add
camera motion that the capture never had, and the temporal RoPE would learn it.

## Data flow

1. Read `bbox.npy`: a per-frame xyxy box and a valid flag.
2. Take the union over the valid frames.
3. Make a centered square of side `max(w, h) × pad_factor`.
4. `fit_square_to_canvas` shifts the square, then caps it.
5. `build_guidance` and `precompute` read the resulting `XYXY`.

The module is pure numpy: no GPU, no torch and no dataset access. This is
deliberate. It makes the rule testable without the corpus, so the same
arithmetic can be pinned in two conda environments.

## Organization logic

### Core crop calculation

1. Select valid bbox rows with no NaN coordinate. Reject an empty selection.
2. Form their union: minimum left/top and maximum right/bottom across the complete video.
3. Let union width/height be `w,h` and center be `cx,cy`.
   Request a square side `max(w,h)*pad_factor`, centered at that point.
4. Set fitted side `s=min(round(requested_side), canvas_width, canvas_height)`.
5. Set left to `min(max(round(cx-s/2),0), canvas_width-s)`.
   Apply the same rule to top with the canvas height.
6. Return `[left,top,left+s,top+s]` and crop only pixels inside the original canvas.
7. Calculate effective padding `s/max(w,h)`.
   A value below one means the subject union does not fit; the corpus caller excludes that view.

Use Python's rounding rule. Recorded boxes depend on this exact arithmetic.
`canonical_crop_box` calculates/checks geometry; capture precompute is the producer of the saved box.
Guide generation reads that saved box and reports a disagreement rather than choosing another crop.

`canonical_crop_box` is **the whole rule in one call**, and it is the only copy
of the rule. `precompute.py --process_gt_latent` calls it to compute the box
that it records. `build_guidance.py` renders into that recorded box. So there is
one producer and one arithmetic. `test_geometry.py` is a golden test of exact
box values. A change to the rule would crop again, without an error, a corpus
whose latents are already encoded.

`crop_from_canvas` crops a fitted box and never pads. Capture targets never get
synthetic pixels: shift and cap, then exclude a subject that does not fit.

## Invariants

- **Shift-and-cap, never white-pad.** No synthetic pixels are ever invented for a loss
  target. A subject union wider than the canvas is flagged (`effective_pad_factor < 1.0`) and
  **excluded**, not silently encoded — 15 of 3,360 views corpus-wide.
- **1024 % 32 == 0**, the LTX VAE requirement. There is no edge sweep; 1024² is the geometry.
- The box is computed from **valid, finite** bboxes only.

## Gotchas

- `effective_pad_factor` distinguishes two very different outcomes that look alike: padding
  *capped by the canvas* (12.9 % of views, harmless) versus the *subject itself* not fitting
  (0.4 %, must be excluded). Only the second is a data-quality exclusion.
- The render must map the same box through the intrinsics (`fx,fy *= s; cx -= x0; cy -= y0;
  cx,cy *= s`) while leaving `raw_size`/`K_raw` at the full frame, or ARGAvatar's `full_K`
  normalisation is wrong. That conversion lives in `build_guidance.py`, not here.

## Tests

`tests/test_geometry.py` — golden tests on exact box values. `precompute._capture_box` calls
`canonical_crop_box`, so the two cannot disagree.
Worked check: canvas `100x80`, union `[0,20,40,60]`, and padding `1.2`
request `[-4,16,44,64]`. Fitting returns `[0,16,48,64]` with no synthetic pixels.
A 90-pixel-wide union on that canvas gets square side 80 and effective padding `80/90 < 1`.
That view is excluded rather than accepted as a clipped training target.
