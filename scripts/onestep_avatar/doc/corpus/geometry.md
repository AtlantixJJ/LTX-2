# `corpus/geometry.py` — the square crop rule

## Objective

SS1.7's crop geometry, as one callable rule: **one square box per (clip, view), fixed for the
whole clip**, sized to contain the subject across every frame, resized to 1024².

Fixed-for-the-clip is not a simplification. A per-frame rect would inject camera motion the
capture never had, and the temporal RoPE would learn it.

## Data flow

`bbox.npy` (per-frame xyxy + valid) → union over the valid frames → a square of
`max(w, h) × pad_factor`, centred → `fit_square_to_canvas` (shift, then cap) → the `XYXY`
consumed by `build_guidance` and `precompute`.

Pure numpy: no GPU, no torch, no dataset access. That is deliberate — it makes the rule
unit-testable without the corpus, which is what lets the same arithmetic be pinned across
two conda envs.

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

Use Python's existing rounding rule. Keep exactly the same arithmetic for old recorded boxes.
`canonical_crop_box` calculates/checks geometry; capture precompute is the producer of the saved box.
Guide generation reads that saved box and reports a disagreement rather than choosing another crop.

`canonical_crop_box` is **the whole rule in one call**, and since 2026-09-15 it is the *only*
copy of it. `precompute.py --process_gt_latent` calls it to compute the box it records, and
`build_guidance.py` renders into that recorded box — one producer, one arithmetic.

It used to be transcribed into `precompute.py`, with a test pinning the two spellings,
because the two modules lived in different trees and different conda envs. Consolidating the
package removed that seam; the two were verified identical over **all 3,360 corpus views × 3
canvas shapes** before the copy was deleted, so no recorded box moved. `test_geometry.py` is
now a golden test on exact box values, which is what still bites: changing the rule would
silently re-crop a corpus whose latents are already encoded.

The legacy `crop_with_padding` helper can still fill out-of-frame areas for older callers.
It is not the rule for producing capture training targets.
Inventory callers before removing it; never replace shift-and-cap with synthetic target pixels.

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

`tests/test_geometry.py` — golden tests on exact box values. There is no longer a pin against
`precompute.py`'s arithmetic: `precompute._capture_box` calls `canonical_crop_box`, so the two
cannot disagree and a test comparing them would be tautological.
Worked check: canvas `100x80`, union `[0,20,40,60]`, and padding `1.2`
request `[-4,16,44,64]`. Fitting returns `[0,16,48,64]` with no synthetic pixels.
A 90-pixel-wide union on that canvas gets square side 80 and effective padding `80/90 < 1`.
That view is excluded rather than accepted as a clipped training target.
