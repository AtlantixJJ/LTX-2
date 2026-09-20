# `geometry.py` — the square crop rule

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

`canonical_crop_box` is **the whole rule in one call**, and since 2026-09-15 it is the *only*
copy of it. `precompute.py --process_gt_latent` calls it to compute the box it records, and
`build_guidance.py` renders into that recorded box — one producer, one arithmetic.

It used to be transcribed into `precompute.py`, with a test pinning the two spellings,
because the two modules lived in different trees and different conda envs. Consolidating the
package removed that seam; the two were verified identical over **all 3,360 corpus views × 3
canvas shapes** before the copy was deleted, so no recorded box moved. `test_geometry.py` is
now a golden test on exact box values, which is what still bites: changing the rule would
silently re-crop a corpus whose latents are already encoded.

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
