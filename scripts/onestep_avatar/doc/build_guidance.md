# `build_guidance.py` — render the guide (stage B2b)

## Objective

Produce, for one (clip, driving view) and **one objective**, the guide video the model is
conditioned on — plus the alpha and QA the rest of the pipeline needs. This is the expensive
half of corpus building (~20 min/view, GPU).

It **consumes** `precompute.py --capture-only`'s crop box and never re-derives it.

## Data flow

```
capture_latent_manifest.json ─▶ resolve_box ──────────────┐  (asserted, not recomputed)
pose3d.npy ─▶ motion.build_motion ─▶ motion.pth ─┐        │
recon views {00,02,04} frame-0 crops ─▶ reconstruct_avatar│
                                                  ▼        ▼
                              pipeline.render_motion_window(box, 1024²)
                                                  │
                                         RGBA frames (temp, full res)
                                                  │
              ┌───────────────────────────────────┼──────────────────────────────┐
              ▼                                   ▼                              ▼
   qa.mask_iou(alpha, mask.mp4)      alpha → 256² grid              composite_guide_frame(
   → IoU percentiles → sidecar       → argavatar_alpha.mp4 (lossless)            render, alpha,
                                                                      guide_background(obj))
                                                  │                    → overwrites the PNG
                                                  ▼
                                     ffmpeg crf-12 → argavatar_render[_white].mp4
                                                  + argavatar_render[_white].json
```

## Organization logic

**One pass over the RGBA frames does all three things** (IoU, alpha, composite). This is not
an optimisation, it is a correctness constraint: the full-resolution alpha exists *only*
inside this loop — only a 256² grid is persisted — so deferring the composite to a later pass
costs a full re-render. The same trap already bit the alpha itself once.

**The objectives diverge in exactly one function.** `guide_background(objective, …)` returns
what sits behind the render — frame 0 of the driving view for `bg`, a white frame for
`white` — and `composite_guide_frame` is one blend for both. An objective is a *choice of
background*, not a second guide-construction code path that could drift from the first.

**The renderer's RGB is already composited over white, never straight foreground.**
ARG-Avatar's own rasterizer (`forward.cu`: `C[ch] + T * bg_color[ch]`, with
`bg_color=torch.ones(3)`) blends over white before this module ever sees a pixel, so
`composite_guide_frame` does background *replacement* —
`guide = R_white + (1 - alpha) * (B - white)` — not a second alpha blend on top of what it
is handed. Guide contract **v2** (2026-09-18, 2026-09-18 audit finding F1); v1 treated the
input as straight foreground and blended it again, which applied alpha twice and left a
white fringe on every soft edge, including in the `white` objective, which should have been
a no-op. `dataset.GUIDE_COMPOSITING_VERSION` is stamped into every sidecar's
`compositing_version` and checked by `_render_is_complete` the same way `objective` is — a
render built under the retired formula is rebuilt, not silently trained against. Unlike
`objective`, a missing `compositing_version` is never inferred as current: every render
built before this fix used the wrong formula, so there is no legacy value to grandfather in.
**The 19 renders on disk predate this fix and are stale under it** — they were reviewed and
accepted only under the retired v1 contract; the next step in the September 18 plan is one
reviewed real bg/white pair under v2, then a rebuild of the affected guides (capture bundles
are unaffected — only the guide's own compositing changed).

**Failures are per-pair exclusions, never batch-fatal.** A pose-tracking gap or a
reconstruction failure loses that pair (or that clip's pairs), is collected, and is reported
at the end. A review batch over many actors must not die on one bad clip.

## Invariants

- **The box is asserted against the manifest's record**, not matched by equal defaults. A
  mismatched `--pad-factor` between the two runs is the one way a pair silently desyncs, and
  it happened once.
- **Use the render's alpha, never the capture's mask**, for the composite. At deployment
  there is no capture mask, so training on one breaks train/deploy parity; and the render's
  alpha is anti-aliased where the capture matte is a hard threshold off lossy video.
- **Alpha is used continuous, never thresholded** — a soft boundary rather than a hard edge
  quantised to the 32× latent cell.
- Latents are the artifact; pixels are for review. Default runs write no capture video.

**The alpha is stored as a lossless grayscale MP4**, not a raw array — 42× smaller, bit-exact,
and the soft edge survives untouched. See [mask_video.md](mask_video.md) for the measurements
and why lossy was rejected. `--migrate-alpha` re-encodes the legacy `.npy` grids (verified
bit-exact before anything is removed; `--prune-npy` deletes them). It needs no GPU and no
renderer, so it runs before every import-heavy step in `main()`.

## Resumability

`_render_is_complete` accepts a view only if its sidecar and its video agree on frame count,
geometry, alpha grid, **objective, and compositing_version**, so a killed ffmpeg never looks
finished, a render built for the other objective is rebuilt rather than trained against, and
a render built under the retired v1 compositing formula is rebuilt rather than accepted as
current. The alpha check accepts **either** storage form: a render is not stale merely for
predating the MP4 format.

A sidecar with `composited: true` and no `objective` predates the split; that *is* the `bg`
render, so it is read as one. Re-rendering the 19 views on disk for a field name would cost
a GPU-day for nothing.

## Gotchas

- **The alpha/IoU loop zips the render frames against `mask.mp4` with `strict=True`.**
  `alpha_grid` is `np.empty`, so a mask short of the render would leave its tail
  uninitialised and persist that as the view's alpha — complete-looking and wrong. A
  frame-count disagreement is a corpus defect, and raising makes it this pair's exclusion.
- The composite overwrites the render PNG **in place** — the composited frame *is* the guide,
  not a second artifact. Anything that re-reads those PNGs after this loop sees composites.
- `--visualize` writes `qa/overlay_view<D>.mp4` by cropping `rgb.mp4` on the fly; no
  persisted `capture.mp4` is needed or produced.

## Tests

`tests/test_build_guidance.py` — the v2 background-replacement blend against
renderer-faithful (already-white-composited) fixtures including the audit's exact CPU repro
numbers, the `white` objective's exact identity, `guide_background`'s white path, and the
`_render_is_complete` objective and compositing-version cases.
