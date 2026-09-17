# `mask_video.py` — the mask storage format

## Objective

Persist a coverage mask as a **lossless** grayscale MP4 instead of a raw array. Two masks use
it: the render's own alpha (`argavatar_alpha.mp4`) and the capture matte cropped to the same
box (`capture_mask_crop.mp4`), both at 256².

## Why, in one table

Measured on real corpus alphas (256², uint8):

| | 150 frames | 225 frames |
|---|---|---|
| raw `.npy` | 9.83 MB | 14.75 MB |
| lossless gray MP4 | **0.232 MB** | **0.361 MB** |
| | 42× | 41× |

The comparison that makes the point: the dataset's own `mask.mp4`, at **4096×3000**, is
330–570 KB — a mask 180× larger in pixels than our 256² grid, in a fraction of the space. The
raw form was never justified; at corpus scale it is tens of GB of almost entirely flat data.

## Lossless, and not negotiable

Lossy settings were measured and rejected:

| | size | max error | mean error on the soft edge |
|---|---|---|---|
| lossless (crf 0) | 0.232 MB | **0** | **0** |
| crf 12 | 0.136 MB | 58 | 3.20 |
| crf 18 | 0.085 MB | 60 | 6.74 |

crf 12 buys 1.7× over lossless and pays for it on **exactly** the 1.4 % of pixels that carry
the anti-aliased silhouette edge — the part the composite's smooth boundary and the latent
coverage both come from. And masks here are already one generation of lossy video away from
the truth (the capture matte is a hard threshold off h264 — risk 8); §1.7's rule is that this
pipeline does not add a second. 42× for free beats 72× for a corrupted edge.

## Why MP4 rather than `.npz` or FFV1

It reads through the same OpenCV path every other video here uses — no new dependency, no new
failure mode — plays in any viewer for a review pass, and decodes frame by frame rather than
materializing the whole clip.

## Data flow

```
build_guidance.py   alpha_grid [N,256,256] uint8 ─▶ write_mask_video ─▶ argavatar_alpha.mp4
precompute.py       cropped matte [N,256,256]    ─▶ write_mask_video ─▶ capture_mask_crop.mp4
train.py / stats.py read_mask(stem) ─▶ pool spatially + over causal frame groups in memory
```

## Organization logic

**`MASK_ENCODE_ARGS` is asserted by a test, not just its effect.** The module was transcribed
into both of the package's former trees (different conda envs, no shared import); consolidating
removed the copy, but the risk it guarded did not go away — a drift to a lossy `crf` while
"tuning storage" would be invisible in every downstream number, so the setting itself is
pinned.

**`read_mask(stem)` takes a suffix-less path** and prefers the MP4, falling back to a legacy
`.npy`. That fallback is what lets renders predating the format keep working untouched — a
`.npy` is the same array, just 42× larger, and rebuilding one costs a full re-render.

**Writes are atomic** (temp file, then rename), for the same reason every other artifact here
is: a killed ffmpeg must not leave something that looks complete.

**Latent grids are derived, never stored.** `pool_to_latent_grid` applies area pooling and the
causal VAE's `1, 8, 8, ...` temporal grouping when a reader asks for coverage. This keeps the
MP4s as the only mask artifacts and permits a geometry change without mask regeneration.

## Migration

`build_guidance.py --migrate-alpha` re-encodes legacy `.npy` grids, **verifying each round
trip is bit-exact before counting it**; `--prune-npy` deletes the originals only after that
check passes. It needs no GPU, no renderer and no ARGAvatar import, so it runs anywhere.

Run on the 19 views on disk: **240 MB → 5.9 MB**, 0 failures.

## Tests

`tests/test_mask_video.py` — bit-exactness, the soft edge specifically, frame-count round trip,
the encode setting itself, the compression floor, and both legacy-fallback behaviours.
