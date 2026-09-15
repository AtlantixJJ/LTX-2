# `precompute.py` — one continuous VAE encode per view

## Objective

Turn corpus pixels into the master latents the trainer reads, for **one or both** objectives,
resumably. Every product lives **beside its source video**, one per view; there is no
experiment-side latent tree.

| Artifact | Written by | Is |
|---|---|---|
| `ltx_vae_latent[_white].pt` | `--capture-only` | `z_y`, the capture master |
| `argavatar_ltx_vae_latent[_white].pt` | the paired pass | `z_g`, the guide master |
| `capture_mask_crop.mp4` | the paired pass | the capture matte cropped to the box, 256², lossless |
| `loss_mask_grids.pt` | the paired pass | `render_alpha` + `capture_mask`, pooled to the latent grid |
| `capture_latent_manifest.json` | `--capture-only` | **the crop box of record** — `build_guidance.py` renders into it |

## Data flow

```
--capture-only:
  bbox.npy ─▶ square box ─▶ manifest          (the box of record, one producer)
  rgb.mp4  ─┐
  mask.mp4 ─┴─▶ crop_source (worker, CPU) ─▶ {bg: frames, white: matted frames}
                                            ─▶ ONE tiled_encode per objective
                                            ─▶ master_record ─▶ atomic save

paired:
  argavatar_render[_white].mp4 ─▶ ONE tiled_encode ─▶ z_g master
  argavatar_alpha.mp4 ─┐
  mask.mp4 ─▶ crop to the box at 256² ─▶ capture_mask_crop.mp4 (stored once)
                       └─▶ both pooled to the latent grid ─▶ loss_mask_grids.pt

--consolidate:
  v1 per-window bundle ─▶ reassembled master (no VAE, no GPU) ─▶ v2 bundle, in place
```

## Organization logic

**One continuous encode per source, and the master is what is stored.** A genuine causal
keyframe only exists at latent frame 0 of a truly continuous encode, and nothing re-keys
mid-rollout. Encoding each window independently *manufactured* a fresh keyframe at every
window's local frame 0 — measured: window 0 sliced vs. independently encoded differs ~0.1 %
(the bf16 noise floor), a mid-clip window differs ~24 %. Since the master is stored, a window
is not a unit of anything and the block geometry can change without re-encoding a source.

**Both objectives come out of ONE decode.** The expensive part is the sequential h264 decode
of a 4096×3000 source; the white matte is a per-frame blend over pixels already in hand. So
`--objective bg white` pays one decode and writes two bundles.

**Resume is a property of what is on disk, not of a progress file.** Currency is checked per
**(source, objective)** and each bundle is written by one atomic save, so a bundle is either
current or absent — there is no half-done state a restart could inherit. Consequences:

- a run killed mid-corpus resumes at whole-bundle granularity;
- adding `white` to a corpus already encoded as `bg` re-encodes only `white`;
- a source is decoded only for the objectives it is actually missing.

**The matte is applied at full resolution, before the resize**, so matte and pixels are
resampled together — the same reason the guide's composite is built while the full-resolution
alpha is still live. It is used **continuous**, not re-thresholded: the stored mask is already
a threshold off lossy video, and hardening it again would quantise the silhouette edge that
the `white` objective makes the whole task.

**The cropped capture matte is persisted, and that is a read optimisation as much as a
storage one.** Rebuilding the loss grids used to mean re-decoding a 4096×3000 `mask.mp4` at
36 MB a frame — the single most expensive read in this pass. It is now cropped to 256² once
and stored as a lossless MP4 beside the render's alpha (~0.23 MB per view; see
[mask_video.md](mask_video.md)), written from the same decode that feeds the grids, so there
is still exactly one producer.

A side effect worth knowing: **both masks now reach the latent grid by the same route** (full
res → 256 → latent). The capture matte used to be pooled straight to latent resolution while
the render's alpha went via 256, so the band the loss weights by — their difference — carried
a small resampling mismatch unrelated to alignment. Measured cost of the extra hop: mean 6e-5,
max 2e-3 of a cell's coverage, far below the mismatch it removes.

**The crop box comes from `geometry.canonical_crop_box`, not a copy of it.** This pass is its
single producer; the arithmetic was transcribed here until 2026-09-15, and the two were
verified identical over all 3,360 corpus views before the copy went. Note the module is
imported as `crop_geometry`: `geometry` is already this module's parameter name for a
`WindowGeometry` (the `k2` window plan — a different thing entirely).

**Cropping runs in worker processes, the VAE in this one.** The pool is entered *before* the
encoder so workers never fork after this process has touched CUDA.

## Invariants

- `z_y` is never re-encoded by the paired pass. The capture pass is its only producer and the
  trainer reads that bundle directly; a copy is free to disagree with its original.
- The two mask grids are stored **uncombined**. Which disagreement region the loss covers is a
  training decision (SS1.5); pre-combining here would bake one answer into the corpus.
- Every source's windows share one fixed crop box.
- `--consolidate` **refuses** disagreeing overlaps rather than stitching them: a pre-09-11
  independently-encoded bundle is not a master in disguise.

## Gotchas

- **Host RAM, not GPU, is the failure mode.** A severe leak in `encode_capture_jobs` caused
  two SIGKILLs before it was found. The fixes that matter are still in place and load-bearing:
  chunked `as_completed` (it holds its own internal set of futures, and with it the ~0.5–0.7 GB
  frame array cached on each, for the generator's whole lifetime) and
  `max_tasks_per_child=1` (numpy/cv2 do not reliably return freed heap to the OS between
  tasks in a long-lived worker).
- Asking for both objectives puts a **second** uint8 frame array in each worker (~0.5 GB at
  150 frames). `--crop-workers` is the knob if host RAM is tight.
- `_read_cropped_masks` streams one frame at a time on purpose — a 3000×4096 mask is 36 MB a
  frame, so a whole-clip `get_batch` costs ~11 GB.
- Decoding is **sequential, never seeking**. These sources have extremely sparse keyframes, so
  `CAP_PROP_POS_FRAMES` would silently re-decode from frame 0 anyway — and seeking on B-frame
  content is a known source of off-by-a-few-frames errors.

## Tests

`tests/test_precompute.py`
