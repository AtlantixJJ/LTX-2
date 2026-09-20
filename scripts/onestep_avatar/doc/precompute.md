# `precompute.py` — one continuous VAE encode per view

## Objective

Turn corpus pixels into the master latents the trainer reads, for **one or both** objectives,
resumably. Every product lives **beside its source video**, one per view; there is no
experiment-side latent tree.

| Artifact | Written by | Is |
|---|---|---|
| `ltx_vae_latent[_white].pt` | `--process_gt_latent` | `z_y`, the capture master |
| `argavatar_ltx_vae_latent[_white].pt` | `--process_syn_latent` | `z_g`, the guide master |
| `argavatar_alpha.mp4` | `build_guidance.py` | the rendered alpha, 256², lossless |
| `capture_mask_crop.mp4` | `--process_syn_latent` | the capture matte cropped to the same box, 256², lossless |
| `<subject>/qa/capture_mask_crop.mp4` | `--process_gt_latent` | review crop for view 0 of the first five subjects in each `Part_*` only |
| `capture_latent_manifest.json` | `--process_gt_latent` | **the crop box of record** — `build_guidance.py` renders into it |
| `manifest[.white].json` | `--process_syn_latent` | paired-run provenance manifest at the corpus root (default: `--corpus-root`) |

There is deliberately no `loss_mask_grids.pt`. The two MP4s are the canonical masks;
`stats.py` pools them to the active latent geometry for QA/measurement. `train.py` reads
the latent masters only and computes unweighted full-frame MSE. The white capture target
already incorporates the source matte before VAE encoding.

## Data flow

```
--process_gt_latent:
  bbox.npy ─▶ square box ─▶ manifest          (the box of record, one producer)
  rgb.mp4  ─┐
  mask.mp4 ─┴─▶ crop_source (worker, CPU) ─▶ {bg: frames, white: matted frames}
                                            ─▶ ONE tiled_encode per objective
                                            ─▶ master_record ─▶ atomic save
  mask.mp4 ─▶ sampled QA crop (first 5 / part, view 0; `--mask-qa-only` skips the VAE)

--process_syn_latent (paired):
  argavatar_render[_white].mp4 ─▶ ONE tiled_encode ─▶ z_g master
  argavatar_alpha.mp4 ──────────────────────────────────────────┐
  mask.mp4 ─▶ crop to the box at 256² ─▶ capture_mask_crop.mp4 ─┴─▶ readers pool on demand
```

## Organization logic

**One continuous encode per source, and the master is what is stored.** “Source” here means
the planned prefix through the last complete fixed-stride window; trailing frames that cannot
form a complete window are not encoded. A genuine causal
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

“Current” is an explicit provenance check, not “a tensor with the expected shape.” A bundle
must match `ENCODE_CONTRACT_VERSION`, source, objective, input fingerprint (RGB, plus the matte for
`white`), VAE fingerprint, crop box, pixel/latent frame counts, FPS, edge, channels and spatial
scale. A bundle missing that metadata or disagreeing on any field is atomically regenerated.

**Crop boxes are read through `dataset.CaptureManifest.load(root).boxes`, not a second
parser.** `discover_pairs` and the paired-encode loop carried their own `manifest_boxes`, a
byte-for-byte second reader of `capture_latent_manifest.json`, from when the two halves lived
in different trees and conda envs; S1(8) of the 2026-09-17 cleanup plan retired it.

**Multi-GPU ownership is `items[rank::n_rank]`.** Discovery and ordering happen before the
slice, and capture mode slices whole sources rather than windows. Therefore every window and
both requested objectives for one source stay on exactly one rank. Paired mode applies the
same rule to its sorted pair list. All ranks write the same full-corpus manifest through
PID-unique temporary files; identical atomic replacements are safe, whereas rank-local
manifests would omit other ranks' crop boxes. Rank 0 alone writes the sampled QA gallery.

For four GPUs, launch four processes with unique ranks (the `run_b2a.sh` wrapper was retired;
call the module directly, and detach a long run with `setsid` so a signal to the shell does not
take it down):

```bash
for rank in 0 1 2 3; do
  setsid -f conda run -n ltx python -m scripts.onestep_avatar.precompute \
    --process_gt_latent --objective bg white --gpu-id "$rank" --rank "$rank" --n_rank 4 \
    </dev/null >/dev/null 2>&1 &
done
```

**The matte is applied at full resolution, before the resize**, so matte and pixels are
resampled together — the same reason the guide's composite is built while the full-resolution
alpha is still live. It is used **continuous**, not re-thresholded: the stored mask is already
a threshold off lossy video, and hardening it again would quantise the silhouette edge that
the `white` objective makes the whole task.

**The two mask MP4s are the only stored mask representations.** The capture matte is cropped
to 256² once and stored losslessly beside the render alpha (~0.23 MB per view; see
[mask_video.md](mask_video.md)). Persisting a latent-grid `.pt` duplicated these masks and
bound them to one geometry. Readers instead pool both MP4s by the same spatial and causal
temporal rules when a latent grid is needed.

**Mask visualization is deliberately sampled.** The canonical per-view MP4 remains beside
each paired view for training. A human-review copy is written to
`<Part_*>/<subject>/qa/capture_mask_crop.mp4` only for view 0 of the lexicographically first
five subject directories in each `Part_*`. No QA copy is written for other subjects or views.

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
- The two masks remain **uncombined MP4s**. Which disagreement region the loss covers is a
  training decision (SS1.5); pre-combining or persisting a derived grid would bake one answer
  and one geometry into the corpus.
- Every source's windows share one fixed crop box.
- **The on-disk plan cache (`enumerate_capture_jobs`, `.capture_plan_cache.json`) is keyed on
  BOTH `rgb_fingerprint` and `bbox_fingerprint`** (F5, 2026-09-18 audit fix). `plan_source`
  computes the box from `bbox.npy`, so keying on the RGB file's fingerprint alone (the
  original design) let a corpus re-ingest that corrected a bbox with the RGB file untouched
  keep serving the OLD cached box indefinitely — silently, since nothing else re-derives it.
  A cache entry from before this field existed has no `bbox_fingerprint` and therefore always
  misses (replanned once, then cached again), the same no-legacy-value rule as
  `dataset.GUIDE_COMPOSITING_VERSION`.
- A live multi-GPU run uses every rank in `[0, n_rank)` exactly once. Duplicate ranks would
  duplicate work even though atomic writes prevent partial bundles.
- **`capture_latent_manifest.json` validation and direct overwrite**: `--process_gt_latent` always
  processes all available views (the `--views` CLI argument has been removed). When an existing
  `capture_latent_manifest.json` is present at `--corpus-root`, its validity is checked via
  `is_capture_manifest_valid` against the current run's `geometry`, `resolution`, and `pad_factor`.
  If the existing manifest is valid and `--overwrite` is not requested, the manifest is overwritten
  directly without forcing bundle re-encoding (unmodified bundles are skipped). If the existing
  manifest is invalid (or if `--overwrite` is passed), `overwrite` is set to `True`, replacing the
  manifest and re-encoding all capture bundles. Still open: the write happens before
  `encode_capture_jobs` confirms the bundles it describes actually exist/match (so a killed or
  partially-failed run can still publish a box ahead of its bundle).
- Obsolete per-window latent bundles are regenerated; `precompute.py` no longer carries a
  migration path for them.

## Gotchas

- **Host RAM, not GPU, is the failure mode.** A severe leak in `encode_capture_jobs` caused
  two SIGKILLs before it was found. The fixes that matter are still in place and load-bearing:
  chunked `as_completed` (it holds its own internal set of futures, and with it the ~0.5–0.7 GB
  frame array cached on each, for the generator's whole lifetime) and
  `max_tasks_per_child=1` (numpy/cv2 do not reliably return freed heap to the OS between
  tasks in a long-lived worker).
- Asking for both objectives puts a **second** uint8 frame array in each worker (~0.5 GB at
  150 frames). `--crop-workers` is the knob if host RAM is tight.
- `--limit` is applied after rank sharding and counts whole sources/views, never windows.
- `_read_cropped_masks` streams one frame at a time on purpose — a 3000×4096 mask is 36 MB a
  frame, so a whole-clip `get_batch` costs ~11 GB.
- The **corpus decode paths are sequential, never seeking** (`crop_source`,
  `_read_cropped_masks`). These sources have extremely sparse keyframes, so
  `CAP_PROP_POS_FRAMES` would silently re-decode from frame 0 anyway — and seeking on B-frame
  content is a known source of off-by-a-few-frames errors. `VideoReader.get_batch` *does* call
  `CAP_PROP_POS_FRAMES`, and is safe only because every caller here starts its range at frame
  0 (`plan_source`/`_video_info` take frame 0; `encode_pairs` takes `range(pixel_frames)`).
  Asking it for a mid-clip range would reintroduce exactly the seek this rule forbids.

## Tests

`tests/test_precompute.py`
