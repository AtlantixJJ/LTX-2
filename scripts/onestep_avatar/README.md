# `onestep_avatar` — the one-step LTX-2.5 avatar renderer pipeline

Implements `plans/2026-09-15-ltx25-one-step-argavatar-lora-core.md` (design, file structure,
next steps; `plans/2026-09-10-...md` is the long-form original). Read the plan for *why*; this
file is the run order and the things that will bite you.

**One package, two conda envs.** The ARGAvatar renderer and the LTX VAE cannot share a
process, but that is a *runtime* constraint, not a layout one — exactly one module needs
ARGAvatar. Every command below runs from the **LTX-2 repo root**:

| env | runs |
|---|---|
| `argavatar` | **`build_guidance.py` only** (and `run_b2b.sh`, which wraps it) |
| `ltx` | everything else, including `windows.py` and the tests |

*(Until 2026-09-15 the corpus half lived at the workspace root in its own tree. Consolidating
removed four transcribed copies of shared knowledge — artifact names, the crop box, the block
plan, the mask codec — and the tests that existed only to pin them together.)*

Seven files are the pipeline's internal contract, and each has exactly **one producer**. Five of
the seven live **beside the source video**, one set per view: since the causal rewrite
(2026-09-14) a training sample is a span of one clip's continuous encode, so there is no
per-window latent tree and `expr/onestep_avatar/precomputed/` is no longer produced or read.

| file | produced by | consumed by |
|---|---|---|
| `<corpus>/capture_latent_manifest.json` | `precompute.py --capture-only` | `build_guidance.py`, `windows.py`, `precompute.py` (paired) |
| `<view>/ltx_vae_latent[_white].pt` — the capture master `z_y` | `precompute.py --capture-only` | `train.py`, `stats.py`, `precompute.py` (paired) |
| `<view>/argavatar_ltx_vae_latent[_white].pt` — the guide master `z_g` | `precompute.py` (paired) | `train.py`, `stats.py` |
| `<view>/argavatar_alpha.mp4` — the render's alpha, 256², lossless gray | `build_guidance.py` | `train.py`, `stats.py` |
| `<view>/capture_mask_crop.mp4` — the capture matte cropped to the box, 256², lossless gray | `precompute.py` (paired) | `train.py`, `stats.py` |
| `<subject>/qa/capture_mask_crop.mp4` — sampled review crop | `precompute.py --capture-only` | human QA; first five subjects per `Part_*`, view 0 only |
| `expr/onestep_avatar/windows/<name>.json` | `windows.py` | `train.py` |

Every past bug in this pipeline has been the same shape: **two producers of something that must
have exactly one.** If you are about to recompute a crop box, a block plan, or a target latent
somewhere else — don't. Read it from the file above.

Bundles written in the obsolete per-window format are regenerated. The active precompute path
stores only continuous schema-v2 masters and no longer carries migration-only code.

## Run order

```bash
# 1. Capture target latents, from raw rgb.mp4 (days; --capture-only is idempotent per source).
#    Use the detached supervisor, not a bare invocation -- a plain foreground/backgrounded run
#    shares this shell's process group, so a signal to the shell (closed terminal, killed job)
#    takes the retry loop down with the job it is supervising (this happened twice, see the plan
#    plan's §1.3). setsid gives the loop its own session so it survives that.
for rank in 0 1 2 3; do
  setsid -f scripts/onestep_avatar/run_b2a.sh "$rank" 2 "$rank" 4 \
    </dev/null >/dev/null 2>&1 &
done
disown
# tail -f expr/onestep_avatar/logs/precompute_capture_only_gpu1.log to watch it.
# Equivalent bare invocation, for reference (don't launch it this way for a multi-day run):
conda run -n ltx python -m scripts.onestep_avatar.precompute \
  --capture-only --objective bg white --views 0 1 2 3 4 5 6 7 \
  --edge 1024 --pad-factor 1.2 --crop-workers 2 --gpu-id 0 --rank 0 --n-rank 4

# Regenerate only the sampled mask QA gallery (CPU-only, no VAE):
conda run -n ltx python -m scripts.onestep_avatar.precompute \
  --capture-only --views 0 --mask-qa-only

# 2. Render guides into the manifest's box (~20 min per view). DO --limit 8 FIRST AND LOOK.
#    19 pairs / 12 actors exist as of 09-13; the review gate passed. No longer the binding
#    constraint -- step 1 is, since a pair also needs its capture latents.
LTX-2/scripts/onestep_avatar/run_b2b.sh 3           # gpu, then [limit] [driving-views...]

# 3. Freeze a training subset: chains, the actor split, the sha256 pin.
#    --min-holdout-actors matters below T3 scale: it defaults to 12 (the T3/T4 floor from the
#    plan's SS6.1), which at 8 actors holds out 7 of them, leaving just 1 for training.
conda run -n ltx python -m scripts.onestep_avatar.windows \
  --name t2 --max-actors 8 --require-guide --chain-length 3 --min-holdout-actors 2

# 4. Encode each guide render's master latent and store the cropped capture-mask MP4.
#    --objective is a SET: pass both to build them from one decode per source. Currency is
#    tracked per (source, objective), so this resumes -- and a late-added objective re-encodes
#    only itself.
conda run -n ltx python -m scripts.onestep_avatar.precompute --gpu-id 2 \
  --objective bg white

# 5. Measure before training: r, the latent moments, the base model's excursion
conda run -n ltx python -m scripts.onestep_avatar.stats \
  --pairs ../data/AnimatableHuman/DNARenderingVideo --renders ../../ARG-Avatar/expr \
  --out ../expr/onestep_avatar/analysis_summary.json --gpu-id 2

# 5b. A1 on its own, on the real corpus guides -- characterises Phi at sigma_0 (~2 h, 1 GPU)
LTX-2/scripts/onestep_avatar/run_a1.sh 2

# 5c. Cost per finalized chunk: the causal denoise+refresh pair against k2's two window
#     forwards (~2 min, 1 GPU). NOT YET RUN under §4.4 -- the compute claim moved when the
#     cache landed, and this is the measurement that settles where it moved to. Sweep the
#     cache depth, which is the knob: --context-latent-frames 0 2 4
conda run -n ltx python -m scripts.onestep_avatar.bench_forward --gpu-id 3

# 6. Train (2 GPUs shown; drop --lora-rank when GPUs are scarce, never --chain-length)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --config_file scripts/onestep_avatar/configs/fsdp_2gpu.yaml --main_process_port 29517 \
  -m scripts.onestep_avatar.train \
  --subset ../expr/onestep_avatar/windows/t2.json \
  --output ../expr/onestep_avatar/runs/t2-r16 --lora-rank 16 --steps 2000
#     --objective must match the subset's. --disagreement-weight is the plan's SS1.5 rule and
#     defaults to 0.0 (band excluded, everything else at full weight).
#     The corpus root comes from the subset; --context-latent-frames is the cache depth (and
#     therefore the compute/quality knob), recorded in the checkpoint metadata.

# 6a. D0 one-step initialization sanity check. The run writes both the exactly-no-op
#     step-0 adapter and the adapter after its first optimizer update. A non-zero exported
#     LoRA B at step 0 is a hard error, not a visual judgment call.
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --config_file scripts/onestep_avatar/configs/fsdp_2gpu.yaml --main_process_port 29517 \
  -m scripts.onestep_avatar.train \
  --subset ../expr/onestep_avatar/windows/t2.json \
  --output ../expr/onestep_avatar/runs/d0-init-sanity \
  --guide-mode d0 --save-initial --save-every 1 --steps 1

# 6b. Optional: one adapter across several noise levels, one level per rank for the whole run.
#     sigma=0.0 is refused (it trains on nothing). Stamps sigma0="mixed" in the checkpoint.
#     ... --sigma-levels 0.909375 0.725 0.421875

# 7. Look at a checkpoint. Decodes a fixed chain at a fixed seed into one MP4 per sigma,
#    laid out `capture | frozen base | LoRA`. D0 only today; a D1 counterpart is owed.
conda run -n ltx python -m scripts.onestep_avatar.visualize_d0 \
  --subset ../expr/onestep_avatar/windows/prelim2.json \
  --checkpoint <run>/checkpoints/lora_weights_step_00200.safetensors \
  --output <run>/probes/step_00200 --gpu-id 2

# For the initialization check, render both artifacts together. In each generated MP4,
# step 0's third panel must equal the frozen-base middle panel; step 1 then shows the first
# update on the same target, seed, chain, and sigma.
conda run -n ltx python -m scripts.onestep_avatar.visualize_d0 \
  --subset ../expr/onestep_avatar/windows/t2.json \
  --run ../expr/onestep_avatar/runs/d0-init-sanity --steps 0 1 \
  --output ../expr/onestep_avatar/runs/d0-init-sanity/probes/init --gpu-id 2

# 8. Figures from the rank logs (4 of the plan's 7; no reference lines yet).
conda run -n ltx python -m scripts.onestep_avatar.plot_training --run <run>
```

## Things that will bite you

- **`--crop-workers 3` is load-bearing.** Each worker holds one source's raw-frame batch
  (1–2 GB); the default `os.cpu_count()` fan-out killed the first capture run by exhausting
  host RAM. Check `free -h` before raising it.
- **A `--capture-only` restart looks like a hang** on any source not yet in the plan cache
  (`.capture_plan_cache.json`, corpus root): it must still open + frame-0-decode those before
  checking which bundles exist, with no log line and no bundle written meanwhile, parent at ~0 %
  CPU in `futex_wait`. The tell is the *worker* CPU (`ps --ppid`), which is at several hundred
  percent each. A fully-cached source skips this; the cache is keyed on the source's own
  fingerprint plus pad-factor/window-geometry, so a changed source or a changed `--pad-factor`
  replans just that entry rather than reusing a stale box.
- **A render built at a stale box is refused, not skipped.** `discover_pairs` compares each
  `argavatar_render.json`'s box to the manifest and names the views to re-render. That is the
  guard working — the alternative is training on a pair that is shifted by a few hundred pixels.
- **A render without `argavatar_alpha.mp4` is incomplete, not merely old.** The alpha only
  exists inside the render's own temp frames, so it cannot be back-filled without re-rendering.
  (`_render_is_complete` accepts a legacy `.npy` as well, so a pre-2026-09-15 render is not
  stale merely for predating the MP4 format.)
- **Masks are stored as lossless grayscale MP4, not raw arrays** — ~42x smaller with a
  bit-exact round trip (9.83 MB -> 0.232 MB for a 150-frame alpha). Lossy was measured and
  rejected: it corrupts exactly the soft silhouette edge, and these mattes are already one
  generation of lossy video from the truth. Legacy `argavatar_alpha.npy` files are still read;
  `build_guidance.py --migrate-alpha [--prune-npy]` converts them (verified bit-exact, no GPU).
- **Masks are tracked separately from latents.** A view encoded before `capture_mask_crop.mp4`
  existed has a complete, current guide master; `encode_pairs` checks the two independently so
  such a view is re-visited for its masks rather than skipped forever.
- **Multi-GPU capture preprocessing shards whole sources.** Every process discovers the same
  sorted corpus, then owns `sources[rank::n_rank]`; windows and both objectives never split
  across ranks. Rank 0 alone writes QA. Shared bundle writes are atomic, and resume accepts a
  bundle only when its encode-contract/input/VAE/crop provenance is current.
- **Causality and the K/V cache are one feature, not two.** A cached context token's keys and
  values are only reusable because nothing later can change them, which is exactly what
  block-causal attention guarantees. Turning the mask off and keeping the cache would silently
  compute a different function; `causal_core` owns both and `test_causal_core` pins the cached
  forward against a masked full-sequence one.
- **The cache costs ~0.8 GB of VRAM per retained latent frame** at the 22B geometry (48 layers
  x 1024 tokens x 4096 dims x k and v x 2 bytes), per rank. `--context-latent-frames` is the
  knob; the pinned frame-0 sink is always there on top of it.
- **An adapter is only valid at the cache depth it was trained at.** It is in the checkpoint
  metadata for the same reason sigma_0 is: two frames of context and six are different
  functions, and nothing downstream can tell by looking at the weights.
- **Two `r`s, and only one is about the task.** Before the pixel composite landed, the render
  sat on white and the capture on a dark dome, with the subject at ~12 % of the crop — so
  whole-crop `r` measured a background convention (1.38, above the capture latent's own scale).
  Compositing collapsed it to ~0.49 at n=217, but the subject number barely moved. Use
  `r_subject` either way.
- **fps is RoPE, not metadata** (`VideoLatentTools` divides the temporal axis by it). Never
  default it.
- **The loss weighting is a training choice, not a corpus one.** Both grids are stored
  uncombined; `train.py --disagreement-weight` decides what the band between them is worth.
  That disagreement region is exactly the §B1 IoU gap.
- **There is ONE masking rule now: full frame, down-weight `render ⊖ capture`.** It replaced
  five pre-product *subject* masks (`none`/`render`/`capture`/`union`/`intersection`), every one
  of which gave weight zero outside the subject at time `t` — which is exactly where the ghost
  band (`mask_0` minus `mask_t`) lives, so none of them could ever train the product objective.
  `--disagreement-weight 0.0` (default) excludes the band; `1.0` is a plain full-frame loss.
  **Runs before 2026-09-15 used `--loss-mask union` and are not comparable** on anything the
  ghost band touches.
- **Two objectives, one code path.** `bg` (the product) and `white` (both sides on white)
  differ only in which pixels were encoded and which filename holds them. `bg` keeps the
  unsuffixed names, so nothing already on disk was invalidated. A subset records the objective
  it was frozen against and `train.py` refuses a mismatch.

## Tests

```bash
conda run -n ltx python -m pytest scripts/onestep_avatar/tests -q    # from the LTX-2 root
```

Until the 2026-09-15 consolidation two of these tests existed only to pin transcribed copies
of the crop box and the block plan across the two trees. There is one copy of each now --
`precompute._capture_box` calls `geometry.canonical_crop_box`, `windows.plan_blocks` calls
`causal_core.CausalGeometry.plan` -- so those comparisons would be tautological and are gone.
What replaced them is **golden** tests on the exact values, which is the risk that survives
consolidation: changing either rule would silently re-crop a corpus whose latents are already
encoded, or re-point every chain in every subset already frozen.

`test_causal_core` is the one that matters most: it runs a real (2-layer) `LTXModel` on CPU and
asserts the cached block rollout equals a full-sequence forward under a block-causal mask,
block for block. Everything the cache buys rests on that equality.
