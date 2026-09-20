# `onestep_avatar` — the one-step LTX-2.5 avatar renderer pipeline

**The product.** One real first frame plus a guided ARGAvatar 3DGS render of the motion, turned
into a photorealistic video in **one denoising step** through a LoRA on the distilled LTX-2.5
checkpoint: the subject follows the render, the background is the first frame's. Training is
autoregressive over causal blocks with a clean-latent K/V cache; deployment runs the same blocks
with the same primitives.

**The docs are self-contained — read them, not the workspace plans.**

| Read | For |
|---|---|
| [`doc/core_algorithm.md`](doc/core_algorithm.md) | symbols, the conditioning contract, the block-by-block algorithm, the full data flow, train/probe/deploy parity |
| [`doc/experiments.md`](doc/experiments.md) | what D0/D1, `bg`/`white` and teacher/self forcing mean, and which are implemented |
| [`doc/known_gaps.md`](doc/known_gaps.md) | where the code does not meet the contract |
| [`configs/README.md`](configs/README.md) | the four named run recipes and the Accelerate topology YAMLs |
| [`doc/README.md`](doc/README.md) | the per-module design docs |
| [`CLAUDE.md`](CLAUDE.md) | the maintenance and review rules for this package |

Workspace `plans/` (`2026-09-15-...-core.md`, `2026-09-10-...md`, the 2026-09-18 audit) are
**history and progress records**. They are where a decision's measurements and dates live; they
are not the explanation of anything here, and `SS…` markers in older prose are citations into
them, not definitions.

## Implementation status

> **The supplied real first frame is a clean model condition** (`clean_c0_v1`): at a clip start
> latent frame 0 enters block 0 clean at per-token timestep zero and is preserved through the
> output, the refresh and the pinned cache sink; `onestep_core.rollout` requires an explicit
> `first_frame_latent`. This was
> [**G1**](doc/known_gaps.md#g1--the-supplied-first-frame-is-not-a-model-condition), **verified
> 2026-09-19**. Runs from before that date are not comparable with runs after it.

| | State |
|---|---|
| Causal block training (D0 and D1a, `bg` and `white`, teacher and self forcing) | implemented and runnable |
| Unweighted full-frame latent MSE | implemented; the binding loss decision |
| Clean supplied first-frame condition `c0` | implemented as `clean_c0_v1` (G1 verified) |
| D1 probe; checkpoint-condition enforcement; shared σ validator | owed — G2–G5 |
| Guide artifacts under the v2 compositing contract | 1 of 19 `bg` pairs rebuilt; 0 `white` guides — G6 |
| D1b / D1c | deferred proposals, no code |

Capture is complete: 3,360 bundles per objective. The current frozen subset is `t2r2` (`bg`). The
commands below are pipeline references, not a request to restart completed capture encoding or to
bulk-rebuild guides.

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
| `<corpus>/capture_latent_manifest.json` | `precompute.py --process_gt_latent` | `build_guidance.py`, `windows.py`, `precompute.py --process_syn_latent` |
| `<corpus>/manifest[.white].json` | `precompute.py --process_syn_latent` | paired-run provenance manifest at corpus root (default: `--corpus-root`) |
| `<view>/ltx_vae_latent[_white].pt` — the capture master `z_y` | `precompute.py --process_gt_latent` | `train.py`, `stats.py`, `precompute.py --process_syn_latent` |
| `<view>/argavatar_ltx_vae_latent[_white].pt` — the guide master `z_g` | `precompute.py --process_syn_latent` | `train.py`, `stats.py` |
| `<view>/argavatar_alpha.mp4` — the render's alpha, 256², lossless gray | `build_guidance.py` | `stats.py` (QA/measurement only — `train.py` does not read it; see the loss rule below) |
| `<view>/capture_mask_crop.mp4` — the capture matte cropped to the box, 256², lossless gray | `precompute.py --process_syn_latent` | `stats.py` (QA/measurement only — `train.py` does not read it; see the loss rule below) |
| `<subject>/qa/capture_mask_crop.mp4` — sampled review crop | `precompute.py --process_gt_latent` | human QA; first five subjects per `Part_*`, view 0 only |
| `expr/onestep_avatar/windows/<name>.json` | `windows.py` | `train.py` |

Every past bug in this pipeline has been the same shape: **two producers of something that must
have exactly one.** If you are about to recompute a crop box, a block plan, or a target latent
somewhere else — don't. Read it from the file above.

Bundles written in the obsolete per-window format are regenerated. The active precompute path
stores only continuous schema-v2 masters and no longer carries migration-only code.

## Run order

```bash
# 1. Capture target latents, from raw rgb.mp4 (days; --process_gt_latent is idempotent per source).
#    Capture is COMPLETE for both objectives (3,360 bundles each) -- this is the reference, not
#    a job to restart. For a long re-run, detach it with `setsid`: a plain backgrounded run
#    shares this shell's process group, so a signal to the shell takes the job down with it
#    (this happened twice). --process_gt_latent discovers every bbox-bearing view itself;
#    --rank/--n_rank shard that one plan across processes.
python -m scripts.onestep_avatar.precompute  --process_gt_latent --objective white
# python -m scripts.onestep_avatar.precompute  --process_gt_latent --objective white --rank 0 --n_rank 1

# Regenerate only the sampled mask QA gallery (CPU-only, no VAE):
conda run -n ltx python -m scripts.onestep_avatar.precompute \
  --process_gt_latent --mask-qa-only

# 2. Render guides into the manifest's box (~20 min per view). DO --limit 8 FIRST AND LOOK.
#    The September 18 audit found a compositing defect (F1), now fixed and versioned as guide
#    contract v2 (dataset.GUIDE_COMPOSITING_VERSION). One bg/white pair has been rebuilt and
#    reviewed under v2; the other 18 bg pairs are still stale and will be rebuilt on the next
#    run (doc/known_gaps.md G6). There are no white guide renders yet.
scripts/onestep_avatar/run_b2b.sh 3           # gpu, then [limit] [driving-views...]

# 3. Encode each guide render's master latent and store the cropped capture-mask MP4.
#    --objective is a SET: pass both to build them from one decode per source. Currency is
#    tracked per (source, objective), so this resumes -- and a late-added objective re-encodes
#    only itself.
conda run -n ltx python -m scripts.onestep_avatar.precompute --gpu-id 2 \
  --process_syn_latent --objective bg white

# 4. Freeze a training subset AFTER paired encoding: chains, actor split, sha256 pin.
#    K (blocks per training sample) is fixed HERE by --chain-length, not by train.py.
#    t2r2 already exists: reuse it for the current dry-run gate, do not overwrite it.
#    For a new subset, choose an unused name and change the training --subset path below.
#    --min-holdout-actors defaults to 12; at 8 actors use 2 to avoid a one-actor train split.
conda run -n ltx python -m scripts.onestep_avatar.windows \
  --name t2r2 --max-actors 8 --require-guide --chain-length 3 --min-holdout-actors 2

# 5. Measure before training: r, the latent moments, the base model's excursion
conda run -n ltx python -m scripts.onestep_avatar.stats \
  --pairs ../data/AnimatableHuman/DNARenderingVideo --renders ../../ARG-Avatar/expr \
  --out ../expr/onestep_avatar/analysis_summary.json --gpu-id 2

# 5b. A1 on its own, on the real corpus guides -- characterises Phi at sigma_0 (~2 h, 1 GPU)
scripts/onestep_avatar/run_a1.sh 2

# 5c. Cost per finalized chunk: the causal denoise+refresh pair against k2's two window
#     forwards (~2 min, 1 GPU). NOT YET RUN under the causal scheme -- the compute claim moved
#     when the cache landed, and this is the measurement that settles where it moved to. Sweep the
#     cache depth, which is the knob: --context-latent-frames 0 2 4 8
conda run -n ltx python -m scripts.onestep_avatar.bench_forward --gpu-id 3

# 6. Train (2 GPUs shown; drop --lora-rank when GPUs are scarce, never K).
#     The four NAMED recipes -- d0/d1 x teacher/self forcing, with every flag spelled out and
#     the bg/white substitution -- are in configs/README.md. This is the short form.
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --config_file scripts/onestep_avatar/configs/fsdp_2gpu.yaml --main_process_port 29517 \
  -m scripts.onestep_avatar.train \
  --subset ../expr/onestep_avatar/windows/t2r2.json \
  --output ../expr/onestep_avatar/runs/t2-r16 --lora-rank 16 --steps 2000
#     --objective must match the subset's. Training uses unweighted full-frame latent MSE, no
#     mask and no disagreement weighting. The corpus root comes from the subset;
#     --context-latent-frames is the cache depth (the compute/quality knob), recorded in the
#     checkpoint metadata. The supplied first frame is a clean condition (clean_c0_v1).

# 6a. D0 one-step initialization sanity check. The run writes both the exactly-no-op
#     step-0 adapter and the adapter after its first optimizer update. A non-zero exported
#     LoRA B at step 0 is a hard error, not a visual judgment call.
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --config_file scripts/onestep_avatar/configs/fsdp_2gpu.yaml --main_process_port 29517 \
  -m scripts.onestep_avatar.train \
  --subset ../expr/onestep_avatar/windows/t2r2.json \
  --output ../expr/onestep_avatar/runs/d0-init-sanity \
  --guide-mode d0 --save-initial --save-every 1 --steps 1

# 6b. Optional: one adapter across several noise levels, one level per rank for the whole run.
#     sigma=0.0 is refused (it trains on nothing). Stamps sigma0="mixed" in the checkpoint.
#     ... --sigma-levels 0.909375 0.725 0.421875

# 7. Look at a checkpoint. Decodes a fixed chain at a fixed seed into one MP4 per sigma,
#    laid out `capture | frozen base | LoRA`. D0 only today; a D1 counterpart is owed (G4),
#    and it validates none of the adapter's recorded conditions (G3).
conda run -n ltx python -m scripts.onestep_avatar.visualize_d0 \
  --subset ../expr/onestep_avatar/windows/t2r2.json \
  --checkpoint <run>/checkpoints/lora_weights_step_00200.safetensors \
  --output <run>/probes/step_00200 --gpu-id 2

# For the initialization check, render both artifacts together. In each generated MP4,
# step 0's third panel must equal the frozen-base middle panel; step 1 then shows the first
# update on the same target, seed, chain, and sigma.
conda run -n ltx python -m scripts.onestep_avatar.visualize_d0 \
  --subset ../expr/onestep_avatar/windows/t2r2.json \
  --run ../expr/onestep_avatar/runs/d0-init-sanity --steps 0 1 \
  --output ../expr/onestep_avatar/runs/d0-init-sanity/probes/init --gpu-id 2

# 8. Figures from the rank logs (4 of the 7 the long-form plan sketched; no reference lines yet).
conda run -n ltx python -m scripts.onestep_avatar.plot_training --run <run>
```

## Things that will bite you

- **`--crop-workers 3` is load-bearing.** Each worker holds one source's raw-frame batch
  (1–2 GB); the default `os.cpu_count()` fan-out killed the first capture run by exhausting
  host RAM. Check `free -h` before raising it.
- **A `--process_gt_latent` restart looks like a hang** on any source not yet in the plan cache
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
  knob; the pinned frame-0 sink is always there on top of it. The default 8 + the sink is a
  retained history of 9 latent frames. It is bounded by what fits: measured 2026-09-19 at LoRA
  rank 32 on 4x49 GB, depth 15 OOMs in `backward` and depth 7 ran flat at ~45.1 GB.
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
- **Training uses one unweighted full-frame loss.** Alpha and capture masks remain corpus/QA
  artifacts; `train.py` does not read them and has no loss-mask or disagreement-weight option.
- **The pinned frame-0 sink is a retention policy, not image conditioning.** It keeps whatever
  the first refresh wrote there — a *generated* frame 0 under self forcing. `keyframes_mask` is
  the VAE's geometry mark, not a "hold this image" instruction. See
  [G1](doc/known_gaps.md#g1--the-supplied-first-frame-is-not-a-model-condition); "the sink is
  pinned" is never evidence that the first frame is conditioned.
- **Teacher forcing does not mean the same thing in both callers.** `train.py` refreshes from
  the target `z_y`; the generic `causal_core.rollout(teacher_forcing=True)` refreshes from the
  guide, which equals the target for D0 only
  ([G2](doc/known_gaps.md#g2--generic-teacher-forced-rollout-refreshes-from-the-guide-not-the-target)).
- **Checkpoint metadata is written, not enforced.** Nothing validates an adapter's σ, geometry,
  arm or objective at load ([G3](doc/known_gaps.md#g3--checkpoint-and-artifact-conditions-are-recorded-but-not-enforced)),
  and a run-directory name is not provenance. Keep `--lora-alpha` equal to `--lora-rank`: the
  scale is stamped but not applied at fusion.
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

**What the suite does not establish.** It pins cache/attention equivalence, the crop box, the
block plan and the mask codec — not conditioning correctness. The cache-parity test passes today
*with* the missing first-frame condition, so it is not evidence about
[G1](doc/known_gaps.md#g1--the-supplied-first-frame-is-not-a-model-condition); neither is "the
sink is pinned" or "training and deployment share `causal_core`". The tests owed for the
first-frame fix are listed with that gap.
