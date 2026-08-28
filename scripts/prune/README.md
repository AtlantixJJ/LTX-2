# `scripts/prune/` — training-free head + FFN pruning of the LTX refiner

Implements [`plans/2026-08-26-refiner-head-ffn-pruning.md`](../../plans/2026-08-26-refiner-head-ffn-pruning.md).
Everything here runs in the `ltx` conda env, from the **LTX-2 repo root**, as a module:

```bash
conda activate ltx
cd LTX-2
python -m scripts.prune.<script> --model 2.5 --gpu-id 2
```

`-m` from the repo root is required — these modules import each other as
`scripts.prune.*` and do not manipulate `sys.path` (the run scripts one level up do).

Artifacts land under `expr/refiner_prune/<model-key>/`. Nothing is shared between
generations: head index spaces are not comparable and FFN activation statistics do
not transfer, so every file is namespaced by `2.3` / `2.5` and carries a
`provenance` block naming the transformer fingerprint that produced it.

## Phase 0 — run order

| # | Command | Gate it satisfies |
|---|---|---|
| 1 | `python -m scripts.prune.preflight --model 2.5 --dump-caps` | `ModelCaps` on disk per generation |
| 2 | `python -m scripts.prune.prompt_cache --model 2.5 --gpu-id N --verify` | prompt-context cache is bit-exact vs the text encoder |
| 3 | `python -m scripts.prune.video_only_check --model 2.5 --gpu-id N` | video-only build matches audio-video within bf16 noise, 3 distinct subjects |
| 4 | `python -m scripts.prune.parity_check --model 2.3 --gpu-id N` | registry refactor reproduces the pre-refactor script bit-for-bit |
| 4b | `python -m scripts.prune.method_parity --model 2.5 --gpu-id N` | **the gates roll out the deployed method**, bit-for-bit vs `scripts/vae_refine_sliding_window.py` |
| 5 | `python -m scripts.prune.bench_refiner --model 2.5 --gpu-id N` | baseline latency / FLOP / memory table, incl. the K/V-cache axis |
| 6 | `python -m scripts.prune.source_target --model 2.5 --freeze` (+ `--validate --gpu-id N`) | source-target recorded and frozen; calibration split fixed |
| 7 | `python -m scripts.prune.summarize_phase0` | every gate + number collected into `analysis_summary.json` |

The `torch.compile` / CUDA-graph axis is a separate, smaller sweep so it never
overwrites the eager baseline table:

```bash
python -m scripts.prune.bench_refiner --model 2.5 --gpu-id N --tag compile \
    --chunk-latent-frames 1 4 --ctx-latent-frames 0 4 --no-kv-cache --compile-chunks 1 4
```

## Modules

| File | What it owns |
|---|---|
| `refine_task.py` | The deployment conditions — prompt, k-step, window geometry. Single source of truth (§1). |
| `refine_core.py` | The ONE implementation of "refine one sliding window": geometry, tools (fps is required), state construction, the k-step loop. Imported by `scripts/vae_refine_sliding_window.py` *and* by the gates, so the two cannot drift. |
| `method_parity.py` | The gate that proves they haven't: same clip, same geometry, same seed, `torch.equal` on the refined latents. |
| `model_registry.py` | `--model {2.3,2.5}` → `ModelPaths` + sigmas + sampler + probed `ModelCaps` + probed scale factors (§4). |
| `geometry.py` | Probed latent geometry: pixel↔latent frames, token counts, window-grid rules. No literal `8`/`32` anywhere else. |
| `preflight.py` | Path / caps / GPU validation, called at the top of every entry point (§3). |
| `provenance.py` | Run stamp + checkpoint fingerprint embedded in every artifact (§4). |
| `prompt_cache.py` | The constant prompt's context, cached to disk, with a bit-exactness gate (§5.3). |
| `cross_kv_cache.py` | Per-sigma `attn2` K/V memoization, with a bit-exactness gate (§5.4). |
| `timing.py` | `StageTimer` + FLOP counting (§5). |
| `bench_refiner.py` | The Phase 0 baseline table (§5.5). |
| `video_only_check.py` | The audio-branch-drop gate (§5.2). |
| `parity_check.py` | The refactor-parity gate (§5.1). |
| `source_target.py` (deprecated alias: `teacher.py`) | Source-target definition, corpus freeze, plus reproducible on-policy and renoised AR-state cache (§5.6, §6). |
| `chunk_states.py` | Persistent patchified state/x0* records, including the frozen-context keyframe caveat (§6). |
| `losses.py` | Fresh-token-only x0 MSE and T0 relative L2 (§6). |
| `metrics.py` | T0 latent, T1 decoded pixels, T2 sequential-rollout slopes, and T3 review grids (§6). |
| `sampler_ab.py` | Reproducible 2.5 Euler/ancestral T0 comparison and recorded sampler decision (§4, §6). |
| `phase1_gates.py` | Runs the **unpruned** student through T0/T1/T2/T3 — the §6 gate itself, and the reference level every pruned candidate is measured against. |
| `summarize_phase0.py` | Collects every artifact into `analysis_summary.json` + markdown tables (§12). |

Phase 2 is implemented in `hooks.py`, `head_scores.py`, `lstsq.py`, and
`prune_schedule.py`.  Its estimators can be distributed across GPUs; for
example, run contribution/Michel/Gauss--Newton on GPUs 0/1/2 and the exact
ablation/leave-one-out validation on GPU 3.  Phase 3+ modules are not written
yet.

```bash
python -m scripts.prune.head_scores --model 2.5 --gpu-id 0 --methods contribution
python -m scripts.prune.head_scores --model 2.5 --gpu-id 1 --methods michel
python -m scripts.prune.head_scores --model 2.5 --gpu-id 2 --methods gauss_newton --gauss-newton-projections 16
python -m scripts.prune.head_scores --model 2.5 --gpu-id 3 --methods contribution --ablate-layers --validate-heads 200

# Compare estimates from matched scoring runs visually (block rows × head columns).
python -m scripts.prune.plot_head_scores \
  --scores expr/refiner_prune/2.5/*-head-scores/head_scores.json \
  --output-dir expr/refiner_prune/2.5/head-score-figures

# Functionally remove selected heads, then compare x0 latents and a decoded MP4
# with the unpruned result. (Structural tensor export is Phase 4.)
python -m scripts.prune.head_ablation_eval --model 2.5 --gpu-id N \
  --remove-head 7.attn2:14 --split held_out --max-records 8
```

## The deployed method, and why `method_parity.py` exists

Every number in this package is a *delta* against
`scripts/vae_refine_sliding_window.py` — the script that produced
`expr/sam3dgs_vae_refine/`, including the reference
`4D-Dress_00129_0__woman_dance_2_crop/k2_longform_v3_carryover/decode_full.mp4`.
If the harness rolls out something else, the deltas describe a model nobody ships.

The first published sweep did exactly that. `phase1_gates` had its own rollout,
and it differed from the run script in four ways that each change the
transformer's input and none of which showed up in any JSON it wrote:

| | run script (deployed) | old `phase1_gates` rollout |
|---|---|---|
| fps | the clip's own (30 for 41/44 corpus clips) | hardcoded `24.0` — and fps divides the temporal RoPE axis (`VideoLatentTools`: `positions[:, 0] /= fps`) |
| latent index 0 | a genuine causal keyframe: each window is re-encoded from pixels | a *regular* latent frame spliced off the rollout stream into the keyframe slot |
| geometry | 25-frame window / 9-frame overlap = keyframe + **1** frozen + **2** fresh latent frames | **4** frozen + **1** fresh — a window the refine script has never run |
| seed | one fixed seed per run | `seed + chunk_index` |

Its T2 rollout was visibly softer than `decode_full.mp4`, which is how this was
caught. The fix was to give both callers one implementation (`refine_core.py`)
and a gate (`method_parity.py`) that runs the refine script in a subprocess and
`phase1_gates._encode_windows` + `._rollout` in-process, then compares the refined
latents with `torch.equal`. Two windows minimum — one window never exercises the
carryover.

```bash
python -m scripts.prune.method_parity --model 2.5 --gpu-id 0 --windows 3
# -> expr/refiner_prune/2.5/method_parity.json, "pass": true
```

**`run_head_sweep.sh` refuses to start unless that file says `"pass": true`.**

## Phase 1 — build calibration data

First freeze the corpus split, then cache the real states.  A smoke cache is
useful before launching the complete 52-clip build:

```bash
python -m scripts.prune.source_target --model 2.5 --freeze
python -m scripts.prune.source_target --model 2.5 --gpu-id N --validate --num-clips 3
python -m scripts.prune.source_target --model 2.5 --gpu-id N --build-calibration --max-clips 2
python -m scripts.prune.source_target --model 2.5 --gpu-id N --build-calibration
python -m scripts.prune.sampler_ab --model 2.5 --gpu-id N
python -m scripts.prune.phase1_gates --model 2.5 --gpu-id N     # the §6 gate
python -m scripts.prune.summarize_phase0 --model 2.5            # renders the gate table
```

Calibration records are versioned (`chunk_states.RECORD_FORMAT`). Format 1 caches
were built at the pre-parity geometry (4 frozen latent frames, 24 fps) and are
refused rather than migrated — their tensors are states the deployed refiner never
sees. Rebuild instead.

## Phase 2 — the sparsity sweep

```bash
bash scripts/prune/run_head_sweep.sh 2.5 "0 1 2 3"
```

One sparsity target per GPU: `head_scores --target-sparsity P` (iterative Michel,
re-scoring the currently masked model each round) then the full
`phase1_gates --head-masks` T0/T1/T2/T3 gate. The script exists mostly to avoid
two operational traps the first sweep hit: `conda run`'s trailing blank line makes
`$(... | tail -1)` capture `""` (so `--head-masks ""` becomes `Path(".")`), and
`provenance.run_id()` only has 1-second resolution, so simultaneous launches can
land in the same directory and clobber each other's `head_scores.json`. It greps
for the path and re-reads each schedule's own `target_sparsity` to prove it got
the file it asked for.

`--max-clips 2` takes the manifest's *first* two clips, which both fall in the
held-out subject set — a smoke cache built that way contains **zero**
calibration-split records and cannot drive Phase 2. `summarize_phase0`'s
`usable_for_phase2` flag exists to catch exactly that; check it before assuming a
cache is real.

The cache contains both on-policy k2 states and independently renoised x0*
states at every deployed nonterminal sigma.  Every future scorer must load these
``.pt`` records via `chunk_states.load_record`, rather than generate its own noise.
The current corpus holds 44 clips; its frozen 29/15 subject-disjoint split is
recorded in `teacher_manifest.json` and each cache index.  It intentionally
supersedes the plan's stale 52-clip / 40–12 estimate.

Phase 0 summaries save latency/FLOP charts in `expr/refiner_prune/<model>/figures/`.
For Phase 1 visual review, call `metrics.t3_grid(...)` and
`metrics.t3_video(source, teacher, candidate, output)` to save PNG grids and
aligned `source | teacher | candidate` MP4s under the candidate run's `figures/`.
`teacher --validate` does this automatically at
`expr/refiner_prune/<model>/teacher/figures/`.

## Two places the plan is wrong, and what was done instead

1. **§5.4's K/V cache sketch cannot run.** It assigns `attn.to_k = lambda ...`;
   `nn.Module.__setattr__` raises `TypeError` when replacing a registered
   submodule with a non-Module, and its `id(context)` memo key would never hit
   (the modulated context is freshly allocated per call) and could hit *wrongly*
   after a free. `cross_kv_cache.py` swaps in a real `nn.Module` wrapper keyed on
   a caller-declared sigma, and asserts bit-exactness before anything trusts it.
2. **§6's teacher premise does not hold.** The on-disk `k8` outputs are not
   converged refinements: `k8` starts at sigma 1.0 and `FINDINGS.md` measures it
   at 4.44 dB vs source — the latent is erased and regenerated from the prompt.
   `teacher.py` instead subdivides the interval below the *student's* sigma_0.
   See its module docstring.

## What `--validate` actually measured (2.5, 3 subjects, 2026-08-27)

`teacher --validate` was written to be falsifiable and it should be read that way,
because the result is **not** the one its own docstring predicts.

| clip | VAE round-trip | student k2 | teacher (16 steps) | teacher vs student | on-disk k8 |
|---|---:|---:|---:|---:|---:|
| `2K2K_00052_0__man_dance_2_crop` | 36.42 | 27.21 | **27.13** | 33.34 | 4.00 |
| `2K2K_01331_0__woman_mocap_1_c5_original` | 36.80 | 26.38 | **26.38** | 33.76 | — |
| `2K2K_01465_0__man_dance_2_crop` | 34.37 | 25.49 | **25.43** | 31.44 | — |

(dB PSNR vs source.) Two things follow, and both matter for how Phase 2 reads its
own numbers:

* **The k8 critique is confirmed** — 4.00 dB measured here against 4.44 dB in
  `FINDINGS.md`. Dropping the plan's k8 teacher was right.
* **But the replacement teacher is not a better refinement either.** Eight times
  the compute moves PSNR-vs-source by −0.08/−0.00/−0.06 dB: within noise, and if
  anything slightly *down*. The 16-step solution is a different point (31–34 dB
  away from the student) but not a better one. Both sit ~9 dB below the plain VAE
  round-trip, i.e. at `k2` this pipeline *degrades* the latent it is given rather
  than repairing it.

  So `x0*` is a **stable, seed-independent reference point on the deployment
  manifold** — which is all §7's estimators need, and it does keep the
  mask-gradient loss nonzero at ξ=1 — but it is **not a quality ceiling**, and no
  Phase 2 conclusion may be phrased as "closer to the teacher ⇒ better output".
  Quality claims have to come from T1/T2/T3 against the *source*, not from T0.
