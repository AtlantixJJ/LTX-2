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
| 5 | `python -m scripts.prune.bench_refiner --model 2.5 --gpu-id N` | baseline latency / FLOP / memory table, incl. the K/V-cache axis |
| 6 | `python -m scripts.prune.teacher --model 2.5 --freeze` (+ `--validate --gpu-id N`) | teacher recorded and frozen; calibration split fixed |
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
| `refine_task.py` | The deployment conditions — prompt, k-step, AR chunk sizes. Single source of truth (§1). |
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
| `teacher.py` | Teacher definition, corpus freeze, plus reproducible on-policy and renoised AR-state cache (§5.6, §6). |
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
```

## Phase 1 — build calibration data

First freeze the corpus split, then cache the real states.  A smoke cache is
useful before launching the complete 52-clip build:

```bash
python -m scripts.prune.teacher --model 2.5 --freeze
python -m scripts.prune.teacher --model 2.5 --gpu-id N --validate --num-clips 3
python -m scripts.prune.teacher --model 2.5 --gpu-id N --build-calibration --max-clips 2
python -m scripts.prune.teacher --model 2.5 --gpu-id N --build-calibration
python -m scripts.prune.sampler_ab --model 2.5 --gpu-id N
python -m scripts.prune.phase1_gates --model 2.5 --gpu-id N     # the §6 gate
python -m scripts.prune.summarize_phase0 --model 2.5            # renders the gate table
```

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
