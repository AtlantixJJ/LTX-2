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
| `summarize_phase0.py` | Collects every artifact into `analysis_summary.json` + markdown tables (§12). |

Phase 2+ modules named in §12 (`hooks.py`, `head_scores.py`, `ffn_scores.py`, `lstsq.py`, `prune_schedule.py`,
`export_pruned.py`, `gates.py`) are not written yet.

## Phase 1 — build calibration data

First freeze the corpus split, then cache the real states.  A smoke cache is
useful before launching the complete 52-clip build:

```bash
python -m scripts.prune.teacher --model 2.5 --freeze
python -m scripts.prune.teacher --model 2.5 --gpu-id N --build-calibration --max-clips 2
python -m scripts.prune.teacher --model 2.5 --gpu-id N --build-calibration
```

The cache contains both on-policy k2 states and independently renoised x0*
states at every deployed nonterminal sigma.  Every future scorer must load these
``.pt`` records via `chunk_states.load_record`, rather than generate its own noise.

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
