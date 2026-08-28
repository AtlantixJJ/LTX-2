# `scripts/prune/` — training-free head + FFN pruning of the LTX refiner

Implements [`plans/2026-08-26-refiner-head-ffn-pruning.md`](../../plans/2026-08-26-refiner-head-ffn-pruning.md),
refactored per [`plans/2026-08-28-prune-package-refactor.md`](../../plans/2026-08-28-prune-package-refactor.md).
Everything here runs in the `ltx` conda env, from the **LTX-2 repo root**, as a module:

```bash
conda activate ltx
cd LTX-2
python -m scripts.prune.<subpackage>.<script> --model 2.5 --gpu-id 2
```

`-m` from the repo root is required — these modules import each other as
`scripts.prune.*` and do not manipulate `sys.path` (the run scripts one level up do).
`scripts/` and `scripts/prune/` are themselves `__init__.py`-free PEP 420 namespace
packages; each of the six subpackages below (`core/ data/ score/ evaluate/ checks/
report/`) carries its own `__init__.py` and is where the module in question actually
lives — see "Modules" for the full file → subpackage map.

Artifacts land under `expr/refiner_prune/<model-key>/`. Nothing is shared between
generations: head index spaces are not comparable and FFN activation statistics do
not transfer, so every file is namespaced by `2.3` / `2.5` and carries a
`provenance` block naming the transformer fingerprint that produced it. Path
construction goes through `artifacts.py` only — see "Modules" below.

## Tests

```bash
python -m pytest scripts/prune/tests -q          # Tier A: CPU only, < 30 s
python -m pytest scripts/prune/tests -q -m gpu    # Tier B: needs a GPU + checkpoint, ~10 min
```

No mocks: Tier A runs the real production classes at small dimensions and reads
the real checkpoint's safetensors headers and the real calibration cache; it skips
(never fails) an assertion whose artifact is missing from disk. Tier B loads the
real 22B transformer and VAE. `scripts/prune/tests/conftest.py` documents the
fixtures. A third, non-pytest tier is the regression gate below.

## Phase 0 — run order

| # | Command | Gate it satisfies |
|---|---|---|
| 1 | `python -m scripts.prune.core.preflight --model 2.5 --dump-caps` | `ModelCaps` on disk per generation |
| 2 | `python -m scripts.prune.data.prompt_cache --model 2.5 --gpu-id N --verify` | prompt-context cache is bit-exact vs the text encoder |
| 3 | `python -m scripts.prune.checks.video_only_check --model 2.5 --gpu-id N` | video-only build matches audio-video within bf16 noise, 3 distinct subjects |
| 4 | `python -m scripts.prune.checks.parity_check --model 2.3 --gpu-id N` | registry refactor reproduces the pre-refactor script bit-for-bit |
| 4b | `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N` | **the gates roll out the deployed method**, bit-for-bit vs `scripts/vae_refine_sliding_window.py` |
| 5 | `python -m scripts.prune.evaluate.bench_refiner --model 2.5 --gpu-id N` | baseline latency / FLOP / memory table, incl. the K/V-cache axis |
| 6 | `python -m scripts.prune.data.source_target --model 2.5 --freeze` | corpus split frozen (`--build-calibration` next; see Phase 1) |
| 7 | `python -m scripts.prune.report.summarize_phase0` | every gate + number collected into `analysis_summary.json` |

The `torch.compile` / CUDA-graph axis is a separate, smaller sweep so it never
overwrites the eager baseline table:

```bash
python -m scripts.prune.evaluate.bench_refiner --model 2.5 --gpu-id N --tag compile \
    --chunk-latent-frames 1 4 --ctx-latent-frames 0 4 --no-kv-cache --compile-chunks 1 4
```

## Modules

The directory tree physically groups modules under six subpackages
(`core/ data/ score/ evaluate/ checks/ report/`); each file below lives at
`scripts/prune/<subpackage>/<name>.py`, not at the old flat `scripts/prune/<name>.py`
path. There is no compatibility shim for the pre-relocation flat path or CLI —
callers (including `scripts/vae_refine_sliding_window.py` and
`scripts/prune/run_head_sweep.sh`) were updated to the new paths instead, since
the goal of the relocation was the physical layout, not preserving an external
command line. Each subpackage's `__init__.py` documents its modules but does not
eagerly import them (`core.session` needs `data.prompt_cache`, and
`data.source_target` needs `core.session`, so eager cross-package imports at
package-init time would cycle); import the specific module you need, e.g.
`from scripts.prune.core import session`.

**`core/`** — bootstrap, geometry, the model registry, the private-API quarantine:

| File | What it owns |
|---|---|
| `session.py` | **Start here for a new entry point.** One bootstrap (`open_session`) replacing the preamble nine scripts used to write out by hand: preflight + device + prompt-context + the deployed sigma schedule, plus `Session.transformer()`/`.decoder()` context managers and `.stamp()` for provenance. `DTYPE = torch.bfloat16` is declared exactly once, here (`tests/test_session.py::test_dtype_is_declared_exactly_once` enforces it). |
| `artifacts.py` | The single source of truth for every path under `expr/refiner_prune/<key>/` — names only, no I/O logic. Add a name here, not a path literal at a call site. |
| `ltx_adapter.py` | Quarantines every underscore-prefixed `ltx_core`/`ltx_pipelines` symbol this package touches (`_build_state`, `_step_state`, `DiffusionStage._transformer_ctx`, `VideoDecoder._decoder_builder`, `ImageConditioner._build_encoder`, `should_use_ancestral_sampler`) behind six public wrappers. When the LTX-2 submodule pin moves, this is the one file that needs re-checking; `tests/test_ltx_adapter.py` enforces that nothing else imports a private name. |
| `refine_task.py` | The deployment conditions — prompt, k-step, window geometry. Single source of truth (§1). |
| `refine_core.py` | The ONE implementation of "refine one sliding window": geometry, tools (fps is required), state construction, the k-step loop. Imported by `scripts/vae_refine_sliding_window.py` *and* by the gates, so the two cannot drift. |
| `model_registry.py` | `--model {2.3,2.5}` → `ModelPaths` + sigmas + sampler + probed `ModelCaps` + probed scale factors (§4). |
| `geometry.py` | Probed latent geometry: pixel↔latent frames, token counts, window-grid rules. No literal `8`/`32` anywhere else. |
| `preflight.py` | Path / caps / GPU validation, called at the top of every entry point (§3). |
| `provenance.py` | Run stamp + checkpoint fingerprint embedded in every artifact (§4). |

**`data/`** — the corpus, calibration-record selection, persisted states, the prompt cache:

| File | What it owns |
|---|---|
| `corpus.py` | The sam3dgs refine corpus: clip listing, subjects, the frozen calibration/held-out split (read from `artifacts.manifest()`), and each clip's real fps — never a literal, since fps is RoPE. |
| `records.py` | Calibration-record selection: split/family/step filtering and an evenly-strided, family-balanced `--max-records` sample. `records.select` is the one implementation the five duplicated copies collapsed into. |
| `chunk_states.py` | Persistent patchified state/x0* records (`RECORD_FORMAT == 2`), including the frozen-context keyframe caveat (§6). |
| `prompt_cache.py` | The constant prompt's context, cached to disk, with a bit-exactness gate (§5.3). |
| `source_target.py` | Source-target definition, corpus freeze, plus reproducible on-policy and renoised AR-state cache (§5.6, §6). Renamed from `teacher.py`; the abandoned 16-step-teacher design it replaced is described under "Two places the plan is wrong" below. No `teacher.py` alias remains. |

**`score/`** — Phase 2/3 pruning:

| File | What it owns |
|---|---|
| `hooks.py` | Activation-collection and head/FFN mask context managers on `to_out[0]`/`ff.net[2]` (§2). |
| `losses.py` | Fresh-token-only x0 MSE and T0 relative L2 (§6). |
| `lstsq.py` | Ridge least-squares accumulator and masked attention accumulator for the linear-map estimators. |
| `head_scores.py` | Contribution / Michel / Gauss–Newton head-importance estimators, iterative pruning, leave-one-out validation (§7). |
| `ffn_scores.py` | FFN-channel scoring and mask construction. |
| `prune_schedule.py` | Iterative head-mask pruning to a target sparsity, re-scoring the currently masked model each round. |
| `export_pruned.py` | Structural export: bakes a mask into a narrower checkpoint (Phase 4). |

**`evaluate/`** — the T0–T3 gate metrics and the candidate gates themselves
(`evaluate/`, not `eval/` — `eval` shadows the builtin and trips ruff's `A` rules):

| File | What it owns |
|---|---|
| `metrics.py` | T0 latent, T1 decoded pixels, T2 sequential-rollout slopes, and T3 review grids (§6). |
| `decode.py` | The one decoder implementation: `decode_latent` (a dense latent → `[F,H,W,C]` pixels, the `phase1_gates` rollout path) and `decode_token_latent` (a token-space x0 → `[F,C,H,W]` pixels, `head_ablation_eval`'s path, built on top of `decode_latent`). |
| `phase1_gates.py` | Runs the **unpruned** student through T0/T1/T2/T3 — the §6 gate itself, and the reference level every pruned candidate is measured against. |
| `head_ablation_eval.py` | Functionally removes selected heads (a zero mask at `to_out[0]`, equivalent to deletion) and compares against the unpruned result, plus a review MP4. |
| `sampler_ab.py` | Reproducible 2.5 Euler/ancestral T0 comparison and recorded sampler decision (§4, §6). |
| `gates.py` | The pass/fail verdict logic a pruned candidate is judged by (rollout-length floor, minimum speedup, T0 delta). |
| `bench_refiner.py` | The Phase 0 baseline table: ms/step, peak memory, FLOPs per geometry, incl. the `torch.compile`/CUDA-graph and K/V-cache axes (§5.5). |
| `timing.py` | `StageTimer` + FLOP counting (§5). |
| `cross_kv_cache.py` | Per-sigma `attn2` K/V memoization, with a bit-exactness gate (§5.4). |

**`checks/`** — the three bit-exactness gates:

| File | What it owns |
|---|---|
| `parity_check.py` | The refactor-parity gate (§5.1): the model-registry refactor reproduces the pre-refactor script bit-for-bit. |
| `method_parity.py` | **The gate that proves the harness rolls out the deployed method**: same clip, same geometry, same seed, `torch.equal` on the refined latents vs `scripts/vae_refine_sliding_window.py` run as a subprocess. |
| `video_only_check.py` | The audio-branch-drop gate (§5.2): a video-only build matches an audio-video build within bf16 noise. |

**`report/`** — collecting results:

| File | What it owns |
|---|---|
| `summarize_phase0.py` | Collects every gate + number into `analysis_summary.json` + markdown tables (§12). Reads paths only through `artifacts.py` — `tests/test_artifacts.py::test_summarize_reads_only_paths_artifacts_can_produce` enforces it. |
| `plot_head_scores.py` | Compares estimates from matched scoring runs visually (block rows × head columns). |

Phase 2's estimators can be distributed across GPUs; for example, run
contribution/Michel/Gauss–Newton on GPUs 0/1/2 and the exact ablation/leave-one-out
validation on GPU 3. Phase 3+ (beyond `export_pruned.py`) is not written yet.

```bash
python -m scripts.prune.score.head_scores --model 2.5 --gpu-id 0 --methods contribution
python -m scripts.prune.score.head_scores --model 2.5 --gpu-id 1 --methods michel
python -m scripts.prune.score.head_scores --model 2.5 --gpu-id 2 --methods gauss_newton --gauss-newton-projections 16
python -m scripts.prune.score.head_scores --model 2.5 --gpu-id 3 --methods contribution --ablate-layers --validate-heads 200

# Compare estimates from matched scoring runs visually (block rows × head columns).
python -m scripts.prune.report.plot_head_scores \
  --scores expr/refiner_prune/2.5/*-head-scores/head_scores.json \
  --output-dir expr/refiner_prune/2.5/head-score-figures

# Functionally remove selected heads, then compare x0 latents and a decoded MP4
# with the unpruned result. (Structural tensor export is Phase 4.)
python -m scripts.prune.evaluate.head_ablation_eval --model 2.5 --gpu-id N \
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
python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id 0 --windows 3
# -> expr/refiner_prune/2.5/method_parity.json, "pass": true
```

**`run_head_sweep.sh` refuses to start unless that file says `"pass": true`.**

Two remaining `fps=24.0` literals were audited and left alone deliberately, each
with a comment explaining why: `metrics.t3_video`'s default (an ffmpeg display
rate, not a model input) and `bench_refiner`'s `--fps` default (a synthetic
benchmark that never opens a clip, so no real fps applies).

## Phase 1 — build calibration data

First freeze the corpus split, then cache the real states.  A smoke cache is
useful before launching the complete build:

```bash
python -m scripts.prune.data.source_target --model 2.5 --freeze
python -m scripts.prune.data.source_target --model 2.5 --gpu-id N --build-calibration --max-clips 2
python -m scripts.prune.data.source_target --model 2.5 --gpu-id N --build-calibration
python -m scripts.prune.evaluate.sampler_ab --model 2.5 --gpu-id N
python -m scripts.prune.evaluate.phase1_gates --model 2.5 --gpu-id N     # the §6 gate
python -m scripts.prune.report.summarize_phase0 --model 2.5            # renders the gate table
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
recorded in `artifacts.manifest()`'s target file (`source_target/manifest.json`)
and each cache index. It intentionally supersedes the plan's stale 52-clip / 40–12
estimate.

Phase 0 summaries save latency/FLOP charts in `expr/refiner_prune/<model>/figures/`.
For Phase 1 visual review, call `metrics.t3_grid(...)` and
`metrics.t3_video(source, teacher, candidate, output)` to save PNG grids and
aligned `source | teacher | candidate` MP4s under the candidate run's `figures/`.

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
   `source_target.py` instead uses the VAE-encoded source latent directly as the
   target (`||D(z_sigma, sigma) - x_source||²`), rather than a deeper-schedule
   teacher. A 3-subject, 16-step probe of the abandoned teacher design (now
   removed, along with its `--validate`/`--num-clips` flags) measured it moving
   PSNR-vs-source by only ±0.08 dB over the plain student — within noise, and a
   worse target than the plain source, not a better one. See the module
   docstring for the full rationale.

## Path-contract drift, found and fixed

Before `artifacts.py` existed, the corpus/split-freeze manifest path was spelled
out as a string literal in twelve modules, and the writer and reader had already
drifted apart: `source_target.py` wrote `source_target/manifest.json` while
`summarize_phase0.collect` read a stale `teacher/teacher_manifest.json` — and
because the read path swallowed `FileNotFoundError`, this failed silently rather
than erroring. `analysis_summary.json` therefore described an abandoned
16-step-teacher design rather than the source-target formulation the calibration
cache was actually built under. Fixed: `summarize_phase0.collect` now reads
`artifacts.manifest(key)`, and the stale `teacher/` output directory was archived
(not deleted) as `_archive_teacher_20260827/`.
