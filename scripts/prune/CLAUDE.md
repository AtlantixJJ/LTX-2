# CLAUDE.md — `scripts/prune/`

Guidance for Claude Code when working inside `LTX-2/scripts/prune/`. Read this before
editing; read [`README.md`](README.md) for the per-file module table, the Phase 0/1/2 run
order, and the post-mortems behind the invariants below.

## What this package is

A training-free head + FFN pruning harness for the **LTX refiner** — the k2 sliding-window
denoise that `scripts/vae_refine_sliding_window.py` runs. 29 modules across 6 subpackages,
5.7k production lines, 17 CLI entry points, 0.8k lines of tests.

**The binding constraint:** every number this package produces is a *delta* against
`scripts/vae_refine_sliding_window.py` (which produced everything under
`expr/sam3dgs_vae_refine/`). If the harness rolls out something else, the deltas describe a
model nobody ships — that has already happened once (see README § "The deployed method").
So: **any change that can move a tensor is not done until `checks.method_parity` passes
again.**

## Running anything here

```bash
conda activate ltx                      # never base, never `uv run`, never another env
cd LTX-2                                # the repo root, not scripts/prune
python -m scripts.prune.<subpackage>.<module> --model 2.5 --gpu-id N
```

`-m` from the repo root is mandatory: modules import each other as `scripts.prune.*` and do
not touch `sys.path`. `scripts/` and `scripts/prune/` are deliberately `__init__.py`-free
PEP 420 namespace packages — **do not add an `__init__.py` to either**; the six subpackages
each have one, and it is a docstring only (no eager `from . import x`, which would cycle:
`core.session` → `data.prompt_cache`, `data.source_target` → `core.session`).

Before any GPU run, check `nvidia-smi`; `--gpu-id` is checked for free memory by
`core.preflight` and fails in ~1 s rather than 25 s into a load.

## Where code goes

| | |
|---|---|
| `core/` | bootstrap (`session`), paths (`artifacts`), the deployed task constants (`refine_task`), the one window implementation (`refine_core`), the registry, the private-API quarantine (`ltx_adapter`) |
| `data/` | corpus, record selection, the on-disk calibration cache, the prompt-context cache |
| `score/` | Phase 2/3: mask hooks, losses, head/FFN estimators, iterative schedule, checkpoint export |
| `evaluate/` | T0–T3 metrics, decode, the Phase 1 gate, the latency/FLOP baseline, K/V cache |
| `checks/` | the three bit-exactness gates |
| `report/` | `analysis_summary.json` + figures |

Prefer adding a function to an existing module over adding a module. The package already
has more files than concepts; a new file needs to own something none of the above does.

## Rules that must not be broken

1. **One rollout implementation.** `core/refine_core.py` owns "refine one sliding window"
   (geometry, tools, state, k-step loop). `vae_refine_sliding_window.py` *and* the gates
   both import it. Never write a second noise/step/carryover loop — that is exactly the
   drift `checks/method_parity.py` exists to catch.
2. **`core/artifacts.py` is the only source of paths** under `expr/refiner_prune/<key>/`.
   No path literal at a call site, ever — writer and reader have silently diverged here
   before. A new stable gate file means a new name in `artifacts.GATES`.
3. **`core/ltx_adapter.py` is the only place that may touch an underscore-prefixed
   `ltx_core`/`ltx_pipelines` symbol.** `tests/test_ltx_adapter.py` enforces it. When the
   LTX-2 submodule pin moves, this is the one file to re-check.
4. **`DTYPE = torch.bfloat16` is declared exactly once**, in `core/session.py`
   (`tests/test_session.py` enforces it). Import it; do not re-declare `torch.bfloat16`.
5. **Every model forward runs inside `torch.no_grad()`** unless the estimator genuinely
   needs a VJP (`score/head_scores.py` re-opens it with `torch.enable_grad()` and says
   why). An unwrapped `PromptEncoder`/transformer call OOMs at ~45 GB vs ~25 GB —
   `.eval()` does *not* clear `requires_grad` on parameters.
6. **fps is RoPE, not metadata.** `VideoLatentTools` does `positions[:, 0] /= fps`. Never
   default it; pass the clip's own `corpus.fps(source)`. 41 of the 44 corpus clips are
   30 fps, 3 are 24. The two surviving `24.0` literals (`metrics.t3_video`'s display-rate
   default, `bench_refiner --fps` on a synthetic latent) are documented as inert — leave
   them and their comments alone.
7. **`chunk_states.RECORD_FORMAT == 2`.** Format-1 caches were built at a wrong geometry;
   they are refused, not migrated. Bumping the format invalidates the 528-record cache.
8. **Score the chunk, not `denoise_mask`.** `chunk_states.chunk_token_mask(state, meta)` is
   the set the deployed AR refiner predicts; `state.denoise_mask` also includes the index-0
   keyframe (half the fresh-token mass at `n_new=1`).

## Writing a new entry point

Start from `core/session.py`, never from another script's preamble:

```python
ap = argparse.ArgumentParser(description=__doc__)
session.add_model_args(ap)                 # --model / --gpu-id / --seed
session.add_record_args(ap)                # --states / --split / --max-records
args = ap.parse_args()

s = session.open_session(args, script="my_thing")   # preflight + device + prompt ctx + sigmas
paths = records.select(s.states_root(args.states), split=args.split, limit=args.max_records)
with s.transformer() as transformer:       # video-only build, no_grad, freed on exit
    ...
out = artifacts.run_dir(s.key, "my-thing", script="my_thing", argv=sys.argv[1:])
(out / "my_thing.json").write_text(json.dumps({"provenance": s.stamp(), ...}, indent=2))
```

Stable one-per-generation gate files go to `artifacts.gate(s.key, "<name>")` instead, and
a gate `main()` returns `0`/`1` on pass/fail. Every artifact carries `s.stamp()` —
artifacts are per checkpoint and head index spaces are not comparable across generations.

## Tests and lint

```bash
python -m pytest scripts/prune/tests -q            # Tier A: CPU, < 30 s
python -m pytest scripts/prune/tests -q -m gpu     # Tier B: real 22B model, ~10 min
uv run ruff check scripts/prune                    # ruff is NOT in the ltx env
```

No mocks: Tier A runs the real classes at small dimensions against the real checkpoint
headers and the real calibration cache, and **skips** (never fails) when an artifact is
missing from disk — so `-rs` and an empty skip list is the real signal on this host.
Tier C is the regression gate: `checks.method_parity --windows 3` must say `"pass": true`.

## Gotchas that have cost time before

- `conda run` appends a blank line to stdout, so `$(... | tail -1)` captures `""` → a
  `Path("")` downstream. `run_head_sweep.sh` greps for the filename instead.
- `provenance.run_id()` carries a PID because two jobs finishing in the same second landed
  in one directory and clobbered each other's `head_scores.json`. Twice.
- `source_target --build-calibration --max-clips 2` takes the manifest's *first* two clips,
  which are both held-out → **zero** calibration records. Check
  `summarize_phase0`'s `usable_for_phase2` before trusting a cache.
- `checks.parity_check` is 2.3-only (2.5 has no pre-refactor baseline);
  `checks.method_parity` is the one that matters for 2.5.
- `bench_refiner --sampler ancestral` raises on purpose — the ancestral step needs a
  per-step noise draw the bench loop does not supply.
- LTX-2.5 is the default and preferred checkpoint; 2.3 is only for reproducing old results.
