# `scripts/prune` design docs

For the active native bidirectional D0 workflow, read [METHODS](METHODS.md),
[ARCHITECTURE](ARCHITECTURE.md), [VALIDATION](VALIDATION.md), and
[HISTORY](HISTORY.md). Keep the per-module pages below: each is the design note
for one production module. Historical k2 pages document existing code and a
separate deployed-refiner path; they do not define active D0 acceptance.

These pages describe the native D0 path and retained shared/historical code. Read
[`../CLAUDE.md`](../CLAUDE.md) for binding invariants and [`../README.md`](../README.md) for
the active run order. Each production Python module and the retained sweep launcher has one page here;
`__init__.py` files are docstring-only package markers, and tests are indexed by the module
they exercise.

## The cross-module contract

```mermaid
flowchart LR
  B[saved baseline capture + epsilon] --> I[data.whole_clip]
  I --> S[score.whole_clip_d0_scores]
  S --> M[score.hooks + estimators]
  M --> X[score.export_pruned]
  I --> P[checks.export_parity]
  X --> P
  I --> E[evaluate.whole_clip_d0]
  X --> E
  I --> T[evaluate.bench_whole_clip_d0]
  X --> T
  K[core.refine_core] --> D[deployed k2 refiner + checks.method_parity]
```

`refine_core.py` owns the deployed sliding-window rollout. `checks/method_parity.py` compares
its output with `vae_refine_sliding_window.py` at the latent level; rerun it after any change
that can alter a tensor. `artifacts.py` owns output paths, `session.py` owns dtype and model
lifetime, and `ltx_adapter.py` owns upstream private API access. Format-1 calibration records
are refused because they were made with different geometry and fps.

## Files

| Area | Module docs |
|---|---|
| Core | [artifacts](artifacts.md), [geometry](geometry.md), [ltx_adapter](ltx_adapter.md), [model_registry](model_registry.md), [preflight](preflight.md), [provenance](provenance.md), [refine_core](refine_core.md), [refine_task](refine_task.md), [session](session.md) |
| Data | [chunk_states](chunk_states.md), [corpus](corpus.md), [prompt_cache](prompt_cache.md), [records](records.md), [source_target](source_target.md), [whole_clip](whole_clip.md) |
| Scoring | [estimators](estimators.md), [export_pruned](export_pruned.md), [ffn_scores](ffn_scores.md), [head_scores](head_scores.md), [hooks](hooks.md), [losses](losses.md), [lstsq](lstsq.md), [prune_schedule](prune_schedule.md), [whole_clip_d0_scores](whole_clip_d0_scores.md) |
| Evaluation | [bench_whole_clip_d0](bench_whole_clip_d0.md), [decode](decode.md), [gates](gates.md), [metrics](metrics.md), [phase1_gates](phase1_gates.md), [timing](timing.md), [whole_clip_d0](whole_clip_d0.md) |
| Checks | [method_parity](method_parity.md), [export_parity](export_parity.md), [profile_export](profile_export.md), [video_only_check](video_only_check.md) |
| Reports and launcher | [summarize_phase0](summarize_phase0.md), [run_head_sweep](run_head_sweep.md) |

## Reading the pages

Each page names its owner, input/output flow, invariants, and a relevant check. Code is the
source of current behavior. Defects found in the review are listed in
[`known_gaps.md`](known_gaps.md), with proposed work in the workspace plan. Update a page
when its module's objective, flow, invariant, or interface changes.
