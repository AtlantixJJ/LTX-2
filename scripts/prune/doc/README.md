# Whole-clip pruning design docs

Start with the [literature review and planned methods](METHODS.md), [architecture](ARCHITECTURE.md) and
[validation](VALIDATION.md). [../README.md](../README.md) gives the run order.
The folders mirror the source tree: each retained production module
has a design page at `doc/<area>/<module>.md`. Package-wide guides
stay at the documentation root.

| Area | Module docs |
|---|---|
| core | [artifacts](core/artifacts.md), [geometry](core/geometry.md), [ltx_adapter](core/ltx_adapter.md), [model_registry](core/model_registry.md), [preflight](core/preflight.md), [provenance](core/provenance.md), [session](core/session.md) |
| data | [prompt_cache](data/prompt_cache.md), [whole_clip](data/whole_clip.md) |
| score | [estimators](score/estimators.md), [export_pruned](score/export_pruned.md), [hooks](score/hooks.md), [whole_clip_d0_scores](score/whole_clip_d0_scores.md) |
| evaluate | [bench_whole_clip_d0](evaluate/bench_whole_clip_d0.md), [decode](evaluate/decode.md), [metrics](evaluate/metrics.md), [whole_clip_d0](evaluate/whole_clip_d0.md) |
| checks | [export_parity](checks/export_parity.md) |
