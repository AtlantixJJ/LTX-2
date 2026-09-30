# Pruning experiment history

The original package calibrated and judged the deployed k2 sliding-window refiner. A prior harness rollout diverged from deployment in fps, first-frame keyframe source, window geometry and seed. `core/refine_core.py` and `checks/method_parity.py` protect that separate deployed path. The [historical k2 report](../../../../expr/refiner_prune/2.5/HISTORICAL_K2.md) records its 200-window findings and source provenance.

The active decision moved to native whole-video bidirectional D0: an original capture is noised at exact sigma levels, denoised once over the complete clip, and compared by noise direction and VAE decode. The [current findings](../../../../expr/refiner_prune/2.5/FINDINGS.md) distinguish the transferred p05 control from the D0-calibrated head/FFN screen. The [simplification plan](../../../../plans/2026-09-29-prune-simplification.md) records proposed retirements. No historical captures, model checkpoints, latents or logs are deleted by the documentation change.

The old k2 command-line modules remain available until their reusable logic, external call sites and tests have been migrated or explicitly retired. Git history preserves prior source revisions, but it is not a substitute for the attributed experiment results above.

## Retired standalone k2 frontends

The native D0 workflow now covers functional head/FFN ablation, direct dense-forward timing and native export parity. A call-site audit found no live external imports of the following standalone experiment commands. `report/summarize_phase0.py` continues to read existing historical `parity_check`, `sampler_ab` and benchmark artifacts; their path names remain in `core/artifacts.py` for that reader. The old files remain recoverable in Git history. SHA256 values below identify the removed worktree content, including its paired module page.

| Removed path | SHA256 |
|---|---|
| `evaluate/sampler_ab.py` | `a7f1a76ebe5a64b7b3cc3b039f67ce3ff8a9686d7d521d085aa469b1795fdf95` |
| `doc/sampler_ab.md` | `c0d28c5b10f355e53f511a71817933c7095e827e50568c0e55101e9e051d3295` |
| `evaluate/head_ablation_eval.py` | `7af8e77200a41d1e32ca9b8d667b45e0f63c0caf1f7cc6070fa306a4de993dba` |
| `doc/head_ablation_eval.md` | `aacf94aea22d6d2dd42b26d4470a980a2c704ad558175b0dc128497932312a31` |
| `evaluate/bench_refiner.py` | `17a692659e6f0b080450bd96a1449cbddd251bc0a8ace2a8da3071d642be8229` |
| `doc/bench_refiner.md` | `007ec78376e188a8104bb1b65a1cdf74a37edef5d0784551769cc5f281300bc7` |
| `evaluate/cross_kv_cache.py` | `b45cd7925ea021b8f87e401743b0f32fe45b0e832939a9dc0c20cbd37d1551ba` |
| `doc/cross_kv_cache.md` | `c3c277ad13d4b125b70e059b292349fa0d1bf1dce5551199f00d828b89a20d4b` |
| `report/plot_head_scores.py` | `2f3c8f85f4d1772411fcca7dd68c9e2d9e73b6490f945e2182f408b2471d7590` |
| `doc/plot_head_scores.md` | `21daf70f5442a022d45ad352f9e7d1b4fe9ef860cb3ea564397783e690807930` |
| `checks/parity_check.py` | `b8c12e1dc0b1659ba63f00884b75731d7d0ea0ad2d0d2d11c01defdb2f7152d4` |
| `doc/parity_check.md` | `0088c9a02e786d561b90988abf53f66f388e8c814e1f95f668bd5d10341053d6` |

These removals do not alter checkpoints, captures, latents, reports or logs. The retained k2 source/cache/phase modules still support `checks/method_parity`, the explicit historical export-parity mode, and existing historical readers. They should be reconsidered only after those dependencies have a tested replacement.
