# D1 implementation review — 2026-09-27

Snapshot: LTX-2 revision `5452107`; the active plan has uncommitted updates. New work is in experiment-local scripts. This review inspected those scripts, manifests and saved tensors, and one continuation contact sheet. It did not rerun transformer/VAE inference or modify the other agent's implementation/jobs. The sigma-sweep analysis was actively decoding during review; its final metrics/report and the decoder diagnostic results were not yet available.

Current execution note (2026-10-07): the legacy sigma-sweep executors are retired. Their bytes are retained as provenance text; [current commands](../../../expr/onestep_avatar/d1_selfrollout_sigma_sweep_20260926/README.md) use the package dependency queue. The exact self-matching watcher described in finding 4 was verified idle and stopped during retirement. These changes do not revise the dated scientific observations below.

## Findings, in priority order

### 1. Quality acceptance metrics do not yet measure identity or guide motion

The [sweep scorer](../../../expr/onestep_avatar/d1_selfrollout_sigma_sweep_20260926/provenance/retired_execution_sources/analyze.py.txt) calls adjacent unaligned RGB change `motion_over_capture`, and last-frame versus first-frame RGB error `drift_vs_c0_last`. The [continuation scorer](../../../expr/onestep_avatar/d1_continuation_confirm_20260926/analyze.py) uses equivalent definitions. Texture flicker contributes to the first score; legitimate pose changes contribute to the second. Neither can establish preserved motion or identity. Shared masks include generated outputs, so the measurement region also changes when new conditions are added to the sweep.

Keep these as explicitly named RGB diagnostics. Add tracked face/clothing comparisons, guide/capture pose trajectories and visibility confidence, using fixed source-derived regions. Inspect boundary-adjacent lossless frames. Do not select a sigma or declare the AR seam fixed from these scores alone. In the inspected `0025_11` boundary sheet, capture itself changes arm pose near the boundary; raw RGB jump size therefore cannot isolate an appearance reset.

### 2. The new latent boundary statistic changes the earlier metric definition

`latent_residual_frame9` in the continuation scorer is `mean(abs(pred[9] - pred[8]))`. The earlier diagnostic was `mean(abs((pred[9] - pred[8]) - (GT[9] - GT[8])))`. The new statistic measures change magnitude, not the earlier GT-adjusted temporal error. Less movement can improve it.

Rename the new field to `latent_step_mae` and also report the original metric for comparability. CPU recomputation from the saved D1 outputs gives:

| Actor / seed | Cache GT-adjusted residual | Joint GT-adjusted residual | Reduction |
|---|---:|---:|---:|
| 0008_01 / 42 | 0.237721 | 0.228905 | 3.71% |
| 0012_09 / 42 | 0.133605 | 0.116815 | 12.57% |
| 0025_11 / 42 | 0.171071 | 0.160959 | 5.91% |
| 0008_01 / 43 | 0.232820 | 0.230205 | 1.12% |
| 0008_01 / 44 | 0.233089 | 0.224094 | 3.86% |

Joint improves this diagnostic in all five cases. Existing RGB boundary magnitude improves in three cases and worsens in two. These results support a repeatable latent-level benefit, but not a reliable visual discontinuity fix. The GT-adjusted metric is itself spatially unaligned and remains secondary evidence.

### 3. The short/full decoder comparison still lacks aligned explicit noise

The [decoder diagnostic](../../../expr/onestep_avatar/d1_decoder_stability_20260926/run.py) resets the generator seed for each decode but does not inject a canonical noise field sliced by absolute frame coordinates. Its docstring correctly says the short/full comparison mixes noise alignment with context. Consequently, it does not complete the length-isolation experiment in Action 4.

The same-length future-content substitution is useful: identical latent prefixes and equal shapes/seeds, together with the repeatability control, isolate the effect of changed future latent content. Complete the explicit-noise short/full control only before making a streaming-decoder or padding conclusion; otherwise label this action partial. Save resolved decoder fingerprint/settings with the results.

### 4. The background dependency wait can match itself indefinitely

Observed process 3049045 had already run for over ten hours with:

```bash
while pgrep -f d1_selfrollout_sigma_sweep_20260926/run.sh >/dev/null; do sleep 20; done
```

Its own `bash -c` command line contains the search string, so `pgrep` can continue matching that shell after generation finishes. A separate manual analysis process was running during review, so decoding had resumed through a workaround. Another broad `run_generate.sh` watcher had the same structural problem.

Use a captured child PID and `wait`, or explicit success/failure markers written after artifact validation. Retire stale watchers deliberately; this review did not kill them or launch duplicate GPU work.

### 5. Several acceptance controls are recorded rather than enforced

The continuation analyzer stores `first_block_latents_equal` without stopping on false. The sweep analyzer similarly records `c0_equal` and sigma-1 equality after decoding. It does not validate cross-invocation noise/model/source equality before scoring. Current saved artifacts pass the independent checks below, so this does not invalidate them. Add fail-fast checks before expensive decoding to prevent a future invalid comparison from producing apparently normal metrics.

## Independently verified controls

CPU tensor/manifest checks passed:

- All eight sweep cells exist: D0/D1 at 0.421875, 0.725, 0.909375 and 1.0, with one/two/three/eight denoising calls respectively.
- Both sweep invocations use the same frozen base with no adapter, generated history, cache mode, geometry, prompt, source hashes and identical saved epsilon tensors. The model metadata differs only in its timestamp.
- Every sweep latent is finite, contains 17 frames and preserves the supplied `c0` exactly. Sigma-1 D0/D1 outputs are bit-identical.
- The new 0.909375 D1 output exactly reproduces the earlier generated-history output.
- All five continuation pairs have matching source/noise/geometry/prompt settings and bit-identical first nine raw latent frames.
- The fixed-length intervention preserves noise in generated frames 1–2 and changes only frames 3–8. Joint-repeat and causal-control prefixes are bit-exact. Independent recomputation gives early joint relative L2 changes of 0.23431, 0.31235 and 0.30999, confirming the report's main claim.

No generation-path defect was found in these inspected controls. This is not a claim that every implementation path is bug-free.

## Next actions

1. Finish the already-running sigma decode/review, add the quality measurements above, and produce the source-by-sigma report. The requested self-rollout experiment is now implemented and generated; its quality conclusion is pending.
2. Preserve both latent metric definitions and report the five-pair continuation result with tracked visual/motion evidence. Joint history is a candidate with modest repeatable latent improvement, not an established fix.
3. Promote input invariants to assertions and replace the self-matching wait loops.
4. Complete explicit decoder-noise alignment only where required by the streaming question. Avoid claiming Action 4 is fully resolved by the current script.
5. Select a training pilot after the sigma result distinguishes residual boundary jumps, appearance drift and motion failure. The completed future-noise test establishes dependence; it does not diagnose the appearance-jump cause.
