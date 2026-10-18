# Whole-clip pruning verification

## CPU gate

Run from `LTX-2` with the `ltx` interpreter:

```bash
conda activate ltx
CUDA_VISIBLE_DEVICES='' python -m pytest scripts/prune/tests -q -m 'not gpu'
ruff check --select F,I scripts/prune
```

The collection hook skips only tests already marked `gpu` when CUDA is hidden.
CPU fixtures retain import, metadata, shape and small-model checks.
A marker does not disable imports. Ruff follows the repository's dev-tool setup when unavailable in `ltx`.

For documentation changes, check source coverage, required headings, local links and documented CLI options.
Render and inspect changed Mermaid diagrams.
Documentation verification does not require expensive GPU work.

The clipped-reset CPU controls are in `tests/test_clipped_reset.py`.
They verify clipping and surrogate endpoints, reset-free regret windows, gate Adam resets, adapter continuity, and transaction replay.
They also verify the global 500+300 bound and pre-event versus canonical checkpoint representations.
These controls do not qualify native clipped-reset execution. Training rejects an acceptance receipt without its exact policy hash.
Preparation, fixed-test evaluation, media, and binary export need their own evidence before this experiment is complete.

`test_clipped_preparation.py` verifies initial-state replacement, untrained source rejection, and exact-step guards at zero, 500, 550, and 800.
`test_fixed_test_inputs.py` rejects changed frozen noise, targets, schedules, context, incomplete cases, and historical four-actor monitor data.
The saved historical preflight verifies all 17 checkpoint bytes, initial adapters, ordered draw prefixes, and permanent-mask sanity counts.
These CPU checks do not validate historical prediction reproduction or qualify new model predictions.

## Learned FFN controls

`tests/test_ffn_learning.py` checks aligned counts, stable ties and binary surrogate gradients.
It compares the live post-GELU boundary with independent native FFN algebra.
It verifies frozen base weights, exact resumed updates across freezing, original-ID Adam transfer and FP32 fusion.
Four real CPU DDP ranks compare joint gradients with independent serial four-draw accumulation.

`learn.ffn --phase native --testing` uses GPUs 4–6 for disposable native tests.
The formal owner requires a separate four-rank receipt on GPUs 0–3 before its first training update.
Both checks exercise all-one parity, checkpointed gradients, exact serialized resume and fixed-mask recovery.
F18 and cropped-F28 inputs and full-source inference retain separate memory evidence.
`score.export_learned_ffn --probe --parity-gpu-id 4` records split/fused/compact native parity separately.
Preserve parity failures with the unchanged 0.02 tolerance. Training success does not qualify compact execution or speed.

## Report verification

Apply the [report and wording contract](reporting.md).
Check question structure, defined terms, figure data, matched identities and scientific coverage.
Inspect synchronized media and wide/narrow layouts. Preserve aspect ratios and readable labels.
Rebuild from saved artifacts with model, VAE and queue loaders blocked.
A missing required input must produce a named error and zero model or decoder jobs.
The report validator runs after each change. Its result does not replace the native gates below.

## G1: native baseline reproduction

Reconstruct the exact capture, VAE, fps, geometry, context bytes, seed and saved epsilon.
Use one bidirectional block and the exact `[sigma,0]` schedule.
Inspect the actual float32 sigma and token timesteps, BF16 latents and clean capture frame zero.

Calibration forwards must reproduce their saved D0 outputs.
The scorer rejects nonfinite maximum differences and values above 0.02.
Use the existing avatar `prune_baseline` queue experiment for native reproduction.
Saved noise and D0 tensors must be compared directly across old/new runs.
Equal seeds alone do not prove equal noise.

The saved completion reader checks structure, coverage, hashes, finite BF16 tensors and exact c0.
It opens no model and cannot regenerate device-specific CUDA noise.
The wrapper separately checks full component and software pins.
See [producer verification](evaluate/whole_clip_baseline.md#saved-completion-verification)
and [queued baseline](usage.md#queued-baseline-reproduction).

## G2: artifact identity and held-out scope

Require `whole_clip_d0_mask_v1` for width or `whole_clip_d0_depth_v1` for intact-block removal.
Check native task, attention, conditioning, model/VAE identity and full calibration-manifest content.
Check source, noise and context identities.

Exclude calibration actors across clips, Parts, views and aliases.
Also reject a held-out capture reused from calibration.
Held-out sigmas must occur in calibration.

The selected baseline must retain the pinned calibration rows for distribution validation.
The saved quality analyzer visits all rows. Identify held-out rows separately in the report.
See [native input contract](data/whole_clip.md) and [mask checks](score/hooks.md).

## G3: export parity

Use identical saved inputs for the in-memory intervention and exported model.
Width reference is the functional mask.
Depth reference is the ordered retained block list.
Load one transformer at a time.

The default gate requires matching shapes and maximum absolute latent difference at most 0.02.
Report relative L2 and finite values as additional evidence.
Include a no-prune export control.
A changed normalization or fitted weight requires its own reference.

Depth architecture verification checks inventory, config, mappings and accounting.
It does not compare payload bytes or establish numerical parity.
The native parity gate remains separate.
See [export parity](checks/export_parity.md).

## G4: held-out decoded quality

Report generated-frame direction relative L2/cosine, capture latent MSE and prediction disagreement.
Exclude clean frame zero.
State sample size, actor split, seeds and exact schedules.

Inspect synchronized capture VAE, baseline D0 VAE and candidate D0 VAE panels.
Judge identity, detail, pose, motion and temporal consistency.
Identify VAE decoding explicitly. Include matched capture RGB when a report assesses original capture fidelity.
Verify preprocessing coordinates and source hashes before interpreting alignment.

Latent metrics measure numerical agreement. They do not prove correct action or useful motion.
Predeclare acceptance thresholds before blind tests.
See [comparison logic](evaluate/whole_clip_d0.md).

## G5: measured cost

Measure repeated warmed and synchronized forwards on the same GPU.
Keep input, software, precision and numerical policy fixed.
Use ABA and BAB brackets when gains are close to measured drift.
The benchmark requires at least one warmup and three timed calls.

A speed claim must exceed drift and timing spread.
Report checkpoint bytes, resident-video parameters and peak allocated memory separately.
A compact file or mask is not a speed result.
See [benchmark logic](evaluate/bench_whole_clip_d0.md).

## Worked evidence checks

- Rehashing a malformed tensor does not make its shape, dtype, finiteness or c0 valid.
- A depth export can match retained source blocks while losing quality against the full teacher.
- A local ridge fit can reduce projection MSE without improving held-out model output.
- A benchmark can improve its median within the bracket drift. That is insufficient speed evidence.

## Test ownership and limits

The [module index](README.md) links each production owner and its Tests section.
Depth tests use real small video/AV models, strict reload, cache comparisons and backward gradients.
Reconstruction tests compare dual ridge with an independent primal solve.
Saved-baseline tests exercise a tiny real transformer and model-free completion refusals.

Those CPU results establish focused correctness.
Native BF16 baseline, export, quality and timing evidence are independent acceptance gates.
Do not label a candidate accepted while a required gate remains unrun.

## MobileWan helper checks

The focused CPU controls compare exact projected energy with independent direct projection.
They verify channel cancellation, both full token spans, equal-view pooling and actor-cluster intervals.
They also verify exact official-grid preservation and nested global allocation with stable ties.
The source audit rejects changed membership pins and records all row failures.
Native dev direct-call parity, hook transparency, full dataset scoring and held-out media remain unverified.

Direct dev CPU producer checks verify one call, exact sigma, missing-guide independence and saved completion.
Per-view score tests reject missing blocks, heads, sigmas, nonfinite energy and changed noise identities.
Native queue acceptance and full-span dev parity remain separate gates.

## Learned head controls

The focused CPU controls cover exact +3 initialization, independent microbatch graphs and analytic sparsity gradients.
They compare tiny legal-count selection with brute force and verify binary-forward STE gradients.
The official FP32 loss fixture matches `FlexibleStrategy._compute_modality_loss`.
Other controls verify original-ID adapter/moment slicing, zero delta, exact resumed updates and fusion payloads.
GT region tests cover source coordinates, expansion, crop mapping, visibility and empty coverage.
Teacher tests verify immutable reuse and bounded evidence storage.

Native feasibility remains required at both 18 and 28 latent frames and all eight sigmas.
It separates ordinary wrapper parity, all-one gates, zero adapters and the attenuated sigmoid(3) student.
Ten complete-clip backwards per length must leave at least 2 GiB of device headroom.
Execution probes test the three declared head layouts, all-one export and FFN-only widths.
Use both ABA and BAB; drift above two percent increases repetitions to twenty.

Check structural transfer, adapter transfer, fusion and merged reload independently.
Keep failures rather than treating later recovery as proof of earlier parity.
Merged reload must have maximum absolute difference at most 0.02 for every case.
Recovered validation requires complete update/draw coverage and full source/noise/context/tensor hashes.
Required decoded region metrics and reviewed videos cannot pass with missing coverage.

## Uniform-F18 DDP and logging

The four-rank CPU DDP fixture compares real reducer gradients with independent serial four-draw accumulation.
It includes native-style non-reentrant checkpointing and hook-owned gates, verifies frozen state and exact checkpoint resume.
Crop fixtures check F18 identity, nonzero longer-source windows, local conditioning, region alignment and saved source hashes.
Logger fixtures check one rank line per series and globally averaged losses and unique validation cases.
No online W&B request or GPU workload is needed for those CPU checks.

`checks.ddp_native` is the separate native BF16 gate for a newly frozen DDP study.
It checks F18 and cropped-F28 sources, nonzero crop conditioning, all-one parity, soft/binary gradients, memory and exact resume.
The saved acceptance must match the study, DDP method and configured world size before sustained training.
Two-rank full-F18 memory evidence does not establish four-rank crop execution or full-source validation memory.
Keep those native checks pending when the user has stopped GPU training.

## Progressive gate removal

Before sustained progressive training, require all-free parity, exact removed-zero execution and zero frozen gradients and Adam moments.
Require legal irreversible decisions and exact resumed native updates across a decision, including checkpoint recomputation.
CPU checks alone cannot satisfy this gate. Save the native `progressive_execution` acceptance flag under the new study identity.
Every 50 formal updates, inspect events, losses and actual-state capture MSE. Few eligible removals limit the evidence.
Latent MSE does not establish decoded video quality.

Zero/one continuation additionally requires bit-identical parent/migrated native predictions,
exact one execution and zero fixed-one logit gradients/moments. The disposable event must
select three zeros and two ones. Resume must reproduce the event and next update exactly.
