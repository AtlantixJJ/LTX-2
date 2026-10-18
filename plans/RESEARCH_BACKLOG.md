# Dormant research backlog

Consolidated on 2026-10-17 from the retired September and October plans.
These proposals are unrun or unqualified follow-ups, not an execution order.
They carry no old GPU allocation, launch permission or promise of acceptance.
The workspace [handoff](../../plans/2026-10-07-onestep-avatar-development-experiment-handoff.md)
owns avatar status. The [historical record](HISTORY.md) identifies the original sources.

## Persistent motion conditioning and a task teacher

D0 noises future capture frames. It is a diagnostic oracle, not a deployment input.
D1 noises the render guide. In the existing mixture, the guide has no weight at sigma one.
Preserving D0 predictions therefore does not establish a motion-controlled avatar product.

Propose a separate motion condition that remains available at sigma one.
Begin with camera-aligned 2D pose; compare MHR depth, normals or silhouette separately.
Compare ARGAvatar RGB as an appearance-rich alternative. Verify landmark ordering,
left/right topology, crop, occlusion and timestamps before training.

The October proposal uses the trainer's IC-LoRA reference-token path for an offline pilot.
It supplies the guide independently and noises the capture target during supervised training.
Its proposed velocity target is `epsilon - capture`, with clean first-image tokens excluded.
This is a candidate design, not a shipped avatar condition or runnable training recipe.
Streaming use needs explicit reference-token visibility and cache ownership.

First require a useful multistep task teacher on held-out actors.
A mismatched guide must degrade motion; the matched guide must preserve identity and detail.
Then compare compact task recovery at fixed steps before reducing the step count.
Do not transplant an adapter between dev, distilled and compact bases without requalification.

## Alternative compression and recovery

Recovered depth pruning and aligned FFN reconstruction remain alternatives to learned head pruning.
The October recommendation to prioritize depth is historical; current head studies do not settle this comparison.
The depth exporter exists, but selection and recovery remain separate scientific work.
See the current [pruning limits](../scripts/prune/doc/known_gaps.md).

For depth, retain original block IDs, remap block-indexed settings and verify missing tensors after reload.
For FFNs, compare source-order retained channels with hardware-aligned widths and calibration-only reconstruction.
Compare no recovery, reconstruction and teacher-output recovery on matched inputs and budgets.
Use separate checks for trained save/reload parity and preservation of teacher or task quality.

Report stored bytes, resident parameters, executed computation, memory and measured forward time separately.
Faithful padding can preserve a function without reducing its GEMM work.
An unchanged export's parity gate cannot qualify a trained model's quality.
Do not resume the retired October candidates from missing reports or checkpoints.

## Data quality and task evaluation

Split by bare actor identity across clips and views before selection or teacher training.
Check every reused teacher against its actual training membership.
The October reserved actors overlapped the then-running D0 corpus training; that split did not support an unseen-actor claim.

Audit crop, camera, matte, timing and guide/capture alignment in the recorded model coordinates.
Keep fitted, moderate, loose and unknown clothing labels separate from fit quality.
Compare quality filtering at matched training size before considering permanent garment exclusions.
Keep rejected cases and reasons in a reversible manifest.

Measure pose trajectory, identity, moving detail, guide following, resets and long-rollout drift separately.
Use source-derived regions, visibility confidence, capture RGB/VAE references and mismatched controls.
Latent MSE, gradient energy and adjacent RGB change remain supporting diagnostics.
They can reward blur, static output or incorrect appearance.
Aggregate by actor; frames and seeds are not independent people.

## End-to-end streaming and detail preservation

Keep the current cached-history contract; recomputed and joint history remain diagnostics.
The [core algorithm](../scripts/onestep_avatar/doc/core_algorithm.md) owns the implemented calculation.
The [known gaps](../scripts/onestep_avatar/doc/known_gaps.md) owns current implementation limits.
Do not reopen completed cache sweeps as prerequisites for every training experiment.

For a new condition, verify that every token path sees only available current and past inputs.
Audit future access in crop unions, leading-gap filling and full-sequence pose refinement.
Declare multiview reconstruction and offline-prepared driving streams explicitly.
Train and test generated-history continuations beyond eviction before claiming streaming quality.

Before online RGB emission, isolate decoder length effects with one canonical noise field indexed by absolute coordinates.
Compare short/full decodes using corresponding noise slices and an explicit padding convention.
Separately vary future latent content at fixed total length and noise.
Measure the required overlap and delay; preserve already emitted pixels.
This explicit-noise isolation was not completed in the retired September record.
Equal-length complete decoding remains appropriate for offline comparisons.

If a future loss comparison needs a frozen-base preservation term, give the teacher its own K/V cache.
Refresh both caches from the same student-generated history; do not reuse student features as teacher features.
Motion-aligned perceptual losses need explicit correspondence, visibility and gradient designs.
Adjacent-frame equality and high-frequency maximization are not substitutes for correct motion and detail.
Distribution distillation needs a verified score teacher and auxiliary score model; a few-step generator alone is insufficient.

Measure first-output, per-block and total latency, including condition preparation, history refresh and decoding.
Transformer causality or checkpoint size alone cannot establish end-to-end real-time operation.

## External baseline comparisons

Keep separate tracks for pose driving, rendered-RGB driving and optional pose-plus-audio driving.
Match permitted condition sources, reference image, physical timestamps, crop and common output coverage.
Retain native schedules and temporal conditioning; relabeling MP4 playback does not repair a temporal mismatch.
Equal seed numbers across architectures do not imply equal noise.

Historical MimicMotion and UniAnimate-DiT MHR outputs with wrong joint topology are excluded from matched comparisons.
Corrected guide files alone do not repair those generations.
Capture-extracted pose or face streams are privileged relative to render/MHR-derived driving.
Overlapping offline RGB clips do not demonstrate block-causal cached generation.
Score raw outputs separately from capture-mask or background postprocessing.
Start with matched zero-shot cases before funding several fine-tuning projects.

## Deferred CPU test-tool options

The T0–T3 selector, runtime pinning, shared test builders and child-process consolidation are implemented.
Use the [selector design](../scripts/onestep_avatar/doc/tests/select.md), not the retired proposal.
The [saved evidence](../../expr/onestep_avatar/test_suite_20261012/README.md) records acceptance and runtime limits.

Three optional items remain deferred from that task:

- Move the 28 report-reader tests to their study validator while retaining their assertions and ownership.
  They currently load gitignored `expr/` code, which makes clean-checkout testing depend on local study files.
- Evaluate `pytest-xdist` only after an explicit environment-change decision and process-lifecycle isolation checks.
- Consider a coverage comparison after an explicit environment-change decision.

This cleanup changes no tests, dependencies, deadlines or scientific tolerances.
