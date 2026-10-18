# Retained plan history

Consolidated on 2026-10-17. This is a provenance record, not a new experiment report or execution plan.
No historical number below is newly measured or promoted to current acceptance.
The [dormant backlog](RESEARCH_BACKLOG.md) retains unfinished ideas.
The [workspace handoff](../../plans/2026-10-07-onestep-avatar-development-experiment-handoff.md) owns avatar status.

## Source attribution and cleanup

The source revision is LTX-2 `77c5c6a5694cdd52912c46ec4d663d8af150ee57`.
Every removed tracked file matched that revision before cleanup.
The [inventory](cleanup_2026-10-17.tsv) records all 23 original paths, byte counts,
SHA-256 hashes, Git blobs, retained destinations and removal reasons.
It includes the replaced README and the four relocated evidence files.
The [cleanup validation receipt](../../expr/onestep_avatar/plan_cleanup_20261017/cleanup_validation.json)
records the original-inventory comparison, retained hashes and local-link checks.
Isolated controls confirmed refusal of restored obsolete files and changed evidence.

Recover a historical source for inspection with `git show <revision>:<original-path>`.
Use the revision above and a path from the inventory. Old executable scripts are retired.
Inspect their Git bytes or save them as non-executable provenance outside the live package.
Do not restore them into this folder or use old plans as current launch instructions.

The removed sources have these roles:

- `2026-10-04-prune-finetune-causal-avatar.md`: compression, task conditioning and baseline proposals, with dated CPU/GPU progress.
- `2026-10-12-onestep-avatar-test-suite-for-agents.md`: implemented CPU test optimization and measurements; three options remain deferred.
- `d1-next-actions.md`: September consolidation; several completed actions still had prospective instructions.
- `archive/d1-2026-09-26/`: six superseded plans and their index. The appearance-jump analysis contains original diagnostic measurements.
- `assets/2026-09-27-d1-implementation-review.md`: metric corrections, five-pair continuation measurements and decoder caveats.
- `assets/2026-09-26-d1-boundaries/`: seven investigative scripts and four saved evidence assets.

Historical model manifests that name a removed plan remain unchanged.
Resolve their source attribution through the inventory and pinned revision.
Experiment snapshots, native results and their producer hashes were not rewritten.
The [read-only validator](validate_cleanup.py) rejects restored retired paths and changed evidence.
It also checks recoverable Git blobs and the replacement documents' local links.

## September D1 observations

D0 uses capture as the noising source; D1 uses the ARGAvatar render.
`c0` is the clean capture first-image input. GT history supplies capture frames;
generated history supplies earlier predictions. The following observations come from the dated September sources.

### Cache semantics and future influence

The appearance-jump analysis first reproduced the prompt-conditioning cache discrepancy on a small model.
The September review then recorded real-checkpoint divergence before eviction.
Clean history token timesteps were zero in both paths, but whole-model sigma changed prompt AdaLN.
At sigma 0.909375, block-1 relative latent L2 differences were 0.12084 for D0 and 0.11428 for D1.
At sigma one, both were 0.23675. Block 0 and cached repeats agreed exactly.
These are semantic differences, not proof that recomputation fixes visible jumps.
The current [known-gap description](../scripts/onestep_avatar/doc/known_gaps.md#cached-history-and-causal-recomputation) retains that distinction.

The completed fixed-length intervention replaced only generated noise frames 3–8.
Early joint outputs changed by relative L2 0.23431, 0.31235 and 0.30999 across three replacements.
Joint repeats and causal first-block controls were bit-exact.
This establishes future-noise dependence for those joint runs, not a cause of appearance jumps.
The surviving [intervention report](../../expr/onestep_avatar/joint_future_noise_influence_20260926/REPORT.md)
and [current guide-study rerun](../../expr/onestep_avatar/ar_sigma_rollouts_rerun_20261012/REPORT.md) provide further context.

### Continuation and metric corrections

The September 27 review checked five cache/joint pairs at sigma 0.909375.
They used the same frozen distilled base, white objective, generated history and saved global noise.
The exact tail was `[0.909375, 0.725, 0.421875, 0]`.
Each 129-frame output had two generated blocks with identical first nine latent frames.
The transition was at pixel frame 65.

The GT-adjusted residual was `mean(abs((pred[9]-pred[8])-(GT[9]-GT[8])))`.
The recorded cache-to-joint values were:

- Actor `0008_01`, seed 42: 0.237721 to 0.228905, a 3.71% reduction.
- Actor `0012_09`, seed 42: 0.133605 to 0.116815, a 12.57% reduction.
- Actor `0025_11`, seed 42: 0.171071 to 0.160959, a 5.91% reduction.
- Actor `0008_01`, seed 43: 0.232820 to 0.230205, a 1.12% reduction.
- Actor `0008_01`, seed 44: 0.233089 to 0.224094, a 3.86% reduction.

The alternative `mean(abs(pred[9]-pred[8]))` measures change magnitude, not GT-adjusted error.
Both are spatially unaligned diagnostics; less motion can improve them.
RGB boundary magnitude improved in three pairs and worsened in two.
The user selected cached history; joint and recomputed history remain diagnostic paths.
These observations do not reopen their deployment selection.

The later September action list records the four-case source-by-sigma sweep as complete.
Its original summary report is now absent at its cited path.
Its sigma recommendation is historical, not a current default.
Use the fresh guide-study rerun for current evidence; keep forcing policies and schedules separate.

### Decoder caveat

The September action list records bit-exact repeat decoding at fixed shape.
It reports future-content effects near the endpoint, and larger short/full differences with the same seed.
The cited decoder report is unavailable at its original path, so these remain attributed observations.
Changing noise shape while keeping a seed does not isolate decoder context or padding.
Canonical absolute-coordinate decoder noise was not tested in that record.
The [streaming backlog](RESEARCH_BACKLOG.md#end-to-end-streaming-and-detail-preservation) preserves this unresolved question.

### Retained boundary evidence

Four files were relocated byte for byte to the workspace's ignored evidence directory:

- [Per-video measurements](../../expr/onestep_avatar/plan_cleanup_20261017/evidence/2026-09-26-d1-boundaries/boundary_metrics.csv).
- [Aggregate measurements](../../expr/onestep_avatar/plan_cleanup_20261017/evidence/2026-09-26-d1-boundaries/summary.json).
- [Small-model sigma-one controls](../../expr/onestep_avatar/plan_cleanup_20261017/evidence/2026-09-26-d1-boundaries/sigma1_invariance.json).
- [Historical contact sheet](../../expr/onestep_avatar/plan_cleanup_20261017/evidence/2026-09-26-d1-boundaries/official_sigma0909375_contact.png).

The appearance-jump analysis measured nine views from three clips at sigma 0.909375.
Its three-step frozen-base arm had boundary/interior temporal residuals of 0.07102/0.06278 for D0
and 0.07687/0.06752 for D1. The metric subtracts the GT frame difference before taking absolute error.
Panels were downsampled to 256 square; source-derived foreground masks were dilated seven pixels.
Boundary regions used plus/minus two frames; each video contributed 35 boundary and 89 interior transitions.
There was no optical-flow compensation. Motion, alignment, compression and background affect this diagnostic.
The CSV also contains older one-step LoRA rows; they cannot establish a same-base sampler comparison.

The retained sigma-one JSON records four small-model combinations of one/eight steps and GT/generated history.
All record zero source difference and exact clean-first-image preservation.
These controls do not establish full-checkpoint video quality.
The two JSON files and PNG were ignored before cleanup; Git never contained their bytes.
They remain local evidence, with hashes in the inventory, and are not added to version control.

## October compression and conditioning record

The October roadmap cites a method screen with 27 blind actor/seed/sigma cases per finalist; none passed all gates.
It records about 1.090x speedup for four-block bypass while retaining tensors and failing quality.
An aligned FFN reduction reached 1.056–1.057x but failed functional/export parity; one-head export was slower.

The October 4 roadmap reports a physical export removing original blocks `[3,7,8,15]`, leaving 44 blocks.
It records 8.19% fewer resident-video parameters and 7.36% less stored tensor payload.
The METHODS inventory records 13,123,337,344 original and 12,048,169,856 retained resident-video elements.
All 4,013 retained tensors were recorded as byte-identical; file sizes were 42,018,190,584 and 38,923,958,248 bytes.
The roadmap's later GPU entry reports exact source/export parity for actor 19 at sigmas 0.725, 0.909375 and 1.
Allocated peaks were 28.343 GiB for the ordered-source intervention and 26.340 GiB for the export.
Its earlier checklist still called native parity pending. The later entry supersedes that dated status.
The deletion set retained its failed quality gate; no recovered or speed-qualified candidate was established.

The roadmap also records twenty genuine 129-frame, 512-square pose references encoded as BF16 `[128,17,16,16]`.
That entry supersedes its earlier missing-reference status, but does not establish teacher-training quality.
Its MimicMotion sample covered one actor/view, 33 frames at 512 square and a cold call of 81.752 seconds.
It used capture-derived conditions and FPS microconditioning 7 with playback 30, limiting any common-task timing claim.

The five October evidence bundles are absent at their original paths under `expr/refiner_prune/2.5/`:
`next_stage_20261004`, `data_audit_20261004`, `depth_export_20261004`,
`pose_condition_20261004` and `baseline_inventory_20261004`.
The source screen and whole-clip reports cited by the roadmap are also absent there.
These facts are retained as dated source claims; this cleanup cannot revalidate their missing native receipts.
The roadmap's fit audit recorded two eligible moderate-clothing training actors instead of the planned four,
no fitted-clothing coverage and test-actor overlap with its separate D0 corpus training.
Do not infer a complete data foundation or unseen-actor qualification from the CPU preparation record.

## Completed CPU test optimization

The October 12 implementation is committed in `40f52fb`, superseding its plan's no-commit statement.
It pins tiny-model CPU threads, selects T0–T3 checks, consolidates child processes,
shares software metadata and tamper builders, and uses unique process tokens.
It replaces the separate claims/TTL test implementation with the production process registry.
Scientific tolerances and supervision deadlines were unchanged.

The [saved evidence](../../expr/onestep_avatar/test_suite_20261012/README.md) records 2,104 avatar tests
and 198 subtests in 178.56 seconds, 219 prune tests with one skip, and 95 blocked boundary tests.
The 100–120 second target was an estimate that was not reached.
Ten selector faults, ten profile inventories, thirteen help rows, six dry-run rows,
eight refusal controls and twenty supervision repeats passed in that dated task.
These counts are historical, not the current collection census.
The three [deferred options](RESEARCH_BACKLOG.md#deferred-cpu-test-tool-options) remain separate from the implemented selector.
