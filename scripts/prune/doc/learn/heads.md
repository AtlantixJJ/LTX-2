# learn/heads

## Clipped-reset execution

The explicit clipped-reset method constructs all-one raw scores with `clip01_v1` activation.
Sustained training requires a fresh four-rank acceptance receipt bound to the exact policy config hash.
The selection owner keeps global updates through the adapter-only recovery tail; compact recovery is a separate workflow and is rejected here.
The gate optimizer stops stepping during recovery. Adapter warmup and moments retain their global counters.
The task graph keeps a masked zero-gradient gate path so DDP continues to see its registered parameters.

At each selection boundary, publish `transactions/pre_event_NNNN.pt` before the reset.
It holds pre-event scores, adapters, both optimizer states, rank RNG, and the pending policy.
Publish the event receipt after reset, then the canonical checkpoint and actual-state monitor.
A restart replays from its last canonical checkpoint and verifies any already-published transaction or receipt.
Pre-event transaction files never match the canonical `checkpoints/update_*.pt` search.
Trace row zero contains post-optimizer scores before the event reset. Canonical boundary checkpoints contain post-reset scores.
Receipts label both representations and keep gate Adam local steps separate from global updates.
Initial checkpoint-zero monitoring uses the actual all-one gates.

## Objective

Run bounded resident-model phases for the learned-head experiment.

## Usage

Started by: a direct human or agent CLI. Run from `LTX-2` in conda environment `ltx`.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx python -m scripts.prune.learn.heads --study ../expr/refiner_prune/2.5/learned_heads_dev8_20261012_r2/configs/frozen_study.json --phase feasibility --gpu-id 0
```

`--phase` selects one bounded phase. `--arm`, `--replicate-seed` and `--removed-heads` select the frozen arm. `--branch-prefix` reuses a verified soft-phase prefix. `--max-updates` bounds a smoke run.

Outputs: Phase receipts, learning state, observations, monitoring predictions and timing probes.
See the [run guide](../usage.md#learned-head-phases).

## Data flow

The frozen study selects explicit dev components and draws. The shared input owner constructs every complete native D0 input.
`training_input_case` applies the frozen F18 crop only to optimizer inputs.
The current default uses full-replica DDP with one draw per rank and native gradient checkpointing.
Four ranks consume four distinct draw indices; no rank duplication occurs.

## Organization logic

Feasibility checks native velocity conversion, all-one gates and zero adapters. Zero-adapter parity runs under `no_grad`; only optimizer forwards enable autograd. Ten backward updates test each complete clip length.
Execution probes export three 152-head layouts and four FFN widths. ABA/BAB timing repeats twenty times when drift exceeds two percent.
The frozen config sets the budget: `soft_updates`, `binary_updates` and `compact_updates` give global boundaries through `protocol.phase_budget`. The default preset uses 700 soft and 300 binary updates; an extended study can use 2,000 soft and 0 binary. Soft-only keeps soft gates through the selection end. Fixed energy and random arms ramp for 300 updates.
Recovery slices adapters and Adam moments at original head IDs. Fresh controls reset adapters and moments. Sequential recovery adds 1,000 adapter updates beyond the compact end.
Every attachment and fresh source-state reset consumes the frozen adapter rank, including rank 128.

## Invariants

The caller enables autograd locally. Base weights stay frozen. Non-reentrant block checkpointing is enabled. There is one legal decision per optimizer update.

## Gotchas

A failed memory or execution gate prevents sustained speed training. Timeout retains progress. Regional training rejects unavailable reviewed tracks. Monitoring never creates teacher substitutes.

## Tests

Run `test_learned_gates.py, test_head_learning_state.py, test_bench_learned_heads.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.

## Numerical and execution records

Feasibility saves native velocity, ordinary clean predictions and initialized sigmoid-gate predictions at each exact sigma.
Its ten optimizer updates per length use two sequential backward graphs with unchanged logits per update.
Execution probes preserve functional-mask predictions separately from dense and compact outputs.
They record both the functional/export difference and the intentional dense/pruned difference.
Fresh and sequential arms retain their extra-compute lineage.
Monitoring uses precomputed original-teacher references on four validation actors at sigma indices 0, 4 and 7.
It saves trained and hardened predictions with the same adapter weights at each selection checkpoint.
A timeout checkpoints the latest completed update; completed evidence remains immutable.

## Native placement diagnostics

The four-GPU diagnostic wraps the shared Session with twelve consecutive blocks per device.
Feasibility compares every native prediction against saved single-GPU references before optimization.
It publishes each completed backward update and per-device peak allocated/reserved bytes and free headroom.
It rejects zero gate/adapter gradients and changes to frozen parameter version counters.
Failures retain current clip length, update and sigma in the diagnostic receipt.

## Sustained four-GPU execution

The frozen `native_placement` selects the same twelve-block layout for feasibility, selection and compact recovery.
Teacher references, calibration and speed probes retain independent single-GPU execution.
Training records allocated, reserved and free bytes on each used device after every update.
Every device must retain the frozen two-GiB minimum headroom. CUDA RNG state spans all adapter and gate owners.
Device memory and timing are excluded from exact replay comparison; scientific update records must agree.
Phase GPU-hours count all resident execution devices.

## FSDP and GT monitoring

`distributed_training=fsdp_frozen_replicated_adapters` selects a direct four-process torchrun launch.
FSDP shards frozen BF16 blocks and ignores the small FP32 trainable parameters.
Two ranks consume each of the two saved draws. One local sample loss is multiplied by two to undo accumulation division.
Average replicated gradients once, then use the existing separate clipping and AdamW owners.
Only rank zero publishes common updates. All ranks contribute RNG state and participate in monitoring forwards.
`monitor_reference=gt_capture` removes teacher-cache and speed-gate requirements for this declared quality experiment.
Monitor official velocity MSE, generated-frame capture MSE and exact clean-frame equality in trained and hardened states.
Use shared-input file locks to prevent duplicate ranks from publishing half-written input bundles.

The GT preset defers foreground replay and bucket preparation because its main loss consumes only capture latents and noise.
Regional and balanced arms reject that deferred scope. They require a separately prepared study.

## Raw-logit sparsity

The frozen config selects the sparsity penalty and logit offset. Pass the raw FP32 logits to the shared loss owner.
Record the penalty form, offset and actual warmup coefficient with each update.
Pass the frozen ramp duration and start multiplier to the shared coefficient owner for loss, gradient and receipt.
The adapter LR uses its existing phase warmup. Gate LR is constant from the first update.
Use the matching analytic sparsity gradient before clipping to separate velocity and sparsity gradient norms.
Save `sparsity_logit_gradient_norm` and `velocity_logit_gradient_norm` after distributed gradient averaging and before gate clipping.
The sparsity gradient includes the current lambda. The velocity component is the total gradient minus the analytic sparsity gradient.
The shared W&B writer logs both as `train/` scalars at every update.
Continuation restores the saved optimizers, masks, RNG state and global update count. A changed loss requires a separate frozen study.
The continuation owner preserves source update receipts in checkpoint lineage; never rewrite their original study identities.

## Frozen sigma grid

Read the exact noise levels from the frozen study, never from the historical score grid.
The current ten-level preset omits sigma 0.1 and includes all five official dev levels in [0.9, 0.95].
Its input manifests declare the complete frozen training grid. Draw indices select that grid.
Quick monitors use explicit `monitor_sigma_indices` when supplied; historical configs default to zero, four and the last level.
The ten-level preset selects indices [0,7,9], preserving sigmas 0.9489720463752747, 0.7663628458976746 and 0.48016369342803955.
For a fresh run, initialize adapters, logits, optimizer moments and RNG state from the declared seed.
Updates up to `soft_updates` use soft gates. The next update starts binary refinement with sparsity disabled; with zero binary updates, compact recovery starts directly from the soft selection checkpoint.

## Uniform F18 and DDP

The `ddp_full_replicas` preset requires `uniform_f18=true`, accumulation one and a four-draw global batch on four GPUs.
`learn.protocol` selects each rank's draw and checks complete schedule coverage.
`data.training_windows` slices aligned continuous capture/noise tokens, resets positions and inserts the selected window's clean first latent.
Preserve full source manifests. Each consumed draw records window bounds, coordinate policy and cropped tensor hashes.
Resume verifies both full source bytes and the actual consumed slices.

`TrainingStep` registers native model and pruning logits together inside the DDP forward.
The local loss has no accumulation division. DDP averages gradients once across ranks.
The runtime checks those gradients before separate clipping and Adam steps; it never reduces them a second time.
The four copies have identical trained state. Freeze gate parameters when no gate optimizer is active.
Checkpoint cursor uses four consumed draws per update; legacy FSDP retains its two-draw contract.

Quick GT monitoring assigns different actors to ranks and gathers all unique actor/sigma cases into one receipt.
Monitoring still uses full source clips. It runs native no-grad forwards directly on replicas, without a DDP backward.
Each rank publishes only its own case tensors; rank zero publishes the gathered monitor JSON.
Every optimizer receipt contains global draws and separate rank sigma/timing records for W&B.
Total loss and validation errors are global means. Rank sigma and memory appear as multiline charts.

The recipe keeps rank/alpha 128, gate LR 0.01 and `lambda(t)=0.01*(0.1+0.9*min(t/100,1))`.
Existing FSDP, block-placement and older losses remain explicit reproduction paths.

## Gate trace

When gate logits train, the primary rank writes `training/<arm>/replicate_<seed>/gate_trace/NNNN.pt` for every update.
It is a (3, blocks, heads) FP32 tensor: row 0 holds the logits after the optimizer step, row 1 the total logit gradient
after DDP averaging and before clipping, and row 2 the analytic sparsity gradient (zero outside the soft phase).
The velocity part is row 1 minus row 2. A replayed update must reproduce the same tensor; the update receipts are unchanged.
Report code uses the trace for per-update regret and direction changes; without it, analysis falls back to checkpoints.

## Progressive gate removal

The `learned_progressive` arm records every post-update logit matrix and commits global removals at the configured interval.
It publishes event receipts before canonical checkpoints. Resume validates the complete saved event history and replays receipts exactly.
After each event and at the final update, monitoring saves only the actual mixed state at the three source sigmas for the first four validation actors.
It uses seed 2042 and preserves Python, NumPy, CPU and local CUDA RNG state, including on failure.
A resumed boundary repeats its immutable monitor before continuing. It never globally hardens free progressive gates.
Source, noise, context and prediction hashes accompany each case.

Progressive training bypasses legacy removal budgets and legal-width checks. It records no target total removal count.
Canonical checkpoints and actual-state monitors follow the configured interval and always include the final update.

Zero/one continuations restore the versioned policy from the saved global checkpoint.
Each update logs zero and one selection/eligibility/shortfall fields separately.
The global update owns learning-rate and sparsity schedules; migration does not restart warmup.
After the free set is empty, zero logit gradients remain valid and LoRA training continues.

## Eligible-population continuation

The explicit eligible policy selects each percentage from its own eligible branch, with a configured minimum count.
See [eligible policy](eligible.md) for the terminal override and exact fixed-gate tail.
The trainer stops at the saved dynamic final update. Completion at the draw envelope without a terminal event is an error.
