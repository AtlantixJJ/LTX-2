# learn/heads

## Objective

Run bounded resident-model phases for the learned-head experiment.

## Usage

Started by: a direct human or agent CLI. Run from `LTX-2` in conda environment `ltx`.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx python -m scripts.prune.learn.heads --study ../expr/refiner_prune/2.5/learned_heads_dev8_20261012_r2/configs/frozen_study.json --phase feasibility --gpu-id 0
```

`--phase` selects one bounded phase. `--arm`, `--replicate-seed` and `--removed-heads` select the frozen arm. `--branch-prefix` reuses a verified 700-update prefix. `--max-updates` bounds a smoke run.

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
Selection uses 700 soft and 300 binary updates. Soft-only uses 1,000 soft updates. Fixed energy and random arms ramp for 300 updates.
Recovery slices adapters and Adam moments at original head IDs. Fresh controls reset adapters and moments. Sequential recovery uses 1,500 adapter updates.
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
Continuation restores the saved optimizers, masks, RNG state and global update count. A changed loss requires a separate frozen study.
The continuation owner preserves source update receipts in checkpoint lineage; never rewrite their original study identities.

## Frozen sigma grid

Read the exact noise levels from the frozen study, never from the historical score grid.
The seven-level preset omits sigma 0.1. Its input manifests also declare only these seven levels.
Draw indices select the frozen grid. Quick monitors use indices zero, four and the last selected level.
For a fresh run, initialize adapters, logits, optimizer moments and RNG state from the declared seed.
The first 700 updates use soft gates. Update 701 starts binary refinement with sparsity disabled.

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

The recipe keeps rank/alpha 128, gate LR 0.1 and `lambda(t)=0.1*(0.1+0.9*min(t/100,1))`.
Existing FSDP, block-placement and older losses remain explicit reproduction paths.
