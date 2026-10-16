# learn/run_full

## Objective

Run the frozen GT training recipe through direct prune CLIs and one CPU logger per arm.

## Usage

Launch directly in ltx from LTX-2. No avatar queue is used.

```bash
CUDA_VISIBLE_DEVICES= CUBLAS_WORKSPACE_CONFIG=:4096:8 setsid python -u -m scripts.prune.learn.run_full --study /abs/frozen_study.json > /abs/run/logs/full.log 2>&1 < /dev/null &
```

Arguments select the study, arm list and replicate seed. Defaults are learned_joint and energy, seed 42.
Use `--arms learned_joint` for the learned arm alone.
Outputs include phase logs, immutable launch records, mutable thirty-second `monitoring/status.json` and PID records.
See the [run guide](../usage.md#uniform-f18-ddp-training). A user-stopped run remains stopped until the user requests a launch.

## Data flow

The frozen config selects uniform-F18 DDP or the retained legacy FSDP method.
CPU historical score reuse, native acceptance, selection, export, compact recovery and final merge follow the same declared study.
The DDP recipe uses one F18 optimizer draw per rank. Source manifests, calibration and quality evaluation retain full sources.

## Organization logic

Read devices and world size from the config. Check free memory before initial native loading.
Select `checks.ddp_native` for uniform-F18 DDP; select `checks.fsdp_native` for FSDP.
Require matching study identity and method. DDP acceptance must also match the frozen world size.
Native acceptance runs separately and its trained tensors never seed formal training.

Start the configured ranks through torchrun. Internal twelve-hour learning bounds save the latest completed checkpoint.
An external thirteen-hour timeout has ten-minute termination grace.
Resume only TimeoutError exits with increased published progress. Other errors stop the assigned run.
The frozen config sets the phase budget through `protocol.phase_budget`. Selection ends at `soft_updates + binary_updates` (1,000 for the 700/300/500 preset) and recovery ends 500 updates later. Export the compact base between phases and merge afterward.
Compact export lazily saves structural inputs on GPU zero without model forwards; CPU surgery follows.

When `wandb.enabled=true`, start one CPU receipt logger per arm with an arm-specific W&B run.
The logger owns rank line-series charts and averaged global metrics. It does not load a model or decoder.
The owner publishes a final stopped/failed/completed status; loggers close after their final receipt pass.
If a logger has not exited, termination is bounded after a thirty-second wait.

## Invariants

Use only the frozen authorized devices, normally GPUs zero through three. No teacher preparation or speed testing in GT mode.
No avatar queue or unrelated process control. Do not edit production sources during a live frozen study.
W&B metric tabs omit arm prefixes; separate runs avoid collisions between learned and energy metrics.
Numeric update filenames are the only published training receipts. Ignore atomic temporary files.

## Gotchas

A native diagnostic does not establish pruning quality. Numeric validation does not establish perceptual quality.
Historical energy can cover only part of the learning pool; the imported record declares actual coverage.
The earlier two-rank F18 memory pass is supporting evidence, not native acceptance for a new four-rank crop study.

## Tests

CPU DDP tests check averaging, frozen state and exact resume. Crop tests check source alignment and window identity.
Logger tests verify rank charts and global means. Native DDP acceptance remains a separate authorized GPU gate.

## Progressive gate removal

A progressive study accepts only `--arms learned_progressive --replicate-seed 42`.
It requires fresh progressive native acceptance, runs 500 soft updates and their saved monitors, then ends.
It performs no compact export, binary training or recovery. Read `progressive_training_complete` as training completion only.

Zero/one continuations use global soft budgets and require fresh fixed-one execution/migration acceptance.
The owner can cross bounded twelve-hour segments by resuming its last canonical state without counter reset.
It runs no binary refinement or compact recovery for this policy.
