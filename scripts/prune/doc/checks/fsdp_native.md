# checks/fsdp_native

## Objective

Accept four-rank native FSDP for unchanged whole-clip learned-head training.

## Usage

Run directly in ltx from the LTX-2 root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc_per_node=4 -m scripts.prune.checks.fsdp_native --study /abs/frozen_study.json
```

Outputs are saved controls, update receipts, a native resumed-update receipt and checks/feasibility.json.
See the pruning run guide. This command has no avatar queue dependency.

## Data flow

The frozen study supplies one learning clip of each length, original components and exact sigmas.
Read all noise levels and index bounds from the study, including the seven-level preset.
An unchanged unsharded native model saves predictions before sharding. Those predictions provide the independent parity reference.
A frozen `native_baseline_study` can reuse those controls only after matching stock numerical dependencies, components, clips and noise schedules. GT capture and saved epsilon supply the loss.

## Organization logic

Verify ordinary native-wrapper conversion. Save both full lengths at every sigma before adapter injection and sharding.
Initialize zero-delta adapters and shard the frozen model.
Compare all-one predictions at both lengths and every sigma with the saved unmodified native velocity.
Run ten two-microbatch updates per length, five soft and five binary, with native checkpointing.
Average FP32 replicated gradients and clip adapters and gates separately. Verify frozen versions and two-GiB headroom on every rank.
Checkpoint after update one. Replay update two from that checkpoint and require exact adapter/logit equality before continuing.
Reset adapters, logits and optimizers before the next clip length.
Initialize and reset adapter factors at the frozen rank; never substitute the default rank 16.

## Invariants

No shortened clip, altered schedule, teacher training target or timing prerequisite.
Only rank zero publishes shared evidence. All ranks contribute memory and RNG evidence.
A failed control or replay leaves no passed feasibility receipt.

## Gotchas

Four-rank FSDP requires every rank to execute the same forwards and checkpoint collectives.
Earlier block-placement acceptance is an independent reference, not acceptance of FSDP memory or gradients.

## Tests

The optimizer controls consume the frozen sparsity coefficient, ramp duration, start multiplier, penalty form and raw logits.
Thus a new acceptance pass checks the configured loss instead of a hard-coded historical default.

The small distributed check precedes this gate. This entrypoint verifies actual BF16 native controls, full gradients and exact resume.
