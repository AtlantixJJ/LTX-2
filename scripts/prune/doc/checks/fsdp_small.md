# checks/fsdp_small

## Objective

Verify four-rank FSDP, two-clip gradients and exact checkpoint replay on a small BF16 model.

## Usage

Run directly from LTX-2 in the ltx environment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc_per_node=4 -m scripts.prune.checks.fsdp_small --output ../expr/refiner_prune/2.5/fsdp_small_check
```

The output contains an immutable checkpoint and passed.json. See the prune run guide.

## Data flow

Two deterministic clips enter serial accumulation and a four-rank FSDP model with identical frozen base weights.
The first update must match within the declared FP32 rounding tolerance. A saved checkpoint then replays the next update exactly.

## Organization logic

Use real ProjectionAdapter modules, frozen FP32 modulation tables, live head hooks, native FSDP and non-reentrant block checkpointing.
Require frozen version counters to stay unchanged and all ranks to agree on final logits.

## Invariants

No native checkpoint is loaded. Exactly two unique samples contribute to an update.
Checkpoint replay restores adapters, gates, both optimizers and each rank's RNG state.

## Gotchas

This small gate cannot establish native memory feasibility. It requires four visible GPUs.

## Tests

The entrypoint executes gradient equivalence and exact resumed update checks and raises on any failed condition.
