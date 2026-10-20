# learn/ddp

## Objective

Run one uniform-F18 training draw per GPU with full native model replicas.

## Data flow

`DDPRuntime` extends `learn.ddp_owner.Runtime` for exact device assignment, one 15-minute NCCL group and record collectives.
Torchrun assigns ranks and local devices. Each process owns the frozen BF16 backbone, FP32 adapters, gates and Adam states.
`TrainingStep` registers the model and hook-owned gates together. DDP sees both the velocity and sparsity gradient paths.
One local undivided loss enters one backward. DDP averages trainable gradients across ranks.

## Organization logic

Use `broadcast_buffers=False` and `gradient_as_bucket_view=True`. Keep native non-reentrant block checkpointing enabled.
The reducer owns gradient averaging. `reduce_gradients` checks gradients; it never applies a second all-reduce.
Register gates as trainable only when the phase has a gate optimizer. Fixed and compact phases optimize adapters alone.
Use inherited coordination for stop flags, legal masks, state agreement and checkpoint publication.
Gather one consumed draw per rank in rank order. Four ranks yield four draws per update.
Rank zero saves a checkpoint with all ranks' RNG states and the actual draw count.

```mermaid
%%{init: {"themeVariables": {"fontSize": "14px"}}}%%
flowchart LR
  I("four saved source/noise draws") --> C["one F18 crop per rank"]
  C --> M["full model replica and loss"]
  M --> G["DDP mean gradients"]
  G --> U["same optimizer update on each rank"]
  U --> R(["rank receipts and checkpoint"])
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef tensor fill:#dcf5df,stroke:#43854b,color:#173b1c;
  classDef out fill:#ece0f8,stroke:#7048a0,color:#26123f;
  class C,M,G,U proc;
  class I tensor;
  class R out;
```

## Invariants

The frozen protocol requires uniform F18, accumulation one and rank assignments `[0,1,2,3]` on four GPUs.
Two-rank diagnostic studies can use `[0,1]` and two draws. A different world size fails before training.
Every rank stores a complete model. No frozen weight sharding or duplicate draw assignment occurs.
The sparse penalty is included once in the averaged update, rather than divided by the world size twice.

## Gotchas

F18 memory evidence from two GPUs does not prove native four-rank or long validation acceptance.
DDP broadcasts during setup change tensor version counters. Record frozen versions after wrapping.
Native no-grad validation runs directly through each replica; it does not activate the DDP backward reducer.

## Tests

`test_ddp_training.py` compares actual four-rank CPU DDP gradients against serial four-draw accumulation.
It checks block checkpointing, hook-owned gates, frozen weights, equal rank state and exact resumed updates.
`checks.ddp_native` owns the separate BF16 crop, gradient, memory and native resume acceptance.
