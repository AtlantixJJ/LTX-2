# learn/distributed

## Objective

Shard frozen native weights across four GPU processes and preserve the two-clip update contract.

## Data flow

Torchrun starts four ranks. Each rank loads the native BF16 video model and identical FP32 adapters.
FSDP wraps the velocity model and its transformer blocks. It ignores small trainable adapters and frozen FP32 modulation tables.
The stock Session preserves those native table values. Keep their original precision and replicate them rather than flattening with BF16 matrices.
Rank zero and two consume draw zero; rank one and three consume draw one.
Each rank computes the undivided sample loss. Averaged FP32 gradients therefore equal the mean of the two unique draws.

## Organization logic

Use FULL_SHARD, original parameters, backward prefetch and native block checkpointing.
Register Modality and TransformerArgs with the public torch.export dataclass API before wrapping.
The frozen-weight reshard hooks flatten inputs through pytree; registration exposes each block input activation.
Without it, gathered frozen blocks can survive until the final backward callback and consume excess peak memory.
Keep root and child input casting disabled. Preserve FP32 conditioning fields.
Release unused initialization cache after sharding so device-free headroom measures current model storage.
The replicated trainable states use an explicit averaged all-reduce before separate adapter and gate clipping.
Broadcast legal masks. Require all ranks to agree on stop requests and state hashes.
Only rank zero publishes common artifacts. Every rank contributes its RNG state to each resumable checkpoint.

## Invariants

Exactly four ranks and two unique full-clip draws per update. No teacher-cache or speed-test prerequisite for GT mode.
Frozen matrices stay BF16; native frozen modulation tables stay FP32. Adapter and gate masters and optimizer moments stay FP32.
No avatar queue or training-engine imports. Model input and loss owners stay in prune.

## Gotchas

FSDP parameter storage changes do not imply optimizer changes to the frozen base.
Ranks must issue the same count of forwards and collectives, including monitoring and checkpointing.
Replicated adapters are deliberately not managed by FSDP gradient reduction; reduce them exactly once.
A checkpoint retains all four RNG records and rejects a different rank count on restore.

## Tests

The small four-rank check compares one update against serial two-clip accumulation, frozen weights and exact resumed next updates.
The native check verifies saved one-process parity, soft/binary gradients, full clip lengths and per-rank headroom.
