# learn/joint_training

## Objective

V3 numerical updates and exact state.

## Data flow

JointStep attaches head and FFN providers at native output-projection boundaries. Frozen BF16 weights retain FP32 gate and adapter masters. GateOptimizers owns two independent Adam instances.
The shared [preview owner](visualization.md) can copy native x0 from the training forward without retaining its graph.
The DDP owner pairs it with that draw's noisy input and GT, then decodes after the update.

## Organization logic

The loss adds head penalties plus channel penalties times 1/704, divided by 1536. Both families reset their scores and Adam state after an event. Gate-output hooks collect task gradients before the surrogate. A diagnostic all-reduce averages these observations; DDP alone reduces parameter gradients.

## Invariants

Projected masks run under `torch.no_grad()` during baseline and periodic monitoring.
Register gate-output hooks only when the current mask requires gradients.
Training keeps its sigmoid surrogate and captured task gradients.

Keep saved input hashes, exact schedules and original unit IDs.
Keep native loss reduction in FP32. Preserve the frozen backbone and complete rank RNG.

Optimizer damping summaries move to the CPU before their median reduction.
The CUDA median-with-indices kernel is not deterministic under the required numerical policy.

## Gotchas

The checkpoint stores policy summaries, both optimizers, adapter state, rank RNG, sample IDs and draw cursor. One training update consumes four frozen draws. A three-rank native check has no formal training authority. FP32 gradient files preserve values near 1e-9.

## Tests

Run `tests/test_readmission.py` with CUDA hidden.
Native acceptance and held-out export parity remain independent evidence requirements.
