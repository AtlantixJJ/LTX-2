# checks/joint_faults

## Objective

Locate the first changed operation in native compact attention.

## Usage

The user or agent runs the standalone CLI from `LTX-2` in conda environment `ltx`.
Check GPU occupancy before native execution. Use physical GPUs 4–6 for tests.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx python -m scripts.prune.checks.joint_faults --study ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261018_launch_r2/preflight/configs/frozen_study.json --output ../expr/refiner_prune/2.5/joint_clip_reset_readmission_fix_20261019/diagnosis --gpu-id 4
```

See the [run guide](../usage.md#joint-clipped-gates-with-re-admission).

## Data flow

The saved study supplies native inputs, exact schedules and original unit IDs.
The producer saves predictions, numerical comparisons and immutable source receipts.

## Organization logic

The producer captures the exact first attention input and rotary tensors. It compares Q, K, V, attention, native gate logits, gated values and the output projection. Each comparison maps retained coordinates to original IDs. It repeats projection arithmetic in FP32 on the same input. The saved fixture allows later kernel controls without another full-model load. Historical study identity supplies input provenance; the current diagnostic code identity is recorded separately.

## Invariants

Preserve original Q/K normalization and video residual width 4096.
Keep the acceptance tolerance at 0.02. Check finite differences before accepting them.
A replay pass does not establish decoded quality or speed.

## Gotchas

Split BF16 adapter arithmetic and merged BF16 arithmetic have different rounding points.
A compact bundle must load both its base and its adapters through the bundle reader.
Historical inputs can be reused by hash. Historical native acceptance cannot qualify changed model code.

## Tests

`test_joint_exports.py` checks original gate geometry, retained coordinates, split arithmetic and payload hashes.
Native acceptance and the full compact parity matrix remain separate required checks.
