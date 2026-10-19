# checks/joint_split_parity

## Objective

Check compact split-adapter replay against the original masked computation.

## Usage

The user or agent runs the standalone CLI from `LTX-2` in conda environment `ltx`.
Check GPU occupancy before native execution. Use physical GPUs 4–6 for tests.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx python -m scripts.prune.checks.joint_split_parity --study ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261018_launch_r2/preflight/configs/frozen_study.json --output ../expr/refiner_prune/2.5/joint_clip_reset_readmission_fix_20261019/compact_split_parity --gpu-id 5
```

See the [run guide](../usage.md#joint-clipped-gates-with-re-admission).

## Data flow

The saved study supplies native inputs, exact schedules and original unit IDs.
The producer saves predictions, numerical comparisons and immutable source receipts.

## Organization logic

The check uses zero removal, head removal, FFN removal, combined removal and the aligned pilot mask. Random noncontiguous IDs and uneven block widths test coordinate handling. Every condition uses three exact development sigmas for one actor and an additional actor at the highest sigma. Each condition has independent functional and reloaded bundle forwards. The zero and random cases use r3 update-50 adapters. The pilot case uses its own saved adapters. The complete five-condition matrix has twenty comparisons. All must pass the unchanged 0.02 limit.
`--kinds` runs named conditions independently. The aggregate review rejects missing or repeated conditions.
`--reuse-zero-head PATH` reuses eight exact matched zero/head observations.
The added FFN reduction branch is a no-op there because those FFN masks are all one.
Every reused observation retains its own source hash. The twelve channel cases run again. The receipt records current code, inputs, source weights, checkpoint and bundle hashes. It does not qualify merged BF16 weights or timing.

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
