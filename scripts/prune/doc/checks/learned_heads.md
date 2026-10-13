# checks/learned_heads

## Objective

Reject incomplete recovered candidates and compare independent numerical references.

## Usage

Started by: a direct human or agent CLI. Run from `LTX-2` in conda environment `ltx`.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx python -m scripts.prune.checks.learned_heads --study ../expr/refiner_prune/2.5/learned_heads_dev8_20261012_r2/configs/frozen_study.json --arm learned_joint --replicate-seed 42
```

`--gpu-id` adds native numerical checks. Omit it for CPU artifact checks.

Outputs: Inventory receipt and optional structural, transfer, fusion and reload evidence.
See the [run guide](../usage.md#learned-head-phases).

## Data flow

The envelope connects source components, binary IDs, compact base, recovery state, draw evidence and merged tensor inventory.

## Organization logic

Validation checks format, source, actor separation, update coverage, adapters, original IDs and every consumed noise file.
Structural inventory independently checks each compact source tensor. Native checks compare masked/compact, transferred, fused and reloaded networks.

## Invariants

A merged candidate has no external adapter. Reload uses the in-memory merged reference. Unknown formats and nonfinite comparisons fail.

## Gotchas

A 0.02 maximum difference is a numerical gate. Passing it cannot prove human quality or speed.

## Tests

Run `test_learned_head_checks.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.
