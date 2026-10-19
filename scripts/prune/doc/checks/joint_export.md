# checks/joint_export

## Objective

Isolate adapter merging, full-width reload and uneven compact surgery.

## Usage

The user or agent runs this standalone owner from `LTX-2` in environment `ltx`.
Check the physical GPUs before model work.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx python -m scripts.prune.checks.joint_export --study ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261018_preflight/configs/frozen_study.json --output ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261018/checks/p2 --gpu-id 6
```

See the [run guide](../usage.md#joint-clipped-gates-with-re-admission).
Outputs include predictions, transition receipts, tensor inventories and artifact hashes.

## Data flow

The frozen study supplies source weights, saved context, source noise and exact schedules.
The checkpoint reader binds policy state, binary masks and adapters to that study.

## Organization logic

The same saved development case uses adapters off and on. The on case loads r3 update-50 adapters. Each family has zero-removal, head-only, uneven FFN-only and combined cases. Merge the adapter before slicing. Save separate split-to-merged, merged-to-reloaded and functional-to-compact errors. A controlled FP32 calculation compares split and merged arithmetic on identical projection inputs. Predictions and exact mask inventories remain saved. Disposable checkpoint payloads are removed after their hashes are recorded.

## Invariants

Preserve original IDs, all Q/K normalization rows and video residual width 4096.
Keep the native maximum absolute latent difference limit at 0.02.
A smaller checkpoint does not establish reduced latency.

## Gotchas

BF16 merge and compact execution can change arithmetic.
A failed transition remains visible and blocks its corresponding compact or timing claim.
The FP32 control tests an arithmetic explanation; a failed full-model check alone does not prove that explanation.

## Tests

Run the CPU suite and native acceptance separately.
Use staged saved transition receipts before a deployment claim.
