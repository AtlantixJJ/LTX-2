# score/export_joint_split

## Objective

Export compact base weights and FP32 adapters while preserving the split calculation.

## Usage

The user or agent runs the standalone CLI from `LTX-2` in conda environment `ltx`.
Check GPU occupancy before native execution. Use physical GPUs 4–6 for tests.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx python -m scripts.prune.score.export_joint_split --study ../expr/refiner_prune/2.5/joint_clip_reset_readmission_main/configs/frozen_study.json --checkpoint ../expr/refiner_prune/2.5/joint_clip_reset_readmission_main/training/checkpoints/update_1700.pt --output ../expr/refiner_prune/2.5/joint_clip_reset_readmission_main/export/final.json
```

See the [run guide](../usage.md#joint-clipped-gates-with-re-admission).
The final update in the export example is the maximum-length case. Use the actual prespecified final checkpoint.

## Data flow

The frozen study pins the source checkpoint and training identity.
The terminal training checkpoint supplies binary masks, original unit IDs and FP32 adapters.
The exporter saves two tensor payloads and an envelope with their content hashes.

## Organization logic

The bundle has an envelope, base safetensors and adapter safetensors. It slices base weights and low-rank tensors at the same original IDs. It does not merge the low-rank update into BF16 weights. The reader checks both file hashes before loading. Use open_bundle(current, envelope_path) for inference. Passing only the base file to the stock loader omits the adapters and is not this model. The default `compact_preserve_ffn_reduction` mode reduces V, attention, output-projection and FFN up widths.
The FFN down projection restores original input coordinates before both its base and adapter reductions.
Its GEMM work stays at the original width; stored columns remain compact.
Explicit `compact` mode reduces that GEMM too, but needs separate parity qualification for scattered channels. The native gate projection retains its original execution width. Optional `compact_faithful` mode restores all execution widths and qualifies storage only. Final CLI export requires terminal aligned masks and complete recovery.

For example, a block can retain 12288 of 16384 channels.
The up projection produces 12288 activations. The down projection stores 12288 columns.
The default reader restores zeros at removed coordinates and executes a 16384-input down projection.
Its output must match the masked source within 0.02 on the saved native parity cases.

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
