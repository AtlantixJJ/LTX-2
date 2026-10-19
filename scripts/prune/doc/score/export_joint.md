# score/export_joint

## Objective

Export terminal aligned V3 masks without weakening historical readers.

## Usage

The user or agent runs this standalone owner from `LTX-2` in environment `ltx`.
Check the physical GPUs before model work.

```bash
CUDA_VISIBLE_DEVICES= conda run --no-capture-output -n ltx python -m scripts.prune.score.export_joint --study ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261018/configs/frozen_study.json --checkpoint ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261018/training/checkpoints/update_1700.pt --output ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261018/export/final.safetensors
```

See the [run guide](../usage.md#joint-clipped-gates-with-re-admission).
Outputs include predictions, transition receipts, tensor inventories and artifact hashes.

## Data flow

The frozen study supplies source weights, saved context, source noise and exact schedules.
The checkpoint reader binds policy state, binary masks and adapters to that study.

## Organization logic

The terminal checkpoint must have binary aligned masks and complete recovery. Its actual selection endpoint determines the final update; 1700 is only the hard-limit example. Merge FP32 adapter deltas into full-width weights before slicing coupled projections. Save a distinct V3 family envelope with original IDs, actors and exact tensor inventory. Probe exports never qualify for final evaluation. Numerical parity and timing remain separate gates.

## Invariants

Preserve original IDs, all Q/K normalization rows and video residual width 4096.
Keep the native maximum absolute latent difference limit at 0.02.
A smaller checkpoint does not establish reduced latency.

Use [the split bundle exporter](export_joint_split.md) when exact trained adapter arithmetic is required.
This merged exporter changes BF16 rounding points and needs independent parity acceptance.

## Gotchas

BF16 merge and compact execution can change arithmetic.
A failed transition remains visible and blocks its corresponding compact or timing claim.
The FP32 control tests an arithmetic explanation; a failed full-model check alone does not prove that explanation.

## Tests

Run the CPU suite and native acceptance separately.
Use staged saved transition receipts before a deployment claim.
