# score/export_learned_ffn

## Objective

Export both learned mask families, fuse adapters and validate the resulting payload.

## Usage

A human or agent runs CPU surgery directly from `LTX-2` in `ltx`.

```bash
conda run --no-capture-output -n ltx python -m scripts.prune.score.export_learned_ffn --study ../expr/refiner_prune/2.5/learned_ffn_joint_20261018/configs/frozen_study.json --checkpoint ../expr/refiner_prune/2.5/learned_ffn_joint_20261018/training/checkpoints/update_1300.pt --output ../expr/refiner_prune/2.5/learned_ffn_joint_20261018/exports/final_merged.safetensors
```

`--probe` permits disposable native-check checkpoints. Such exports cannot qualify as final candidates.
`--parity-gpu-id 4` also runs one held-out native comparison on an allowed testing GPU.
Outputs are a compact checkpoint, a versioned envelope, file hashes and optional parity results.
See the [run guide](../usage.md#learned-ffn-masks).

## Data flow

Read a completed frozen checkpoint. Validate original masks against source tensor shapes.
Slice head and FFN adapters at their original retained IDs.
Use existing coupled structural surgery, then merge each effective weight in FP32 and round once.
The new envelope pins study, components, recovery checkpoint, complete masks and merged tensor inventory.
`stream_merge` plans shapes without loading the full source dictionary, then transforms one tensor at a time.
FFN recovery slices first and merges compact adapters afterward. V3 requests merge-first arithmetic through the same owner.
`adapter_targets` validates every FP32 A/B pair against the coordinates where its merge will run.

## Organization logic

The format is `whole_clip_d0_learned_head_ffn_v1`; the candidate family is `width_head_ffn_recovered`.
The loader checks source identity, final recovery, consumed input chain, actor separation and tensor hashes.
FFN up B loses removed rows. FFN down A loses removed columns.
For retained channels [1,3], both projections use that ascending order.
The down bias stays unchanged. Head Q/K normalization retains its original width and positional identities.

Native parity separates split adapters versus fused masked weights from fused masked versus compact execution.
Each comparison keeps the same source, context, noise and direct schedule.
Each maximum absolute latent difference must be at most 0.02.
The command records failed parity without claiming an accepted candidate.
The writer hashes payloads as it writes them. Its final header binds the envelope after that inventory exists.
Whole-file hashes remain provenance; exact payloads and parsed metadata define export-content acceptance.

## Invariants

Never rerank during export. Reject incomplete adapter inventories and nonbinary masks.
Final exports require the exact recovery endpoint. Probe envelopes remain disposable.
Reload has no external adapters, live selectors or full-width reconstruction.

## Gotchas

BF16 fusion and changed reduction widths can change outputs even with correct tensor slices.
The native inference reader does not restore RNG from a different device mapping.
Export and candidate validation can read large shared checkpoint files.
Payload correctness does not prove decoded quality or reduced latency.

## Tests

`test_ffn_learning.py` checks independent merge algebra, bias preservation and original-channel adapter/moment transfer.
`test_safetensors_stream.py` checks both slice/merge orders and the envelope binding against in-memory controls.
Native tests separately measure functional, fusion and compact discrepancies.
