# `score/export_pruned.py` — width-mask checkpoint export

## Objective

Write a checkpoint that executes the declared native head and FFN intervention.
The exporter and ltx-core runtime share a metadata contract, so sliced tensors keep their original normalization and positions.
The [glossary](../glossary.md) defines functional masks, export modes and original IDs.

## Usage

A human or agent can use the direct CLI from `LTX-2` in `ltx`.
The current CLI calls CUDA preflight, although tensor surgery runs on CPU.
There is no `--dry-run` option.

```bash
conda run -n ltx python -m scripts.prune.score.export_pruned \
  --model 2.5 --masks <mask.json> \
  --mode compact_faithful \
  --output <candidate.safetensors>
```

| Argument | Meaning |
|---|---|
| `--model` | Registered key; default 2.5 |
| `--masks` | Complete native width-mask artifact |
| `--transformer-path` | Optional source checkpoint override |
| `--mode` | `masked_full`, `sparse`, `compact` or `compact_faithful` |
| `--output` | Exported safetensors path |

Output is the checkpoint. Stdout reports its fingerprint and peak host RSS.
See [run guide](../usage.md#width-export-and-parity).

## Data flow

Read source tensor shapes and validate the mask.
Plan coupled slicing on meta tensors, or record full-width active indices.
Update transformer metadata and stream one tensor at a time through `safetensors_stream`.

## Organization logic

`checkpoint_mask_widths` probes per-layer Q and FFN shapes without loading payload tensors.
`export` uses source-order retained indices.
`stream_plan` uses the existing surgery on meta tensors and returns final shapes and independent per-tensor slices.
The retained `load`/`export_tensors` helpers also support explicit in-memory callers and baseline controls.

Q/K RMSNorm spans the original heads. Keep full Q/K projections and normalization weights.
Compact attention slices V rows, output-projection columns and optional gate rows.
At execution, the native gate pads its rows to the original head count before its GEMM, then selects retained logits.
For example, a 31-head compact layer executes its native gate projection with 32 outputs.
This preserves BF16 gate rounding while V, attention and the output projection remain compact.

Compact FFN slices input-projection rows and bias, then output-projection columns.
Residual widths and output biases stay unchanged.

| Mode | Storage and execution |
|---|---|
| `masked_full` | Store full weights; compute all units and zero inactive activations before output projections |
| `sparse` | Store full weights; select active attention heads after full projections, scatter back, and keep FFN execution full |
| `compact` | Store sliced V/gate/output/FFN weights; execute retained widths while Q/K normalization stays full |
| `compact_faithful` | Store compact slices; pad weights to original GEMM shapes and execute original attention widths |

Metadata records per-layer widths, positional head identities, active units and source/mask provenance.

Worked example: keeping heads 0 and 2 at head dimension 4 retains channels 0–3 and 8–11.
Q/K normalization still sees the original width.

### Exporter/runtime keys

All keys below belong to the safetensors metadata `config.transformer` object.
Per-layer arrays have one entry per retained block. Entries use original ascending unit IDs.

| Key | Written meaning | Runtime consumer |
|---|---|---|
| `per_layer_video_attn1_heads` | Configured execution head count for video self-attention | Transformer builder → attention |
| `per_layer_video_attn2_heads` | Configured execution head count for video text attention | Transformer builder → attention |
| `per_layer_ff_inner_dim` | Configured FFN execution width | Transformer builder → feed-forward |
| `per_layer_video_attn1_rope_head_indices` | Original self-attention head IDs for compact positions; otherwise null | Pre-attention Q/K selection and rotary frequencies |
| `per_layer_video_attn2_rope_head_indices` | Original text-attention head IDs for compact positions; otherwise null | Pre-attention Q/K selection |
| `video_pruning_preserve_qk_norm` | Keep full original Q/K projections and normalization before selecting head slices | Attention construction and pre-attention operations |
| `video_pruning_shape_faithful` | Re-pad compact stored weights into original linear shapes each forward | `ShapeFaithfulLinear` in attention and FFN |
| `per_layer_video_attn1_active_head_indices` | Kept self-attention IDs for a full execution width; omitted for `compact` | Attention output mask or active-head selection |
| `per_layer_video_attn2_active_head_indices` | Kept text-attention IDs for a full execution width; omitted for `compact` | Attention output mask or active-head selection |
| `per_layer_video_ffn_active_channels` | Kept FFN IDs for a full execution width; omitted for `compact` | Mask post-activation channels before down projection |
| `video_pruning_select_active_heads` | True only for `sparse`; compute attention on selected heads then scatter back | Attention forward |
| `pruning` | Mode, model key, task and mask/source provenance | Candidate validators and downstream loaders |

The runtime owners are [model construction](../../../../packages/ltx-core/src/ltx_core/model/transformer/model.py),
[attention](../../../../packages/ltx-core/src/ltx_core/model/transformer/attention.py),
[pre-attention and native gates](../../../../packages/ltx-core/src/ltx_core/model/transformer/ops.py),
and [feed-forward / ShapeFaithfulLinear](../../../../packages/ltx-core/src/ltx_core/model/transformer/feed_forward.py).
These links identify the exporter/runtime boundary even though the runtime lives outside this package.

`compact` keeps full Q/K tensors because RMS normalization spans their original head width.
The runtime selects retained Q/K after normalization and preserves original rotary head coordinates.
Its native head-gate projection pads sliced gate rows to the original output count before the BF16 GEMM, then selects retained logits.
This keeps gate rounding compatible while V, attention and output projection use retained widths.

For `compact_faithful`, `ShapeFaithfulLinear` scatters compact parameters into zero-filled original row/column shapes each forward.
Its retained indices are Python metadata, not checkpoint tensors, so meta-device loading adds no unmaterialized buffers.
This mode executes full attention and GEMM geometry, including temporary padded weights.

### Learned exports and split bundles

Learned exporters reuse the same coupled surgery, then add a versioned recovered envelope.
A merged export forms FP32 adapter deltas and stores BF16 fused weights; this changes rounding points from trained split arithmetic.
`score.export_joint_split.open_bundle` loads base weights and separate adapters to preserve those operations.
Its default FFN down projection restores original reduction coordinates, so that full-width work belongs in benchmarks.
The [verification guide](../VALIDATION.md) owns qualification for each representation.

## Invariants

- Binary masks keep at least one unit per branch.
- Preserve original Q/K normalization and positional identity.
- The CLI checks complete families and content-pinned native provenance.
- Validate the exported model against its functional reference.

## Gotchas

BF16 GEMM rounding can change with compact shapes.
`compact_faithful` improves storage without promising reduced GEMM work.
The writer holds one changed tensor and a bounded copy buffer; the depth exporter separately owns block renaming.
Atomic publication permits an identical replay but rejects different bytes at an existing output. Select a fresh path.
The direct Python `export` function does not perform all CLI provenance checks.

## Tests

`tests/test_export_pruned.py` checks coupled slicing and real tiny-model export behavior.
`tests/test_mask_artifact.py` checks native artifact validation.
`tests/test_safetensors_stream.py` compares all four modes with the earlier in-memory serializer.
The [export gate](../VALIDATION.md#g3-reproduce-the-functional-export-reference) defines native parity coverage.
