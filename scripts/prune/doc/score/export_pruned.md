# `score/export_pruned.py` — width-mask checkpoint export

## Objective

Write a checkpoint that executes the declared native head and FFN intervention.

## Usage

A human or agent can use the direct CLI from `LTX-2` in `ltx`.
The current CLI calls CUDA preflight, although tensor surgery runs on CPU.
There is no `--dry-run` option.

```bash
conda run -n ltx python -m scripts.prune.score.export_pruned \
  --model 2.5 --masks ../expr/refiner_prune/2.5/prune_doc_example/mask.json \
  --mode compact_faithful \
  --output ../expr/refiner_prune/2.5/prune_doc_example/candidate.safetensors
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
Load the source tensor dictionary on CPU.
Apply coupled slicing or record active indices, update transformer metadata, then save the checkpoint.

## Organization logic

`checkpoint_mask_widths` probes per-layer Q and FFN shapes without loading payload tensors.
`export` uses source-order retained indices.

Q/K RMSNorm spans the original heads. Keep full Q/K projections and normalization weights.
Compact attention slices V rows, output-projection columns and optional gate rows.
At execution, the native gate pads its rows to the original head count before its GEMM, then selects retained logits.
For example, a 31-head compact layer executes its native gate projection with 32 outputs.
This preserves BF16 gate rounding while V, attention and the output projection remain compact.
Compact FFN slices input-projection rows and bias, then output-projection columns.
Residual widths and output biases stay unchanged.

| Mode | Storage and execution |
|---|---|
| `masked_full` | Full stored widths; active indices encode the control |
| `sparse` | Full projections; select active attention heads |
| `compact` | Compact stored and supported executed widths |
| `compact_faithful` | Compact storage; restore original execution geometry |

Metadata records per-layer widths, positional head identities, active units and source/mask provenance.

Worked example: keeping heads 0 and 2 at head dimension 4 retains channels 0–3 and 8–11.
Q/K normalization still sees the original width.

## Invariants

- Binary masks keep at least one unit per branch.
- Preserve original Q/K normalization and positional identity.
- The CLI checks complete families and content-pinned native provenance.
- Validate the exported model against its functional reference.

## Gotchas

BF16 GEMM rounding can change with compact shapes.
`compact_faithful` improves storage without promising reduced GEMM work.
The exporter loads the full tensor dictionary; it is not the bounded streaming depth exporter.
`save_file` can replace an output. Select a fresh path.
The direct Python `export` function does not perform all CLI provenance checks.

## Tests

`tests/test_export_pruned.py` checks coupled slicing and real tiny-model export behavior.
`tests/test_mask_artifact.py` checks native artifact validation.
Run [native export parity](../checks/export_parity.md) before interpreting quality or timing.
