# `score/export_depth.py` — intact-block physical depth export

## Objective

Write a shorter native LTX transformer checkpoint by removing complete blocks,
without loading weight tensors into host RAM or allocating CUDA memory. This is
an architecture diagnostic with evidence assessed by the [verification guide](../VALIDATION.md).
The [glossary](../glossary.md) defines depth artifacts and original block IDs.

`create_artifact(baseline_root, calibration_views, sigmas, removed_blocks)` binds
an explicitly chosen, sorted deletion set to a saved native D0 baseline.
`read_artifact` validates that binding. `export(source, artifact_path, output)`
writes the checkpoint and returns its fingerprint, file sizes and parameter
accounting. `score.export_depth.verify_export` validates architecture/provenance for native
consumers. `retained_blocks` supplies a reversible in-memory reference for
export parity; it restores the original `ModuleList` even when the forward fails.

## Usage

A human or agent can run this CPU export directly from `LTX-2` in `ltx`.
There is no `--dry-run` option. The command reads saved calibration inputs and checkpoint metadata.

```bash
conda activate ltx
CUDA_VISIBLE_DEVICES='' python -m scripts.prune.score.export_depth \
  --baseline <baseline_root> --calibration-views <calibration_view> \
  --sigmas 0.725 0.909375 1.0 \
  --remove-blocks 1 \
  --artifact <depth_candidate.json> --output <depth_candidate.safetensors> \
  --purpose calibration_candidate --quality-status not_evaluated
```

This example supplies original block 1 as the deletion set; the saved baseline must contain the listed exact sigmas.
The resulting artifact starts unqualified. For a no-prune architecture control, supply
`--remove-blocks` with no indices and `--purpose no_prune_control`.

| Argument | Meaning |
|---|---|
| `--baseline` | Saved native D0 baseline |
| `--calibration-views` | Views recorded in that baseline |
| `--sigmas` | Exact saved calibration levels |
| `--remove-blocks` | Original block indices; empty for a no-prune control |
| `--artifact` | New depth JSON path |
| `--output` | New checkpoint path |
| `--purpose` | `diagnostic_architecture`, `calibration_candidate` or `no_prune_control` |
| `--quality-status` | Declared current quality status |

Outputs are the artifact and streamed checkpoint. Stdout records sizes, counts and fingerprint.
See the [depth run guide](../usage.md#depth-export).

## Data flow

The source must be unpruned, have contiguous block indices matching `num_layers`
and load through the native video configurator. Sources with existing pruning
metadata are rejected; composing multiple pruning operations is not supported.
The depth artifact uses `candidate_format=whole_clip_d0_depth_v1` and
`family=depth`, separately from `whole_clip_d0_mask_v1` width artifacts. It pins
the source fingerprint/header hash, complete calibration-manifest content and
distribution, source parameter accounting, retained order and both directions
of the original/compact mapping. Every initial artifact is `unqualified`.

## Organization logic

The exporter removes all tensor keys below each deleted block, including audio
and AV branches, renames retained blocks contiguously, sets the new `num_layers`,
and slices all eight supported `per_layer_*` lists. Unknown block-indexed list
fields are rejected. All non-block tensors and unrelated metadata remain in the
checkpoint. Native D0 requires CFG 1/STG 0; original guidance indices therefore
cannot silently select different compact blocks. A future guided contract must
explicitly map such indices or reject deleted ones.

Only the JSON header and bounded payload chunks are resident during copying.
Payload dtype, shape and bytes remain unchanged. A temporary file is format
validated and atomically published without overwriting an existing destination.

Source paths, symlinks and hard-link aliases cannot be overwritten. Source inode,
size and modification/change times are checked before and after copying; artifact
and manifest content must still match before publication. Failed copying removes
its temporary file and leaves no output checkpoint. The CLI writes a new artifact
with exclusive creation and never calls CUDA preflight.

Worked example: deleting block 1 from four blocks keeps original blocks `[0,2,3]`.
The compact indices become `[0,1,2]`, and `num_layers` becomes 3.

## Invariants

- Keep at least one block and preserve retained order.
- Bind mappings, source identity and calibration content to the artifact.
- Preserve retained tensor dtype, shape and payload bytes.
- Refuse existing destinations, source aliases and changed inputs.
- Mark new artifacts `unqualified`.

## Gotchas

Parameter accounting measures storage and architecture, not speed or quality.
This module does not select the deletion set.

### Parameter accounting

`checkpoint_tensors` counts all stored tensors. `resident_video_parameters`
counts exactly the keys/shapes expected by `LTXVideoOnlyModelConfigurator` on the
meta device, excluding unused audio and connector tensors. Block subtotals and
source/export totals report tensor count, elements and stored bytes. Actual
whole-file bytes are returned separately. These denominators establish storage
and architecture changes; forward speed and peak VRAM require measurement.

`verify_export` checks artifact content, pinned source, complete retained key
inventory, tensor shapes/dtypes, transformed config and parameter counts. It
does not compare tensor payloads or establish numerical parity.

## Tests

`tests/test_export_depth.py` compares
retained tensor values and real small video/AV model forwards, strict reload,
cache entries/full block-causal references, cache eviction, retained-block
backward gradients, clean c0, aliases, source mutation,
manifest/artifact/config mutation and interrupted-copy cleanup.
`tests/test_safetensors_stream.py` separately checks bounded streaming payloads and serialization.
A production BF16 held-out run through `checks.export_parity --depth-artifact` remains required
under the [standard export tolerance](../VALIDATION.md#g3-reproduce-the-functional-export-reference), followed by quality and timing.
