# `score/export_depth.py` — intact-block physical depth export

## Objective and interface

Write a shorter native LTX transformer checkpoint by removing complete blocks,
without loading weight tensors into host RAM or allocating CUDA memory. This is
an architecture diagnostic until native BF16 parity, quality and timing are run.

`create_artifact(baseline_root, calibration_views, sigmas, removed_blocks)` binds
an explicitly chosen, sorted deletion set to a saved native D0 baseline.
`read_artifact` validates that binding. `export(source, artifact_path, output)`
writes the checkpoint and returns its fingerprint, file sizes and parameter
accounting. `verify_export` validates architecture/provenance for native
consumers. `retained_blocks` supplies a reversible in-memory reference for
export parity; it restores the original `ModuleList` even when the forward fails.

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.prune.score.export_depth \
  --baseline <saved-native-D0-baseline> \
  --calibration-views <calibration-view-1> <calibration-view-2> \
  --sigmas 0.725 0.909375 1.0 \
  --remove-blocks 3 7 8 15 \
  --artifact <new-depth-artifact.json> --output <new-depth-checkpoint.safetensors> \
  --purpose diagnostic_architecture --quality-status failed_prior_functional_gate
```

The example set is the previously studied four-block diagnostic, whose quality
gate failed. It is not a newly selected recovery candidate. An eight-block set
requires further calibration. For a no-prune architecture control, supply
`--remove-blocks` with no indices and `--purpose no_prune_control`.

## Data flow and invariants

The source must be unpruned, have contiguous block indices matching `num_layers`
and load through the native video configurator. Sources with existing pruning
metadata are rejected; composing multiple pruning operations is not supported.
The depth artifact uses `candidate_format=whole_clip_d0_depth_v1` and
`family=depth`, separately from `whole_clip_d0_mask_v1` width artifacts. It pins
the source fingerprint/header hash, complete calibration-manifest content and
distribution, source parameter accounting, retained order and both directions
of the original/compact mapping. Every initial artifact is `unqualified`.

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

## Accounting and validation

`checkpoint_tensors` counts all stored tensors. `resident_video_parameters`
counts exactly the keys/shapes expected by `LTXVideoOnlyModelConfigurator` on the
meta device, excluding unused audio and connector tensors. Block subtotals and
source/export totals report tensor count, elements and stored bytes. Actual
whole-file bytes are returned separately. These denominators establish storage
and architecture changes; forward speed and peak VRAM require measurement.

`verify_export` checks artifact content, pinned source, complete retained key
inventory, tensor shapes/dtypes, transformed config and parameter counts. It
does not compare tensor payloads or establish numerical parity. CPU tests compare
retained tensor values and real small video/AV model forwards, strict reload,
cache entries/full block-causal references, cache eviction, retained-block
backward gradients, clean c0, aliases, source mutation,
manifest/artifact/config mutation and interrupted-copy cleanup. A production
BF16 held-out run through `checks.export_parity --depth-artifact` remains required
at maximum absolute latent difference 0.02, followed by separate quality/timing.
