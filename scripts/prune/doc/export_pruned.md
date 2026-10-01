# `score/export_pruned.py`

## Objective and data flow

Validate a native mask against checkpoint header widths, slice supported tensors,
write execution metadata and pin task/source/mask identity in safetensors.
The CLI reports the output fingerprint and peak process RSS.

## Organization and invariants

`masked_full` is the default control. `sparse` selects attention heads with
full-width projections. `compact` slices V, gate, output and FFN dimensions while
preserving full Q/K normalization and retained RoPE identity. `compact_faithful`
stores compact tensors with original execution geometry. It saves persistent
parameter storage; padded temporary weights and full activations still cost memory.

All CLI masks require native whole-clip D0 provenance. No record-derived mask or
reconstruction-state interface is supported. Export materializes weights in host
RAM. Passing numerical parity does not establish quality or speed.

## Verification

`tests/test_export_pruned.py` checks structural metadata, faithful linear behavior,
mask validation and numerical controls. Run `checks.export_parity` on fresh held-out
whole-clip inputs before interpreting an export's results.
