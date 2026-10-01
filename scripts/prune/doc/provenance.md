# `core/provenance.py`

## Objective and data flow

Identify checkpoints, saved inputs and execution context on artifacts. Checkpoint
fingerprints hash the safetensors header, file size and fixed sampled data regions.
`file_sha256` pins small files in full. `stamp` records model, VAE, geometry, revision,
host and device; `run_id` combines timestamp and PID.

## Invariants and verification

A sampled fingerprint is an identity cue, not whole-file integrity proof.
Manifest/noise/capture hashes are validated by `data.whole_clip`. No renderer or
window-source dependency is included. Native manifest and mask tests cover
identity rejection; real saved-output checks verify model-facing input equality.
