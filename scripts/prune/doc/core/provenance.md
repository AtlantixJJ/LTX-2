# `core/provenance.py` — file identities and execution stamps

## Objective

Record checkpoint identity, file content and execution context on saved artifacts.

## Data flow

A checkpoint path produces a short fingerprint.
A file path produces a full SHA-256.
A resolved model and optional device produce a provenance dictionary.

## Organization logic

`checkpoint_fingerprint` hashes file size, the safetensors header and three sampled payload regions.
Each region is at most one MiB. The result contains 16 hexadecimal characters.
`file_sha256` reads the whole file in one-MiB chunks.

`run_id` combines a local timestamp, PID and optional prefix.
`stamp` records model paths, VAE scales, revision, host, runtime versions and time.
A CUDA device adds its index and name. Caller fields extend the record.

Worked example: processes finishing in the same second get different run IDs because their PIDs differ.

## Invariants

- Use full hashes for pinned manifests, source captures and saved tensor files.
- Record the actual selected checkpoint.
- A dirty revision is execution context, not a full source-content pin.

## Gotchas

A sampled checkpoint fingerprint does not prove whole-file integrity.
The avatar queue wrapper separately hashes complete components for native acceptance.
Caller `extra` fields can override stamp fields; review them at the call site.

## Tests

Manifest, mask and depth-export tests check changed-file refusals.
`tests/test_artifacts.py` checks run attribution.
Native input reconstruction must also check actual tensor values.
