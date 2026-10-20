# core/records

## Objective

Hash scientific records and publish immutable evidence without replacing completed files.

## Data flow

`digest` encodes sorted compact JSON with NaN disabled, then hashes its UTF-8 bytes.
`tensor_hash` binds the shape, dtype and contiguous raw CPU bytes.
`unchanged_state_hash` recursively encodes tensors, NumPy arrays, lists and tuples.
It omits only the top-level `identity` and `lineage` fields.

## Organization logic

`publish` writes and fsyncs a temporary file beside the destination, then creates a hard link.
An existing destination must have the same file hash. The temporary file is removed on every exit.
`publish_json` uses indented sorted JSON with a final newline.
`publish_tensor` accepts an existing tensor only when its semantic hash matches.
`publish_payload` compares an existing payload through `unchanged_state_hash`.
Both torch writers use file streams to avoid archive-name changes from temporary filenames.
`write_json` replaces mutable status through a process-specific temporary file after fsync.
`read_json` reads one JSON object without additional schema checks.

Worked example: publishing the same JSON twice passes. Changing one value raises `completed artifact differs`.

## Invariants

Keep byte encodings stable. Immutable publication never replaces completed evidence.
Tensor hashes include dtype and shape. Equal flattened values alone do not establish identity.

## Gotchas

`unchanged_state_hash` deliberately ignores lineage and identity. Validate those fields separately when reading checkpoints.
A create race uses file-byte equality, while an existing tensor uses semantic equality.
`write_json` is for mutable status, not immutable scientific receipts. Parent directories must share the destination filesystem.

## Tests

Run the prune CPU suite, including checkpoint resume, tensor publication and immutable receipt controls.
Native numerical parity remains independent of these file operations.
