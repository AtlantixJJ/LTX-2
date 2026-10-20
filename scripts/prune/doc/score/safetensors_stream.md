# score/safetensors_stream

## Objective

Publish a transformed checkpoint with bounded host memory and attributable tensor hashes.
Keep the source file unchanged and publish the destination atomically.

## Data flow

`inspect` validates the safetensors header and returns meta tensors for shape and dtype planning.
`write` reserves the final header and visits tensors in the installed serializer's dtype/key order.
Unchanged payloads use an 8 MiB copy buffer. A changed tensor is loaded, transformed and released before the next tensor.
Tensor digests bind shape, dtype and the exact bytes written.

```mermaid
flowchart LR
  S("source header and payload") --> P["plan final shapes"]
  P --> W["copy or transform one tensor"] --> H["hash written bytes"]
  H --> E(["envelope inventory"])
  E --> B["bind final header"] --> O(["atomic checkpoint"])
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef tensor fill:#dcf5df,stroke:#43854b,color:#173b1c;
  classDef out fill:#ece0f8,stroke:#7048a0,color:#26123f;
  class P,W,H,B proc;
  class S tensor;
  class E,O out;
```

## Organization logic

Empty tensors let the official serializer choose tensor ordering without allocating source weights.
The writer preserves the supplied metadata order because Rust metadata maps otherwise randomize key order.
After payload hashes exist, an optional callback publishes the envelope and returns its final metadata binding.
The header is rewritten in place before publication. Its reserved byte length must remain unchanged.
For example, replacing a 64-character placeholder with an envelope SHA-256 keeps that length fixed.

Peak tensor memory depends on the largest transformed projection and its FP32 intermediates, plus caller-owned adapter state.
It does not depend on the complete transformer dictionary. The source checkpoint remains memory-mapped only for transformed tensors.
Each changed tensor gets a separate mapping. Closing it prevents touched source pages from accumulating in process RSS.
Full-file sidecar hashing remains a separate pass after publication.

## Invariants

- The source and destination cannot name the same file or a hard-link alias.
- Final tensor names equal the source inventory; this owner does not rename depth blocks.
- Every transformed tensor must match its planned shape and dtype.
- A changed source stat key stops publication. This check does not replace full component integrity validation by the caller.
- A replay must match the published destination bytes; a different artifact cannot replace it.
- Adapter arithmetic, mask selection and model qualification belong to the callers.

## Gotchas

The existing depth owner retains its block-renaming writer. This helper serves width and adapter exports.
An envelope can be published before a later checkpoint-publication failure, as with the earlier exporter.
Preserve that partial evidence and diagnose it rather than replacing it.
Whole-file hashes can differ from the old serializer solely because metadata key ordering differs.
Export acceptance compares exact tensor payloads and parsed metadata; raw hashes remain provenance.
Empty payloads and scalar tensors still have valid shape/dtype digests.

## Tests

`tests/test_safetensors_stream.py` checks mixed dtypes, scalar/empty tensors, chunk boundaries and recorded tensor digests.
It compares all four width modes with the prior in-memory serializer.
Separate controls compare both adapter/slice orders, envelope binding and failure before publication after source mutation.
Real saved V3 export checks compare payload, metadata, wall time and peak host RSS against the preserved exporter.
Native numerical parity and decoded quality remain separate gates.
