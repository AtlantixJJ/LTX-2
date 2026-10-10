# `experiments/spec_protocol.py` — pinned checker specifications

## Objective

Give queued checkers one strict data-only interface. Keep scientific argument
parsing with each checker. This helper owns no model, GPU claim, child process,
supervisor or numerical comparison.

## Data flow

A queue-normalized spec and output enter the checker parser. The exact spec
selects its scientific arguments. The checker writes original scientific
artifacts, then publishes a manifest. Model-free completion checks the spec,
settings, complete local inventory and the checker's scientific verifier.

```mermaid
flowchart LR
  S[("pinned spec")] --> P["checker scientific parser"]
  P --> C["checker kernel"]
  C --> A[("scientific artifacts")]
  A --> H["publish"]
  H --> M[("bound manifest")]
  A --> V["verify_completion"]
  M --> V
  S --> V
  V --> O(["checked completion"])
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef disk fill:#eceff3,stroke:#6b7280,color:#1f2937;
  classDef terminal fill:#e9ddff,stroke:#7253a0,color:#281a3f;
  class P,C,H,V proc;
  class S,A,M disk;
  class O terminal;
```

## Organization logic

Parse `--spec`, `--output`, local `--gpu-id` and `--dry-run` only. First-pass
relative paths stay unopened for queue normalization. Once the spec path is
absolute, parse its scientific fields without loading inputs or a model.
Require `schema_version: 1`, the exact literal protocol and a string argument list.
The optional `operation` must belong to the selected checker's fixed allowed
set; it selects steps such as serial reference or saved comparison. Refuse
duplicate JSON fields, unknown fields, duplicate flags, execution overrides and
non-absolute scientific paths.
The original scientific parser decides valid settings and defaults.

`publish` rechecks the pinned spec bytes, records scientific settings and all
local output files, and refuses escaped/symlinked artifacts. `inspect` repeats
this exact inventory and hash check; it never recreates a missing result. Each
checker then verifies actual tensors, controls or metrics. A manifest hash by
itself does not establish a scientific pass. Completion evidence includes the
manifest and every bound artifact, so queue dependency receipts bind the full
result. Preserve original producer source manifests; add this helper to new
experiment extras only.

Worked refusal: a spec containing `--output /other/path` fails before any output
or native loader. Changing a tensor and rewriting its file digest still must
fail the checker comparison when the tensor violates the control.

## Invariants

The spec byte identity covers scientific settings. Output and GPU come only
from the queue interface. No checker launches another process. Native
measurements and fixed tolerances remain in the original scientific owners.

## Gotchas

Relative spec arguments cannot depend on a launch cwd. All scientific paths
are absolute. Runtime/notification files outside the output are queue evidence,
not extra checker artifacts. A dry-run publishes no manifest.

## Tests

Test first/second parser passes, overrides, changed specs, missing/escaped
artifacts, exact evidence inventories and rehashed scientific negative controls.
Verify completion with model loaders and process launches forbidden.
