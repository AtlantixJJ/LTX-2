# `experiments/capacity_review.py` — saved D0 learning review

## Objective

Review a separate D0-trained adapter at updates 0, 20 and 60 against its base.
D0 uses the capture as the noising source. It is a capture-provided capacity
test, unavailable to product generation. This review never substitutes a
same-adapter D0/D1 oracle for D0 training.

## Data flow

```mermaid
flowchart LR
  E[("ordinary D0 evaluation jobs and saved results")] --> C["verify exact base, zero, twenty and sixty"]
  C --> D["public native VAE decoder"]
  R[("checked capture pixels and crop")] --> P["media.prepare_training_references"]
  D --> V["media.render_panels"]
  P --> V
  V --> O(["full and narrow learning review"])
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef disk fill:#eceff3,stroke:#6b7280,color:#1f2937;
  classDef out fill:#ece0f8,stroke:#7048a0,color:#26123f;
  class C,D,P,V proc;
  class E,R disk;
  class O out;
```

## Organization logic

The existing `conditioning_check` queue selector dispatches the explicit
`capacity_review` operation. Its specification names a frozen list of ordinary
evaluation jobs and one mode. Declare every evaluation output as an in-list
dependency. The list contains data only. This module loads no transformer,
text encoder or adapter and does not generate missing results.

Before native decoding, normalize the ordinary jobs and verify their exact
result inventory through the public evaluation verifier. Require D0, white,
dev LTX-2.5, direct `[0.725, 0]`, seven encoded frames, one source per job,
base followed by same-lineage checkpoints 0, 20 and 60. Inspect the actual
contracts, output tensors and consumed input hashes. Zero output equals base
bit-for-bit. Exact native call dictionaries are one denoise and one forward
for bidirectional, or three denoises, three refreshes and six forwards for
causal, with no priming. Each saved metric is recomputed. The adapter changes;
capture, clean first image, noise, text, geometry, history and schedule remain
fixed within a case.

The review software profile pins this owner, the conditioning selector, the
reference adapter, the specification protocol and the adapter-effect owner's
checkpoint-selection helper. A change to that helper invalidates the capacity
review profile. It does not change ordinary training or other control profiles.

Prepare matched capture RGB and VAE reconstruction with the public media owner
in the recorded white objective and crop. Preserve reference provenance and
full RGB pixel tensors. Decode base, zero, twenty and sixty with equal fresh
seeded generators. Save one base-versus-twenty review and one
base-versus-sixty review. Each has capture RGB, capture VAE, zero output, base
output and changed output. The full layout has three columns. The narrow
layout has two columns, an explicit unused slot, and the base/changed pair
beside each other. Keep equal display heights and aspect-preserving padding.
All panels cover recorded RGB frames 0 through 48 at the original FPS.

Completion repeats ordinary scientific verification and reads saved pixels
only. Check source/reference/VAE/decoder identities, panel tensors, labels,
positions, padding, synchronized coverage, actual video dimensions/FPS/frame
count and reconstructed posters. Original capture and guide videos remain in
the corpus. Publication uses the conditioning selector's byte-pinned manifest.
The queue's generic runtime bounds the complete decoder job. Decoder cost
keeps cumulative allocated/reserved peaks without resetting the queue's peak.

Worked check: D0 base and zero outputs both equal 2, while twenty equals 3
and sixty equals 4 under identical input hashes. Zero is a no-op and the
trained correction is measurable. This establishes an effect. It does not
establish correct identity or action; inspect the matched videos separately.

The [reference input owner](reference_inputs.md) derives a copied source row
from the saved capture manifest and original capture/guide encode records before
public RGB replay. It preserves original membership bytes and content identity
and records the derivation with saved reference provenance.

## Invariants

- No missing evaluation is rebuilt and no model training runs here.
- D0-trained contracts must actually record D0; a research override cannot
  convert a D1-trained adapter into learning-capacity evidence.
- Exact checkpoint inventory, source pins and consumed inputs are checked.
- Native source attribution is preserved; review software has its own identity.
- Numerical sensitivity and reduced MSE alone never pass visual quality.

## Gotchas

A seven-frame pilot makes 49 RGB frames. Do not extend this coverage in a
caption. Unused guide pixels may remain in the prepared reference bundle but
are omitted from the D0 comparison. A missing twenty or sixty checkpoint is
incomplete coverage, not permission to substitute a shorter update.

## Tests

Focused tests check source selection, exact four-state inventory, fixed input
comparisons, zero no-op, native counts and recomputed metrics. Saved rendering
tests cover adjacent base/changed output positions and explicit empty narrow
slots. Source-attribution tests reject a changed checkpoint-selection helper
while the unrelated training, conditioning and reference profiles stay equal.
Native results, visual identity/action review and cost remain separate
acceptance evidence.
