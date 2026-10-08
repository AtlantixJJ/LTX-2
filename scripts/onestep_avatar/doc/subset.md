# `subset.py` — write a fixed video list

Status: **Version-two conversion, saved-probe membership and typed runtime integration implemented.**
`convert_legacy` creates membership and reproduction records from an old subset.
`from_saved_probe` creates one-view membership from original paired evidence.
Typed training and ordinary evaluation use these version-two records.
`windows.py` still owns the legacy corpus survey. A direct version-two survey
producer and legacy caller retirement remain incomplete.

## Objective

Select video/view files and split people into training and evaluation groups.
Save their content hashes.
Both modes reuse this list and select their own encoded frame ranges.

## Data flow

```mermaid
flowchart LR
  S[("original block-chain subset")] --> C["convert_legacy"]
  M[("checked crop records and masters")] --> C
  C --> O[("fixed video membership")]
  C --> R[("original selection reproduction record")]
  O --> P["training.config.build_frame_plan"] --> F(["mode frame selection plan"])
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef disk fill:#eceff3,stroke:#6b7280,color:#1f2937;
  classDef out fill:#ece0f8,stroke:#7048a0,color:#26123f;
  class C,P proc;
  class S,M,O,R disk;
  class F out;
```

The code calls the fixed video list `membership`.
The mode's frame selection plan is separate.
Neither operation changes the crop or runs the VAE again.

## Organization logic

Current conversion keeps the old selection, people, groups and content pins.
It reads encoded frame counts from masters, not raw-video length.
It adds a separate reproduction record for the old frame selections.
Saved-probe conversion checks exact paired input pins and records its historical
group. No current conversion reselects or surveys the corpus.

### Select and group videos

**Required direct survey; not yet implemented.**
The following procedure defines the eventual version-two survey producer.
The current module has no `survey_sources`, `split_actors` or `pin_content`
functions. Until this producer exists, use checked conversion or saved-probe
membership. Do not describe the old `windows.py` survey as a new schema producer.

1. Discover relative video/view paths in stable order and read the saved crop/master records.
2. Apply the current crop, completeness, and guide/provenance checks.
   Record every excluded path and reason before assigning groups.
3. Use the current bare person ID across corpus parts and all views.
   Cap a requested person tier in the existing SHA-256 person-ID order, not discovery order.
4. If explicit train/validation/test lists are supplied, check disjointness and surviving sources.
   Keep those exact groups. Record sources outside them as excluded.
5. Otherwise, preserve `windows.split_actors`:
   sort unique person IDs by SHA-256 of the ID;
   choose `h=min(max(round(N*f), min(m,N-1)), N-1)` held-out people.
   Here `N` is person count, `f` the holdout fraction, and `m` the holdout floor.
   The first `h` IDs in hash order are held out; the remainder train.
   Fewer than two people is an error. Record when the requested floor cannot fit.
6. Hash selected producer files and records, then write the fixed list atomically.

Every view of a person inherits that person's group.
Do not repartition people after seeing model outputs.

### Freeze data identity

`from_saved_probe` creates a one-view replay membership from a saved paired
probe manifest. Require every recorded row to identify the same capture/guide
paths and original bundle hashes. Resolve the corpus root from the capture
bundle's recorded relative source; never guess it from an actor name. Read
the actual actor ID from clip metadata. Preserve the original encoded bytes
and use the canonical source checker for current paired-guide provenance.
All replay sources use a `historical` group: this is bookkeeping for an exact
saved comparison, not a new training/validation split. Record the original
probe hash and current raw/crop metadata identities separately. Current raw
file pins cannot establish their unrecorded historical byte identities.
Do not drop any original encoded pin or repair a changed producer artifact.

The source checker verifies original bundle hashes, dimensions/fps, paired
encode records, render/sidecar identities and compositing version before
returning the membership. Recheck the original probe bytes after those reads.
Tests require exact source/person/hash retention and fail on changed capture,
guide or inconsistent manifest rows before any model execution.

Implemented schema version two has kind `onestep_avatar_membership`.
It records background, corpus root, crop-record identity, sorted video IDs, person/group,
frame rate, dimensions, encode records, hashes, exclusions, and hash scope.
It has no attention, cache depth, block size, block chains, or K.

The list hash covers background, videos, groups, exclusions, and hash rules.
Video IDs use relative corpus paths.
Record root-directory changes without changing video/group assignments.

Use a declared versioned hash payload with sorted JSON keys and stable video/group ordering.
Hash content identities and relative paths, not local absolute root paths or modification times.
Record hash scope so readers can reproduce the exact payload.
Root relocation changes the locator, not video membership, grouping, or producer bytes.
Verification resolves the relative paths, recalculates hashes, and returns all missing/mismatched paths.
It does not repair files or omit a mismatched video.

A separate frame-plan hash covers mode, selection rules, ordered samples, and old index mappings.
D0 can use a capture-only list.
Compared D0/D1 runs use the same paired list.
A D1 run fails before model loading if any required guide is missing or outdated.
It must not skip videos during training.

Convert old subsets to new files:
copy exact videos, groups, and hashes; keep the original hash;
write the old block indices into a reproduction frame plan.
[V8](verification.md) checks this conversion.
A new selection rule is a new plan, not an undocumented reproduction of old samples.

`convert_legacy` returns two records without writing. It reads each capture
bundle to record its actual shape and encoding fields. Existing content pins
must still match. Missing old encoded-file pins are added with full SHA-256.
Preserve original raw-video/guide hashes and split lists exactly.
Move `n_blocks`, `span_latent_frames`, and unused-tail facts into the reproduction
record. They do not belong to the fixed video list.
The reproduction plan keeps every old source record, chain index, block index,
and half-open encoded range. Its hash is separate from the membership hash.
Membership hashes exclude `corpus_root` so relocation does not change membership.
The CLI refuses used destinations and writes only new versioned JSON files.
No conversion re-encodes a master, changes a split, or infers adapter mode.

## Invariants

- One person and all views belong to one data group.
- Record both list and frame-plan hashes.
- Conversion preserves stored crop, VAE, and master data.
- Report exclusions/errors before loading the model.
- State which files are hashed. Encoded-file hashes do not prove raw-video identity.

## Gotchas

Do not change an old subset's schema label in place.
Bidirectional mode can use final frames that a causal block plan discards.
Record actual output frame coverage.
The same video list does not prove the same selected frames.

## Tests

[V8](verification.md) defines unchanged people, videos, and encoded data.
`tests/test_subset.py` checks exact old people, pins and frame ranges, changed
producer bytes or coverage, duplicate groups, D0 without guides, and CLI
preservation/refusal behavior. Preserve those controls through source migration.
Compare two frame-selection rules over one fixed list.
Reject missing guides and outdated encode records.
Keep original subset files unchanged.
Worked check: one person has views in two corpus parts.
All those views receive the same group. Relocating an unchanged corpus preserves the list identity.
Changing one capture-master byte fails verification for that video.
Changing mode leaves this list unchanged and creates a different frame-plan record.
