# Native gate study preparation

## Objective

`data/native_gate_study.py` freezes the native diagnostic against existing training evidence.
It does not launch a model or change a source study.
The [entrypoint](../score/native_head_gates.md#usage) exposes preparation on CPU.

## Data flow

Read the r3 frozen study and its complete 5,200-record draw list.
Verify source identity, components, saved context and learning captures.
Reconstruct the primary pruning lineage and pin binary masks to canonical checkpoints.
Save an immutable plan copy, source study, original draw stream, comparison masks and event records.
Freeze current executable code separately from the historical training code identity.

## Organization logic

The original training cycle has ten sigma indices. Retain source indices `[1,4,7,9]` only.
The four sigma values are copied exactly from the source configuration.
Save both selected and skipped global draw IDs; retain the original four-record step convention.
A 100-step milestone has 40 observations per selected sigma, without resampling or reassignment.

The primary mask reader compares update-600 and update-800 monitor masks with the canonical update-600 checkpoint.
It replays selection events through update 500 from the original run and events 550/600 from the continuation.
The superseded recovery is excluded. Eligible-head records retain the pre-reset drop and selected removal reason.
An available r3 binary checkpoint is pinned separately, including its phase and update.
A binary but unfinished r3 checkpoint is never labeled the final model.

## Invariants

- Learning membership stays at 256 views from 93 actors under the saved sampler.
- Native gates come from the dev checkpoint with no trained adapters.
- Primary pruning has 292 removed and 1,244 kept original head IDs.
- Source component and context byte hashes must match before freezing.
- New collection inputs belong to the new run, never the active r3 output root.

## Gotchas

The source study's code identity is historical. It is not a native-observer acceptance receipt.
The saved plan copy is immutable; live progress amendments belong in the workspace plan.
A missing future A/B/C result does not block the primary comparison.
`saved_input_root` locates existing noise bundles; it never creates or repairs one.

## Tests

The collector tests enforce exact source-prefix semantics, original IDs and model-free report boundaries.
Preparation itself verifies the complete draw cycle, membership counts, full component/capture hashes and mask lineage.
