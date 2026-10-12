# MobileWan raw score records

## Objective

Save complete raw head energies with strict identity and coverage checks.
Support both complete-view and individual-sigma resume.

## Data flow

The worker supplies original view, actor and split plus capture, noise, context, transformer and source hashes.
It also supplies the exact sampler record and discovered self-attention widths.
Records retain squared projected energy, without normalization or rank selection.

## Organization logic

A view record requires all eight exact sigmas and every head in each discovered self-attention block.
An observation record requires one exact selected sigma index and all the same heads.
Validate the caller's complete identity before accepting either record on resume.
Reject missing blocks, missing heads, changed sigma identities and nonfinite or negative energies.

Write a private temporary file, flush it and synchronize its bytes to disk.
Create the destination with an atomic hard link. This refuses overwrite by competing workers.
When a destination already exists, its identity and energies must match exactly.
Remove the private temporary file even after an error.

## Invariants

Raw observations precede pooling and normalization.
A completed view cannot be inferred from one completed sigma.
The complete record validator checks the worker's sampler identity, including stride one and clean-frame exclusion.
The worker compares that identity with the full audited token record.
A changed model, source, epsilon or code hash invalidates resume.

## Gotchas

View filenames use the full relative-ID hash. Repeated view basenames do not collide.
Per-sigma filenames include the selected index, not a rounded sigma label.
A worker may resume seven observations and run only the missing eighth.
No partial record is presented as a final dataset result.
These records do not verify original pixel files themselves; the input owner verifies those pins.

## Tests

`tests/test_mobilewan_records.py` checks atomic resume, changed provenance, incomplete heads and sigmas, and nonfinite energies.
It also checks independent sigma records and rejects substitution of another sigma index.
The tiny worker test confirms the exact number of resumed forwards.
