# data/extend_learned_heads

## Objective

Continue a completed soft checkpoint under a longer phase budget, in a new frozen study.
Keep the source run immutable. Only `soft_updates`, `binary_updates` and `compact_updates` change.

## Usage

Historical study plans were consolidated in the [workspace history](../../../../../plans/history/README.md).
For fresh preparation, replace `/path/to/reviewed-study-plan.md` with the actual reviewed specification.
Retired reference pointers are not plan inputs, and retired source folders must not be recreated from these examples.

Launch directly on CPU from `LTX-2` in conda environment `ltx`, after the source owner has stopped.

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.prune.data.extend_learned_heads \
  --previous ../expr/refiner_prune/2.5/<source>/configs/frozen_study.json \
  --checkpoint ../expr/refiner_prune/2.5/<source>/training/learned_joint/replicate_42/checkpoints/update_0700.pt \
  --plan /path/to/reviewed-study-plan.md \
  --output ../expr/refiner_prune/2.5/<new study> \
  --soft-updates 2000 --binary-updates 0 --compact-updates 500
```

The output must be new. The checkpoint must be the source's soft checkpoint at its `soft_updates` boundary.
Output: the new frozen study, extended draw lists, the rebound checkpoint, reused native acceptance and `provenance/continuation.json`.
Then start the direct owner (`learn.run_full`) on the new study. See the [run guide](../usage.md#extend-the-soft-phase).

## Data flow

The source study, checkpoint, update receipts and consumed inputs enter the CPU validator.
It regenerates each replicate's ordinary draws for the new total and requires the old list as an exact prefix.
It rebinds the checkpoint identity and keeps every state payload bit.

## Organization logic

Verify the source study digest and frozen plan. Restrict changed production files to the budget amendment owners.
Verify the training identity, the completed soft phase and every consumed input and noise hash for updates 1 to N.
Replace only the source run's own draw and config pins; shared corpus and component pins stay.
Reuse the source native acceptance with an explicit reuse record. Execution, inputs and loss are unchanged.

A source checkpoint at update 700 stays update 700 with draw cursor 2800 (four draws per update).
The next update is 701. With 2,000 soft and 0 binary updates, soft training continues to 2000 and compact recovery covers 2001-2500.
There is no optimizer reset, RNG reset or fresh gate initialization.

## Invariants

Preserve gate logits, adapters, optimizer moments, mask, draw cursor, phase and all rank RNG states.
The new checkpoint payload fingerprint must equal the source fingerprint after excluding identity and lineage.
The extended draw list must start with the exact frozen draw list. Never modify source receipts.

## Gotchas

The extended study has a mixed history: updates 1-700 come from the source run under its budget.
The source run's binary refinement and later checkpoints are not part of the new study.

## Tests

Run `test_phase_budget.py` with CUDA hidden. It checks budget boundaries and the exact draw prefix.
A dry run on a scratch output verifies state equality, prefix receipts and the reused acceptance record.
