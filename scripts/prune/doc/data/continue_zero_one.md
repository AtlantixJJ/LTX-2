# data/continue_zero_one

## Objective

Freeze a fresh continuation study from the completed removal-only checkpoint at global step 500.

## Usage

Run the direct CPU preparation owner from LTX-2 in the ltx environment:

```bash
conda run --no-capture-output -n ltx python -m scripts.prune.data.continue_zero_one \
  --previous ../expr/refiner_prune/2.5/progressive_head_gate_removal_f18_20261015/configs/frozen_study.json \
  --plan ../plans/2026-10-16-progressive-zero-one-resume-from-step500.md \
  --output ../expr/refiner_prune/2.5/progressive_zero_one_f18_from_u500_20261016 \
  --max-step 4000
```

The arguments pin the parent, plan and fresh output root. `--max-step` is the final global step.
Outputs include frozen study/config/draws, migrated checkpoint 500, inherited events and preparation evidence.
No acceptance receipt is copied. See the [run guide](../usage.md#zeroone-continuation).

## Data flow

Verify the parent study digest, complete strict audit, pinned files and full component hashes.
Load the canonical checkpoint through the checkpoint owner. Deterministically extend ordinary seed-42 draws.
Verify the entire existing ancestor and parent draw prefixes before freezing the extension.

## Organization logic

Preserve adapters, logits, optimizers, all rank RNG states and draw cursor 2000.
Upgrade only the policy envelope, identity and lineage. Hash the remaining payload before and after publication.
For final step 4000, save 16,000 draws. The next update consumes draws 2000 through 2003.
Copy parent event receipts with exact bytes represented by canonical JSON. New decisions begin at 550.
The environment and all corpus/context/component pins remain inherited. Model execution is separate.

## Invariants

Require source step 500, 160 removed heads, 1,376 free heads and zero fixed-one heads.
Reject a changed draw prefix or state payload. Executed gates must remain bit-identical at migration.
Require fresh four-rank native acceptance before any sustained continuation update.
Keep all parent artifacts immutable and create a new study identity.

## Gotchas

The original source may have fewer draws than the new budget. Extend with the same deterministic generator;
never repeat its first 2,000 draws. Budgets use global steps, not a counter reset at the fork.
The parent study's frozen code differs after the policy amendment; verify parent bytes against its own pins.

## Tests

Preparation checks full bytes, ancestor/parent draw equality, unchanged state payload and policy migration.
Focused CPU policy tests cover migration and replay. Fresh native acceptance proves executed parent parity.
