# learn/checkpoints

## Objective

Publish immutable records and restore exact learning state.

## Data flow

Gate, adapter, optimizer, RNG and draw state enter a same-directory temporary file. Atomic linking publishes complete bytes.

## Organization logic

Readers verify the file receipt and tensor hashes. Resume restores state before the next ordered draw.
A prefix branch requires update 700, the soft phase and a matching budget-independent prefix identity.

## Invariants

Completed files never change. A changed checkpoint fails. Only the active CUDA device contributes RNG state.

## Gotchas

Optimizer/RNG records use trusted local pickle loading. Tensor artifacts use weights-only loading. No full backbone is saved per update.

## Tests

Run `test_head_learning_state.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.

## Blind finalist freeze

`validate_finalists` checks the frozen study, settings hash, exact reserved actor list and every selected checkpoint, mask and envelope hash.
A marker file without these fields fails.
Candidate inference additionally checks the arm and replicate seed against that frozen list.
Teacher reference generation uses the same freeze before producing blind evidence.
