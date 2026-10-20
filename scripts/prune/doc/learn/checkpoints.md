# learn/checkpoints

## Objective

Publish immutable records and restore exact learning state.

## Data flow

Gate, adapter, optimizer, RNG and draw state enter a same-directory temporary file. Atomic linking publishes complete bytes.

`save` and `load` own `head_learning_state_v1`. `load` rejects retired progressive policy payloads.
`publish_state` and `read_state` own shared versioned state publication and receipt verification.
FFN state uses `head_ffn_learning_state_v1`; V3 state uses `joint_learning_state_v3`.
Both numerical owners call the shared state I/O with their own format and validation rules.

## Organization logic

Readers verify the file receipt and tensor hashes.
The shared reader also checks unchanged-state hash, optional study and world size, rank RNG coverage and draw cursor.
FFN cursor stride is the saved world size. V3 preserves its versioned four-draw cursor, including three-rank pilots. Resume restores state before the next ordered draw.
A prefix branch requires update 700, the soft phase and a matching budget-independent prefix identity.

## Invariants

Completed files never change. A changed checkpoint fails. Save RNG states for every CUDA device that owns an adapter or gate. Restore each state on its recorded device.
Legacy single-device records remain readable. Do not initialize unrelated GPU contexts.

## Gotchas

Optimizer/RNG records use trusted local pickle loading. Tensor artifacts use weights-only loading. No full backbone is saved per update.

## Tests

Run `test_head_learning_state.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.

## Blind finalist freeze

`validate_finalists` checks the frozen study, settings hash, exact reserved actor list and every selected checkpoint, mask and envelope hash.
A marker file without these fields fails.
Candidate inference additionally checks the arm and replicate seed against that frozen list.
Teacher reference generation uses the same freeze before producing blind evidence.

## Distributed resume

Full-replica DDP keeps adapters, gates and Adam moments replicated in FP32.
The runtime gathers each rank's Python, NumPy, CPU and local CUDA RNG state before saving on rank zero.
Restore selects the recorded rank's state and rejects a different world size.
All ranks wait until immutable checkpoint publication finishes.
Store `draws_per_update` and `draw_cursor=update*draws_per_update` in new checkpoints.
Four-rank DDP checkpoints use four. Legacy checkpoints without the field use two.
Verify the count against the receipt and frozen batch before resume. Replaying a changed batch requires a separate study.
