# learn/checkpoints

## Clipped-reset transactions

`save_pre_event` publishes `clipped_reset_transaction_v1` files outside the canonical checkpoint directory.
These files preserve post-optimizer scores, gate values, adapters, optimizer states, rank RNG, and the pending policy.
The stage is `post_optimizer_pre_event`. The canonical checkpoint uses the completed post-reset policy instead.
Replay verifies the complete saved state and existing sidecar hash. It can finish a missing sidecar after interruption.
Transactions are evidence for deterministic replay; they are not completed checkpoints and cannot advance the training cursor.

Loading a canonical clipped-reset checkpoint checks its policy replay and gate Adam local cursor.
At an event or during recovery, the complete gate Adam state must be empty.
Within a window, its local step must equal the global update modulo the window interval.
Moments must be finite, have the raw-score shape, and remain zero on removed coordinates.

## Objective

Publish immutable records and restore exact learning state.

## Data flow

Gate, adapter, optimizer, RNG and draw state enter a same-directory temporary file. Atomic linking publishes complete bytes.

## Organization logic

Readers verify the file receipt and tensor hashes. Resume restores state before the next ordered draw.
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

FSDP and full-replica DDP keep adapters, gates and Adam moments replicated in FP32.
The runtime gathers each rank's Python, NumPy, CPU and local CUDA RNG state before saving on rank zero.
Restore selects the recorded rank's state and rejects a different world size.
All ranks wait until immutable checkpoint publication finishes.
Store `draws_per_update` and `draw_cursor=update*draws_per_update` in new checkpoints.
Four-rank DDP checkpoints use four. Legacy checkpoints without the field use two.
Verify the count against the receipt and frozen batch before resume. Replaying a changed batch requires a separate study.

## Progressive gate removal

Progressive checkpoints include policy masks, frozen logits, the interval-dependent window cursor and full event history.
The receipt binds the policy hash. Loading checks mask/history agreement, deterministic rankings, threshold compliance and the update cursor.
Publishing an existing progressive checkpoint compares the complete trainable, optimizer, RNG and policy payload.

Progressive resume requires the frozen hyperparameters. Policy receipts record candidate ranks and original selected IDs.
