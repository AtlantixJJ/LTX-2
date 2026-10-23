# Joint selection state and reset

## Objective

Choose which video self-attention heads and FFN channels to keep by observing trained score movement.
Removal is reversible until a returned unit is removed again; terminal alignment then prepares fixed-mask adapter recovery.
The [glossary](../glossary.md) defines V3, events, give-back and permanence.
The [algorithm](../core_algorithm.md#joint-clipped-selection-with-return) owns equations, thresholds, schedule defaults and the worked event.

## Data flow

1. `learn.joint_training.JointStep` supplies post-optimizer FP32 head/channel scores to `learn.readmission.JointPolicy.record`.
2. Each `learn.readmission.Family` accumulates only minimum, maximum, final and late-start scores.
3. At an event, compute removal and return candidates from the same pre-event masks.
4. Rank candidates, apply block guards and family caps, then commit both directions together.
5. Reset scores and gate optimizer state. Return the full event record to the training owner for immutable publication.
6. At selection end, align FFN widths, reset again and enter fixed-mask recovery.

## Organization logic

Each family stores `kept`, `permanent`, `returned`, `persist`, `removal_drop`, `removed_at` and `last_rise` in original coordinates.
`persist` remembers whether the preceding event met the weaker removal condition.
`returned` remembers any return, so a later removal needs the strong condition and becomes permanent.
`removal_drop` and `last_rise` order terminal alignment; `removed_at` attributes the latest removal.

`JointPolicy` stores the global update cursor, consecutive quiet count, selection endpoint and alignment result.
The state includes partial-window summaries and sample count; resuming mid-window must continue those observations.
`JointPolicy.from_state` checks format, shape, finiteness, masks, minimum widths, cursor bounds and partial-window length.
The numerical checkpoint additionally stores both gate optimizers, adapters, adapter optimizer, rank RNG and consumed-input chain.

For example, a resume at update 73 of the standard event schedule must retain 23 observations after the last reset.
Restarting that window from reset values would change eligibility at the next event.
The [worked head event](../core_algorithm.md#one-selection-event) shows the actual candidate calculation and cap.

`budget_joint_v3` sets `projection_mode` because it selects a budgeted prefix each update rather than observing event windows.
The [budget algorithm](../core_algorithm.md#matched-budget-control) owns its cost and prefix rules.
Its frozen reference receipt supplies budget and selection endpoint.

### Reviews belong to the training owner

New review records use descriptive names:

- `allocation_review`: check early block allocation, repeated channel-cap pressure and monitor quality.
- `alignment_review`: compare the pre-alignment and aligned result at selection end and verify prerequisite acceptance.

`learn.joint.reviews` publishes these records and requests a stop when triggers remain pending.
`learn.joint.check_start` also reads legacy review filenames, so a renamed record cannot bypass an unresolved saved review.
The frozen study supplies the prerequisites; CLI flags for optimizer replay, channel ablation and export isolation retain their recorded names.
Review pauses do not change the [five evidence gates](../VALIDATION.md#evidence-gates).

## Invariants

- Make all event choices from pre-event state; mutating a mask earlier would change return eligibility and caps.
- Preserve original IDs and stable ordering so equal scores have reproducible outcomes.
- Keep window summaries on CPU with memory proportional to unit count; a full trajectory is unnecessary for decisions.
- Reset gate Adam state only. Preserve adapter optimizer history so recovery learning continues across selection events.
- Keep a zero-gradient score path during fixed-mask recovery so DDP sees every registered parameter in each backward.

## Gotchas

A removed, non-permanent unit can have a nonzero continuous gate before a recorded binary return.
Permanence applies to selection; final alignment can restore a permanent FFN channel to meet the executable width multiple.
Quiet events count eligible score candidates before caps, so a capped-away candidate can prevent early convergence.
A mismatched state format or partial-window cursor fails before continuing optimization.

## Tests

`tests/test_readmission.py` checks clipped forward/backward, strong and persistent candidates, returns, permanence,
stable ties, block/global caps, quiet/hard stopping, alignment, budget projection and exact serialized state.
It also compares joint distributed gradients with independent serial controls.
`tests/test_joint_monitor_gradients.py` checks the retained task-gradient evidence.
`tests/test_ddp_owner_families.py` checks hook boundaries and review behavior.
`tests/test_ddp_owner_loop.py` checks complete update/receipt/input-chain equivalence across resume.
