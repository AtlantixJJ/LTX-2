# learn/clipped_reset

## Objective

Remove original head IDs from clipped raw scores. Reset each survivor to one after every selection window.
Keep this policy separate from all saved sigmoid policies. It has no permanent fixed-one branch.

## Data flow

FP32 scores and an alive mask enter `clipped_gate`. The forward uses clip [0, 1].
The backward uses the derivative of clip [-3, 3]. PyTorch gives unit derivatives at both endpoints.
Removed heads have zero forward values and zero task gradients.

`record` receives protected post-optimizer scores in global update order.
An event consumes the initial snapshot and every update snapshot in its window.
Its receipt stores that complete trajectory, pre-event scores and gates, and the post-reset scores.
The next window starts with exactly one post-reset snapshot.
The training step retains the executed-gate gradient. The trainer averages it across ranks and saves
`executed_gate_gradient/NNNN.pt` per update. Negative values on removed heads ask a removed gate to return.
This is logging only; it never changes an update.

## Organization logic

The versioned config names the activation, surrogate, reset, and optimizer reset rules explicitly.
Unknown fields and versions fail. Resume must use the same complete config.
The default window has 50 updates and 51 snapshots. Scores can leave [0, 1] between events.

`drop_statistics` computes FP64 drop, give-back, late change, travel and path regret per head.
The window start is the post-reset snapshot, so the drop is one minus the pre-event score.
Give-back is the share of the deepest drop returned toward the start; it is null for a head that never fell.
A head is regret-free when give-back is at most `giveback_limit` (0.10) and the late change over `late_updates` (10) is negative.

`drop_selection` admits a regret-free head with a strong or a persistent drop.
A strong drop is at least `strong_fraction * full_speed_drop` (0.5 x 0.5 = 0.25) in one window.
A persistent drop is at least `persistent_fraction * full_speed_drop` (0.15) in this window and the preceding one.
The full-speed drop must equal gate LR x interval; `protocol.phase_budget` checks this.
Sort candidates by descending drop (lowest score first), then give-back, path regret, block ID and head ID.
Remove `min(C, floor(cap_fraction * alive))`. For 1,536 alive heads the cap is 61. There is no minimum.
An event with zero removals still resets survivors. Path regret is recorded only.

Persistence flags carry qualified survivors, including capped candidates, into the next window.
An event is empty when it has no candidates and sets no new flag.
Commit removed IDs, set their scores to zero, and set every survivor score to one.
Clear the entire separate gate Adam state. This clears moments and the bias-correction step together.
Keep optimizer parameter groups, adapter state, RNG, and draw order unchanged.
All survivors remain eligible in the next window.

Selection ends after `empty_events_to_stop` (3) consecutive empty events, at no alive heads, or at the selection cap.
The cap event still commits its selected removals. The default cap is update 500.
The terminal reset starts exactly 300 adapter updates. There are no later selection events or score resets.
`protect` preserves removed coordinates during selection and all binary scores during recovery.
The training owner disables gate stepping and sparsity in recovery.

## Invariants

Fresh scores must be an FP32 all-one matrix. Historical masks or logits cannot initialize this policy.
The mask uses original block and head IDs. A temporary clipped zero is still alive.
The canonical event state is post-reset. Gradient traces use post-optimizer, pre-event scores.
Only a separate gate-only Adam or AdamW optimizer can be reset by this owner.
No operation resets adapter moments, adapter counters, the global update, or rank RNG.

State loading replays the complete saved trajectories through the same event owner.
It checks every receipt, mask, reset, stop reason, partial window, and transaction stage.
A completed event cannot run twice after resume. A partial window cannot start over.

## Gotchas

The trajectory `[1, 0.8, 0.6]` has path regret and give-back zero. Adding its reset to one gives one for both.
The reset jump must belong to neither window. A head that never falls has null give-back and cannot be selected.
A small alive count can floor the cap to zero: four percent of fewer than 25 heads removes nothing.
Adam moves a score about the gate LR per update, so one window cannot lower a score much below 0.5.
The zero clip and the minus-three surrogate bound therefore do not activate in this pilot.
Raw-score sparsity uses `relu(alpha + 3)` and the original head-count denominator.
Its derivative is independent of the task surrogate, including scores above three.
Binary functional removal does not prove compact export parity or speed.

## Tests

Run `test_clipped_reset.py` with CUDA hidden in `ltx`.
It checks exact clipping, inclusive surrogate endpoints, removed protection, and unprojected scores.
It checks give-back and late-change limits, threshold equality, strong and persistent tiers, capped flags,
lowest-first ranking, the alive cap, empty-event stops, resets, and exact replay across events and recovery.
It checks analytic sparsity gradients and adapter Adam continuity.
Transaction tests distinguish pre-event optimizer state from canonical post-reset state.
`checks.ddp_native` runs the native four-rank acceptance. Sustained training, fixed-test scoring, and export remain separate gates.
