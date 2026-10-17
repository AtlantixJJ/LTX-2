# learn/eligible

## Objective

Select fractions of each eligible population and train an exact fixed-gate tail.

## Data flow

Saved source state enters the policy migration. Saved draws retain their original global indices.

## Organization logic

Restore both parent masks and every partial-window snapshot. Preserve the global update cursor.
Each branch requests max(minimum_count, ceil(fraction * its own eligible count)). Clip to that branch's candidates.
For 1,200 zero-eligible heads, 2% requests 24. For 100 one-eligible heads, the floor requests 16.
The legacy mode uses disjoint regret thresholds on the pre-commit free mask. Gate-only mode uses the zero-first order below.
When zero eligibility is empty, fix all remaining free heads to one, including low-regret and constant heads.
Record ordinary and forced one IDs separately. Record the event cursor T and immutable final cursor T+300.
The same tail starts if an ordinary event exhausts all free heads.
During the tail, advance the cursor without further decisions. Gate logits and their Adam moments stay fixed.
Adapters continue ordinary optimization. State loading replays decisions and validates masks, receipts and finalization bounds.

## Invariants

Fixed masks never reopen. Execute free*sigmoid(alpha)+one at original widths.
Keep learning rates and consumed input order unchanged. Reject altered configuration on resume.

## Gotchas

Regret measures trajectory reversals, not perceptual importance. The terminal override ignores the ordinary one eligibility threshold.
The 300 updates start after the event, not at its optimizer update. A resumed tail keeps its original end.

## Tests

`test_eligible_gates.py` checks eligible denominators, floors, partial migration, inherited ones, forced remainder, tampering and exact 300 updates.
Fresh four-rank native acceptance verifies migrated predictions and actual fixed-gate adapter training.

## Gate-value-only fixed-one mode

With fixed_one_selection=gate_only, first select zeros with the unchanged trajectory rule.
Exclude these selected zero IDs from one candidates. Every remaining free head is eligible for one.
Ignore regret and travel, including constant trajectories. Rank descending CPU FP32 sigmoid gate, then block/head IDs.
For 900 free heads and 16 selected zeros, E1 is 884; two percent requests 18 one heads.
The configuration stores a null fixed_one_regret_threshold; no regret tie break applies.
Legacy events retain their saved high-regret mode. Migration preserves both masks and the partial window.
The terminal override and exact finalization tail do not change.
