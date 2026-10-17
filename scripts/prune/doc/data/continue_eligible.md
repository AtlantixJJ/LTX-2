# data/continue_eligible

## Objective

Freeze a continuation from a stopped checkpoint without resetting training state.

## Usage

Run in ltx from LTX-2 after the source owner stops:

```bash
python -m scripts.prune.data.continue_eligible \
  --previous ../expr/refiner_prune/2.5/progressive_zero_one_f18_from_u500_20261016/configs/frozen_study.json \
  --plan ../plans/2026-10-16-eligible-two-percent-zero-stop-final300.md \
  --output ../expr/refiner_prune/2.5/progressive_eligible2pct_min16_final300_20261016_from_u1357 \
  --resume-update 1357
```

The output must be fresh. This CPU command does not start training.
See [run guide](../usage.md#eligible-population-continuation).

## Data flow

Saved source state enters the policy migration. Saved draws retain their original global indices.

## Organization logic

Require a stopped source owner and a passing independent audit at the requested source cursor.
Verify source study digest, pinned inputs and components, source checkpoint identity and draw cursor.
Migrate both masks and the partial decision window through EligiblePolicy. Preserve training tensors, optimizers and rank RNG.
Generate a deterministic schedule with the whole prior draw list as an identical prefix.
The finite envelope allows at most one nonterminal event per initially free head, plus a terminal event and 300 updates.
The envelope is a guard. Actual completion requires the recorded terminal event plus exactly 300 updates.
Keep the original step-500 comparison study separate from the immediate training source.
Publish a fresh study identity, migrated checkpoint, copied historical event receipts, environment and preparation evidence.

## Invariants

Fixed masks never reopen. Execute free*sigmoid(alpha)+one at original widths.
Keep learning rates and consumed input order unchanged. Reject altered configuration on resume.

## Gotchas

Regret measures trajectory reversals, not perceptual importance. The terminal override ignores the ordinary one eligibility threshold.
The 300 updates start after the event, not at its optimizer update. A resumed tail keeps its original end.

## Tests

`test_eligible_gates.py` checks eligible denominators, floors, partial migration, inherited ones, forced remainder, tampering and exact 300 updates.
Fresh four-rank native acceptance verifies migrated predictions and actual fixed-gate adapter training.

## Gate-only continuation

Pass --gate-only and --resume-update N to continue a stopped eligible-policy run with gate-only fixed-one selection.
The source must have a matching stop audit and no started finalization tail.
Preserve the entire prior draw schedule by keeping at least its full mathematical envelope.
Keep comparison_parent_study across repeated migrations so videos still compare with the original step500 checkpoint.
Example from LTX-2 in ltx:

```bash
python -m scripts.prune.data.continue_eligible \
  --previous ../expr/refiner_prune/2.5/progressive_eligible2pct_min16_final300_20261016_from_u1357/configs/frozen_study.json \
  --plan ../plans/2026-10-16-gate-value-only-fixed-one-final300.md \
  --output ../expr/refiner_prune/2.5/progressive_gate_only2pct_min16_final300_20261016_from_u1668 \
  --resume-update 1668 --gate-only
```
