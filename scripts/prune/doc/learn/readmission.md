# learn/readmission

## Objective

V3 family decisions and joint selection.

## Data flow

Family keeps original IDs, first-removal drops, return flags, permanence and four window summaries. JointPolicy owns the global cursor, quiet events and terminal alignment.

## Organization logic

A kept score of 0.6 has drop 0.4. It passes the strong head test if give-back is at most 0.1 and late change is negative. A removed score of 0.4 passes the mirrored return test. The global removal cap is floor(0.04 times pre-event kept count). Block limits and minimum widths apply before the global cap.

## Invariants

Keep saved input hashes, exact schedules and original unit IDs.
Keep native loss reduction in FP32. Preserve the frozen backbone and complete rank RNG.

## Gotchas

Do not use V2 states here. Events compute both directions before changing any mask. Returned units require a strong second removal. The second removal is permanent. Recovery preserves a zero-gradient graph. Alignment restores returnable channels before permanent channels.

## Tests

Run `tests/test_readmission.py` with CUDA hidden.
Native acceptance and held-out export parity remain independent evidence requirements.
