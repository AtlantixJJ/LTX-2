# `sampling.py`

## Objective

Own one-step avatar sigma-grid and checkpoint-condition validation.

## Data flow and organization logic

`one_step_schedule` checks the requested operating point against the supplied
model grid and returns `[sigma0, 0]`. `assert_one_step_conditions` checks a
fixed-sigma checkpoint against an explicit schedule. Deployment consumes the
single nonzero level. The training/probe grid policies remain documented in
[known_gaps.md](known_gaps.md).

## Invariants and gotchas

There are no window, carryover or pruning-task constants. An unlabelled checkpoint
passes the sigma metadata check but still must have one step. This helper does
not add enforcement at callers that do not invoke it.

## Tests

`tests/test_sampling.py` checks grid rejection, schedules and metadata conditions.
