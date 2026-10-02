# `sampling.py`

## Objective

Own one-step avatar sigma-grid and checkpoint-condition validation.

## Data flow and organization logic

`one_step_schedule` checks the requested operating point against the supplied
model grid and returns `[sigma0, 0]`. `assert_one_step_conditions` checks a
fixed-sigma checkpoint against an explicit schedule. Deployment consumes the
single nonzero level. The training/probe grid policies remain documented in
[known_gaps.md](known_gaps.md).

### The adapter condition reader (G3, 2026-10-02)

`read_adapter_metadata` reads a LoRA's safetensors metadata without its tensors.
`adapter_condition_problems` lists every way the requested evaluation differs from the
recorded training conditions: base variant and **fingerprint** (from `backbone.identity`),
model key, objective, arm, `clean_c0_v1`, loss, attention, history computation, block /
context / sink geometry, training history policy, `alpha == rank`, the calibrated σ (fixed
`sigma0`, or one of the levels of a `mixed` adapter) and the schedule (`ONE_STEP` means exactly
`[σ, 0]`). `check_adapter_conditions` raises unless `override=True`, and returns the list so
the caller records it. σ calibration is separate from base support: dev accepts any start in
`(0, 1]`, the distilled grid nine points. `visualize_d1.py` calls it before loading the 22B
base; `--off-condition-override` turns the refusal into a recorded `off_condition` label.

## Invariants and gotchas

There are no window, carryover or pruning-task constants. An unlabelled checkpoint
passes the sigma metadata check but still must have one step. These helpers do
not add enforcement at callers that do not invoke them: `visualize_d0.py` and
`onestep_core.rollout` do not call the condition reader yet (G3 stays in progress). An
adapter written before 2026-10-02 has no `onestep_avatar_base_*` or history-computation keys
and is refused unless overridden.

## Tests

`tests/test_sampling.py` checks grid rejection, schedules and metadata conditions;
`tests/test_train.py::test_condition_reader_refuses_a_dev_adapter_on_distilled_weights` checks
the reader against metadata written by `train.checkpoint_metadata`.
