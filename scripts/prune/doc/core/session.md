# `core/session.py`

## Objective and data flow

Bootstrap the model, selected prompt and context, then own resident transformer
and decoder lifetimes. `DTYPE` and `DEFAULT_PROMPT` are defined here. Avatar and
whole-clip consumers use the same argument helpers and context managers.

## Invariants and verification

A Session has no default sampling schedule, AR record root or window geometry.
Callers supply model-facing inputs and schedules. Forwards normally run under
`no_grad`; explicit gradient estimators must reopen autograd. LoRAs fuse at load.
`tests/test_session.py` checks dtype ownership and optional GPU lifetime behavior;
`onestep_avatar/tests/test_prompt_and_whole_clip.py` checks prompt selection.
