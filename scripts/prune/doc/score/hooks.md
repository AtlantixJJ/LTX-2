# `score/hooks.py` — functional masks and activation observation

## Objective

Observe or mask native attention and FFN branches while keeping checkpoint weights unchanged.

## Data flow

`read_mask_artifact` validates a saved width-mask report and returns masks plus the full file SHA-256.
Attachment helpers install pre-hooks at executed output-projection inputs.
A context manager removes all hook handles on exit.

## Organization logic

The model can be an `X0Model` wrapper or its underlying velocity model.
Branch names are `<layer>.attn1`, `<layer>.attn2` and `<layer>.ff`.

Head masks reshape `[B,T,heads*dim_head]` to separate heads before multiplying.
FFN masks multiply post-activation channels before `ff.net[2]`.
Masks are float32 leaves; multiplication casts them to activation dtype.
`collect_activations` observes the same boundaries without applying a mask.

Artifact validation requires native D0 provenance and the selected source fingerprint.
Attention masks must cover both attention kinds in every layer when present.
FFN masks must cover every FFN when present. An omitted complete family remains unpruned.
Each branch has binary values, the exact width and at least one retained unit.

Worked example: head mask `[1,0]` keeps the first head and zeros every channel of the second head.
Output-projection bias remains unchanged.

## Invariants

- Validate saved artifacts before attaching hooks.
- Keep checkpoint parameters unchanged.
- Remove hooks after success or failure.
- Reject calibration actors and source reuse in held-out checks.
- Held-out sigmas must belong to the calibration levels.

## Gotchas

The low-level attachment helpers allow differentiable masks.
They do not replace strict saved-artifact validation.
A functional mask keeps full execution widths and does not establish a speed gain.
Actor identity spans clips, Parts, views and filesystem aliases.

## Tests

`tests/test_hooks.py` checks branch boundaries, widths and cleanup.
`tests/test_mask_artifact.py` checks schema, content identity and incomplete families.
`tests/test_actor_holdout.py` checks actor separation.
