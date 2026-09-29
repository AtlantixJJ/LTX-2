# `checks/export_parity.py`

## Objective

Prove that a checkpoint export computes the same function as the
source checkpoint with its masks attached.

## Data flow

An attributable mask report, exported checkpoint, one held-out frozen record,
and one held-out two-window source produce `export_parity.json`. The check
compares the frozen-state denoiser output and both sequential rollout latents,
and records timed window refinements and peak allocated GPU memory for both models.

## Organization

The source and exported transformers load sequentially so only one 22B model is
resident. The source uses `hooks`; the exported model has no runtime masks.
Both paths use `phase1_gates._rollout` and the same encoded windows and sigmas.

## Invariants and gotchas

The mask must match the source checkpoint fingerprint. A compact bf16 export
can differ numerically from its full-width masked counterpart; the sparse p05
export matched exactly on the tested record and two windows. The default
maximum absolute tolerance is explicit in the artifact. This is an executable
parity check, not a perceptual quality gate.

## Verification

Run from LTX-2 in the `ltx` env with `--model 2.5 --gpu-id N --masks
<head_scores.json> --exported-checkpoint <checkpoint>`. Inspect every
comparison and the recorded source/export window times.
