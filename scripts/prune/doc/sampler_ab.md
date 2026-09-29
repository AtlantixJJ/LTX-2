# `evaluate/sampler_ab.py`

## Objective

Compare Euler and ancestral steps at the same cached initial states.

## Data flow

Format-2 records and both steppers -> T0 sampler comparison artifact.

## Organization

A seeded noise stream makes ancestral draws reproducible; post-step re-freezing preserves context.

## Invariants and gotchas

Without re-freezing, ancestral changes the frozen context and the experiment stops isolating sampler choice.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
