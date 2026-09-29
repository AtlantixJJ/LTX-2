# `score/head_scores.py`

## Objective

Estimate attention-head importance with contribution, Michel, and Gauss-Newton methods.

## Data flow

Frozen records and chunk mask -> per-head scores, optional iterative mask, ablations, and rank correlation.

## Organization

The estimators share hooks and loss; gradient methods reopen autograd only around required VJPs.

## Invariants and gotchas

Every method must score the same cached tensors. Iterative rescoring uses the currently masked model; labels and provenance travel with results.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
