# `score/prune_schedule.py`

## Objective

Iteratively remove low-scoring heads/channels and re-fit output projections.

## Data flow

Current model and scorer callback -> mask history; records and keep indices -> fitted attention projection.

## Organization

Head selection is global; FFN selection is per layer; each round re-scores the masked model.

## Invariants and gotchas

The callback must measure the current mask. Preserve at least one executable unit per branch and verify final achieved sparsity.

## Verification

Check [`tests/test_prune_schedule.py`](../tests/test_prune_schedule.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
