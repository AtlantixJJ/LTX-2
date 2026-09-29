# `score/losses.py`

## Objective

Compute fresh-token x0 MSE and relative L2 on an explicit token set.

## Data flow

Predicted x0, source target, state, optional chunk mask -> scalar loss.

## Organization

Shape guards normalize masks and reject mismatched targets.

## Invariants and gotchas

Pass chunk_token_mask for AR scoring; the default denoise_mask also includes the keyframe.

## Verification

Check [`tests/test_losses.py`](../tests/test_losses.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
