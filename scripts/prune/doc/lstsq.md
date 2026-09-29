# `score/lstsq.py`

## Objective

Stream ridge regression statistics for reduced attention and FFN output maps.

## Data flow

Retained activations, original outputs, and optional token mask -> fitted projection matrix.

## Organization

RidgeAccumulator collects fp32 normal equations; specialized closures map head/channel indices.

## Invariants and gotchas

An empty accumulator is invalid. Fitted shape must match retained channel count and model output width.

## Verification

Check [`tests/test_lstsq.py`](../tests/test_lstsq.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
