# `data/records.py`

## Objective

Select a balanced subset of frozen calibration records for scoring.

## Data flow

Calibration index/root, split and optional positive limit -> deterministic record paths.

## Organization

Selection is shared by score and evaluate entry points.

## Invariants and gotchas

A limit must stay positive and split filtering must prevent held-out/calibration
leakage. The missing-cache error names the current `data.source_target`
producer.

## Verification

Check [`tests/test_records.py`](../tests/test_records.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
