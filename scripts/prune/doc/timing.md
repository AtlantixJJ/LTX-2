# `evaluate/timing.py`

## Objective

Time CUDA stages and count model FLOPs for benchmark reports.

## Data flow

Already-warmed callable and CUDA device -> synchronized latency, peak memory, or FLOPs.

## Organization

StageTimer synchronizes before and after; FlopCounterMode runs a real forward.

## Invariants and gotchas

A FLOP-counted call is not a timing dry run. Keep timing and FLOP sampling separate.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
