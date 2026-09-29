# `core/refine_task.py`

## Objective

Name the deployed refiner task and its supported schedule/geometry variants.

## Data flow

Model sigmas and scale factors -> k2/one-step schedule and deployed or calibration WindowGeometry.

## Organization

This module owns prompt, window size, overlap, context/chunk widths, and one-step metadata assertions.

## Invariants and gotchas

The constant prompt and 25/9 geometry define calibration meaning. Do not silently swap to full-window or other conditioning.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
