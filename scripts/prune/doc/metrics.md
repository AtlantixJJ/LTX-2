# `evaluate/metrics.py`

## Objective

Provide common T0 latent, T1 pixel, T2 rollout, and T3 visual metrics.

## Data flow

Predictions, teacher/source tensors, rollout rows -> numeric summaries, grids, and synchronized MP4.

## Organization

No model loading here; callers own lifecycle and pass tensors.

## Invariants and gotchas

T3 fps is display metadata. Match shapes and source-frame alignment before interpreting image/video deltas.

## Verification

Check [`tests/test_metrics.py`](../tests/test_metrics.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
