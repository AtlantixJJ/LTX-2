# `evaluate/bench_refiner.py`

## Objective

Measure baseline refiner time, memory, and FLOPs across geometries and execution modes.

## Data flow

Synthetic state, model caps, chosen sampler/cache mode -> bench JSON.

## Organization

Separates analytic FLOPs, real timed denoiser runs, compile/CUDA graph, and K/V-cache axes.

## Invariants and gotchas

The synthetic --fps default is not a corpus fps. Ancestral benchmarking is rejected without per-step noise. Match hardware and geometry for speed comparisons.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
