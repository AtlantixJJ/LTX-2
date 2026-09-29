# `data/prompt_cache.py`

## Objective

Build and verify the one constant text-conditioning tensor.

## Data flow

Model key, refiner prompt, dtype, and device -> cached prompt context tensor and bit-exact check.

## Organization

Cache keys bind prompt and model; verification recomputes the encoder output.

## Invariants and gotchas

Changing REFINE_PROMPT changes the cache key. Verify the cached tensor against the active checkpoint before trusting a run.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
