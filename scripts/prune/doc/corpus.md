# `data/corpus.py`

## Objective

Locate and select source clips from the frozen subject-disjoint corpus.

## Data flow

Manifest and requested geometry/window count -> source video paths.

## Organization

Helpers keep source lookup and clip picking out of GPU gates.

## Invariants and gotchas

Use the frozen manifest split and enough source frames for each requested window; do not infer identity from adjacent filenames.

## Verification

Check [`tests/test_corpus.py`](../tests/test_corpus.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
