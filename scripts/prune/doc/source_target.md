# `data/source_target.py`

## Objective

Freeze the corpus split and build source-latent targets plus calibration records.

## Data flow

Manifest clips, per-clip fps, VAE encodes, deployed geometry, and prompt -> split manifest and format-2 records.

## Organization

The source VAE latent is the target; record families include on-policy and independently renoised states.

## Invariants and gotchas

The old k8 teacher is not the target. A small --max-clips sample can contain zero calibration records; inspect the index before scoring.

## Verification

Check [`tests/test_source_target.py`](../tests/test_source_target.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
