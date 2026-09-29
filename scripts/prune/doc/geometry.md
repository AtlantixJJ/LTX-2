# `core/geometry.py`

## Objective

Derive video VAE spatial and temporal scales from checkpoint metadata.

## Data flow

VAE and transformer safetensors headers -> scale factors and source label; dimensions -> checked latent geometry.

## Organization

Prefer the VAE block list, fall back to transformer metadata, then report the default explicitly.

## Invariants and gotchas

A default scale is a reported fallback, not proof of the installed VAE geometry. Validate F % time == 1 and overlap alignment before a rollout.

## Verification

Check [`tests/test_geometry.py`](../tests/test_geometry.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
