# `evaluate/decode.py`

## Objective

Decode dense or token-space video latents through the shared video VAE path.

## Data flow

Dense latent or positioned token latent -> RGB frames.

## Organization

Token decoding reconstructs a dense frame grid and calls the dense decoder.

## Invariants and gotchas

Position grids must cover every token; decode shape conventions differ between dense and token entry points.

## Verification

Check [`tests/test_decode.py`](../tests/test_decode.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
