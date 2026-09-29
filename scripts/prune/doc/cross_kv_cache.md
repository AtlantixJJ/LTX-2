# `evaluate/cross_kv_cache.py`

## Objective

Memoize cross-attention text K/V at an explicitly declared sigma.

## Data flow

Transformer attn2 modules and sigma scope -> reusable K/V wrapper and bit-exact check.

## Organization

A registered nn.Module wrapper replaces each projection temporarily; context exit restores originals.

## Invariants and gotchas

Do not key by Python object identity: modulated context is rebuilt per call. Cache validity depends on sigma and prompt context.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
