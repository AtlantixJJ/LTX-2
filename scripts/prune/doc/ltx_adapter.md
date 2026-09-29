# `core/ltx_adapter.py`

## Objective

Quarantine calls to upstream private LTX pipeline APIs.

## Data flow

Pipeline stages, state specifications, and samplers -> build/step contexts and encoder/decoder access.

## Organization

Thin wrappers centralize underscore-prefixed access and lifetime management.

## Invariants and gotchas

When the upstream LTX pin changes, review this module first and run test_ltx_adapter.py; no other prune module should import private LTX symbols.

## Verification

Check [`tests/test_ltx_adapter.py`](../tests/test_ltx_adapter.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
