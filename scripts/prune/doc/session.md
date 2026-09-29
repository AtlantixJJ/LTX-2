# `core/session.py`

## Objective

Give CLI entry points one bootstrap path for model, prompt context, sigmas, and transformer/decoder lifetime.

## Data flow

CLI args -> preflighted Session -> no-grad resident transformer or decoder.

## Organization

Shared argument helpers, DTYPE, open_session, and context managers prevent per-script setup drift.

## Invariants and gotchas

DTYPE is declared here once. Model forwards normally run in no_grad; gradient estimators open grad explicitly. Empty LoRA tuple preserves the baseline path.

## Verification

Check [`tests/test_session.py`](../tests/test_session.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
