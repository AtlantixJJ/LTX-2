# `checks/parity_check.py`

## Objective

Check the registry refactor against the earlier 2.3 refine script.

## Data flow

Git baseline script and current script on same clip/seed -> latent equality report.

## Organization

Writes a temporary baseline copy in scripts/ and removes it in finally.

## Invariants and gotchas

This is 2.3-only history; 2.5 has no pre-refactor baseline. Use method_parity for deployed 2.5 behavior.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
