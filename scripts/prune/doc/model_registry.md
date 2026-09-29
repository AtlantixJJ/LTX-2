# `core/model_registry.py`

## Objective

Resolve a 2.3 or 2.5 key into checkpoint paths, sampler policy, sigma values, and measured ModelCaps.

## Data flow

Model key and optional checkpoint override -> RefinerModel with metadata-derived capability fields.

## Organization

ModelCaps reads transformer config from safetensors; resolve handles split and monolithic checkpoint packs.

## Invariants and gotchas

Head counts and widths come from checkpoint metadata, not plan tables. Model keys and head indices are generation-specific.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
