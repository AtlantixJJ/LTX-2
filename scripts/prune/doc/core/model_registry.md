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

Run the package CPU suite and native export parity when model-facing behavior changes. Run `python -m pytest scripts/prune/tests -q -m 'not gpu'` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For model-facing changes, run the native checks in [VALIDATION](../VALIDATION.md).
