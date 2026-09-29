# `score/ffn_scores.py`

## Objective

Score and optionally reconstruct FFN channels at deployed chunk states.

## Data flow

Frozen records -> post-GELU RMS, output-weight scores, masks, T0, and fitted projection weights.

## Organization

Hooks collect chunk-token activations; iterative allocation removes an equal fraction per layer.

## Invariants and gotchas

Per-layer allocation avoids erasing a whole branch. Reconstruction must be checked on held-out states before export.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
