# `score/ffn_scores.py`

## Objective

Score and optionally reconstruct FFN channels at deployed chunk states.

## Data flow

Frozen records -> post-GELU RMS, output-weight scores, masks, T0, and fitted projection weights.
The T0 sweep includes an unpruned `0.0` reference. `--held-out-max-records N`
evaluates the same score-derived masks on a separate held-out sample.

## Organization

Hooks collect chunk-token activations; iterative allocation removes an equal fraction per layer.

## Invariants and gotchas

Per-layer allocation avoids erasing a whole branch. The held-out sweep is a
T0 screen, not a T1 pixel or long-form quality verdict. Reconstruction must
be checked on held-out states before export. In the 2026-09-29 four-record
calibration and twelve-record held-out screen, 10%/25%/50% FFN pruning raised
held-out T0 relative L2 by 0.010/0.034/0.105 respectively. A 10% compact
FFN export failed the two-window numerical parity check.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
