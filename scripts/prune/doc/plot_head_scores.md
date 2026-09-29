# `report/plot_head_scores.py`

## Objective

Render head-importance rankings and cross-method agreement.

## Data flow

One JSON score report per method on matched data -> rank heatmaps and agreement plots.

## Organization

Ranks normalize unlike estimator units before comparing block rows and head columns.

## Invariants and gotchas

Do not combine different checkpoints or record sets; plots are comparative evidence, not a numeric gate.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
