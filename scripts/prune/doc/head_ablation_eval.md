# `evaluate/head_ablation_eval.py`

## Objective

Compare selected zeroed heads against an unpruned transformer on held-out records.

## Data flow

Head selectors, frozen records, model -> paired x0 metrics and review media.

## Organization

CLI parses layer.attn kind:index, installs masks via hooks, and uses shared metric/decode paths.

## Invariants and gotchas

Functional ablation is equivalent to deletion at the mask site, but structural export still needs a separate load/forward check.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
