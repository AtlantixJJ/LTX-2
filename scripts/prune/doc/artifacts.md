# `core/artifacts.py`

## Objective

Own every stable path and per-run directory beneath expr/refiner_prune/<model>.

## Data flow

Model key, gate name, or run label -> namespaced paths and runs/index.jsonl.

## Organization

Path constructors separate stable gates from unique run outputs; run_dir records argv, revision, and PID.

## Invariants and gotchas

Add stable gate names to GATES. Readers and writers must use the same constructor; run directories are unique but their JSONL index is append-only.

## Verification

Check [`tests/test_artifacts.py`](../tests/test_artifacts.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
