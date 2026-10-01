# `core/artifacts.py`

## Objective and data flow

Own `expr/refiner_prune/<model>` roots, capability/prompt-cache verification
paths and attributable run directories. `run_dir` records script, argv, revision
and PID in `runs/index.jsonl` and avoids collisions.

## Invariants and verification

Explicit native output paths are allowed. No AR calibration, phase or report
paths are defined. `tests/test_artifacts.py` covers named gates and run indexing.
