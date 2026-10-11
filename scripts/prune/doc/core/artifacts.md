# `core/artifacts.py` — output paths and run records

## Objective

Keep pruning outputs under one workspace root. Give each automatic run a directory and an index entry.

## Data flow

`OUT_ROOT` is `WORKSPACE_ROOT/expr/refiner_prune`.
`root(key)` selects a model directory. `gate(key, name)` selects a named JSON file.
`run_dir` creates a directory and appends one record to `runs/index.jsonl`.

## Organization logic

Only `caps` and `prompt_cache_check` are named gates.
Run names use the timestamp, PID and caller prefix. Existing names receive a numeric suffix.
The index records the script, arguments, revision and PID.

Worked example: `gate("2.5", "caps")` returns `expr/refiner_prune/2.5/caps.json`.
An unknown gate raises `KeyError`.

## Invariants

- Model keys separate output paths.
- The caller supplies the script and arguments.
- Explicit output paths remain the responsibility of each entry script.

## Gotchas

An index entry does not prove completion or scientific acceptance.
This module defines no queue, sampling schedule or report layout.

## Tests

`tests/test_artifacts.py` checks gate names, paths and run indexing.
