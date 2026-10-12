# Changed-file test selection

## Objective

Select CPU checks from changed files. Selection accelerates edits; the full T2
suite remains the final gate. This test tool does not change production rules.

## Usage

From the LTX-2 root in the `ltx` environment:

```bash
conda run -n ltx python -m scripts.onestep_avatar.tests.select --tier T0
conda run -n ltx python -m scripts.onestep_avatar.tests.select --tier T1 --run
```

Without `--run`, the command prints the plan and makes no writes. `--base`
(default `HEAD`) selects the Git comparison. `--tier` chooses T0 through T3.
`--run` executes the printed commands and returns the first failure.
See [the run guide](../../configs/README.md#10-check-the-code).

## Data flow

Git changed and untracked paths enter an area table and an AST direct-import
scan. Changed tests enter directly. Shared test infrastructure escalates to T2.
Added or removed import lines and dispatch owners add the blocked T3 command.
The output lists the effective tier, selected paths, and exact commands.

## Organization logic

Model, corpus, training, run support, media and execution each have a fixed
area. Experiments select their own test plus direct consumers. Imports include
`from owner import module`, relative imports and imports inside functions.
The scan adds direct test consumers; it does not expand a transitive closure.
Unknown production owners use T2, so they cannot silently receive no checks.
Docs and comments alone select no tests. T2 always includes the prune suite.
A prune-imported owner also adds prune to T1. T0 excludes subprocess tests.
T3 sets the experiment blocker only for the boundary command.

For example, changing `model/causal.py` selects model tests and direct test
consumers. T1 also selects prune. Changing `tests/support/model.py` selects
T2. Adding an import to causal adds a separate blocked boundary check.

## Invariants

Do not import production modules to select tests. Resolve paths against the
canonical repository roots. A failed command stops execution with its code.

## Gotchas

Area selection can miss transitive consumers. Run T2 before reporting done.
Untracked source imports count as boundary changes. Git errors are failures.

## Tests

`tests/test_select.py` checks areas, direct and relative imports, shared-owner
escalation, changed tests, documentation changes and prune/boundary selection.
