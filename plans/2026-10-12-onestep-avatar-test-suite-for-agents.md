# One-step avatar CPU tests: faster, selective runs for agents

Date: 2026-10-12. Status: implemented in the `main` worktree, with the
approval-only report-test migration and optional environment installs held.
The original measurements below are the pre-change baseline from `main`
(`11f4c8c`) on this host while other users' GPU jobs were running (load average
12–55). Scope: `LTX-2/scripts/onestep_avatar/tests/` (2,320 tests in 100 files,
27.6k lines) and the run guide's check recipe
(`scripts/onestep_avatar/configs/README.md` §10).

## Implementation result — 2026-10-12

The implementation is in the LTX-2 worktree. No commit or workspace submodule
pin update was made. The full evidence is in
[`expr/onestep_avatar/test_suite_20261012/README.md`](../../expr/onestep_avatar/test_suite_20261012/README.md).

1. CPU defaults: set OMP/MKL defaults before Torch, hide CUDA unless opted in,
   and set Torch to one thread. The 225-case comparison was 11.09/11.08 seconds.
2. T0–T3: registered markers, area/direct-import selector and unit tests,
   shared-helper escalation, dispatch/import boundary escalation, and slow-test
   reporting. The run guide and package guidance use the tiers. T2 selects
   touched-file Ruff as well as avatar and prune tests.
3. Children: batch help and dry-run rows; use one warm third-party parent and
   clean forks for all ten profiles. Fresh-child equality covers ten inventories,
   thirteen help rows and six dry runs. Fresh import-light children remain where
   preloading Torch would contradict an assertion. Two owners no longer import
   the evaluator merely to exercise the execution protocol.
4. Expensive inputs: session source-stat/version caching, inline executors for
   cache/publication checks, real spawn coverage, and build-once tamper matrices
   with original-path restoration. Eight named refusal fault controls each
   fail exactly their matching subtest. Ten selector fault controls all fail
   within T1 selections.
5. Redundancy: eight non-queued training cases cover every level pair, two
   queued cases retain the original refusal, and E4 keeps four replay cases
   plus one refusal. Removed-attribute checks are gone while parser refusals
   remain. Ordinary profile exclusion has one table-driven owner.
6. Helpers and isolation: model, contracts, corpus, training, saved-result and
   import helpers live under non-collected `tests/support/`. Test modules no
   longer supply library builders to other tests. Unique per-test tokens reach
   helpers; the embedded child probe also generates its own UUID. The old
   claims/TTL implementation was removed. Current lifecycle tests use the
   production ProcessRegistry; historical ledger-free rows use a thin adapter
   over that same registry. Supervision passed 20 repeats with unchanged limits.

Final gates: **2,104 avatar tests and 198 subtests passed in 178.56 s**;
**219 prune tests passed, 1 skipped in 3.77 s**; **95 blocked boundary tests
and 24 subtests passed in 36.90 s**. This task's touched Python files pass Ruff, including the pruning experiment
tests (67 passed after annotation-only lint repairs).
The baseline/current collection census accounts for every changed file, including
two independently added queue CLI tests. The measured full-suite runtime is
above the original 100–120 s estimate; no deadline or assertion was relaxed to
meet the estimate.

Held exactly as the proposal requires: the 28 report-reader tests were not
migrated without explicit approval, and pytest-xdist/coverage were not installed.
Concurrent queue, pruning and report work was preserved.

## Recommendation

The suite is not slow because it has too many tests. The tests are mostly
distinct and cheap. 495 of the 1,159 tests with a measurable time take under
50 ms, and 164 tests (7% of the suite) use 80% of the time. The cost has four
causes, and all four can be fixed without removing checks:

1. **Torch runs tiny CPU models with 48 threads.** Nothing pins the thread
   count, so a 2×2-latent, 2-layer model pays OpenMP synchronization on every
   operation. The same 225 in-process tests take **25.0 s with 1 thread, 24.3 s
   with 4 threads and 209.4 s with the default 48**. The default-thread
   full suite used about 37 cores. It had finished only 300 tests after 6 minutes
   when I stopped it.
2. **About 32 tests start a Python child that imports the torch/peft stack**,
   at about 5.3 s each (`import peft` alone costs 5.3 s). They take about
   170 s, 44% of the 385 s single-thread run.
3. **Expensive producers are rebuilt for every tamper case.** These are
   `software.capture` (4,510 calls, 51 s), rendering, `spawn` process pools and
   real owner processes. Each parametrized "change one field" case rebuilds the
   same artifacts.
4. **Every change runs everything twice.** §10 runs the full suite. It then runs
   the 1,565 ordinary tests again with experiment imports blocked (about 275 s
   more at 1 thread). Agents run both after each behavior change. The handoff
   records repeated full runs within single tasks, such as ten consecutive
   cancellation runs and an interrupted full run that was then repeated.

Fixing (1) is one conftest edit and gives the largest win. Fixing (2)–(3) brings
the full single-thread suite from 385 s to an estimated 100–120 s on this host.
A tiered policy (4) then means that the full suite runs once per task, not once
per edit. The full suite stays the final gate. Partial runs make iteration fast;
they do not replace that gate.

## Measurements

| Run | Result |
|---|---|
| Collection | 2,320 tests, 100 files (71 ordinary: 1,565 tests; 29 experiment: 755 tests), 4.4 s |
| Full suite, default threads (current recipe) | Stopped after 300 tests in 6 min; process at ~3,700% CPU |
| Full suite, `OMP_NUM_THREADS=1` | 2,320 passed in 385.2 s; repeated with an operation census: 384.7 s |
| Same 225 in-process model tests | 1 thread 25.0 s; 4 threads 24.3 s; default 209.4 s |
| Experiments-blocked rerun (§10, second command) | about 275 s more at 1 thread (sum of the ordinary files) |
| Prune suite (§10, third command) | 219 passed, 1 skipped, 4.1 s |
| Ruff (§10, fourth command) | < 1 s, but **currently reports 292 findings** (8 in avatar files from `997807e`, the rest in `scripts/prune`), so it is not a usable pass/fail gate today |

Historical data point: the active handoff records "2,293 tests in 517 seconds"
for one earlier full run.

Where the 385 s goes (categories overlap slightly):

| Cause | Tests / calls | Time |
|---|---|---|
| Python children importing the torch/peft stack (`--help`, dry-run, import-closure, profile and two-owner checks) | ~32 tests | ~170 s |
| `spawn` process pools in corpus planning/survey (`test_precompute`, `test_subset`) | 14 pools | ~35 s |
| Repeated `software.capture` (hashes ~170 source files plus 17 `importlib.metadata` lookups per call; `runtime_versions` alone is 24 s) | 4,510 calls | ~51 s |
| Producer or render rebuilt for each tamper case | ~200 cases | ~60 s |
| Real owner processes in `test_guarded_recovery` (one per case) | 21 children | ~23 s |
| Everything else | ~2,050 tests | ~50 s |

Method: `--durations=0` on the full single-thread run, cProfile on two slow
files, and a census plugin that counted calls/time of `software.capture`,
`media.render_panels`, `media.save_render`, `LTXModel()`, `subprocess.Popen` and
`ProcessPoolExecutor`. The scripts and raw outputs were in the session
scratchpad and are not part of the repository. Tiny-model construction is not a
cost: 638 `LTXModel()` builds took 1.5 s.

## Answers to the three questions

### 1. Is each test necessary?

Almost all of them are. Most files check one owner's contract: refusal branches,
c0/first-frame invariants, provenance binding, process ownership. One case per
branch is the package's verification style (`doc/verification.md`). About 125
test IDs (5%) are redundant or in the wrong place:

- `test_training_runtime.py::test_one_bounded_update_saves_real_zero_and_updated_lora`
  is the full 2×2×2×2×3×2 product (96 cases). The 48 `queued=True` cases all
  stop at the same "original dispatch launch evidence" refusal before any model
  work, so 2 cases (one per mode) are enough. The other 48 need a covering set
  of about 10 named cases. Every level and every pair of levels of mode,
  preview failure, update evidence, resource check and consumer trace must
  still appear once.
- `experiments/test_training_update_runtime.py` (16 cases) reruns
  `run_bounded_update` and then does the replay only when
  `resource_check == "pass" and not preview_failure` (4 cases). The other 12
  only repeat one refusal. Keep the 4 replay cases and 1 refusal case.
- Duplicate assertions:
  `not hasattr(continuation_check, "supervised_run")` appears in both
  `test_checker_entrypoints.py` and `test_checker_queue_protocol.py`.
  "Ordinary software profiles contain no experiment owner" is asserted in five
  places: `test_profile_coverage`, `test_execution_ownership`,
  `test_checker_entrypoints`, `test_short_causal_check` and
  `test_capacity_review`. Keep one table-driven test in `test_software.py`, and
  keep each checker's own `EXTRA_SOURCES` binding test.
- "Removed attribute stays removed" checks (`assert not hasattr(...)`) protect
  nothing a parser test does not already protect. Keep the parser-rejection
  tests (for example `--supervise`) and drop the `hasattr` lines.
- `test_report_saved_media.py` (28 tests) loads
  `expr/onestep_avatar/dev_training_20261001/code/stage2_update.py`. That
  file is git-ignored workspace report code. By the package's own boundary,
  `expr/` code is checked by its report validator, not by package tests. This
  test also fails on a clean checkout. Move it to that study's validator.
  **This needs user approval because it removes package tests.**

The per-file verdicts are in the appendix.

### 2. Can we run partial tests for some modifications?

Yes. The package is split into areas whose tests are cheap. Measured times are
at 1 thread, before any of the speed fixes below:

| Area | Source files | Tests | Time now |
|---|---|---|---|
| model | `model/*` | 178 | 3 s |
| corpus | `corpus/*` | 120 | 36 s (about 35 s of it is `spawn` pools) |
| training | `training/*`, `train.py`, `plot_training.py` | 452 | 27 s |
| evaluation/product | `evaluate.py`, `infer.py`, `previews.py`, `prepare_inputs.py`, `bench.py`, `metrics.py` | 294 | 26 s |
| media | `media.py`, `comparisons.py`, `decode_saved.py` | 151 | 17 s |
| execution | `execution/*` | 411 | 58 s |
| one experiment | `experiments/<name>.py` | 4–124 | 0–15 s |
| boundary | import edges, software profiles, entry CLIs | 98 | 151 s (→ ~25 s after phase 2) |
| prune suite | avatar `model/*`, `corpus/dataset.py`, `corpus/subset.py`, `media.py` are imported by `scripts/prune` | 220 | 4 s |

A static, transitive import closure does not help. Every module under
`model/`, `training/` and `execution/`, plus `corpus/dataset.py`,
`corpus/subset.py`, `evaluate.py` and `media.py`, is in the closure of 68–85 of
the 100 test files. The reason is that `evaluate` and `comparisons` import
almost everything. Selection therefore uses an area table
plus the test files that directly import a changed module. The full suite stays
as the once-per-task gate that catches the transitive cases. Tiers:

| Tier | When | What | Target time |
|---|---|---|---|
| T0 inner loop | after each edit | selected files, `-m "not subprocess"` | 2–30 s |
| T1 area check | before leaving an area | selected files including child-process tests; plus the prune suite when a prune-imported module changed | ≤ 60 s |
| T2 final gate | once, before reporting done or committing | full suite + prune suite + Ruff on touched files | ≤ 2 min after phases 1–3 |
| T3 boundary | only when import statements, `execution/jobs.py`, `execution/completion.py`, `execution/software.py` profile lists, `tests/blocked_imports/` or conftest's blocker change | `-m boundary` with experiments blocked, not the whole ordinary suite | ~25 s |

Docs-only changes need no pytest. They need the link, symbol and diagram checks
in the package `CLAUDE.md`.

Rules that make this mechanical for agents:

- One command selects the tests, prints them and can run them:
  `"$PY" -m scripts.onestep_avatar.tests.select [--base HEAD] [--tier T0|T1|T2|T3] [--run]`.
  It reads `git diff --name-only <base>` plus untracked files under the
  package. It maps production files through the area table and adds test files
  that import them directly (an AST scan at run time, so it has no stale map).
  It adds changed test files. It escalates to T2 for `conftest.py`,
  `tests/support/*`, `hashing.py`, `__init__.py` or `execution/software.py`. It
  adds T3 when a diff line adds or removes an import or touches the dispatch
  files above.
- After a failure, fix it and rerun with `--lf` (last failed) and the T0 files,
  not the full suite. Run T2 once at the end.
- Do not rerun T2 after a docs-only or comment-only follow-up edit. Rerun only
  the files that the edit touched.
- Write T2 output to a log and quote its summary line in the handoff. Do not
  stream 2,000 dots into the conversation.

### 3. Duplicate tests and duplicate expensive operations

Each item has its evidence and its fix. Times are at 1 thread.

1. **The experiments-blocked full rerun** repeats 1,565 ordinary tests (~275 s)
   to prove one property: no ordinary path imports `experiments`. Static
   imports are already checked by AST in
   `test_production_imports_keep_experiments_out_of_ordinary_owners`. The only
   dynamic import sites are `execution/jobs.py` and `execution/completion.py`,
   keyed by `job["experiment"]`. They are covered by
   `test_ordinary_job_never_looks_up_or_imports_experiments` and the
   blocked-child entry tests. Replace the routine rerun with the T3 `boundary`
   marker set, run under the blocker. Keep the full blocked rerun as an
   occasional audit, for example before a release.
2. **Child processes that import the same stack**, in four files with four
   hand-written environments:
   - `--help` children: 5 in `test_checker_entrypoints`, 6 in
     `test_experiment_import_boundary` and 2 in `test_output_owner_split`.
     That is ~70 s.
   - Dry-run children: 2 typed plus 4 support in
     `test_experiment_import_boundary`. They share 90% of their tokens. That
     is ~31 s.
   - Import-closure children: `test_numerics`, `test_short_causal_check`,
     `test_corpus_model_boundaries`, `test_execution_ownership`, `test_roots`
     and `test_output_owner_split`.
   - Ten profile children in `test_profile_coverage` (52 s).
   Fix:
   - Use one child per environment for the help and dry-run checks. It
     `runpy.run_module`s every CLI with patched `argv` and prints one JSON row
     per CLI. The parent reports each row as a subtest. "No experiment module
     loaded" stays valid as a union check, which is stricter.
   - Use one warm parent for the profile and import-closure checks. It imports
     only third-party modules (torch, peft, transformers, ltx_core,
     safetensors, PIL), never `scripts.*`. It then `os.fork()`s once per
     profile. Each fork still sees exactly the `scripts.onestep_avatar.*`
     modules its entry path loads.
   - Put the helpers in `tests/support/children.py`.
   - Expected: ~170 s → ~25 s.
3. **`spawn` pools for one work item.** `corpus/precompute.enumerate_capture_jobs`
   and the `corpus/subset` survey create a `spawn` `ProcessPoolExecutor` even
   with `max_workers=1`. Each worker re-imports the stack. Three plan-cache tests
   take 25 s and four subset CLI tests take ~10 s.
   - Test-only fix: these tests check cache and publication logic, not pooling,
     so give them an inline executor.
   - Keep one test per module on the real `spawn` pool, so the pickling
     contract stays covered.
   - The production alternative (run inline when one worker is requested)
     changes behavior. It needs its design-doc update and is not required.
4. **`software.capture` recomputed 4,510 times.** It re-hashes the same
   unchanged ~170 source files and re-reads 17 package versions on every call.
   Production calls it only a few times per run, so do not change production.
   - Add an autouse, session-scoped conftest fixture.
   - It caches `software.runtime_versions()`.
   - It also caches `software.sha256` for `.py` files under `LTX_ROOT`, keyed by
     `(path, st_ino, st_size, st_mtime_ns)`.
   - Tests that monkeypatch `software.sha256` or `runtime_versions` to simulate
     a change still replace the cached attribute, so they keep working.
   - Expected: ~51 s → ~5 s.
5. **The same producer reruns for every tamper case.** A fixture builds a
   complete saved result, then each parametrized case changes one field. These
   are the costly ones:
   - `test_conditioning_check` `decoded_saved`: 7 cases, 12 s, about 1.7 s
     per case (`media.render_panels` is about 0.6 s per call).
   - `test_sigma_sweep_results`: 10 tests, 14 s; each fixture build makes
     eight completed evaluations (about 1.1 s).
   - `test_ordinary_queue_completion`: benchmark 18, preview 6 and product 14
     cases, ~11 s.
   - `test_queue_render`: 25 cases, 5.7 s.
   - `test_saved_comparisons` and `test_saved_comparison_references`: 15 + 15
     cases, ~5 s.
   - `test_evaluation_completion`: 20 + 3 + 4 cases. Each case runs a full CPU
     evaluation.
   - `test_guarded_recovery`: 11 + 4 + 3 cases. Each case starts a real owner
     process, ~21 s.

   The saved records pin absolute paths, so copying a pristine tree does not
   work. Fix: build the artifacts once inside one test function, then loop over
   the same case list with pytest 9's built-in `subtests` fixture (it works in
   this environment). Restore the original file bytes after each case.
   `tests/support/tree.py` gives `snapshot(root)` / `restore(root, snapshot)`,
   which also removes files a case created. The case lists, the expected
   error patterns and the refusal messages do not change. Expected: ~60 s →
   ~12 s.
6. **Repeated tiny rollouts.**
   - `experiments/test_continuation_check.captured()` runs two full causal
     samples, 16 model calls each, in eight test cases.
   - `test_checker_queue_protocol.completed_continuation` reruns them for 10
     cases.
   - These are cheap once threads are pinned (≈1.3 s and ≈1.6 s per file), so
     they are low priority. Cache them only if they grow.
7. **Two owner processes import `evaluate`.**
   `test_queue_g8.py::test_two_cpu_queue_owners_share_disjoint_four_device_jobs`
   takes 10.8 s because each owner script imports `scripts.onestep_avatar.evaluate`
   (peft). Patch what it needs through `execution` only, or fork from the warm
   parent of item 2. Expected: ~1–2 s.
8. **Duplicated fixture code (maintenance, not runtime).**
   - Corpus-membership builders are copied in at least 8 places:
     - `conftest.current_membership`
     - `test_training_helpers.two_videos`
     - `test_training_config._membership`
     - two copies inside `test_training_runtime`
     - `test_media.reference_cli_inputs`
     - `test_evaluate_coverage.causal_coverage`
     - `experiments/test_adapter_effect_check.evaluation_membership`
     - `experiments/test_training_slice_check.original_artifacts`
   - AST import walkers are written four times (`test_corpus_model_boundaries`
     and three in `test_experiment_import_boundary`).
   - A membership-schema change today needs eight edits. Move them to
     `tests/support/corpus.py` and `tests/support/imports.py`.
9. **Test modules imported as libraries.** About 60 import lines pull helpers
   out of other test modules:
   - `test_causal_core._model` is used by 20 files.
   - `test_checkpoint_contract._contract/A/B` is used by 8 files.
   - Two experiment tests import `scripts.prune.tests.test_whole_clip_baseline`.

   `tests/` has no `__init__.py`, so pytest imports `test_causal_core` both as
   `test_causal_core` and as `scripts.onestep_avatar.tests.test_causal_core`;
   I confirmed this by checking `sys.modules`. These imports also make selection
   imprecise, because editing a test file silently affects 20 others. Move the
   helpers to `tests/support/` (non-collected modules). Wrapped fixtures such as
   `create_saved_effect.__wrapped__(...)` become plain builder functions.
10. **A second claims implementation.** `tests/claims_fixture.py` (164 lines) is
    a file-claim/TTL version of the `ProcessRegistry` interface. Five queue test
    files use it to drive `queue.run_child`. So those tests exercise a claims
    implementation that production does not use.
    `ProcessRegistry(tmp_path / "ledger.json", inventory=...)` already works on
    a temporary ledger (`test_process_registry.py`). Replace the fake wherever a
    test does not depend on TTL behavior, then delete it.

## Parallel-safety and determinism issues found

- **Fixed tokens.** `execution/processes.live_attempt_processes` finds workers by
  `TOKEN_ENV=<token>` in `/proc/<pid>/environ`. Tests use the literal `"b" * 32`
  in 47 places. Two concurrent runs of the suite would see each other's live
  children. This includes two agents on this host, or a future `pytest-xdist`
  run. Give each test a unique token (a fixture returning `uuid4().hex`), and
  do this before any parallel execution.
- **CUDA visibility is the caller's job.** The handoff records a run that failed
  CPU-only assertions because CUDA was visible, and 16 assertions depend on it.
  Conftest should force `CUDA_VISIBLE_DEVICES=""` before any CUDA
  initialization, unless a test explicitly opts in.
- **Very short deadlines.** `test_supervision.py` supervises real children with
  0.15 s startup/phase deadlines. At load average 55 this is a flake risk.
  Measure the flake rate with 20 repeated runs under load before changing any
  value. The deadlines test production bounds, so keep their meaning.

## Implementation phases

Each phase keeps every remaining case's inputs, expected errors and tolerances.
Each phase ends with a T2 run whose pass count changes only by the IDs that the
phase explicitly removes or folds. Do not edit scientific code, budgets or
tolerances in these phases.

### Phase 0 — pin the runtime (one conftest edit, largest win)

- At the top of `tests/conftest.py`, before `import torch`:
  - `os.environ.setdefault` for `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`;
  - force `CUDA_VISIBLE_DEVICES=""` unless `ONESTEP_AVATAR_TEST_CUDA=1`.
- After the import, call `torch.set_num_threads(1)`.
- Children inherit these through `os.environ`.
- Add one test asserting `torch.get_num_threads() == 1` and no visible CUDA
  device.
- Update the §10 recipe so the env prefix is no longer required (keep it
  harmless).
- Verify: T2 passes. A rerun of the 225-test A/B set at default shell settings
  is within 20% of the 1-thread time.

### Phase 1 — markers, selector, tiers (no test logic change)

- Register markers in conftest (`config.addinivalue_line`):
  - `subprocess`: starts a Python child that imports the model stack;
  - `boundary`: import, profile and entry-CLI architecture checks;
  - `slow`: more than 1 s after phases 2–3.
- Mark the ~32 child tests and the 8 boundary files.
- Add `tests/select.py` as described in answer 2, with its own unit tests for
  the mapping rules.
- Selector acceptance:
  - For 10 representative modules (`model/common.py`, `model/causal.py`,
    `corpus/subset.py`, `training/engine.py`, `evaluate.py`, `media.py`,
    `execution/supervision.py`, `execution/jobs.py`, one experiment and
    `hashing.py`), inject a fault into a function that the tests call.
  - The T1 selection must contain at least one failing test.
  - Record the results.
- Add a conftest session-end report of tests over 1 s that are not marked
  `slow` or `subprocess`. This stops quiet regressions.
- Docs:
  - Rewrite `configs/README.md` §10 as the tier table plus the selector
    command. That file owns runnable recipes.
  - Update the package `CLAUDE.md` "Before calling a change done" and
    "Verification matches the change" sections to name T0–T3.

### Phase 2 — consolidate child processes (~170 s → ~25 s)

- Add `tests/support/children.py` with:
  - `normal_environment()` and `blocked_environment()`;
  - `run_help_rows(modules, env)`;
  - `run_dry_run_rows(cases, env)`;
  - `fork_profiles(profiles)`.
- Port the checks in answer 3, item 2, and the two-owner test of item 7.
- Equivalence check before deleting the old tests:
  - Run old and new side by side once.
  - For each profile, the loaded `scripts.onestep_avatar.*` set from a fork
    must equal the set from today's fresh child.
  - Each help/dry-run row must have the same return code and the same "usage:"
    result.
  - Save the comparison in the plan record.

### Phase 3 — share expensive results (~145 s → ~20 s)

- Add the session cache for `software.capture` inputs (item 4). First grep
  that no test writes a `.py` file under `LTX_ROOT`.
- Use an inline executor in the plan-cache and subset CLI tests (item 3). Keep
  one real `spawn` test in each of `test_precompute.py` and `test_subset.py`.
- Convert the tamper matrices of item 5 to build-once plus `subtests` with
  `snapshot`/`restore`.
- Order: `test_guarded_recovery`, `test_conditioning_check`,
  `test_sigma_sweep_results`, `test_ordinary_queue_completion`,
  `test_queue_render`, `test_saved_comparisons`,
  `test_saved_comparison_references`, `test_evaluation_completion`.
- After each file:
  - The subtest count must equal the old parametrized count.
  - Temporarily break one verifier clause per file and confirm that exactly
    the matching subtest fails.

### Phase 4 — remove redundancy (~125 IDs, ~12 s)

- Apply answer 1: the training-runtime covering set, the E4 replay reduction,
  the duplicated `hasattr` checks and the five profile-exclusion assertions.
- For the covering set, add a small assertion in the test module that every
  level pair appears in the selected case table.
- Move `test_report_saved_media.py` to the study validator only with user
  approval.

### Phase 5 — helper hygiene and parallel safety

- Move the shared helpers into `tests/support/`: the membership builders, AST
  walkers, `_model/_grid/_geometry/_context`, `_contract/A/B`, `CPUAccelerator`,
  `run_bounded_update` and `GPUClaims` users.
- Add the unique-token fixture.
- Replace `claims_fixture.GPUClaims` where possible (item 10).
- Optional, **needs user approval to change the `ltx` env**: install
  `pytest-xdist`. Then run T2 with `-n 8 --dist loadgroup`. Process-lifecycle
  files share one `xdist_group`. Measure the speedup; expect 4–6× on top of
  phases 0–3.
- Optional, same approval: install `coverage`. Compare line coverage of the
  production package before phase 2 and after phase 5. Expect no lost lines
  outside deleted tests.

### Out of scope here

- Fixing the 292 existing Ruff findings. Until someone does, agents should run
  Ruff on touched files and report pre-existing findings separately.
- Changing production timeouts, budgets or verification rules.
- Updating the workspace's recorded LTX-2 submodule pin.

## Expected end state

| Measure | Now (1 thread) | After phases 0–4 |
|---|---|---|
| Full suite (T2) | 385 s (default threads: much longer) | ~100–120 s; ~25 s with optional xdist |
| Gate per behavior change | full + blocked + prune ≈ 665 s, often repeated | T0/T1 per edit (2–60 s); T2 once (~2 min); T3 only for import/dispatch changes (~25 s) |
| Test IDs | 2,320 | ~2,195 (subtests carry the same cases) |

## Appendix — per-file verdicts

Time is the measured single-thread total. "Keep" means that every case checks a
distinct branch. Phase numbers refer to the fixes above.

### Model and numerics

| File | Tests | Time | Purpose | Verdict |
|---|---|---|---|---|
| test_causal_core.py | 43 | 0.5 s | cache/rollout equals block-causal full sequence on a real 2-layer LTXModel; plan, eviction, schedules | Keep; move helpers out (P5) |
| test_causal_train_sample.py | 19 | 0.0 s | causal training sample: c0, D0/D1 source, teacher forcing, loss scale | Keep (invariant 0 owner) |
| test_bidirectional.py | 14 | 0.2 s | prediction/loss/gradient vs original; c0 | Keep (invariant 0 owner) |
| test_sampling.py | 5 | 0.0 s | Euler operation order, sigma grid | Keep |
| test_prompt_and_whole_clip.py | 36 | 0.0 s | prompt flags, guided passes, 25 rescaled-schedule shapes | Keep |
| test_adapters.py | 26 | 0.7 s | unmerged/fused loading equals the training function | Keep |
| test_adapter_only_export.py | 2 | 0.0 s | PEFT adapter payload export | Keep |
| test_consumer_trace.py | 24 | 1.4 s | training consumer trace and hooks | Keep |
| test_update_state.py | 12 | 0.0 s | Adam-moment export | Keep |
| test_numerics.py | 20 | 5.6 s | deterministic numerics policy | Keep; child check → P2 |
| test_applied_runtime.py | 16 | 0.0 s | rank agreement of applied runtime | Keep |
| test_bench.py | 9 | 0.1 s | benchmark CLI and counts | Keep |

### Corpus

| File | Tests | Time | Purpose | Verdict |
|---|---|---|---|---|
| test_dataset.py | 4 | 0.0 s | atomic writes | Keep |
| test_geometry.py | 10 | 0.0 s | canonical crop box | Keep |
| test_mask_video.py | 9 | 0.1 s | lossless mask codec (real ffmpeg) | Keep |
| test_motion.py | 19 | 0.0 s | motion conversion, gap filling | Keep |
| test_qa.py | 5 | 0.0 s | mask IoU | Keep |
| test_hashing.py | 2 | 0.0 s | file digest | Keep |
| test_build_guidance.py | 8 | 0.1 s | guide compositing and render completeness | Keep |
| test_precompute.py | 30 | 25.2 s | capture planning, manifests, CLI | Keep; inline executor in 3 tests (P3) |
| test_subset.py | 33 | 10.3 s | membership survey, grouping, publication | Keep; inline executor in CLI tests (P3) |

### Training

| File | Tests | Time | Purpose | Verdict |
|---|---|---|---|---|
| test_training_config.py | 22 | 0.0 s | settings and frame plans | Keep |
| test_training_helpers.py | 41 | 0.3 s | preflight refusals, noise/sigma draws, exports | Keep |
| test_training_preflight.py | 12 | 0.0 s | VAE alias and input preflight | Keep |
| test_training_runtime.py | 102 | 9.6 s | real tiny bounded updates, FSDP precision, rank archive | Keep; 96-case product → ~12 (P4) |
| test_training_completion.py | 44 | 4.2 s | checkpoint completion binding | Keep; capture cache (P3) |
| test_training_launch_binding.py | 11 | 0.2 s | queued launch binding | Keep |
| test_checkpoint_contract.py | 51 | 0.0 s | adapter contract checker (V7) | Keep; helpers → support (P5) |
| test_resources.py | 21 | 0.0 s | budget phases and evidence | Keep |
| test_plot_training.py | 8 | 0.3 s | training plots | Keep |
| test_required_inputs.py | 4 | 0.0 s | explicit-input CLIs | Keep |

### Evaluation, product, previews

| File | Tests | Time | Purpose | Verdict |
|---|---|---|---|---|
| test_evaluate.py | 53 | 0.4 s | evaluation CLI, guidance, c0, call counts | Keep |
| test_evaluate_coverage.py | 27 | 0.7 s | output coverage options | Keep |
| test_evaluation_completion.py | 56 | 8.4 s | evaluation queue completion | Keep; build once + subtests (P3) |
| test_infer.py | 13 | 0.2 s | product preflight | Keep |
| test_product_coverage.py | 7 | 0.0 s | product coverage | Keep |
| test_prepare_inputs.py | 15 | 0.4 s | supplied-image preparation | Keep |
| test_prepare_preview.py | 14 | 0.7 s | preview inputs | Keep |
| test_previews.py | 19 | 1.2 s | preview queueing and completion | Keep |
| test_lpips_frame_scores.py | 17 | 0.0 s | LPIPS frame scores (fake model) | Keep |
| test_rgb_measurements.py | 8 | 0.0 s | RGB/subject measurements | Keep |
| test_ordinary_queue_completion.py | 56 | 13.8 s | bench/preview/product completion tampering | Keep; build once + subtests (P3) |

### Media and comparisons

| File | Tests | Time | Purpose | Verdict |
|---|---|---|---|---|
| test_media.py | 36 | 0.4 s | panels, decode identity, reference CLI | Keep |
| test_saved_comparisons.py | 27 | 3.9 s | saved comparison rendering and completion | Keep; build once (P3) |
| test_saved_comparison_references.py | 22 | 3.3 s | RGB references | Keep; build once (P3) |
| test_checkpoint_visualization.py | 12 | 0.1 s | checkpoint groups and official tail | Keep |
| test_decode_saved.py | 1 | 0.1 s | decode reuse | Keep |
| test_report_saved_media.py | 28 | 0.0 s | reader in git-ignored `expr/` report code | Move to the study validator (approval) |

### Execution and queue

| File | Tests | Time | Purpose | Verdict |
|---|---|---|---|---|
| test_queue.py | 37 | 0.8 s | journal, receipts, identities | Keep; unique tokens (P5) |
| test_queue_cli.py | 9 | 0.0 s | CLI state paths | Keep |
| test_queue_device_pool.py | 13 | 0.0 s | device pools | Keep |
| test_queue_foreground.py | 8 | 0.4 s | foreground stop and log streaming (real children) | Keep |
| test_queue_g8.py | 17 | 11.2 s | pool, budgets, clocks, two real owners | Keep; light owner children (P2) |
| test_queue_inventory.py | 17 | 0.0 s | claim release | Keep |
| test_queue_launch.py | 118 | 6.8 s | registration/grant protocol, recovery refusal matrices | Keep; unique tokens (P5) |
| test_queue_loop.py | 21 | 0.1 s | loop and recovery CLI | Keep |
| test_queue_missing_supervision_recovery.py | 21 | 0.2 s | missing-supervisor recovery | Keep |
| test_queue_proc_races.py | 7 | 0.1 s | /proc races | Keep |
| test_queue_render.py | 34 | 7.9 s | render job completion | Keep; build once (P3) |
| test_queue_startup.py | 38 | 2.6 s | startup retries | Keep |
| test_queue_zombie.py | 6 | 0.9 s | unreaped children | Keep |
| test_guarded_recovery.py | 20 | 23.2 s | owners killed at launch boundaries | Keep; one owner per phase (P3) |
| test_process_registry.py | 12 | 0.3 s | shared ledger | Keep |
| test_supervision.py | 29 | 3.5 s | real-child supervision | Keep; measure the flake rate under load |
| claims_fixture.py | — | — | second claims implementation | Replace with `ProcessRegistry` (P5) |

### Boundary and architecture

| File | Tests | Time | Purpose | Verdict |
|---|---|---|---|---|
| test_experiment_import_boundary.py | 16 | 57.4 s | AST and blocked-child import boundary | Keep the AST tests; 13 children → 2 (P2) |
| test_profile_coverage.py | 10 | 52.2 s | software profile covers loaded owners | Keep; fork per profile, `boundary` (P2) |
| test_execution_ownership.py | 4 | 0.1 s | execution owners and profile pins | Keep; drop duplicated assertion (P4) |
| test_corpus_model_boundaries.py | 4 | 1.3 s | corpus/model import rows | Keep; shared walker (P5) |
| test_roots.py | 20 | 2.5 s | canonical repository roots | Keep; `boundary` |
| test_output_owner_split.py | 8 | 10.4 s | owner moves, help, tensor hash | Keep; help → consolidated child (P2) |
| test_software.py | 23 | 0.5 s | software manifest | Keep; owner of the profile-exclusion table (P4) |

### Experiments (755 tests, 99 s)

Every selector is still in `execution/jobs.EXPERIMENTS` and in the run guide,
so all of them stay. They run in T1 only when their experiment, or a shared
owner they call, changes.

| File | Tests | Time | Verdict |
|---|---|---|---|
| test_adapter_effect_check.py | 124 | 3.5 s | Keep; builders → support |
| test_capacity_review.py | 26 | 2.4 s | Keep; drop the profile duplicate (P4) |
| test_causality_diagnostic.py | 5 | 0.3 s | Keep |
| test_checker_entrypoints.py | 13 | 26.7 s | Help → consolidated child (P2); drop `hasattr` (P4) |
| test_checker_queue_protocol.py | 46 | 1.6 s | Keep; drop duplicated `hasattr`; no cross-test fixture imports |
| test_conditioning_check.py | 43 | 14.6 s | Keep; build `decoded_saved` once (P3) |
| test_continuation_check.py | 24 | 1.3 s | Keep |
| test_continuation_identity_check.py | 25 | 2.9 s | Keep |
| test_experiment_queue.py | 35 | 0.0 s | Keep |
| test_experiment_required_inputs.py | 3 | 0.0 s | Keep |
| test_extracted_queue_protocols.py | 23 | 3.5 s | Keep |
| test_fusion_diagnostic.py | 4 | 0.0 s | Keep |
| test_future_noise.py | 21 | 2.2 s | Keep |
| test_past_influence_check.py | 49 | 2.8 s | Keep |
| test_prepare_continuation.py | 8 | 0.0 s | Keep |
| test_prune_baseline.py | 45 | 1.9 s | Keep; fix its Ruff findings; prune-test helpers → support |
| test_prune_reproduction.py | 22 | 3.4 s | Keep; same |
| test_reference_inputs.py | 15 | 0.0 s | Keep |
| test_saved_metrics.py | 4 | 0.0 s | Keep |
| test_short_causal_check.py | 34 | 6.6 s | Keep; child check → P2 |
| test_sigma_sweep.py | 24 | 3.9 s | Keep (229 `ffprobe` calls; cheap) |
| test_sigma_sweep_metrics.py | 13 | 0.0 s | Keep |
| test_sigma_sweep_results.py | 10 | 14.1 s | Keep; build the eight receipts once (P3) |
| test_stock_parity.py | 17 | 0.4 s | Keep |
| test_training_checker_queue_protocols.py | 26 | 0.4 s | Keep |
| test_training_slice_check.py | 26 | 0.6 s | Keep |
| test_training_update_check.py | 6 | 0.1 s | Keep |
| test_training_update_launch_binding.py | 48 | 0.7 s | Keep |
| test_training_update_runtime.py | 16 | 4.9 s | Reduce to 4 replay cases + 1 refusal (P4) |
