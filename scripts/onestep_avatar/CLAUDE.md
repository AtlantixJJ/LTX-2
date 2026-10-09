# CLAUDE.md — `LTX-2/scripts/onestep_avatar/`

**Current GPU dispatch policy — user amendment, 2026-10-07:** query
`nvidia-smi` directly and use one shared JSON file to record only processes this
pipeline starts (PID, start ticks, command, GPU IDs and owned descendants).
New launches do not consult reservation files or unrelated process environments.
No privileged access is required. Preserve original claim,
launch, result and acceptance files unchanged. Scientific inputs, budgets,
tolerances and native E1–E5 gates remain unchanged.

**Concurrent runs — user amendment, 2026-10-08:** independent runs may execute
at the same time on GPUs 0–3. Query direct occupancy before each launch and
register each original owner in the shared process ledger. Keep the prescribed
four-rank training topology; concurrent ordinary checks do not change it.

Guidance for Claude Code when working inside this package: the one-step LTX-2.5 avatar
renderer, corpus tooling and model training in **one tree**.

**Implementation authorization (2026-10-05):** the user directed execution of the revised
two-mode plan. The review boundary is passed. Read `doc/README.md` for the current
implemented/proposed mapping. Model execution and launchers belong in this package;
`expr/` code only generates reports from saved results.
The user-directed 2026-10-07 [architecture amendment](doc/architecture.md)
defines the internal boundary and permits a proposed `experiments/` subpackage.
The latest October 8 user direction requires code refactoring first: move
experiment owners, retire duplicate execution after caller/data/code checks,
and validate the final layout. Run fresh affected native checks and the unchanged
E5 pilots afterward. Missing learning or longer-video evidence does not block
reversible source moves. Scientific acceptance requirements remain unchanged.
The current task is the handoff for the next agent; do not describe proposed
paths as shipped.

**Refactor design decisions — user, 2026-10-08:** the
[architecture contract](doc/architecture.md#decisions-recorded-on-2026-10-08)
is the target. It adds proposed `corpus/`, `execution/` and `experiments/`
subpackages, splits ordinary `evaluate.py` into `evaluate.py`, `metrics.py`,
`previews.py` and `comparisons.py`, keeps every historical study as an
experiment module, replaces the `sigma_sweep` queue kind with one `experiment`
kind, moves the whole-clip pruning producer to `scripts/prune/`, and limits
`expr/` cleanup to retiring executors and freezing old reports. The queue
becomes the only launcher for model work, with bounded supervision and an
overall deadline for every kind and one GPU pool (GPUs 0–3); do not write
hand-written launch controllers. The refactoring agent commits one verified
owner group at a time on LTX-2 branch `onestep-avatar-refactor`; it never
pushes and never updates the workspace's recorded LTX-2 commit. The handoff's
owner groups G0–G14 order the work.

## Reading order — binding

Before editing anything here, in this order:

1. [`doc/architecture.md`](doc/architecture.md) — common versus experiment owners, allowed
   dependencies, proposed destinations, code checks and post-refactor native acceptance.
2. [`doc/core_algorithm.md`](doc/core_algorithm.md) — symbols, the conditioning contract, the
   block-by-block algorithm, train/probe/deploy parity.
3. [`doc/training_choices.md`](doc/training_choices.md) — D0/D1, `bg`/`white`, teacher/self forcing, and
   what is implemented, deferred or historical.
4. [`doc/known_gaps.md`](doc/known_gaps.md) — the open contract violations.
5. [`doc/README.md`](doc/README.md) — the per-module index, and the module doc for the file you
   are touching.
6. [`configs/README.md`](configs/README.md) — before writing or quoting any run command.

Items 1–4 and 6 are **required** before changing conditioning, noising, the cache, the loss,
configuration, a probe, or deployment. The single active workspace handoff in
`plans/` owns progress, acceptance and work order.
Algorithm and ownership explanations live here. Superseded plans are under
`plans/history/`; `SS…` markers in older prose are historical citations.

## One package, two conda envs

The package was split across two directories until 2026-09-15 (`scripts/onestep_avatar/` in
the workspace for the corpus half, this one for the model half) because the ARGAvatar renderer
and the LTX VAE cannot share a process. That reasoning confuses a **runtime** constraint with a
**layout** one: exactly one module imports ARGAvatar, and the split cost four transcribed
copies of shared knowledge plus the tests to pin them. It is one package now.

| Env | Runs |
|---|---|
| `argavatar` | **`corpus/build_guidance.py` only** — the one module that imports `scripts.inference.pipeline` |
| `ltx` | everything else, including the tests |

```bash
cd LTX-2                                       # the repo root, not scripts/onestep_avatar
conda run -n ltx       python -m scripts.onestep_avatar.<module> ...
conda run -n argavatar python -m scripts.onestep_avatar.corpus.build_guidance ...
conda run -n ltx       python -m pytest scripts/onestep_avatar/tests -q
```

`-m` from the LTX-2 root is mandatory: modules import each other as `scripts.onestep_avatar.*`
and do not touch `sys.path`.

**The namespace-package detail that makes `build_guidance` work.** `LTX-2/scripts/` has no
`__init__.py`, so `scripts` is a PEP 420 namespace package whose `__path__` recomputes when
`sys.path` changes. `build_guidance` inserts the ARGAvatar root at runtime, and
`scripts.inference` (ARGAvatar's portion) then resolves alongside `scripts.onestep_avatar`
(this one) and `scripts.prune`. **Do not add an `__init__.py` to `LTX-2/scripts/`** — it would
turn the namespace into a regular package and break that merge.

## Where this code lives, and why that matters

This package sits inside the **LTX-2 submodule**, but most of it is workspace-specific:
corpus plumbing for DNARendering and ARGAvatar, not anything upstream would want. It lives
here because the model half genuinely cannot leave — `train.py`, `corpus/precompute.py` and
the `model/` owners import `scripts.prune.core` and `ltx_core`/`ltx_trainer`, all LTX-2-resident.

**Repository boundary:** training, training/evaluation launchers and queues, generation,
model probes, reusable metrics, decoding, and training/inference visualization stay in this
LTX-2 package. Code under `expr/onestep_avatar/` is only for report generation: sections,
captions, report-specific plots, saved-result summaries, and report validation. Study settings,
saved runs, metrics, media, and logs can remain there as data. Package code accepts their paths
and does not import executable study code. Report rebuilds read saved results; missing results
fail instead of launching training, generation, evaluation, or decoding. Move required logic
out of current `expr/` executors during the reviewed implementation and remove obsolete copies
and their docs. Do not leave forwarding execution wrappers in `expr/`.

**Internal boundary:** core training and ordinary evaluation/product must not
depend on experiment modules. Comparison orchestration and historical converters
move to `experiments/` during the refactor; they call shared public
owners. Numerical diagnostic kernels can stay with their model owner. The existing
queue can explicitly dispatch an experiment job without making ordinary jobs
depend on it. Follow `doc/architecture.md` for the map, provenance and worked checks.

Corpus and model edits live in the LTX-2 submodule worktree. A commit changes
that submodule's history. Updating the workspace's recorded commit pin is a
separate change and requires explicit authorization; do not stage or update
the pin automatically.

## The documentation contract

**Production files over 100 physical lines require a source-mirrored design doc. Files with
100 lines or fewer describe their logic in their header docstring.** Count blanks, comments
and headers; exactly 100 uses the header rule. Empty package markers and tests need no
standalone module docs. Cross-module contracts and the documentation index are separate.

| | |
|---|---|
| **Inline docstrings** answer | *why this line, why this constant, why not the obvious alternative* |
| **`doc/<relative-source-path>.md`** for larger files | *what this file is for, what flows through it, how it fits the other files, what breaks if you change it* |
| **Small-file header** | objective, ordered logic, inputs/outputs and important invariants |

The second is what a reader cannot reconstruct from one file, and it is where this pipeline
has gone wrong before — every past bug has been **two producers of something that must have
one**, which is invisible from inside either producer.

**When to update a doc.** Any change to a module's *objective*, *data flow*, *invariants*, or
*contract with another module*. A pure refactor that moves no tensor and changes no artifact
needs no doc edit; a new flag that changes what lands on disk always does.

**Before changing code**, establish its intended logic in the doc folder. Large-file docs
mirror source paths, e.g. `model/causal.py` → `doc/model/causal.md`, and use Objective · Data
flow · Organization logic · Invariants · Gotchas · Tests. Core logic needs readable Mermaid
diagrams plus explicit shape/state facts and worked checks with expected outcomes. Render
and inspect diagrams; they cannot establish runtime numerical/distributed correctness.

**Core logic is required in the design doc.** The Organization logic section states actual
decisions, ordered operations, calculations/state changes, and outputs/failure behavior.
Do not replace that explanation with a file/function list or a reference to source comments.
Give a worked input with its expected result. For visualization, specify the displayed data,
screen positions, labels, timing, size/readability rules, and the exact text selection rule.
Tests must check those decisions and results, not only file existence or diagram syntax.

Proposed modules are labeled Proposed and their absent source paths are text, not broken links.
For planned small files, draft responsibilities/header content in the index's migration notes
before editing the header. Once implemented, update the status/index and actual doc/header.
When a file shrinks to at most 100 lines, preserve its useful explanation in its header and
delete its separate doc. When it grows past 100, write the matching design doc first.
Source moves/deletions update doc ownership and remove obsolete copies. Do not split or pad
files merely to change which documentation rule applies.

**When a doc and the code disagree, decide which kind of disagreement it is.**

> Code establishes **current** behavior; the explicit user-approved contract establishes
> **required** behavior. When they disagree, record a defect with evidence in
> [`doc/known_gaps.md`](doc/known_gaps.md) and keep both descriptions clear. Do not rewrite the
> intended contract to legitimize a bug, and do not describe a planned fix as shipped.

So: a doc that misdescribes what the code *does* is a stale doc — fix it there and then, because
a stale design doc is worse than none. A doc that describes what the code *must* do, and the code
does not, is a **code** defect: label the two plainly (Required / Current) and file the gap. A
known violation stays prominently marked — in the module doc, in the affected recipes, and in
`known_gaps.md` — until a fix is implemented *and* verified.

## Documentation language

Follow the user's target of about 80% ASD-STE100 style.
Use short sentences, active verbs, and one term for each concept.
Define technical terms before use. Preserve exact code identifiers and equations.
Use `video segment` for consecutive encoded frames; explain `span` only as a code/option term.
Define `c0` as the first-image input with no added noise.
Name data explicitly in diagrams, for example `capture or guide frames` and `encoded first image`.
Avoid unexplained labels such as `selected span` or `source span + c0`.

## Diagrams

Draw a diagram only for a **topology** — what flows where, who produces what. Per-block facts
(source, timestep, cache contents) belong in a table, which states them; a picture only gestures
at them.

- **Mermaid in a fenced ` ```mermaid ` block**, never ASCII art and never a committed SVG. It
  renders on GitHub and in editor previews, diffs line-by-line, and reads as a graph for a
  model. ASCII shears on edit and an image rots invisibly.
- **Shape and colour carry meaning**, one vocabulary across the package: rectangle/blue = code,
  cylinder/grey = on-disk artifact, rounded/green = in-memory tensor, hexagon/amber = mutable
  state, dashed = `no_grad`, stadium/purple = terminal consumer. The legend lives in
  [`doc/core_algorithm.md`](doc/core_algorithm.md) §7 — reuse it, do not invent a second one.
- **One branch per figure.** Two regimes that differ structurally (teacher vs self forcing) get
  two figures differing by one edge, not one figure with both drawn. A branch that only changes
  *which tensor* flows down an existing edge (D0 vs D1) is an edge label.
- **Labels are names.** A node is the callable or file (`refresh_block`, `windows.py`, with the
  arg when it selects a pass); an edge is the distinction it makes. Everything else goes in the
  prose under the figure, where it does not compete with the layout.
- **Do not depend on layout hints.** A subgraph's `direction` is ignored once its nodes have
  outside edges, and invisible `~~~` links become rank edges; both render differently across
  Mermaid versions. If nodes must sit a certain way, restructure the graph.
- **Render before committing** — `npx -y @mermaid-js/mermaid-cli -i <file.md> -o /tmp/out.md`
  — and look at the result. Valid syntax is not a readable figure.

## The contract rules

- **One canonical owner per cross-module contract.** `core_algorithm.md` owns the algorithm and
  the conditioning contract; `training_choices.md` owns the arm/objective/forcing definitions;
  `known_gaps.md` owns defect status; `configs/README.md` owns the runnable recipes. A module doc
  summarises and links; it does not restate a parameter table or the whole algorithm.
- **Core behavior and runnable configuration are self-contained in this package.** Do not write a
  doc whose explanation is "see the plan", and do not leave a bare `SS1.6` where a reader needs
  the substance.
- **A contract change updates everything at once**, in the same commit: the core docs, the
  affected module docs, the recipes in `configs/README.md`, and the gap status. The per-module
  documentation contract below and the two-environment rule still apply.
- **Reviewing a core change means tracing three cases**: block 0, block 1, and a **mid-clip**
  chain start. For each, identify every condition's source, its noise level and per-token
  timestep, what it can attend to, whether it is retained in the cache, its role in the loss, and
  whether deployment can supply it at all.
- **The clean first-frame condition `c0` is invariant across arm and forcing policy.** "The sink
  is pinned", "`keyframes_mask` marks frame 0", and "training and deployment share `causal_core`"
  are **not** evidence that it holds; assert the assembled block-0 input and its timesteps.
- **Configuration changes document defaults, explicit recipes and saved metadata together**, and
  keep implemented, proposed, deprecated and historical settings distinguishable. Never invent a
  flag or a loadable config file; `train.py` is CLI-driven.
- **Verification matches the change.** For docs-only changes: check links, cited symbols, and that
  each recipe matches `train.parse_args`. For behavior changes: the focused and full test runs
  below, plus native saved-input/export checks when changing shared whole-clip inputs.
  Passing cache-parity tests never establishes a missing first-frame condition.

## The invariants that are not obvious from one file

0. **Every generated block, including block 0, must have the supplied first-frame clean latent
   as initial conditioning** — independent of D0/D1 and of teacher/self forcing. This is the
   product contract, implemented as `clean_c0_v1`
   ([G1](doc/known_gaps.md#g1--the-supplied-first-frame-is-not-a-model-condition), verified
   2026-09-19). Cache-parity tests do not cover it; the dedicated conditioning tests in
   `tests/test_train.py` do.
1. **Current:** `model/common.py` owns shared conditioning/noise/loss helpers;
   `model/bidirectional.py` and `model/causal.py` own their mode calculations.
   Training and generation remain different operations; extraction alone does
   not prove parity. No CLI may create a second block-state or input path.
2. **One producer per artifact.** The crop box comes from `precompute.py --process_gt_latent`'s
   manifest; `z_y` from the capture pass; the guide and its alpha from `corpus/build_guidance.py`;
   current membership from `corpus/subset.py` and frame selection from the selected mode.
   `windows.py` is a remaining historical caller. Readers never recompute and
   never "reconstruct if missing" —
   they raise with a pointed error.
3. **Shared knowledge has exactly one spelling.** Artifact names live in `corpus/dataset.py`, the
   crop box in `corpus/geometry.py`, the block plan in `model/causal.py`, the mask codec in
   `corpus/mask_video.py`. These were four transcribed pairs before the merge; do not reintroduce a
   copy "to avoid an import".
4. **Both objectives (`bg`, `white`) share every code path**, differing only in which pixels
   were encoded and which filename holds them. An objective is never a second pipeline.
5. **No synthetic pixels ever enter a loss target** — shift-and-cap, never white-pad, and
   exclude a subject that does not fit the canvas.
6. **Masks are stored losslessly** (`corpus/mask_video.py`). These mattes are already one generation
   of lossy video from the truth; the pipeline does not add a second.

## Before calling a change done

- docs-only work checks links, source symbols, mode logic, worked design checks and inspected diagrams;
- behavior changes pass `conda run -n ltx python -m pytest scripts/onestep_avatar/tests -q`;
- touched modules' mirrored docs or small-file headers reflect the change;
- shared whole-clip input or export changes follow [`../prune/CLAUDE.md`](../prune/CLAUDE.md).
