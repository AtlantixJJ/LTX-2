# `onestep_avatar` — current design and acceptance limits

**Current GPU dispatch policy — user amendment, 2026-10-07:** query
`nvidia-smi` directly and use one shared JSON file to record only processes this
pipeline starts (PID, start ticks, command, GPU IDs and owned descendants).
New launches do not consult reservation files or unrelated process environments.
No privileged access is required. Preserve original claim,
launch, result and acceptance files unchanged. Scientific inputs, budgets,
tolerances and native E1–E5 gates remain unchanged.

Status: **Implementation authorized and in progress.**
The user directed execution of the revised plan. Shared input helpers, schedules,
the checked master reader, and bidirectional functions now exist.
The full restructure remains incomplete. Each module records its own status.
The current task prepares a self-contained handoff for the next agent.
The required work order is code refactor, CPU/import/boundary validation, fresh
affected native checks, then GPU experiments on the final source. This update
moves no production source and launches no GPU job.

## Review reading order

Start with the [package README](../README.md#terms-used-here).
It defines terms and shows file flow.
Then read the separate [bidirectional](../README.md#bidirectional-workflow)
and [causal](../README.md#causal-workflow) workflows and [mode selection](../README.md#selecting-a-mode).

1. [Architecture](architecture.md): common versus experiment code, dependencies, migration destinations and gates.
2. [Core algorithm](core_algorithm.md): input, first-image, loss, and cache rules.
3. [Bidirectional model](model/bidirectional.md): process one video segment together.
4. [Causal model](model/causal.md): process blocks in order; keep past-frame data.
5. [Common inputs](model/common.md) and [denoising steps](model/sampling.md).
6. [Engine](training/engine.md), [settings](training/config.md), and [adapter records](training/checkpoints.md).
7. [Video list](corpus/subset.md), [evaluation](evaluate.md), [visualization](media.md), and [timing](bench.md).
8. [Worked checks](verification.md) and [known gaps](known_gaps.md).

Training visualization is in [engine](training/engine.md#visualization-during-training)
and [training previews](media.md#training-previews).
Product visualization is in [inference output](media.md#inference-output).
The [screen layouts](media.md#visualization-layout) specify panel order and readable size.
The [video text rules](media.md#text-inside-the-video) specify roles and changed values.
The [cleanup list](../README.md#remove-old-code-and-docs) names old files and replacement docs.
The [repository boundary](../README.md#code-ownership) keeps training and model execution in LTX-2.
`expr/` code is only for report generation from saved results.

Explicit-mode training/evaluation/product, shared fp32 adapters, input preparation,
preview rendering and software manifests are implemented. Read
[current acceptance and next step](known_gaps.md#current-acceptance-and-next-step)
for verified scope and remaining gates. The
[active handoff](../../../../plans/2026-10-07-onestep-avatar-development-experiment-handoff.md#current-progress--2026-10-08)
owns exact current evidence; source presence alone does not establish acceptance.
Individual docs explain module behavior and link to that status instead of
repeating a changing run timeline.
`subset.py` converts old records into new files without repairing input bytes.
The workspace plan records migration progress; these docs explain the module logic.

## Documentation rule

A production source file over 100 physical lines requires a doc at the matching relative path.
Example: `model/causal.py` → `doc/model/causal.md`.
Count blank lines, comments, and headers.
A file with 100 lines or fewer describes its logic in its header.
Empty package markers and tests need no separate module doc.
Indexes and shared-rule documents are separate.

Large-file docs use Objective, Data flow, Organization logic, Invariants, Gotchas, and Tests.
Organization logic must explain the core behavior: decisions, ordered steps, actual calculations
or state changes, output meaning, and failure cases.
A function list or flow diagram alone is incomplete.
Do not send the reader to source comments for the main algorithm.
Give a worked input and expected result.
For visual outputs, specify what is displayed, screen layout, labels, units/time mapping,
and how the renderer selects important text.
Show core data flow with Mermaid.
Put equations, array shapes, state facts, and worked checks beside the diagrams.
Use the [shared legend](core_algorithm.md#7-end-to-end-data-flow).
Render and inspect each diagram.

Use about 80% ASD-STE100 style.
Write short sentences with direct verbs.
Define technical terms and use one term for each concept.
Keep code names and equations exact.
A proposed rule is not a passing code test.

Final source size decides whether a module keeps a separate doc.
Do not split or pad code to avoid this rule.
Old large-file docs stay until their source moves.
Small existing files already have logic docstrings; three redundant docs were removed.
This phase does not edit source headers.

## Core logic review map

Review these decisions and expected results, rather than only the file-flow diagrams:

| Design doc | Core behavior to review |
|---|---|
| [Common inputs](model/common.md#core-array-calculations) | token order, fp32 noise/velocity/loss calculations, exact first-image replacement |
| [Bidirectional](model/bidirectional.md#organization-logic) | frame-range selection, full attention, one direct backward, fixed first image across steps |
| [Causal](model/causal.md#organization-logic) | block ranges, priming, immediate backward, refresh/removal, retained frame positions |
| [Sampling](model/sampling.md#build-the-exact-schedule) | exact schedule construction/validation and deterministic Euler advancement |
| [Settings](training/config.md#core-resolution-logic) | omitted versus explicit mode options, checked records, seed formulas, side-effect order |
| [Checkpoints](training/checkpoints.md#read-and-decide-before-model-loading) | field comparisons, override decision, atomic save readiness |
| [Engine](training/engine.md#update-results-and-save-order) | sample/block averaging, one optimizer update, checkpoint step, preview job states |
| [Dataset](corpus/dataset.md#checked-reader-procedure) | unsliced master loading, D0/D1 checks, no producer repair or sample selection |
| [Video list](corpus/subset.md#select-and-group-videos) | deterministic person groups, exclusions, stable content identity, conversion |
| [Evaluation](evaluate.md#define-the-comparison-before-execution) | one-factor matching, saved noise reuse, reference/metric definitions |
| [Media](media.md#visualization-layout) | exact panel positions, important video text, synchronized frames, readable compact layout |
| [Training plots](plot_training.md#read-and-aggregate-logs) | process aggregation, incomplete updates, axes, smoothing, raw summary values |
| [Benchmark](bench.md#measurement-boundaries) | timer boundaries, cache reset, repetition/call counts, actual frame-rate arithmetic |
| [Crop](corpus/geometry.md#core-crop-calculation) | union, padded square, rounded shift/cap, subject-fit exclusion |
| [Motion](corpus/motion.md#convert-one-frame) | all three format conversions, body/camera ownership, exact gap holding and timing |
| [Masks](corpus/mask_video.md#derive-encoded-frame-coverage) | lossless bytes, spatial pooling, first-frame/eight-frame temporal groups |
| [Statistics](stats.md#core-measurement-calculations) | RMS/weighted gaps, moment scope, pairwise noise spread, declared diagnostic inputs |
| [Capture/guide encoding](corpus/precompute.md#core-pixel-and-encoding-transformations) | full-resolution target preparation, fixed crop, VAE-aligned prefix and stored shapes |
| [Guide rendering](corpus/build_guidance.md#organization-logic) | refined body plus view camera, one-pass RGBA work, exact background replacement |

Each doc contains a worked check or links to the numeric algorithm checks.
Proposed behavior remains distinct from the current implementation.

## Target source and design ownership

The user-directed [architecture contract](architecture.md) defines the target
layout: proposed `corpus/`, `execution/` and `experiments/` subpackages beside
`model/` and `training/`, and a split of ordinary `evaluate.py` into
`evaluate.py`, `metrics.py`, `previews.py` and `comparisons.py` (user decisions
of 2026-10-08). It defines the ownership map, allowed dependencies and the
complete migration map. The latest user instruction puts Stage D source moves
and duplicate-runtime removal first. Preserve required behavior and callers,
validate the final source with CPU/import/boundary checks, then publish fresh
affected native checks before GPU experiments. The tables
below include current mixed owners; their presence is not final boundary compliance.
The architecture migration map gives their required destinations. New larger
experiment sources require designs under `doc/experiments/` before source changes.

All source owners below are inside this LTX-2 package.
This includes training/evaluation execution, previews, decoding, and reusable metrics.
Report-specific sections, captions, plots, and validation belong to `expr/`.
Study settings and result files can remain there as data.
Neither report scripts nor forwarding wrappers in `expr/` may launch model work.
Package code does not import executable study code from `expr/`. Core training
and ordinary support do not depend on experiment modules. Explicit experiment
queue dispatch follows the exception in the architecture contract.
Model helpers, bidirectional functions, schedules, training owners, and the fixed-list
converter exist. Checked supplied-image and fixed preview preparation are
implemented. Their native preparation evidence does not certify model output.
Complete input production and native model acceptance remain pending.

| Source owner | Doc | Size rule and purpose |
|---|---|---|
| `model/common.py` | [model/common.md](model/common.md) | >100; token layout, noise, first-image input, output conversion |
| `model/adapters.py` | [model/adapters.md](model/adapters.md) | >100; shared unmerged fp32 PEFT configuration/loading, x0 inference wrapper and explicit fusion diagnostic |
| `execution/software.py` | [software.md](execution/software.md) | >100; explicit worktree owner hashes/runtime versions, current verification and historical integrity reading |
| `prepare_inputs.py` | [prepare_inputs.md](prepare_inputs.md) | >100; fixed preview assembly/pinned text-noise-image tensors and actual one-RGB supplied-image VAE preparation |
| `stock_parity.py` | [stock_parity.md](stock_parity.md) | >100; actual stock video sampling components, repeated controls, checked-reference precision comparison and decoded traces; native acceptance open |
| `model/bidirectional.py` | [model/bidirectional.md](model/bidirectional.md) | >100; segment training and generation |
| `model/causal.py` | [model/causal.md](model/causal.md) | >100; block and cache operations |
| `model/sampling.py` | [model/sampling.md](model/sampling.md) | >100; exact denoising levels and steps |
| `training/config.py` | [training/config.md](training/config.md) | >100; typed settings and initial checks |
| `training/engine.py` | [training/engine.md](training/engine.md) | >100; setup, weight updates, logs, saves, preview jobs |
| `training/runtime.py` | [training/runtime.md](training/runtime.md) | >100; actual Accelerator/FSDP policy capture, rank agreement and replay validation |
| `training/resources.py` | [training/resources.md](training/resources.md) | >100; exact budget identity, synchronized local CUDA phase peaks/time and complete rank/phase validation |
| `training/checkpoints.py` | [training/checkpoints.md](training/checkpoints.md) | >100; save/read/check settings and derive checked legacy metadata |
| `corpus/subset.py` | [subset.md](corpus/subset.md) | >100; fixed videos, person groups, hashes |
| `evaluate.py` | [evaluate.md](evaluate.md) | >100; same-input comparisons, training previews, fusion and eight-block causality diagnostics |
| `media.py` | [media.md](media.md) | >100; training/inference visualization |
| `bench.py` | [bench.md](bench.md) | >100; actual mode cost |
| `corpus/dataset.py` | [dataset.md](corpus/dataset.md) | >100; checked unsliced master reader and corpus names |
| `train.py` | header design below | ≤100 target; thin CLI |
| `infer.py` | [infer.md](infer.md) | >100; product CLI/API, strict preflight and generated-only rendering |
| `model/backbone.py` | header design below | ≤100 expected; resolve base weights |

If a thin file needs over 100 lines, write its matching doc before implementation.

## Small-file header designs and migration notes

- **`__init__.py`:** the standard-library-only root owner. Resolve `PACKAGE_ROOT`
  once; derive `LTX_ROOT` and `WORKSPACE_ROOT` from it. All source readers and
  child launchers import these constants. The B4 AST checks filename aliases and
  helpers; root import needs no tensor/model dependency. Keep `scripts/` a
  namespace package.

- **[`training/numerics.py`](../training/numerics.py):** one import-light
  deterministic training policy and required cuBLAS child environment. Configure
  that environment before native imports, apply the proven Torch/cuDNN/matmul
  flags before model work, and capture complete actual native/serial settings.
  Refuse late workspace configuration after CUDA initialization. Preserve and
  compare cuDNN's separate observed TF32 setting. Historical missing policy facts
  cannot acquire current defaults. The implemented small-file design lives in
  its header. Both modes pass the original four-rank numerical comparison;
  complete workflow acceptance remains separate. See the current gap summary.

- **`training/update_state.py`:** reusable optional named Adam-state export.
  Gather unsharded optimizer moments through native FSDP, or map ordinary
  optimizer parameters to model names. Write only main-rank finite fp32 adapter
  moments with exact optimizer/step metadata. No frozen-weight gathering, training
  loop, experiment imports or resume protocol. Small-file design lives in its header.

- **`training_update_check.py` (implemented root owner):** bounded
  serial replay of one saved distributed update using public shared preparation,
  model-loading, token construction and mode functions. Reuse original rank/slot
  noise seeds. Compare named clipped gradients, gradient norm, loss and actual
  exported adapters under predeclared tolerances. Both modes' numerical gates
  pass; preview/product scope remains separate. [Design](training_update_check.md).
  Proposed Stage D destination: `experiments/training_update_check.py`.

- **`sigma_sweep_jobs.py` (generation/dependent decoder preparation implemented;
  native acceptance pending):** [design](sigma_sweep_jobs.md).
  Preserve original paired masters, schedules, prompts and saved noise as
  explicit package evaluation jobs. Preparation never starts a model or queue.

- **`sigma_sweep.py` (saved decoding and queue dispatch implemented; native acceptance
  pending):** [design](sigma_sweep.md). Read hashed saved tensors, use one
  decoder-only session and shared metrics, publish ten videos and checked
  samples. Report-specific sheets read saved samples under expr and cannot
  invoke models. Historical launchers/analyzer are retired; exact source bytes
  remain as non-executable producer provenance text.
  The `sigma_sweep` queue kind pins specs, uses one shared evaluation-device
  claim and delegates complete output verification to this owner. Four saved
  historical decoder jobs are available; future generated-result binding is
  implemented through version-two result-bound specs. `sigma_sweep_results.py`
  describes its small-file design in its header: check frozen/shared cell
  conditions, verify scientific completion and bind all result/text/noise bytes
  without models or writes.

- **`future_noise_study.py` (preparation implemented, migration incomplete):**
  [design](future_noise_study.md). Reconstruct exact saved noise and prepare
  package jobs for the historical intervention/repeat/causal controls. Current
  real conversion refuses a changed guide-render pin in the old subset. Keep
  a still-required old executor until its callers and required behavior have a
  checked final owner; follow the active handoff's disposition rule. Do not retain
  a broken command indefinitely. Planned owner: `experiments/future_noise_study.py`.

- **`convert_progress_jobs.py`:** data-only conversion of historical progress
  rows to package evaluation jobs. Preserve both fixed views, seed 42, sigma
  0.8977352380752563 and exact one/four-call schedules from the shared scheduler.
  Read the run's arm and causal geometry; use an explicit research override for
  these off-condition diagnostics. Require both sources in checked membership.
  Publish a fresh JSON job list; never open models or start a queue. Tests check
  row coverage, source order, schedules, checkpoint paths and completion paths.

- **`execution/queue_protocol.py` (implemented):** small constant owner for queue token/job
  environment names and the startup-event prefix. Import no model libraries.
  `training/startup.py` retains its public imported names for existing consumers.
- **`training_slice_check.py` (bounded numerical localization controls pass):**
  [One-rank update localization](training_slice_check.md) calls shared training
  owners with exact original visits. Ordinary runtime does not import it.
- **`adapter_effect_check.py` (bounded E2 comparison; full E2 incomplete):**
  [Saved adapter correction](adapter_effect_check.md) compares the shared training
  reference, ordinary evaluation and product APIs with independent supplied c0,
  fixed guide/text/noise and original native numerical policy. It checks the fixed
  under-5% effect criterion and measures original-budget process resources. Fusion
  stays diagnostic. Saved verification performs no model/decoder work. This
  experiment owner moves during the next agent's Stage D refactor; ordinary
  runtime must not import it.
- **`continuation_check.py` (bounded E3 observation; native acceptance pending):**
  [Native history observations](continuation_check.md) delegate unchanged cached
  and recomputed sampling, snapshot real K/V and observe native attention inputs
  before/after eviction. Public future/capture controls use the ordinary evaluator.
  Shared resources and own-process supervision preserve the original E3 limits.
  CPU preflight and tiny-native controls do not prove full-weight or seven-frame
  pilot acceptance. The owner moves during Stage D before final-source native
  experiments; ordinary runtime must not import it.
- **`execution/process_registry.py` (implemented, native launch evidence in progress):**
  [Own-process tracking](execution/process_registry.md) in one shared JSON file, direct GPU
  queries and exact targeted descendant identities. No privilege or unrelated
  environment scans. Closed attempt history remains immutable.
- **`execution/supervision.py` (implemented, native acceptance in progress):**
  [Bounded observation](execution/supervision.md) of registered children, rank phases and
  exact descendant shutdown. Resource journals own allocator measurements.
- **`training/consumer_trace.py` (implemented, native comparison in progress):**
  [Consumer observations](training/consumer_trace.md) record actual transformed
  conditioning, adapter storage and forward compute separately. Failed runs keep
  incomplete observations; trace hashes alone do not prove gradient agreement.
- **`execution/queue_launch.py` (implemented and integrated with persistent dispatch):** launch gate, with [queue_launch.md](execution/queue_launch.md).
  A child registers its stable process identity, waits for a journal-bound grant,
  then replaces itself with the exact recorded command. Import no models.
  Persistent dispatch saves registration before approval and checks command transitions.
  Guarded recovery proves absent approval before marking interrupted work failed.
  Explicit ended-attempt recovery has limited bookkeeping scope and cannot
  manufacture continuous supervision. Exact current receipts are in the active handoff.

- **`train.py`:** parse/check settings through `training.config`.
  Print dry runs or call `training.engine`.
  Own no model/cache/update loop or checkpoint-record construction.
- **`infer.py`:** require guide data, supplied-image encoding, background, frame rate,
  mode, frame dimensions, and adapter settings.
  Check D1 and supported conditions before calling the selected generation function.
  Save output records and optionally call `media.py`.
  Product generation has no capture target or capture past frames.
  This implemented large file has [doc/infer.md](infer.md).
- **`model/backbone.py`:** resolve version and dev/distilled weight paths once.
  Keep registry defaults and return the actual weight hash.
  The current resolver has 62 lines.
- **`training/startup.py` (implemented):** the small-file header owns queued startup events.
  Bind an optional attempt token/job hash and rank; mark updates before training.
  Propagate exceptions and emit only the defined contention/failure events.
  Own no retry, archive, model or process launch.
- **`hashing.py`:** stream file bytes through one SHA-256 helper.
  Do not substitute size/time for a content hash. Current size: 22 lines.
- **`qa.py`:** compare masks with the declared threshold and calculate overlap.
  This score is not a loss weight. Current size: 25 lines.
  Compare raw alpha and capture mask in the same crop, not composited RGB.
  Low overlap requires inspection.
- **`decode_saved.py`:** read hashed jobs and call `media`.
  Do not regenerate model output.
  If its current 104 lines become at most 100, move the doc into its header.

These notes describe the intended ownership. Implemented small-file headers
explain their logic; larger sources use the mirrored docs above.

## Current source owners and remaining migration

The tables in this file list **current** paths. Final paths and every
destination are in [architecture: target layout](architecture.md#target-layout)
and [migration map](architecture.md#migration-map); that map wins over the
"Planned change" summaries below.

| Current source | Current doc | Planned change |
|---|---|---|
| `corpus/precompute.py` | [precompute.md](corpus/precompute.md) | move to `corpus/`; keep capture/guide VAE producers |
| `corpus/build_guidance.py` | [build_guidance.md](corpus/build_guidance.md) | move to `corpus/`; keep the only ARGAvatar process; check old branches |
| `corpus/geometry.py` | [geometry.md](corpus/geometry.md) | move to `corpus/`; keep the shared crop rule |
| `corpus/motion.py` | [motion.md](corpus/motion.md) | move to `corpus/`; keep pose conversion |
| `corpus/mask_video.py` | [mask_video.md](corpus/mask_video.md) | move to `corpus/`; keep lossless masks; check old data use |
| `stats.py` | [stats.md](stats.md) | move whole to `experiments/stats.py` (A1/B1c study code; no other consumer) |
| `plot_training.py` | [plot_training.md](plot_training.md) | keep training plots; remove old readers after log conversion |
| `decode_saved.py` | [decode_saved.md](decode_saved.md) | use media; apply final size rule |
| `train.py` | source header | thin CLI implemented; [engine](training/engine.md) and [config](training/config.md) extracted; remaining owners pending |
| `model/causal.py` | [model/causal.md](model/causal.md) | typed training/evaluation/product integrated; native quality/cost acceptance pending |
| `windows.py` | [windows.md](windows.md) | move the legacy subset hash rule to `corpus/subset.py`, then delete |
| `training/checkpoints.py` | [training/checkpoints.md](training/checkpoints.md) | version-two records and tensor preflight stay; legacy conversion moves to `experiments/legacy_adapters.py` |
| removed `onestep_core.py` | [infer.md](infer.md) | explicit-mode product CLI/API and shared model/media owners |
| `visualize_d0.py` | [visualize_d0.md](visualize_d0.md) | retarget needed checks to final owners, then delete |
| `visualize_d1.py` | [visualize_d1.md](visualize_d1.md) | move the `--whole-clip` pruning producer to `scripts/prune/`, then delete |
| `bench.py` | [bench.md](bench.md) | causal-operation diagnostic and explicit-mode whole-generation CLI implemented; native measurement pending |
| `model/backbone.py` | [source docstrings](../model/backbone.py) | moved; keep the weight-identity description |
| `hashing.py` | [source docstrings](../hashing.py) | stays at the root as a leaf utility; gains `tensor_sha256` |
| `corpus/qa.py` | [source docstrings](../corpus/qa.py) | move to `corpus/`; keep mask comparison; caller rules above |

The architecture contract also requires study/diagnostic orchestration to leave
ordinary `evaluate.py`, A1/B1c orchestration to leave the shared root, and historical
adapter conversion orchestration to leave the normal checkpoint owner. Shared
measurement/model/validation primitives keep one common owner. Sweep/progress/
stock-check files move into `experiments/` during the next agent's code refactor.
Their current root paths and commands stay truthful until that actual move.
Fresh affected native evidence is then produced on the final source.

Current commands remain in [configs/README](../configs/README.md).
Explicit `--mode` and the shared `fsdp.yaml` are implemented.
Internal imports use the extracted owners. The subset conversion CLI is runnable.

## External code and doc migration

Current `expr/` scripts include training/evaluation queues, launch scripts, and model probes.
Some combine execution with report assembly.
Move required execution into this package's public CLIs/helpers.
Retain only report-specific code in `expr/`.
Keep original saved results and their producer records.

Use the workspace plan's explicit file inventory and deletion checks during implementation.
Transfer required module explanations into source-mirrored package docs or small-file headers.
Remove old `expr/code/doc/` pages for deleted model executors.
Retained report code keeps only its own report-generation explanation.
If a required queue needs a new package executor, establish its design doc before source changes.
The package executor is specified in [queue](execution/queue.md). Execution requires
`--execute`, exactly one of `--once` or `--loop`, and
`--process-ledger <SHARED_PROCESS_LEDGER.json>`. Current attempts share
`expr/onestep_avatar/processes.json`, query `nvidia-smi` directly and record only
their own PID/start/command/GPU identities and descendants. The loop rereads
append-only jobs and waits without an active process record,
verifies dependency/completion receipts and refuses failed or unrecovered work.
Bounded startup retries preserve failed attempts; historical job conversion
and native retry/live handoff acceptance remain pending.
The design defines saved job settings, exact completion
evidence, GPU restrictions, own-process tracking, child ownership and recovery.
The retained legacy `GPUClaims` helper and reservation files describe historical
attempts; new dispatch does not read them or unrelated process environments.
Live queue
migration remains pending and must preserve recorded scientific conditions.
The initial inventory includes active and archived sources. Ownership classification
and execution migration remain incomplete.
Classify mixed study branches separately: saved-result rebuilding can coexist
with explicit model generation or historical-byte import in the same source.
Stage D must prevent discovery, import and rebuild from restoring retired execution,
in addition to removing its current paths and forwarding wrappers.

## Artifact owners

Capture precompute writes crop records and `z_y`.
ARGAvatar writes guide RGB/alpha.
Guide precompute writes `z_g` and cropped masks.
Dataset code owns filenames and record access.

Subset code writes the fixed video list without changing source data.
Modes select frames.
Checkpoint code writes adapter settings.
Evaluation writes model-result records; media writes rendering records.
`expr/` report code reads those saved records and media; it does not generate missing results.
Readers check missing producer output instead of reconstructing it.

## Verification and evidence

[verification.md](verification.md) lists V1–V8 and their expected results.
The [current gap summary](known_gaps.md#current-acceptance-and-next-step) separates
CPU behavior, scoped native acceptance and remaining full requirements. Exact
receipts and preserved failed attempts belong to the active handoff.
Diagram checks and documentation links do not prove numerical or video quality.

Final migration must audit `expr/` imports and subprocess calls for model
execution, remove obsolete runnable paths, and prevent import/rebuild from
restoring them. Rebuild reports from saved results with training, generation and
decoder loaders disabled. Missing results must fail without starting a job.

### G3 corpus and adapter layering

The eight corpus owners and their larger-file docs now live in `corpus/` and
`doc/corpus/`. Small `qa.py` uses its header. Corpus commands use the new module
paths; guide production remains in `argavatar`. Model adapters own the LoRA
target list; checked contracts and file identity reach the model as data.
Only explicit legacy subset conversion lazily calls the canonical causal
planner, under the documented dependency exception. CPU boundary tests enforce
that exception and model-free ordinary corpus imports.
