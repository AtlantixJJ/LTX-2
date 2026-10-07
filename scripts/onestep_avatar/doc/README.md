# `onestep_avatar` — current design and acceptance limits

Status: **Implementation authorized and in progress.**
The user directed execution of the revised plan. Shared input helpers, schedules,
the checked master reader, and bidirectional functions now exist.
The full restructure remains incomplete. Each module records its own status.

## Review reading order

Start with the [package README](../README.md#terms-used-here).
It defines terms and shows file flow.
Then read the separate [bidirectional](../README.md#bidirectional-workflow)
and [causal](../README.md#causal-workflow) workflows and [mode selection](../README.md#selecting-a-mode).

1. [Core algorithm](core_algorithm.md): input, first-image, loss, and cache rules.
2. [Bidirectional model](model/bidirectional.md): process one video segment together.
3. [Causal model](model/causal.md): process blocks in order; keep past-frame data.
4. [Common inputs](model/common.md) and [denoising steps](model/sampling.md).
5. [Engine](training/engine.md), [settings](training/config.md), and [adapter records](training/checkpoints.md).
6. [Video list](subset.md), [evaluation](evaluate.md), [visualization](media.md), and [timing](bench.md).
7. [Worked checks](verification.md) and [known gaps](known_gaps.md).

Training visualization is in [engine](training/engine.md#visualization-during-training)
and [training previews](media.md#training-previews).
Product visualization is in [inference output](media.md#inference-output).
The [screen layouts](media.md#visualization-layout) specify panel order and readable size.
The [video text rules](media.md#text-inside-the-video) specify roles and changed values.
The [cleanup list](../README.md#remove-old-code-and-docs) names old files and replacement docs.
The [repository boundary](../README.md#code-ownership) keeps training and model execution in LTX-2.
`expr/` code is only for report generation from saved results.

Explicit-mode training/evaluation/product, shared fp32 adapters, preview rendering,
and software manifests are implemented and CPU checked. Native model/distributed
acceptance, complete checked input production and legacy cleanup remain open.
Individual docs distinguish implemented behavior from remaining acceptance.
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
| [Dataset](dataset.md#checked-reader-procedure) | unsliced master loading, D0/D1 checks, no producer repair or sample selection |
| [Video list](subset.md#select-and-group-videos) | deterministic person groups, exclusions, stable content identity, conversion |
| [Evaluation](evaluate.md#define-the-comparison-before-execution) | one-factor matching, saved noise reuse, reference/metric definitions |
| [Media](media.md#visualization-layout) | exact panel positions, important video text, synchronized frames, readable compact layout |
| [Training plots](plot_training.md#read-and-aggregate-logs) | process aggregation, incomplete updates, axes, smoothing, raw summary values |
| [Benchmark](bench.md#measurement-boundaries) | timer boundaries, cache reset, repetition/call counts, actual frame-rate arithmetic |
| [Crop](geometry.md#core-crop-calculation) | union, padded square, rounded shift/cap, subject-fit exclusion |
| [Motion](motion.md#convert-one-frame) | all three format conversions, body/camera ownership, exact gap holding and timing |
| [Masks](mask_video.md#derive-encoded-frame-coverage) | lossless bytes, spatial pooling, first-frame/eight-frame temporal groups |
| [Statistics](stats.md#core-measurement-calculations) | RMS/weighted gaps, moment scope, pairwise noise spread, declared diagnostic inputs |
| [Capture/guide encoding](precompute.md#core-pixel-and-encoding-transformations) | full-resolution target preparation, fixed crop, VAE-aligned prefix and stored shapes |
| [Guide rendering](build_guidance.md#organization-logic) | refined body plus view camera, one-pass RGBA work, exact background replacement |

Each doc contains a worked check or links to the numeric algorithm checks.
Proposed behavior remains distinct from the current implementation.

## Target source and design ownership

All source owners below are inside this LTX-2 package.
This includes training/evaluation execution, previews, decoding, and reusable metrics.
Report-specific sections, captions, plots, and validation belong to `expr/`.
Study settings and result files can remain there as data.
Neither report scripts nor forwarding wrappers in `expr/` may launch model work.
Package code does not import executable study code.
Model helpers, bidirectional functions, schedules, training owners, and the fixed-list
converter exist. Complete input production and native acceptance remain pending.

| Source owner | Doc | Size rule and purpose |
|---|---|---|
| `model/common.py` | [model/common.md](model/common.md) | >100; token layout, noise, first-image input, output conversion |
| `model/adapters.py` | [model/adapters.md](model/adapters.md) | >100; shared unmerged fp32 PEFT configuration/loading, x0 inference wrapper and explicit fusion diagnostic |
| `software.py` | [software.md](software.md) | >100; explicit worktree owner hashes/runtime versions, current verification and historical integrity reading |
| `prepare_inputs.py` | [prepare_inputs.md](prepare_inputs.md) | >100; fixed preview assembly/pinned text-noise-image tensors and actual one-RGB supplied-image VAE preparation |
| `stock_parity.py` | [stock_parity.md](stock_parity.md) | >100; actual stock video sampling components, repeated native control, fixed-input traces and decoded comparisons; native acceptance open |
| `model/bidirectional.py` | [model/bidirectional.md](model/bidirectional.md) | >100; segment training and generation |
| `model/causal.py` | [model/causal.md](model/causal.md) | >100; block and cache operations |
| `model/sampling.py` | [model/sampling.md](model/sampling.md) | >100; exact denoising levels and steps |
| `training/config.py` | [training/config.md](training/config.md) | >100; typed settings and initial checks |
| `training/engine.py` | [training/engine.md](training/engine.md) | >100; setup, weight updates, logs, saves, preview jobs |
| `training/checkpoints.py` | [training/checkpoints.md](training/checkpoints.md) | >100; save/read/check settings and derive checked legacy metadata |
| `subset.py` | [subset.md](subset.md) | >100; fixed videos, person groups, hashes |
| `evaluate.py` | [evaluate.md](evaluate.md) | >100; same-input comparisons, training previews, fusion and eight-block causality diagnostics |
| `media.py` | [media.md](media.md) | >100; training/inference visualization |
| `bench.py` | [bench.md](bench.md) | >100; actual mode cost |
| `dataset.py` | [dataset.md](dataset.md) | >100; checked unsliced master reader and corpus names |
| `train.py` | header design below | ≤100 target; thin CLI |
| `infer.py` | [infer.md](infer.md) | >100; product CLI/API, strict preflight and generated-only rendering |
| `model/backbone.py` | header design below | ≤100 expected; resolve base weights |

If a thin file needs over 100 lines, write its matching doc before implementation.

## Small-file header designs and migration notes

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
  the old executor until checked data and required parity are established.

- **`convert_progress_jobs.py`:** data-only conversion of historical progress
  rows to package evaluation jobs. Preserve both fixed views, seed 42, sigma
  0.8977352380752563 and exact one/four-call schedules from the shared scheduler.
  Read the run's arm and causal geometry; use an explicit research override for
  these off-condition diagnostics. Require both sources in checked membership.
  Publish a fresh JSON job list; never open models or start a queue. Tests check
  row coverage, source order, schedules, checkpoint paths and completion paths.

- **`queue_protocol.py` (implemented):** small constant owner for queue token/job
  environment names and the startup-event prefix. Import no model libraries.
  `training/startup.py` retains its public imported names for existing consumers.
- **`queue_launch.py` (implemented and integrated with persistent dispatch):** launch gate, with [queue_launch.md](queue_launch.md).
  A child registers its stable process identity, waits for a journal-bound grant,
  then replaces itself with the exact recorded command. Import no models.
  Persistent dispatch saves registration before approval and checks command transitions.
  Guarded recovery proves absent approval before marking interrupted work failed.
  Old unguarded recovery remains conservative; native acceptance is pending.

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

| Current source | Current doc | Planned change |
|---|---|---|
| `precompute.py` | [precompute.md](precompute.md) | keep capture/guide VAE producers |
| `build_guidance.py` | [build_guidance.md](build_guidance.md) | keep the only ARGAvatar process; check old branches |
| `geometry.py` | [geometry.md](geometry.md) | keep the shared crop rule |
| `motion.py` | [motion.md](motion.md) | keep pose conversion |
| `mask_video.py` | [mask_video.md](mask_video.md) | keep lossless masks; check old data use |
| `stats.py` | [stats.md](stats.md) | keep measurements; reuse shared helpers later |
| `plot_training.py` | [plot_training.md](plot_training.md) | keep training plots; remove old readers after log conversion |
| `decode_saved.py` | [decode_saved.md](decode_saved.md) | use media; apply final size rule |
| `train.py` | source header | thin CLI implemented; [engine](training/engine.md) and [config](training/config.md) extracted; remaining owners pending |
| `model/causal.py` | [model/causal.md](model/causal.md) | typed training/evaluation/product integrated; native quality/cost acceptance pending |
| `windows.py` | [windows.md](windows.md) | replace with video list and mode plans, then delete |
| `training/checkpoints.py` | [training/checkpoints.md](training/checkpoints.md) | version-two records, tensor preflight and checked fixed/random legacy conversion implemented; ambiguous evidence and bulk conversion pending |
| removed `onestep_core.py` | [infer.md](infer.md) | explicit-mode product CLI/API and shared model/media owners |
| `visualize_d0.py` | [visualize_d0.md](visualize_d0.md) | replace with evaluate/media, then delete |
| `visualize_d1.py` | [visualize_d1.md](visualize_d1.md) | replace with evaluate/media, then delete |
| `bench.py` | [bench.md](bench.md) | causal-operation diagnostic and explicit-mode whole-generation CLI implemented; native measurement pending |
| `model/backbone.py` | [source docstrings](../model/backbone.py) | moved; keep the weight-identity description |
| `hashing.py` | [source docstrings](../hashing.py) | keep the shared content hash |
| `qa.py` | [source docstrings](../qa.py) | keep mask comparison; caller rules above |

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
The package executor is specified in [queue](queue.md). GPU claim helpers are
implemented; execution requires `--execute`, `--once` or `--loop`, and `--claims-dir` with the shared
reservation directory. The loop rereads append-only jobs, waits without claims,
verifies dependency/completion receipts and refuses failed or unrecovered work.
Bounded startup retries preserve failed attempts; historical job conversion
and native retry/live handoff acceptance remain pending.
The design defines saved job settings, exact completion
evidence, GPU restrictions, claims, child ownership and recovery. Live queue
migration remains pending and must preserve recorded scientific conditions.
The initial inventory includes active and archived sources. Ownership classification
and execution migration remain incomplete.

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

## Review gate and evidence

[verification.md](verification.md) lists V1–V8 and their expected results.

- The original review rendered and inspected twelve algorithm/visualization diagrams and four README diagrams.
  The 2026-10-06 core-logic revision rendered and inspected the revised common-input and training-plot diagrams.
  Temporary checkpoint/history/inference layout sketches passed title bounds and scaled-font checks.
  The sketches contain no actual source or generated video; they check the proposed presentation.
- Local Markdown links and heading anchors resolved.
  At the original review, eleven large-module designs had the six sections.
  Module headers now record their individual implementation status.
- Exact arithmetic checked V1–V3. Integer frame/cache traces checked V4–V6.
  The Euler example also matched.
  Added seed, schedule, crop, mask, array-shape, plot, RMS, and frame-rate examples were checked separately.
- V7–V8 are planned checker/conversion tests, not implemented results.
- Cited current symbols were checked against source.
  Removed small-module docs correspond to files with existing logic docstrings and at most 100 lines.
- The initial 42 inventoried source/test/launcher/config hashes matched during documentation checks.
  Subsequent separate edits changed source/tests and added `model/common.py`.
  This task edited no source, tests, launchers, or runtime configuration. Existing edits are preserved.

After implementation, audit `expr/` imports and subprocess calls for model execution.
Rebuild reports from saved results with training, generation, and decoder loaders disabled.
Missing results must fail without starting a job.

These checks validate documented examples and presentation sketches.
They do not establish renderer/model/distributed implementation correctness.
Implementation validation is recorded separately in the plan's progress section.
The user authorized implementation. Remaining checks must establish the full design.
