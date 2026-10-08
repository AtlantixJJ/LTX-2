# Package launch configuration and current commands

Explicit bidirectional and causal training use the typed runtime. New launches
use the shared `fsdp.yaml` and an explicit process count. The old mode-less
entry is temporary for live queues and is not a new-run recipe.

Use the `ltx` environment from the LTX-2 root. Only guide rendering uses
`argavatar`. Query `nvidia-smi` directly and register starts in the single
shared own-process ledger. Independent checks may run concurrently on GPUs
0–3; four-rank training still requires all four. These recipes describe
commands; this refactor does not start campaigns or bulk preprocessing.

Read [core rules](../doc/core_algorithm.md), [mode settings](../doc/training/config.md)
and [adapter checks](../doc/training/checkpoints.md) before changing conditions.

## 1. Accelerate configs

### Shared template and process-count override

`fsdp.yaml` is the package template for new explicit-mode launches. It keeps
GPU checkpoint loading (`fsdp_cpu_ram_efficient_loading: false`), adapter
gathering (`FULL_STATE_DICT`), FSDP version 1, original parameters, and the
`BasicAVTransformerBlock` wrap policy. Its default process count is four.
Pass `--num_processes` to match the selected devices. Installed Accelerate's
launch parser must resolve that override before any model process starts.
Typed training's shared model builder disables mixed precision's root-input
casting before FSDP wrapping. The YAML's bf16 parameter/reduction policy remains;
the incoming modality's float32 sigma, timesteps and positions must survive it.
Do not infer effective forward precision from the YAML or saved scalar alone.

Queued training snapshots the canonical command and original Accelerate bytes at
dispatch. Actual world size, distributed type and mixed precision must match
before model loading. After wrapping, all ranks record matching FSDP policies
and fp32 adapter storage in config and checkpoint markers.

For bounded native update acceptance, add `--resource-budget <FROZEN_PROTOCOL_JSON>`
to the original training job. The existing protocol declares
`wall_seconds_per_phase` and `memory_limit_allocated_bytes`. Queue identity pins
the exact budget bytes; the trainer records synchronized allocated/reserved peaks
for load, each update and each export. Markers bind immutable per-checkpoint
rank snapshots. Sampled total-device occupancy remains a separate observation.
Current native execution still requires authoritative worker ownership and a
package-owned bounded external startup/phase supervisor. The historical
supervisor source snapshots do not satisfy that gate.

After a valid one-update run, check its original job before serial replay:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.training_update_check \
  --job <CHECKED_ORIGINAL_JOB_JSON> --output <FRESH_SERIAL_OUT> --world-size 4 --dry-run
```

This checks completed scientific evidence and current launch/runtime/resource
identities; it does not start a model. Historical runs missing those facts fail
current acceptance and retain their original scope. Reprepare source-bound jobs
after repairs rather than restamping historical records.

For example, from `LTX-2`, after checking free devices:

```bash
CUDA_VISIBLE_DEVICES=<GPUS> conda run --no-capture-output -n ltx accelerate launch \
  --config_file scripts/onestep_avatar/configs/fsdp.yaml \
  --num_processes <COUNT> --main_process_port <PORT> \
  -m scripts.onestep_avatar.train --mode bidirectional \
  --subset <V2_SUBSET> --output <FRESH_OUT> --variant dev \
  --guide-mode d1 --objective white --span-latent-frames 17
```

The three old GPU-count copies are removed after the training caller migration.
The running study uses the separate recorded forward-prefetch setting; preserve
that setting when reproducing its jobs. Do not modify the trainer package's YAML.

### Corpus measurements and guide rendering

Use the public module commands directly from `LTX-2`. The removed corpus shell
launchers had no active callers in the package or current study execution
inventory. They selected environments, checked free GPU memory, and set logging
paths; they did not implement measurement or guide rendering.

Before measurements, check `nvidia-smi` for an available device with at least
34,000 MiB free. Use the `ltx` environment and the corpus path for both inputs:

```bash
conda run --no-capture-output -n ltx python -u -m scripts.onestep_avatar.stats \
  --model 2.5 --gpu-id <GPU_ID> \
  --renders ../data/AnimatableHuman/DNARenderingVideo \
  --render-glob argavatar_render.mp4 \
  --pairs ../data/AnimatableHuman/DNARenderingVideo \
  --max-videos 4 --eps-samples 8 \
  --out ../expr/onestep_avatar/analysis_summary.json
```

Before guide rendering, check for at least 20,000 MiB free. `CUDA_VISIBLE_DEVICES`
maps the selected physical device to local device zero. Use `argavatar`, keep
two separate driving-view values, and save an unbuffered combined log under
`expr/`. Inspect the first batch's QA overlays before starting another batch.

```bash
mkdir -p ../expr/onestep_avatar/logs
CUDA_VISIBLE_DEVICES=<GPU_ID> conda run --no-capture-output -n argavatar \
  python -u -m scripts.onestep_avatar.build_guidance \
  --driving-views 1 5 --limit 8 --visualize --device cuda:0 \
  2>&1 | tee ../expr/onestep_avatar/logs/guide_review.log
```

Choose a fresh log name for another batch. A zero limit requests all available
inputs; do not use it for this refactor's bounded checks. These recipes document
preserved behavior and do not authorize a corpus campaign.

## 2. Fixed inputs and training modes

### Fixed training preview inputs

Prepare RGB reference pixels with the checked `media --prepare-training-references`
command first. Then assemble one selected source using ordinary evaluation flags:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.prepare_inputs preview \
  --references <REFERENCES_JSON> --output <FRESH_FIXED_DIRECTORY> --gpu-id <FREE_GPU> \
  --evaluation-arguments --mode bidirectional --subset <V2_SUBSET> --source <SOURCE_ID> \
  --variant dev --guide-mode d1 --span-latent-frames 7 --schedule 0.725 0 --include-base
```

For causal, select `--mode causal` with the intended block/history settings.
The original one-update E4 causal
adapter records a null training span and frame counts `[6,7]`. Passing
`--span-latent-frames 7` changes that recorded setting and its strict preview
preflight rejected that changed setting. The causal-only
`--output-latent-frames 7` selects physical coverage while leaving the recorded
span null. Public preparation and fresh causal preview execution pass; complete
workflow acceptance and full/narrow inspection are tracked in
[current acceptance](../doc/known_gaps.md#current-acceptance-and-next-step).
For a pilot adapter actually trained with span 7, retain its
`--span-latent-frames 7`. When both flags are supplied their values must match.
Preparation preserves the causal training span and pins physical coverage
separately in its emitted arguments. Fresh preparation records carry the new
producer identity; retain historical records with their original attribution.
References must cover the selected 49 RGB frames. `--include-base` requests the
base comparison; without it the baseline cell is explicitly not requested.
Both positive and, when needed, negative text are pinned. Supply the resulting
`preview.json` as training's `--preview-inputs`; checkpoint completion enqueues
the separate generation/rendering job. Preparation performs no transformer call.

### Supplied-image product input

The image uses the original canvas coordinates of the guide's saved crop. White
requires a matching grayscale matte; bg rejects one. Use a fresh directory:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.prepare_inputs supplied-image \
  --image <RGB_IMAGE> --mask <GRAYSCALE_MATTE> --guide <WHITE_GUIDE_BUNDLE> \
  --output <FRESH_INPUT_DIRECTORY> --gpu-id <FREE_GPU> --review
```

The result `image.pt` is the `infer.py --first-image` input. The prepared PNG
shows actual preprocessing; the decoded PNG shows native VAE reconstruction.
This command prepares exactly one image and does not generate a video.

### Historical sigma-sweep generation preparation

Preserve the four original cases as 32 explicit causal evaluation jobs and four
dependent matched decoder jobs using
their exact recorded schedules, literal prompts, paired master hashes and
saved noise prefixes. The cases file pins the original manifests and the
historical source supporting their unguided configuration. From `LTX-2`:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.sigma_sweep_jobs \
  --cases ../expr/onestep_avatar/d1_selfrollout_sigma_sweep_20260926/configs/generation_cases.json \
  --output <FRESH_PREPARATION_DIRECTORY>
```

This prepares data only. Review `jobs.json` through the package queue's
`--dry-run`; actual execution uses its documented shared own-process ledger and device
policy. Historical GPU numbers are evidence only. The preparation record binds
the thirteen derived files, original inputs and producer sources. Each decoder
waits for eight unchanged verified generation receipts. Its version-two spec
resolves future tensor hashes only from scientifically complete result records.
Reports are rebuilt separately from saved decoded samples. The old callers are
retired; their exact bytes and hashes remain as producer provenance text.
Native replacement parity remains an open acceptance gate.

### Saved sigma-sweep decoding

The package owns saved sweep decoding; reports read its saved PNG samples.
The four historical decodes also have package queue data in
`expr/onestep_avatar/d1_selfrollout_sigma_sweep_20260926/configs/saved_decode_jobs.json`.
These `sigma_sweep` jobs use shared own-process tracking, pin their specs and verify all saved
media and metric controls before publishing receipts. They decode existing
historical tensors. Generation preparation also creates result-bound dependent
decoders for new outputs.
Four checked specs are under
`expr/onestep_avatar/d1_selfrollout_sigma_sweep_20260926/configs/`.
This command uses the original raw tensors and masters and requires a fresh
destination. Check that the selected GPU is free and use the shared own-process ledger
before native execution. From `LTX-2`:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.sigma_sweep \
  --spec <SAVED_SWEEP_SPEC> --output <FRESH_SAVED_MEDIA_DIRECTORY> --gpu-id <GPU>
```

After decoding has completed, assemble report-only sheets from saved samples:

```bash
conda run --no-capture-output -n ltx python \
  ../expr/onestep_avatar/d1_selfrollout_sigma_sweep_20260926/sheets.py \
  --manifest <SAVED_MEDIA_DIRECTORY>/manifest.json --output <FRESH_SHEET_DIRECTORY>
```

The report reader checks hashes and does not open models or repair missing
media. These recipes do not establish native VAE parity. The old analyzer and
generation launchers have been retired after package/report caller migration.

Check completed saved decoding without opening a decoder or discovering GPUs:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.sigma_sweep \
  --spec <SAVED_SWEEP_SPEC> --output <SAVED_MEDIA_DIRECTORY> --verify
```

Schema-two completion requires all ten movies and 120 samples with actual
coverage/geometry checks, current input/VAE/software bindings and saved metric
controls. The report reader runs this same check, including movies it does not
display in a sheet. Old schema-one control artifacts remain historical evidence.

### Historical future-noise study preparation

The package's `future_noise_study` CLI prepares data and evaluation jobs only.
It requires paths to the original study manifest, saved A/B noise archive,
prefix block-noise archive and legacy subset. It never starts a model or queue.
Run from `LTX-2` in the `ltx` environment:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.future_noise_study \
  --manifest <ORIGINAL_STUDY_MANIFEST> --noise <SAVED_NOISE_AB> \
  --blocks <ORIGINAL_BLOCK_NOISE> --subset <ORIGINAL_SUBSET> \
  --output <FRESH_PREPARATION_DIRECTORY>
```

Current real preparation refuses a changed guide-render pin in `t2r2.json`.
Do not remove it to bypass conversion checks. The old launcher remains pending
checked data and parity; original results remain historical evidence. See the
[preparation design](../doc/future_noise_study.md) for the exact noise slicing,
five jobs and seven result roles.

For an already prepared directory, check original and derived bytes without
starting a model or writing files:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.future_noise_study \
  --verify-preparation <PREPARATION_DIRECTORY>
```

The version-two preparation record binds all seven derived files and the
current producer. Verification also reconstructs noise, membership/frame plan
and exact job settings from the original inputs. This does not establish
native generation or parity with historical raw results.

`<V2_SUBSET>` is a checked version-two fixed video list, not an old block-chain
subset. Convert an old list without changing the original or its masters:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.subset \
  --convert <OLD_SUBSET> --output <NEW_V2_SUBSET> \
  --frame-plan-output <REPRODUCTION_PLAN> --require-guide
```

Omit `--require-guide` only for capture-only D0. A paired D0/D1 comparison uses
the same checked capture/guide list. Conversion preserves groups and old sample
ranges; each new mode writes its own frame plan. Use `--frame-plan` only when
its explicit reproduction geometry matches the requested mode.

Legacy adapter conversion uses a reviewed version-two contract and exact
original records. It copies tensors unchanged into a new derived file:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.training.checkpoints \
  --source <OLD_ADAPTER> --output <NEW_DERIVED_ADAPTER> --contract <REVIEWED_CONTRACT_JSON> \
  --original-config <ORIGINAL_CONFIG> --original-subset <OLD_SUBSET> \
  --membership <NEW_V2_SUBSET> --frame-plan <REPRODUCTION_PLAN> --base <BASE_CHECKPOINT>
```

This checks conditions and unused-history evidence rather than inferring a
mode from the run name. Random-window legacy runs require a matching plan
`start_draw` and a reviewed contract with `data.segment_selection`: window
templates, checked master lengths, allowed starts, the original seed key and G9.
Parent-initialized legacy runs still need additional evidence support.
The original config and adapter must explicitly
stamp the same sigma draw rule; missing stamps are refused, including step zero.
Historical mixed-level runs cannot be assigned today's uniform rule by default.
Original adapters, configs, subsets and masters
remain unchanged. This command does not resume training or recalibrate weights.

For a new training run, choose a fresh output directory. Repeat the shared
Accelerate command above and replace its training arguments with one of these:

```text
-m scripts.onestep_avatar.train --mode bidirectional \
  --subset <V2_SUBSET> --output <FRESH_OUT> --model 2.5 --variant dev \
  --guide-mode d1 --objective white --span-latent-frames 17 \
  --start-policy clip_start --sigma0 0.725 \
  --lora-rank 8 --lora-alpha 8 --lora-target attn --seed 42

-m scripts.onestep_avatar.train --mode causal \
  --subset <V2_SUBSET> --output <FRESH_OUT> --model 2.5 --variant dev \
  --guide-mode d1 --objective white --block-latent-frames 2 \
  --blocks-per-sample 3 --context-latent-frames 8 --sigma0 0.725 \
  --lora-rank 8 --lora-alpha 8 --lora-target attn --seed 42
```

D0 changes only `--guide-mode d0`; its target remains the capture. For causal
capture-history training, add `--teacher-forcing`. Bidirectional commands reject
block, cache, and teacher-history options. Switching modes starts a separate
run. A parent adapter initializes a fresh stage; it does not resume optimizer
state or establish calibration in the new mode.

`--corpus-root` is optional when the list's recorded corpus path resolves.
`--dry-run` verifies data, geometry, schedules and adapter conditions without
loading model weights or writing a run directory. Distilled weights require
supported nonzero sigma levels; state `--variant distilled` explicitly when
using that base. LoRA alpha must equal rank. Anchor options are unsupported by
the typed runtime.

Multi-step evaluation and inference use native Euler arithmetic at positive next
levels. The final zero step returns the prediction exactly. This direct endpoint
can differ from the stock bf16 reconstructed endpoint; native stock acceptance
is still open in [G11](../doc/known_gaps.md#g11--euler-rounding-differs-from-the-stock-step).
Older fixed-input records remain historical after a sampling source change;
prepare fresh records for current execution instead of changing their hashes.

For the separate 17-frame base sampling diagnostic, use an independently encoded
image and a fixed text record. This runs the actual stock video sampling
components with audio absent, plus a repeated control and the bidirectional
sampler. It does not run joint audio-video generation or D1 guide mixing:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.stock_parity \
  --first-image <PREPARED_IMAGE>/image.pt --text-record <FIXED_PREVIEW>/preview.json \
  --output <FRESH_STOCK_CHECK> --frames 17 --steps 4 --seed 42 --gpu-id <FREE_GPU>
```

Add `--dry-run` to validate inputs and print the predeclared protocol without
model/GPU work or output writes. The schedule uses the native scheduler without
a latent argument. The current matching path explicitly uses float32 global
sigma. Inspect repeated-control, intermediate-call, raw and decoded evidence
before accepting it. Ordinary product precision, adapters and the seven-frame
pilot remain separate checks. See [the module design](../doc/stock_parity.md).

To check ordinary global-sigma precision, add `--reference-run
<CHECKED_STOCK_CHECK>` and choose a fresh output. This reuses verified stock
controls and fixed noise and generates only the ordinary arm. The corrected
shared default is float32 global sigma and token timesteps. Full fresh checks
run stock, stock repeat, explicit float32 and ordinary default together. Changed
computation owners, runtime or inputs refuse reuse; historical bf16 diagnostics
retain their original software identity. New adapter/execution contracts bind
precision, and unknown historical calibration refuses execution even with a
research override. Fresh native acceptance remains tracked in
[G12](../doc/known_gaps.md#g12--ordinary-global-sigma-loses-stock-precision).

## 3. Evaluation and previews

Ordinary evaluation writes encoded results and numeric records. Prepare RGB
references with `media` and fixed preview records with `prepare_inputs`.
The preview-job route generates and renders using those pinned records;
ordinary raw evaluation does not invent missing references:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.evaluate \
  --mode bidirectional --subset <V2_SUBSET> --source <SOURCE_ID> \
  --model 2.5 --variant dev --guide-mode d1 --span-latent-frames 17 \
  --schedule 0.725 0 --checkpoint <V2_ADAPTER> --include-base \
  --noise-file <SAVED_BF16_TOKEN_NOISE> --output <FRESH_OUT> --gpu-id <GPU_ID>
```

Use the causal mode and its explicit block/cache settings for a causal adapter.
The `--output-latent-frames` option separates a causal output prefix
from its adapter's recorded `--span-latent-frames`. Omission preserves current
selection. Explicit output must be positive, fit complete causal blocks and fit
the master. Bidirectional mode rejects it. Strict adapter checks and saved-noise
shape checks still run before model sessions; the option is not an override.
See [the coverage design](../doc/evaluate.md#causal-physical-output-coverage).
A saved noise file belongs to one video and must match its full token range.
Adapters are checked before loading weights. Research overrides are recorded;
product inference does not permit them. Do not reinterpret historical G7/G8
results as fixed by the refactor.

For a causal adapter trained with clean sigma-zero cache refresh, normal
evaluation uses `--history-mode cache --kv-source refresh` (the defaults).
`--history-mode recompute`, `--history-mode joint`, or cached generated history
with `--kv-source denoise` changes the adapter computation and requires
`--research-override`. Results record both requested fields and every difference.
Base-only diagnostics record the choices without an adapter override.

Normal adapter evaluation now defaults to
`--adapter-application peft_unmerged_fp32`, sharing training's unmerged fp32
PEFT function against frozen bf16 base weights. Product inference always uses
that method. To measure bf16 fusion as a changed research condition, pass
`--adapter-application fused_bf16 --research-override` in evaluation; the
preflight and saved results record the method difference. Neither ordinary
evaluation nor product silently falls back to fusion on a memory failure.
Full E2 views/trained steps and measured-cost acceptance remain required.

Prepare the three fixed reference roles with the package's decoder-only command:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.media \
  --prepare-training-references --subset <V2_SUBSET> --source <SOURCE_ID> \
  --encoded-frames 7 --guide-mode d1 --model 2.5 --seed 42 \
  --output <FRESH_REFERENCE_DIRECTORY> --gpu-id <FREE_GPU>
```

Check GPU availability before executing. Use `--guide-mode d0` for a capture-only
list. An absent guide appears as `Guide not used`; a recorded guide is still
checked. The command uses the saved crop and background and creates
`references.json` with pinned RGB files. It performs no transformer or text work.
This is the reference bundle, not the complete fixed preview record. That record
also pins capture, optional guide, first-image, text and noise identities and
evaluation arguments; bind `reference_bundle.path` to the absolute manifest path
and `reference_bundle.sha256` to its file hash. `prepare_inputs preview`, shown
above, produces this complete fixed record through the public checked path.

Training can pin `--preview-inputs <FIXED_PREVIEW_RECORD>`. A completed adapter
save may enqueue a preview job. Its raw generation stage runs outside training:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.evaluate \
  --preview-job <JOB_JSON> --gpu-id <GPU_ID>
```

With a pinned reference bundle, this command decodes the saved outputs, renders
the training panels, and completes the job after checking the media evidence.
Without that bundle, raw generation leaves the job running until checked
rendering evidence is supplied. The renderer selects the normal or compact
training layout from the actual labels before decoding; labels that fit neither
fail before opening the decoder. Full native preview acceptance remains pending.
A failure records its reason without invalidating the checkpoint.
See [evaluation](../doc/evaluate.md) and [media](../doc/media.md) for evidence
requirements and exact panel layouts. Report code consumes saved results only.

### Generation benchmark

Historical saved-encoding measurements use the package owner directly:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.evaluate \
  --saved-metrics <PROBE_DIRECTORY>
```

Add `--long-metrics` for complete longer two-frame block sequences. This route
checks saved output hashes and opens no model/VAE session. It preserves the
historical generated-frame metric definitions, which differ from training's
full-frame loss.

Use ordinary evaluation arguments with `bench`, plus measured repetitions and
discarded warmup. For example, benchmark a checked bidirectional segment:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.bench \
  --mode bidirectional --subset <VIDEO_LIST> --source <SOURCE_ID> \
  --variant dev --guide-mode d1 --schedule 0.725 0 --span-latent-frames 17 \
  --noise-file <FIXED_NOISE> --output <NEW_OUTPUT> --repetitions 3 --warmup 1
```

Each ordinary result includes `benchmark` measurements and output identities.
The measured boundary includes sampling and complete CPU encoding, excludes
loading/decoding/writes, and uses a fresh cache per causal repetition. One extra
untimed artifact call must match the measured outputs. Synthetic causal
operation timing requires `--operation-timing --mode causal` and does not
measure complete-output latency. Real-weight performance acceptance is pending.

### Future-noise diagnostic

Use saved native-bf16 token tensors for exactly one selected video. The second
tensor keeps all tokens before the boundary unchanged and changes later tokens:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.evaluate \
  --mode causal --subset <VIDEO_LIST> --source <SOURCE_ID> \
  --variant dev --guide-mode d1 --schedule 0.421875 0 \
  --block-latent-frames 2 --context-latent-frames 8 --span-latent-frames 17 \
  --noise-file <ORIGINAL_NOISE> --changed-noise-file <CHANGED_NOISE> \
  --future-noise-start 9 --checkpoint <V2_ADAPTER> --output <NEW_OUTPUT>
```

This compares earlier encoded frames zero through eight against later frames
nine through sixteen. Results include both encodings, their provenance and
`future_noise.json`. It tests dependence on future noise; it does not establish
capture fidelity or universal cache/recalculation equality.

### Historical eight-block diagnostic

For the saved white D1 diagnostic, the package owns the session and two
generated-history rollouts. It checks the adapter's recorded conditions before
opening weights. Masters need at least 17 encoded frames; global noise draws
retain the entire recorded master. Use a new output path:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.evaluate \
  --causality --checkpoint <ADAPTER> --view <VIEW> --sigma 0.421875 \
  --gpu-id <FREE_GPU> --output <FRESH_JSON>
```

This preserves the earlier diagnostic's sigma, seed pair 42/99, B2/D8/sink1,
first eight blocks, and changed noise after frame eight. It reports earlier
bit equality and later differences; it does not establish capture fidelity.

### Product review panels

Prepared guide and supplied-image encodings must have matching crop, background,
frame rate and VAE provenance. Generate and save raw output before decoding:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.infer \
  --mode bidirectional --guide <GUIDE_PT> --first-image <IMAGE_PT> \
  --variant dev --schedule 0.725 0 --output <NEW_OUTPUT> \
  --decode --review --poster-frame 0
```

The generated-only video stays separate from the review under `review/`.
The review shows the decoded supplied-image still, decoded guide and generated
video. Its labels identify VAE reconstructions; it adds no capture metrics.
Omit `--review` for generated-only media. Product adapter checks permit no
research override. Real-weight product/review acceptance remains pending.

## 4. Remaining migration

### Bounded first-update numerical reference

For the native E4 check, use the ordinary explicit-mode four-process training
recipe with `--steps 1 --save-initial --save-update-state --chains-per-rank 2`
and no preview input. Keep gradient checkpointing enabled. The existing queue
uses the fixed training pool 0–3; idle GPUs outside that pool do not change it.
`--save-update-state` writes the actual training text and named fp32 Adam moments.
It changes evidence only and is not a resume option.

After the distributed job completes, save its exact `arguments` in a JSON job.
Run the bounded serial reference in the `ltx` environment on one independently
registered free device, with `CUDA_VISIBLE_DEVICES` set to that physical device:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.training_update_check \
  --job <ORIGINAL_TRAIN_JOB_JSON> --world-size 4 --output <FRESH_SERIAL_DIRECTORY>
```

`--dry-run` verifies the completed distributed inputs/logs/exports/moments and
prints fixed visits and tolerances without opening a model or writing output.
Actual replay compares gradients, norms, loss, export and distributed step-one
reload. A failed comparison exits 2 and retains measured failed evidence.
Native previews/product and perceptual quality are separate acceptance gates.

Review a normalized package queue without starting children:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.queue \
  --jobs <QUEUE_JSON> --state <STATE_JSON> --dry-run
```

It checks package arguments and persisted job identities, then prints planned
commands. It creates no state/output directories, claims no GPU and starts no
child. Planned GPU IDs do not prove availability. Scientific input preflight
and artifact verification are separate from this review.

Saved-comparison rendering uses a `render` job with no model-generation mode:

```json
{"id":"comparison", "kind":"render", "arguments":["--render-saved-comparisons","spec.json","--output","fresh_media","--seed","42"], "output":"fresh_media", "dependencies":[], "completion":{"manifest":"fresh_media/render_manifest.json"}}
```

The version-one list wraps this record in `jobs`. Paths are relative to that
list's directory. Queue preparation pins the spec bytes. Jobs wait for all
saved panel files before acquiring one allowed GPU. Completion requires the
exact requested spec, current decoder/software identity and input/media hashes;
an old unbound render manifest is not adopted. Reports remain separate readers.
The development study's converted list is
`../expr/onestep_avatar/dev_training_20261001/configs/package_render_jobs.json`.
It preserves eight specifications and seed 42, with fresh outputs beneath the
study's `media/videos/package/`. Review it with `--dry-run`; this recipe does
not start a campaign or replace historical media.

Dispatch normalized jobs with direct GPU queries and one shared process ledger:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.queue \
  --jobs <QUEUE_JSON> --state <STATE_JSON> --execute --loop \
  --process-ledger <SHARED_PROCESS_LEDGER.json> --poll-seconds 30
```

Use `--once` instead of `--loop` for at most one dispatch. Every current queue
shares the single `<SHARED_PROCESS_LEDGER.json>` file, currently
`expr/onestep_avatar/processes.json`, and queries `nvidia-smi` directly. Historical
`runs/.gpu_claims` files remain unchanged evidence and are not read by new attempts.
The loop waits without an active own-process record, accepts appended jobs with
unchanged prior identities, and exits
only after verified completion. Failed or running journal entries stop the
loop. Explicit recovery checks terminated child handles and saved outputs:

```bash
conda run --no-capture-output -n ltx python -m scripts.onestep_avatar.queue \
  --jobs <QUEUE_JSON> --state <STATE_JSON> --recover
```

Recovery needs an existing journal. It refuses live children, unresolved launch
windows and changed output content. It neither starts processes nor closes
active process records nor retries failed work. Typed training has a separate automatic
startup contention path: at most three retries, with original failed outputs
and logs preserved under `superseded_startup_contention/`. A token-bound
CUDA OOM or typed port-in-use event must precede every rank's update boundary;
surviving workers and any numeric update forbid retry. Other failures stop.
`--once` returns 2 when a preserved retry is pending and starts no second child.
See [retry evidence](../doc/queue.md#startup-contention-retries). Historical text
job lists and mode-less jobs need conversion before these commands can execute
them. These recipes do not restart an existing queue or start a campaign.

The old GPU-count YAML copies were removed after their training caller moved.
Live training uses its own recorded forward-prefetch setting. Do not silently
substitute the new template for that study. The trainer package's Accelerate
configuration is unchanged.

Old diagnostic and campaign commands are historical evidence, not current
package recipes. Their saved inputs, outputs and attribution stay intact during
execution-owner migration. Joint-history, recalculation and guidance diagnostics
still need complete CLI migration before they can replace all older callers.
`fsdp_forward_prefetch.yaml` is the package-owned four-process template for
converted historical whole-clip jobs. It preserves the old forward/backward
prefetch and full-shard settings. The queue supplies `--num_processes 4` and
the recorded port. Historical configs under `expr/` remain provenance inputs;
they are not imported by package execution.
