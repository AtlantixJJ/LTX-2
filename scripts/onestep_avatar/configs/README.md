# Run guide — workflows and experiments

This guide is the entry point for running the package. It gives one recipe for
each workflow and each experiment. The [package README](../README.md) explains
the terms and the data flow. The [design index](../doc/README.md) links the
module designs.

## 1. Rules for every run

**Directory and environment.** Run every command from the LTX-2 root as a
module: `python -m scripts.onestep_avatar.<module>`. Use the `ltx` conda
environment. Only guide rendering (`corpus.build_guidance`) uses `argavatar`.
Call the environment's interpreter directly for long runs; `conda run` buffers
output until the process ends.

```bash
cd LTX-2
PY=$(conda run -n ltx python -c 'import sys; print(sys.executable)')
```

**Who starts GPU work.**

- The package queue (`execution.queue`) starts training, evaluation, previews,
  product generation, decoding, comparison rendering, cost measurement and every
  experiment that loads a model. Do not start these modules directly; their
  `--gpu-id` belongs to the queue.
- Corpus production (`corpus.precompute`, `corpus.build_guidance`) and input
  preparation (`media --prepare-training-references`, `prepare_inputs`) run
  directly on one free GPU that you select.
- CPU commands run directly: `corpus.subset`, `plot_training`,
  `experiments.saved_probe_metrics`, every `--dry-run` and every `--help`.

**GPUs.** The pool is GPUs 0–7. A GPU is free when `nvidia-smi` shows less
than 1,024 MiB in use. Check before every direct run; the queue checks before
every launch. Training uses exactly four free GPUs, which need not be adjacent.
Never use a busy GPU and never stop another user's process.

**Process ledger.** Every queue owner records its own processes in one shared
file, `../expr/onestep_avatar/processes.json`: PID, start ticks, command, GPUs
and owned descendants. Pass this same path to every queue owner.

**Long runs survive the terminal.** Start a queue owner or a long corpus run in
its own session, with a log file:

```bash
setsid -f "$PY" -m scripts.onestep_avatar.execution.queue --jobs <JOBS.json> --state <STATE.json> \
  --execute --loop --process-ledger ../expr/onestep_avatar/processes.json \
  > <LOG_FILE> 2>&1 < /dev/null
```

A queue owner that ends with its terminal cannot publish its supervisor result.
The queue then records the attempt as failed with reason `missing_supervision`,
even when the scientific output is complete.

**Outputs.** Every run writes a new output directory. Existing outputs are
refused, not overwritten. Keep data, weights, media and logs out of Git.

## 2. The queue

### Job list

A job list is one JSON file:

```json
{"schema_version": 1, "jobs": [ {"id": "...", "kind": "...", "...": "..."} ]}
```

Relative paths resolve against the job list's directory. Each job has:

| Field | Meaning |
|---|---|
| `id` | Unique job name. |
| `kind` | `train`, `evaluate`, `preview`, `product`, `bench`, `decode`, `render` or `experiment`. |
| `arguments` | The module's command-line arguments, including exactly one `--output`. |
| `output` | The same output path as in `arguments`. |
| `dependencies` | Earlier job IDs that must be complete first (optional). |
| `completion` | The files that prove completion (see each kind below). |
| `resource_budget`, `resource_budget_sha256` | Optional budget file for a non-training job. The default is 1,800 seconds per phase and 48,000,000,000 allocated bytes. |
| `numerical_policy` | Optional `"deterministic"`: the queue applies the deterministic numerical environment to the child. |

`arguments` cannot contain `--gpu-id`, `--dry-run`, `--help`, `--overwrite`,
`--saved-metrics` or `--verify`; only a `preview` job may contain
`--preview-job`. The queue chooses the GPU.

A budget file can also set `overall_seconds`. The clock starts at the first
attempt and survives startup retries and queue restarts. When it expires, the
queue stops the job's processes and records reason `overall_deadline`.

### Commands

```bash
# Check identities and print the planned commands; writes nothing and claims no GPU.
"$PY" -m scripts.onestep_avatar.execution.queue --jobs <JOBS.json> --state <STATE.json> --dry-run

# Run ready jobs until every job is complete (use setsid, see section 1).
"$PY" -m scripts.onestep_avatar.execution.queue --jobs <JOBS.json> --state <STATE.json> \
  --execute --loop --process-ledger ../expr/onestep_avatar/processes.json

# Start at most one job, then exit.
"$PY" -m scripts.onestep_avatar.execution.queue --jobs <JOBS.json> --state <STATE.json> \
  --execute --once --process-ledger ../expr/onestep_avatar/processes.json

# After an owner ended: check ended children and saved outputs; starts nothing.
"$PY" -m scripts.onestep_avatar.execution.queue --jobs <JOBS.json> --state <STATE.json> --recover
```

How the queue behaves:

- One queue owner runs one child at a time. To use several GPUs at once, run
  separate owners on separate job lists and state files with the same ledger.
- Ready non-training jobs start first. A job whose GPUs are busy waits; later
  ready jobs can start before it.
- You can append jobs to a running list. Existing job entries cannot change.
- `--loop` stops with exit 1 when any job has failed, and refuses a state with a
  `running` job until `--recover` has checked it. A failed job stays failed; add
  a new job with a new ID and a new output to retry it.
- Training startup contention (CUDA out of memory or a busy rendezvous port
  before any update) is retried at most three times. Earlier outputs move to
  `superseded_startup_contention/`. `--once` exits 2 when such a retry is pending.

### Job records by kind

`train` — four-process Accelerate training:

```json
{"id": "train_bidirectional", "kind": "train",
 "arguments": ["--mode", "bidirectional", "--subset", "/abs/membership.json", "--output", "/abs/run",
               "--variant", "dev", "--guide-mode", "d1", "--objective", "white", "--steps", "60"],
 "output": "/abs/run", "processes": 4, "accelerate_config": "/abs/LTX-2/scripts/onestep_avatar/configs/fsdp.yaml",
 "port": 29785, "completion": {"step": 60, "checkpoint": "/abs/run/checkpoints/lora_weights_step_00060.safetensors"}}
```

Use a unique `port` for each training job that can run at the same time. Put a
resource budget in the arguments as `--resource-budget <BUDGET.json>`.

`evaluate` — `"completion": {"records": [...]}` lists every
`<output>/case_NNNN/variant_NNN/result.json` that the arguments select.

`preview` — arguments `["--preview-job", "<PREVIEW_JOB.json>", "--output", "<OUT>"]`;
completion `{"records": ["<OUT>/preview_completion.json"]}`.

`product` — completion `{"records": [...]}` lists `<OUT>/result.json`, plus
`<OUT>/rendering.json` with `--decode` and `<OUT>/review/rendering.json` with `--review`.

`bench` — `"completion": {"records": [...]}` lists the same
`<output>/case_NNNN/variant_NNN/result.json` files that evaluation would write;
the verifier derives the exact inventory and refuses a different list. Causal
operation timing lists `<output>/operation_timing.json`.

`decode` — arguments `["--jobs", "<DECODER_JOBS.json>", "--output", "<OUT>"]`;
completion `{"manifest": "<OUT>/manifest.json"}`.

`render` — arguments `["--render-saved-comparisons", "<SPEC.json>", "--output", "<OUT>", "--seed", "42"]`;
completion `{"manifest": "<OUT>/render_manifest.json"}`. The job waits until
every saved panel file exists.

`experiment` — one selector from section 9:

```json
{"id": "e1_stock", "kind": "experiment", "experiment": "stock_parity",
 "spec": "/abs/stock_parity_spec.json", "spec_sha256": "<sha256 of the spec file>",
 "arguments": ["--spec", "/abs/stock_parity_spec.json", "--output", "/abs/out/e1_stock"],
 "output": "/abs/out/e1_stock", "completion": {"manifest": "/abs/out/e1_stock/manifest.json"},
 "numerical_policy": "deterministic"}
```

Most experiment specs use one shape:

```json
{"schema_version": 1, "protocol": "<selector>", "arguments": ["--scientific", "arguments"]}
```

Scientific arguments use absolute paths and never contain `--output`,
`--gpu-id` or `--dry-run`. Some checkers also accept `"operation"`.

## 3. Prepare the corpus

The corpus root defaults to `data/AnimatableHuman/DNARenderingVideo` in the
workspace. The order is fixed: capture first, then guides, then guide encodes.

1. **Encode capture masters and record crop boxes** (one free GPU):

   ```bash
   "$PY" -m scripts.onestep_avatar.corpus.precompute --process_gt_latent \
     --objective bg white --model 2.5 --gpu-id <FREE_GPU>
   ```

   This writes `ltx_vae_latent.pt` (`bg`) and `ltx_vae_latent_white.pt`
   beside each view's `rgb.mp4`, the lossless capture masks and
   `capture_latent_manifest.json` (the only crop-box producer) at the corpus
   root. A rerun encodes only what is missing. Lower `--crop-workers` when host
   memory is tight. `--limit N` and `--dry-run` bound a trial.

2. **Render guides** (`argavatar` environment, one free GPU with at least
   20,000 MiB free):

   ```bash
   CUDA_VISIBLE_DEVICES=<FREE_GPU> conda run --no-capture-output -n argavatar \
     python -u -m scripts.onestep_avatar.corpus.build_guidance \
     --objective white --driving-views 1 5 --limit 8 --visualize --device cuda:0
   ```

   Inspect the QA overlays under each clip's `qa/` before a larger batch.
   `--limit 0` renders every remaining pair. `--clips` selects clip names.

3. **Encode guide masters** (one free GPU):

   ```bash
   "$PY" -m scripts.onestep_avatar.corpus.precompute --process_syn_latent \
     --objective white --model 2.5 --gpu-id <FREE_GPU>
   ```

4. **Create a fixed video list (membership)** on CPU:

   ```bash
   "$PY" -m scripts.onestep_avatar.corpus.subset \
     --objective white --require-guide --splits <PERSON_GROUPS.json> --output <NEW_MEMBERSHIP.json>
   ```

   The groups file maps group names (for example `train`, `held_out`) to
   disjoint person-ID lists. Without `--splits`, the command forms
   `train`/`held_out` groups from `--held-out-fraction` and `--min-held-out`.
   `--source <VIEW_ID>` (repeatable) selects exact views; `--max-people` caps the
   person count. Use `--require-guide` for D1 and for paired D0/D1 work. The
   command pins every selected file's hash and refuses an existing output.

## 4. Train an adapter

Training is a queue `train` job (section 2). The arguments select one mode.
Check them on CPU first; a dry run reads the membership, masters, plans and
conditions, loads no weights and writes nothing:

```bash
"$PY" -m scripts.onestep_avatar.train --mode causal --subset <MEMBERSHIP.json> \
  --output <FRESH_RUN> --variant dev --guide-mode d1 --objective white --dry-run
```

Bidirectional arguments (one segment, no cache):

```text
--mode bidirectional --subset <MEMBERSHIP.json> --output <FRESH_RUN>
--model 2.5 --variant dev --guide-mode d1 --objective white
--span-latent-frames 7 --start-policy clip_start --sigma0 0.725
--lora-rank 8 --lora-alpha 8 --lora-target attn --lr 1e-4 --warmup-steps 20
--max-grad-norm 1 --steps 60 --save-initial --save-every 20 --seed 42 --noise-policy fresh
```

Causal arguments (blocks with a bounded cache):

```text
--mode causal --subset <MEMBERSHIP.json> --output <FRESH_RUN>
--model 2.5 --variant dev --guide-mode d1 --objective white
--block-latent-frames 2 --blocks-per-sample 3 --context-latent-frames 8 --start-policy clip_start
--sigma0 0.725 --lora-rank 8 --lora-alpha 8 --lora-target attn --lr 1e-4 --warmup-steps 20
--max-grad-norm 1 --steps 60 --save-initial --save-every 20 --seed 42 --noise-policy fresh
```

| Option | Meaning |
|---|---|
| `--guide-mode d0` | Noise the capture instead of the guide (capacity test). The target stays the capture. |
| `--objective bg` | Use the real background instead of white. |
| `--variant distilled` | Use distilled base weights; sigma must be on its grid. |
| `--sigma-levels A B ...` | Draw one sigma per update and GPU process from these levels. |
| `--teacher-forcing` | Causal only: refresh the cache from the capture target. |
| `--span-latent-frames N` | Train on N encoded frames per segment. |
| `--start-policy random` | Start segments at random encoded frames (see known gaps). |
| `--chains-per-rank N` | Samples accumulated per GPU process per update. |
| `--init-adapter <ADAPTER>` | Start a new stage from a parent adapter's weights. This is not a resume. |
| `--preview-inputs <preview.json>` | Queue a fixed preview after each saved checkpoint (section 6). |
| `--resource-budget <BUDGET.json>` | Bound each phase's time and allocated memory. |
| `--save-update-state` | Save training text and Adam moments for update checks (E4). |
| `--no-wandb` | Disable Weights & Biases logging. |

Bidirectional mode refuses block, cache and teacher-forcing options. A mode
change needs a new run and output directory. LoRA alpha must equal rank.
Outputs are `config.json`, per-rank `metrics_rank*.jsonl` logs and
`checkpoints/lora_weights_step_NNNNN.safetensors`; each adapter stores its
complete training conditions in metadata.

`configs/fsdp.yaml` is the Accelerate configuration for four processes.
`configs/fsdp_forward_prefetch.yaml` adds forward/backward prefetch for jobs
that record that setting. The queue passes `--num_processes 4` and the port.

## 5. Evaluate an adapter

Evaluation is a queue `evaluate` job. It saves encoded outputs and result
records; decoding is separate (section 8). Example arguments:

```text
--mode causal --subset <MEMBERSHIP.json> --split held_out --source <VIEW_ID>
--model 2.5 --variant dev --guide-mode d1 --objective white --schedule 0.725 0
--block-latent-frames 2 --blocks-per-sample 3 --context-latent-frames 8
--checkpoint <ADAPTER.safetensors> --include-base --noise-file <SAVED_NOISE.pt> --output <FRESH_OUT>
```

- The mode and settings must match the adapter's recorded conditions. The check
  runs before any weights load.
- `--include-base` adds the base model without the adapter as a second variant.
- `--noise-file` reuses saved token noise for one video; it must match the full
  token range.
- `--output-latent-frames N` (causal only) sets the generated length in complete
  blocks without changing the adapter's recorded training span.
- Diagnostics that change the adapter's computation need `--research-override`,
  which the results record: `--history-mode recompute|joint`, `--kv-source
  denoise` and `--adapter-application fused_bf16`. The defaults are `cache`,
  `refresh` and `peft_unmerged_fp32`.
- Check the arguments on CPU with `"$PY" -m scripts.onestep_avatar.evaluate <ARGUMENTS> --dry-run`.

## 6. Fixed training previews

A preview generates and renders the same fixed inputs at each saved checkpoint.

1. Prepare reference pixels (one free GPU; decoder only):

   ```bash
   "$PY" -m scripts.onestep_avatar.media --prepare-training-references \
     --subset <MEMBERSHIP.json> --source <VIEW_ID> --encoded-frames 7 --guide-mode d1 \
     --model 2.5 --seed 42 --output <FRESH_REFERENCES> --gpu-id <FREE_GPU>
   ```

2. Pin the fixed inputs (one free GPU; VAE and text encoder only):

   ```bash
   "$PY" -m scripts.onestep_avatar.prepare_inputs preview \
     --references <FRESH_REFERENCES>/references.json --output <FRESH_FIXED> --gpu-id <FREE_GPU> \
     --evaluation-arguments --mode bidirectional --subset <MEMBERSHIP.json> --source <VIEW_ID> \
     --variant dev --guide-mode d1 --span-latent-frames 7 --schedule 0.725 0 --include-base
   ```

   For a causal adapter, use the causal settings it was trained with and
   `--output-latent-frames` for the generated length.

3. Pass `<FRESH_FIXED>/preview.json` to training as `--preview-inputs`. Each
   saved checkpoint then writes a preview job; run it as a queue `preview` job.
   The result shows capture RGB, VAE-decoded capture, guide, base and adapter
   panels with matched frames. A failed preview does not change training.

## 7. Product generation from a supplied image

1. Encode one real RGB image in the guide crop's original canvas (one free GPU).
   The `white` objective needs a matching grayscale matte; `bg` refuses one:

   ```bash
   "$PY" -m scripts.onestep_avatar.prepare_inputs supplied-image \
     --image <RGB_IMAGE> --mask <GRAYSCALE_MATTE> --guide <WHITE_GUIDE_BUNDLE.pt> \
     --output <FRESH_INPUT> --gpu-id <FREE_GPU> --review
   ```

2. Run a queue `product` job (`infer`) with arguments:

   ```text
   --mode bidirectional --guide <GUIDE_BUNDLE.pt> --first-image <FRESH_INPUT>/image.pt
   --checkpoint <D1_ADAPTER.safetensors> --variant dev --schedule 0.725 0 --span-latent-frames 7
   --seed 42 --decode --review --poster-frame 0 --output <FRESH_OUT>
   ```

Product uses D1, the supplied image as `c0`, generated history and the
unmerged adapter. It refuses research overrides and has no capture target.
`--decode` writes the generated video; `--review` adds a separate panel video
with the supplied image, guide and output.

## 8. Decoding, comparison media, cost and plots

- **Decode saved encodings**: a queue `decode` job (`decode_saved`) reads a
  decoder job list and writes videos without running the transformer.
- **Render saved comparisons**: a queue `render` job (`comparisons`) builds
  matched comparison videos from a saved comparison specification.
- **Measure generation cost**: a queue `bench` job takes evaluation arguments
  plus `--repetitions 3 --warmup 1` and records warmed latency and peak memory.
  The measured time covers sampling and CPU encoding, not loading, decoding or
  writes. Causal operation timing (`--operation-timing --mode causal --reps N
  --warmup N`) times single operations and does not measure complete-output latency.
- **Plot training curves** (CPU):

  ```bash
  "$PY" -m scripts.onestep_avatar.plot_training --run <RUN_A> --run <RUN_B> --label A --label B
  ```

## 9. Experiments

Each experiment is a queue `experiment` job with a pinned spec (section 2).
Check a spec on CPU without loading a model:

```bash
"$PY" -m scripts.onestep_avatar.experiments.<selector> --spec <SPEC.json> --output <FRESH_OUT> --dry-run
```

The acceptance gates E1–E5 are defined in
[verification](../doc/verification.md#acceptance-gates).

### E1 — stock sampler parity (`stock_parity`)

Compares the stock LTX video sampler with the package sampler on the same
weights, first image, text, saved noise and schedule.

```json
{"schema_version": 1, "protocol": "stock_parity",
 "arguments": ["--first-image", "/abs/image.pt", "--text-record", "/abs/text.json", "--frames", "17", "--steps", "4", "--seed", "42"]}
```

Add `"--reference-run", "/abs/checked_stock_run"` to reuse a verified stock run
and generate only the ordinary arm. See [design](../doc/experiments/stock_parity.md).

### E2 — learned adapter function (`adapter_effect_check`)

Compares the training reference, ordinary evaluation and product for base, zero
adapter and trained adapter under identical inputs.

```json
{"schema_version": 1, "protocol": "adapter_effect_check",
 "arguments": ["--job", "/abs/train_job.json", "--inputs", "/abs/fixed_inputs.json",
               "--update-check", "/abs/e4_reference/result.json",
               "--checkpoints", "/abs/run/checkpoints/lora_weights_step_00000.safetensors",
               "/abs/run/checkpoints/lora_weights_step_00001.safetensors", "--decode"]}
```

`--update-job` names the one-update E4 job when `--job` is a longer run;
`--subset` selects a separate checked evaluation membership. Declare the E4
reference job as a dependency. See [design](../doc/experiments/adapter_effect_check.md).

### E3 — causal continuation (`continuation_check`, `causality`)

`continuation_check` runs three phases on 17 encoded frames: `capture` saves
cached generation and layerwise K/V; `reference` recomputes from that capture's
snapshot; `control` runs one named control from a control plan.

```json
{"schema_version": 1, "protocol": "continuation_check",
 "arguments": ["--phase", "capture", "--protocol", "/abs/protocol.json", "--sigma", "1.0", "--history", "generated"]}
```

A `reference` job uses `--phase reference --snapshot <CAPTURE_OUT>/<snapshot>`
and depends on its capture job. A `control` job uses `--phase control
--protocol ... --control-plan ... --control-id ...`.

The short controls (seven frames, nine raw outputs: repeat, `c0`, prefix and
changed future noise at boundaries 3 and 5) use the same selector:

```json
{"schema_version": 1, "protocol": "continuation_check",
 "arguments": ["--phase", "short_controls", "--job", "/abs/causal_train_job.json",
               "--inputs", "/abs/fixed_inputs.json", "--update-check", "/abs/e4_causal_reference/result.json",
               "--checkpoints", "/abs/step0.safetensors", "/abs/step1.safetensors",
               "--changed-noise-3", "/abs/noise_changed_3.pt", "--changed-noise-5", "/abs/noise_changed_5.pt"]}
```

`causality` with `"protocol": "future_noise"` keeps earlier saved noise bytes,
changes later ones and runs ordinary evaluation twice. Its `arguments` are
causal evaluation arguments plus `--noise-file`, `--changed-noise-file` and
`--future-noise-start`. `"protocol": "eight_block"` takes the fields
`checkpoint`, `view` and `sigma` and needs an adapter whose recorded conditions
match eight causal blocks over 17 frames. See the
[continuation](../doc/experiments/continuation_check.md),
[short controls](../doc/experiments/short_causal_check.md) and
[causality](../doc/experiments/causality.md) designs.

### E4 — distributed training update (`training_update_check`)

1. Run the one-update `train` job: training arguments plus `--steps 1
   --save-initial --save-update-state --chains-per-rank 2 --resource-budget
   <BUDGET.json>`, with no preview input.
2. Queue a serial reference that depends on it:

   ```json
   {"schema_version": 1, "protocol": "training_update_check", "operation": "reference",
    "arguments": ["--job", "/abs/update_job.json", "--world-size", "4"]}
   ```

3. Queue a comparison that depends on the reference:

   ```json
   {"schema_version": 1, "protocol": "training_update_check", "operation": "compare",
    "arguments": ["--job", "/abs/update_job.json", "--world-size", "4", "--reference", "/abs/reference_out"]}
   ```

The comparison checks 768 named gradients, Adam moments, exports, loss, norm and
the step-one reload against relative L2 0.02 (near-zero RMS `1e-8`). A failed
comparison exits 2 and keeps its measurements. `training_slice_check` (operations
`run` and `compare`) localizes a mismatch on one GPU; it cannot pass E4.
See [design](../doc/experiments/training_update_check.md).

### E5 — bounded learning pilots

Run one 60-update `train` job per mode with the arguments in section 4, four
GPUs each and a budget with `overall_seconds: 7200`. Save updates 0, 20 and 60.
Then compare the saved adapters with `adapter_effect_check`, evaluation and
product jobs on held-out people and seeds, and measure cost with `bench`. The
fixed settings are in [verification](../doc/verification.md#acceptance-gates).

### Pruning baseline (`prune_baseline`)

Runs the prune package's whole-clip producer for one recorded case and checks
its saved noise and raw D0 outputs.

```json
{"schema_version": 1, "protocol": "prune_baseline",
 "arguments": ["--view", "/abs/corpus/Part_1/0008_01/views/view00_cam51", "--model", "2.5",
               "--variant", "distilled", "--objective", "white", "--sigmas", "0.725", "0.909375", "1.0",
               "--context-latent-frames", "8", "--prompt", "", "--seed", "42", "--raw-only", "--arms", "d0"]}
```

The spec allows exactly one view and one seed, raw output, white distilled
weights, no guidance and context eight. `--arms d0 d1` adds the D1 arm.

See [design](../doc/experiments/prune_baseline.md) and the producer
[`scripts/prune/doc/evaluate/whole_clip_baseline.md`](../../prune/doc/evaluate/whole_clip_baseline.md).

### Other studies

- `fusion_parity` — fused bf16 versus unmerged fp32 adapter output on one
  causal block. Spec: `{"schema_version": 1, "protocol": "fusion_parity", "run":
  "/abs/run", "view": "/abs/view", "step": 1}`.
- `sigma_sweep` — decodes and scores a result-bound sigma sweep. Its spec
  embeds the eight evaluation jobs whose verified results it decodes. Check a
  finished sweep with `"$PY" -m scripts.onestep_avatar.experiments.sigma_sweep
  --spec <SPEC> --output <OUT> --verify`.
- `stats` — capture/guide gap and latent statistics. Arguments include
  `--model 2.5`, `--renders` and/or `--pairs`, `--max-videos` and
  `--eps-samples`; `--no-gpu` runs only the pair measurements.
- `saved_probe_metrics` — CPU only, run directly:
  `"$PY" -m scripts.onestep_avatar.experiments.saved_probe_metrics --saved-metrics <PROBE_DIR> [--long-metrics]`.

## 10. Check the code

```bash
CUDA_VISIBLE_DEVICES= "$PY" -m pytest scripts/onestep_avatar/tests -q
CUDA_VISIBLE_DEVICES= ONESTEP_AVATAR_BLOCK_EXPERIMENTS=1 "$PY" -m pytest scripts/onestep_avatar/tests -q \
  --ignore=scripts/onestep_avatar/tests/experiments
CUDA_VISIBLE_DEVICES= "$PY" -m pytest scripts/prune/tests -q
uv run ruff check scripts/onestep_avatar scripts/prune
```

The second command proves that ordinary code runs with experiment imports
blocked. CPU checks do not establish native numerical results or video quality.
