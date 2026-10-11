# `training/config.py` — check run settings

`train.py` is the training CLI. It has fewer than 100 lines, so it has no
design document of its own. Its arguments are parsed here, so its Usage
section is in this document.

## Objective

Resolve one explicit mode and check all settings before model weights load or
output files change. The configuration comes from the command line only. There
is no experiment YAML loader and no implicit resume.

`parse_settings` makes a `RunSettings` value with a `BidirectionalSettings` or
`CausalSettings` value. It is the only training parser. `train.py` refuses a
command without `--mode` before any write. There is no anchor option and no
offline-teacher path. The engine imports `training_sigmas`,
`training_noise_seed` and `sigma_for_rank` from this module.

## Usage

Agents start training as a queue `train` job. Humans can use the
[foreground queue or direct CLI](../usage.md). The queue starts four processes with
Accelerate on four free GPUs. See
[run guide §4](../../configs/README.md#4-train-an-adapter) for the job record.

This is the agent path. A person can also run the same module directly, or start the queue in the foreground; see [usage](../usage.md).

1. Check the arguments on CPU. A dry run reads the membership, the masters, the
   plans and the conditions. It loads no weights and writes nothing:

   ```bash
   cd LTX-2
   "$PY" -m scripts.onestep_avatar.train --mode causal --subset <MEMBERSHIP.json> \
     --output <FRESH_RUN> --variant dev --guide-mode d1 --objective white --dry-run
   ```

2. For managed execution, put the arguments without `--dry-run` in a `train` job.
   Start it with `execution.queue --foreground` to watch and stop it from the terminal.

**One run that you watch.** Put the one `train` job in its own job list.
Start the queue owner in the foreground. It waits for four free GPUs, streams
the training output and stops the run cleanly on Ctrl+C. See
[queue usage](../execution/queue.md#usage).

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" -m scripts.onestep_avatar.execution.queue \
  --jobs <ONE_TRAIN_JOB.json> --state <STATE.json> --foreground \
  --process-ledger ../expr/onestep_avatar/processes.json --gpu-pool 0,1,2,3
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

| Argument | Meaning |
|---|---|
| `--mode {bidirectional,causal}` | Required. The attention mode. |
| `--subset` | Required. A version-two fixed video list. |
| `--output` | Required. A new run directory. |
| `--model {2.3,2.5}`, `--variant {distilled,dev}` | The base weights. Distilled weights need sigma on their grid. |
| `--guide-mode {d0,d1}` | D0 noises the capture (capacity test). D1 noises the guide. The target is always the capture. |
| `--objective {bg,white}` | The background. |
| `--split`, `--frame-plan` | The data group, and an optional saved reproduction plan. |
| `--span-latent-frames N`, `--start-policy {clip_start,random}` | Bidirectional: frames per segment and the start rule. |
| `--block-latent-frames`, `--blocks-per-sample`, `--context-latent-frames` | Causal only: B, K and D. |
| `--teacher-forcing` | Causal only: refresh the cache from the capture target. |
| `--sigma0 S` or `--sigma-levels A B ...` | One fixed sigma, or one draw per update and process from these levels. |
| `--lora-rank`, `--lora-alpha`, `--lora-target {attn,attn_ffn}` | LoRA settings. Alpha must equal rank. |
| `--lr`, `--warmup-steps`, `--max-grad-norm`, `--steps` | Optimizer settings. |
| `--chains-per-rank N` | Samples accumulated per process for each update. |
| `--seed`, `--init-seed`, `--data-seed`, `--noise-seed`, `--noise-policy` | Seeds and the noise rule. |
| `--save-initial`, `--save-every`, `--log-every` | Checkpoint and log intervals. |
| `--init-adapter`, `--allow-cross-mode-init` | Start a new stage from parent weights. This is not a resume. |
| `--preview-inputs <preview.json>` | Write a preview job after each saved checkpoint. |
| `--resource-budget <BUDGET.json>` | Limit the time and allocated memory of each phase. |
| `--save-update-state` | Save training text and Adam moments, for E4. |
| `--consumer-trace` | Record the model's actual conditioning and adapter compute. |
| `--no-wandb`, `--wandb-*` | Weights & Biases logging. |
| `--dry-run` | Print the resolved settings only. |

Bidirectional mode refuses block, cache and teacher-forcing options. A mode
change needs a new run and a new output directory.

Outputs in `<FRESH_RUN>`: `config.json`, the logs `metrics_rank*.jsonl` of each
rank and `checkpoints/lora_weights_step_NNNNN.safetensors`. Each adapter stores
its complete training conditions in its metadata. Use
[plot_training](../plot_training.md#usage) to plot the logs.

## Data flow

The module reads the CLI arguments, the fixed video list and the base weight
identity. It makes checked common settings, mode settings and a frame
selection plan. The engine receives those checked records.

## Organization logic

### Mode

`train.py`, `evaluate.py` and `infer.py` all require `--mode bidirectional` or
`--mode causal`. Training selects `plan_samples` and `train_sample` from that
mode. Evaluation and product generation select its `sample` function and check
the adapter settings. See the [README workflows](../../README.md#workflows).

- **Common settings:** paths, train or evaluation group, model version, base
  variant, D0/D1, background, noise levels, seeds, LoRA, optimizer,
  accumulation and logging. The model version is 2.5, with the current
  distilled default. Examples state their base variant explicitly.
- **Bidirectional settings:** the number of encoded frames and the start rule.
  Explicit block, cache, priming or history options are refused.
- **Causal settings:** block length, blocks per sample, cache depth and history
  policy. Random-start independent segments keep the one-block limit.

### Check order

1. Parse the types. Keep a record of which options were given explicitly.
2. Resolve the defaults. The seeds fall back to `seed`.
3. Check finite noise levels, unique level lists, LoRA values and accumulation
   limits.
4. Check the background, the group, the video hashes, the frame counts, the
   frame rate and the guide availability.
5. Build the frame plan of the selected mode. Check the sample and block counts
   across GPU processes, and the cache capacity.
6. Check the base and LoRA settings of a parent adapter. Record the previous
   training stage.
7. Check the output directory. Archive an old output only after all checks
   pass.

A dry run prints the resolved settings. It does not archive files, create text
caches or load model weights. An accepted run saves the resolved settings in
`config.json`.

### Mode rules

- In bidirectional mode, an explicit causal option fails, even when its value
  equals a causal default. An omitted causal option never enters the resolved
  bidirectional record.
- In causal mode, the block length and K must be positive, the history depth
  must not be negative, and every planned sample needs enough complete blocks.
- In bidirectional mode, each selected master needs a nonempty legal segment.
- An empty frame plan fails before any distributed work.

The causal frame plan records the sample stride (default K), the ordered starts
and the sources that give no full sample. Reproduction plans keep the original
starts, indices and seed maps.

### Checked result

The checked result has four parts:

- **Data and model identity:** resolved paths, background, D0/D1, groups, the
  actual base identity, the video-list hash and the frame-plan hash.
- **One mode record:** the segment length and start rule, or the block length,
  K, history depth and policy.
- **Training runtime:** LoRA and optimizer values, accumulation, steps, seeds,
  noise rule and save and log settings.
- **Preview settings:** fixed inputs, levels, output roles and the checkpoint
  trigger rule. There is no preview model session.

A dry run prints the same records, and an accepted run saves them. The engine
receives them directly. It does not parse the CLI again or infer the mode from
K. Only after all checks pass can the engine create or archive the run output
and build model or text caches.

### Frame plans

`--frame-plan` selects a saved reproduction plan. The reader checks its hash,
membership hash, mode, geometry and sample lengths before the model loads.

For a random-start plan, `window_start_draw` owns the saved draw descriptor:
uniform integer starts, the noise seed and
`onestep_avatar.window:{seed}:{step}:{rank}:{slot}`.

- A reproduction plan must match this descriptor exactly, even when its content
  hash is valid.
- A clip-start plan refuses a random descriptor.
- Adapter records use the same descriptor.
- A causal random-start plan allows only block zero and its `[0,B+1]` template.
  A later block index is refused, because the runtime would apply it to a newly
  sliced independent window.

### Parent adapters

`--init-adapter` starts a new stage from a parent's weights, with a fresh
optimizer, step and random state. `--allow-cross-mode-init` allows a parent from
the other mode only with the same base, rank, alpha and targets. The record
keeps the parent mode. This is not a resume.

### Optional evidence flags

- `--save-update-state` (default false) asks the shared engine for named Adam
  moments after updates. It changes the saved evidence only. It changes no
  noise, sample, optimizer or adapter contract. The resolved settings record
  it. It is not optimizer resume support.
- `--resource-budget PATH` pins the bytes of a frozen budget file. It adds no
  experiment loader and no new limits. The resource owner reads the exact
  allocated-memory and per-phase wall-time limits before execution. It
  measures the load, update and export phases. Current native replay requires
  this evidence. CPU tests and older runs without the option keep their
  original scope.
- `--consumer-trace` (default false) records the model's actual conditioning
  and adapter compute through observational hooks after wrapping. It changes
  the evidence and the measured overhead, not the training calculation.
  Completion binds the fixed per-rank trace of each checkpoint. An incomplete
  or changed requested trace refuses completion. The trace alone does not
  prove native numerical agreement.

### Queue launch record

Queued training also saves `queue_launch`, the original dispatch binding from
the queue owner. It has the exact normalized arguments, processes, port and
command, and the original Accelerate bytes and hash. Applied runtime evidence
is separate from the CLI settings. The engine records the actual world size and
the precision policy of each rank after the Accelerator and FSDP setup. Readers
compare those facts with the launch. They never infer missing settings from the
current file.

### Previews

Visualization settings give the plot output and the optional preview jobs. A
preview job records its checkpoint step, fixed inputs, mode, exact levels, seed
and output path. Preview execution stays separate from distributed training
(see [engine](engine.md) and [media](../media.md)).

`preview_inputs` is an optional JSON path. `preview_record` is the checked,
fixed input record that the checks keep. A dry run reads and validates it, but
adds no job. Preview settings add no training or decoder session.

### Reproducible noise and level choices

The seed calculations are:

- Fixed-per-chain base noise seed: `noise_seed*100003 + chain_index*101`.
- Fresh base noise seed: `((noise_seed*1000003 + step)*1009 + rank)*131 + slot*17`.
- Block noise uses the base seed plus the block index.
- With one sigma level, use that value. With several levels, seed a local
  Python generator with `onestep_avatar.sigma:{seed}:{step}:{rank}` and select
  `rng.randrange(number_of_levels)`.

Fresh noise changes at the next update or visit. Fixed-per-chain noise reuses
the same chain draw. The sigma draw is independent of the noise stream and the
data order. One draw applies to all accumulation slots of that process and
update. Equal exposure to the levels is expected over many draws. It is not
guaranteed for each update or epoch. Reproduction plans keep the original chain
indices where the seed rule needs them.

## Invariants

- One block does not select bidirectional mode automatically.
- A resolved bidirectional record has no causal settings.
- Invalid requests and dry runs do not change an existing output directory.
- A parent adapter starts a new stage with a fresh optimizer, step and random
  state.
- Frame plans use the fixed video list and declared rules, not directory names.
- A mode change starts a new run. It does not convert an existing adapter.

## Gotchas

- An omitted cache option must not make a bidirectional request invalid.
- Keep GPU-process settings in the Accelerate configuration.
- Keep the model mode and the preview settings in the run configuration.

## Tests

[V6–V8](../verification.md) check mode fields, data changes and incompatible
adapters. The tests run these checks before any model load. They check dry-run
and archive behavior. They check that preview jobs record fixed inputs and do
not run inside the training loop.

- Worked mode check: bidirectional mode with an explicit history-depth option
  fails, even if the value is zero. Without that option, it passes when the
  segment and data checks pass.
- Worked side-effect check: an existing output directory and an invalid D1
  guide must stay unchanged.
- A valid dry run prints the same mode and frame plan as execution. It creates
  no model, cache or output files.
- Worked seed check: `noise_seed=2` and `chain_index=3` give the fixed base
  seed `200309`. A change of only sigma does not change that base seed.

Explicit-mode train commands run through the typed engine. They require a
version-two fixed video list. Preview CLI settings and checked input records
are implemented. Native preview output is still an acceptance gate.
