# Usage — human and agent launches

Agents use `execution.queue` for model work. Humans can choose the queue or
run the task module directly. Both paths use the same task code and arguments.

## Set up the shell

Run from the LTX-2 root. Activate `ltx` to get live output without conda buffering:

```bash
cd /home/jianjinx/data2/VideoDiffusionModels/LTX-2
conda activate ltx
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
PY=$(python -c 'import sys; print(sys.executable)')
```

Only `corpus.build_guidance` uses `argavatar`. All other commands use `ltx`.
Replace example paths with your prepared inputs and new output paths.

## Human: one command through the queue

Use an existing job list, or create a one-job list using the
[job records by kind](../configs/README.md#job-records-by-kind).
For example, save this as `/tmp/avatar-train-jobs.json` and adjust the paths:

```json
{
  "schema_version": 1,
  "jobs": [{
    "id": "human_train",
    "kind": "train",
    "arguments": ["--mode", "bidirectional", "--subset", "/abs/membership.json",
                  "--output", "/abs/new_train", "--variant", "dev",
                  "--guide-mode", "d1", "--objective", "white", "--steps", "60"],
    "output": "/abs/new_train",
    "processes": 4,
    "accelerate_config": "/home/jianjinx/data2/VideoDiffusionModels/LTX-2/scripts/onestep_avatar/configs/fsdp.yaml",
    "port": 29785,
    "completion": {"step": 60,
                   "checkpoint": "/abs/new_train/checkpoints/lora_weights_step_00060.safetensors"}
  }]
}
```

Launch, watch and stop from one terminal:

```bash
"$PY" -m scripts.onestep_avatar.execution.queue \
  --jobs /tmp/avatar-train-jobs.json --state /tmp/avatar-train-state.json \
  --foreground --process-ledger ../expr/onestep_avatar/processes.json
```

- The queue waits for free GPUs. It prints the job, GPUs, owner PID and log path.
- Child stdout and stderr appear live. The same bytes remain in the saved child log.
- Press Ctrl+C once to stop. Wait for the owned workers to end and the queue to return.
- Cancellation returns 130 and preserves partial output as a failed attempt.
  Start a new job ID and output path to retry a stopped run.
- Normal completion returns 0 after output verification. A failed job returns nonzero.
- A list with one job runs one task. A larger list runs all its jobs in order of readiness.
  `--foreground --once` instead attempts at most one ready job; it returns 2 if GPUs are busy.
- To stay attached across terminal disconnects, use the same command inside `tmux`.

Inspect a saved log from another terminal with `tail -F <printed-log-path>`.
Run `--dry-run` instead of `--foreground` to inspect identities and current states.
The [queue design](execution/queue.md#usage) lists every control.

## Agent: unattended queue

Use separate job and state files for each owner, with the same process ledger.
Start the owner in its own session:

```bash
setsid -f "$PY" -m scripts.onestep_avatar.execution.queue \
  --jobs /abs/jobs.json --state /abs/state.json --execute --loop \
  --process-ledger ../expr/onestep_avatar/processes.json \
  > /abs/queue-owner.log 2>&1 < /dev/null
```

Child logs are in the state directory's `logs/` folder. The state file records
the running job's `owner_pid` and `log`. To stop an owner you launched,
send `kill -TERM <OWNER_PID>` and wait for cleanup. SIGTERM returns 143.
Keep the owner alive until shutdown finishes. Recovery after an unexpected
owner exit is documented in [run guide §2](../configs/README.md#commands).

## Human: direct CLI

Select free physical GPUs after checking occupancy:

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
```

Use GPUs 0–7 with less than 1,024 MiB in use. Direct commands do not wait for GPUs.
Training uses four devices. Other GPU tasks normally use one.
`CUDA_VISIBLE_DEVICES=5` makes physical GPU 5 the local GPU 0,
so pass `--gpu-id 0`, not `--gpu-id 5`.

### Direct training

Start the shared trainer with the same four-rank Accelerate configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -m accelerate.commands.launch \
  --config_file scripts/onestep_avatar/configs/fsdp.yaml \
  --num_processes 4 --main_process_port 29785 \
  -m scripts.onestep_avatar.train \
  --mode bidirectional --subset /abs/membership.json --output /abs/new_train \
  --model 2.5 --variant dev --guide-mode d1 --objective white \
  --span-latent-frames 7 --start-policy clip_start --sigma0 0.725 \
  --lora-rank 8 --lora-alpha 8 --steps 60 --save-every 20 --no-wandb
```

Choose four currently free GPUs and an unused rendezvous port.
For causal training, use the [causal arguments](../configs/README.md#4-train-an-adapter).
Check inputs first with `python -m scripts.onestep_avatar.train ... --dry-run`.
Output appears in this terminal. Press Ctrl+C to stop the Accelerate launcher and its ranks.

### Direct evaluation, generation, media and experiments

Example evaluation on one free GPU:

```bash
CUDA_VISIBLE_DEVICES=5 "$PY" -m scripts.onestep_avatar.evaluate \
  --mode bidirectional --subset /abs/membership.json --source VIEW_ID \
  --variant dev --guide-mode d1 --span-latent-frames 7 \
  --schedule 0.725 0 --checkpoint /abs/adapter.safetensors \
  --output /abs/new_eval --gpu-id 0
```

The table maps queue kinds to direct modules. Reuse the task arguments from
the linked recipe, including `--output`. Add `--gpu-id 0` for a visible GPU.
Remove `--dry-run` to execute. CPU tasks need no CUDA device selection.

| Task / queue kind | Direct module after `python -m scripts.onestep_avatar.` | Arguments and outputs |
|---|---|---|
| Training / `train` | `train` through Accelerate above | [§4](../configs/README.md#4-train-an-adapter) |
| Evaluation / `evaluate` | `evaluate` | [§5](../configs/README.md#5-evaluate-an-adapter) |
| Preview / `preview` | `previews --preview-job /abs/preview.json` | [§6](../configs/README.md#6-fixed-training-previews) |
| Product / `product` | `infer` | [§7](../configs/README.md#7-product-generation-from-a-supplied-image) |
| Decode / `decode` | `decode_saved --jobs /abs/decoder_jobs.json` | [§8](../configs/README.md#8-decoding-comparison-media-cost-and-plots) |
| Render / `render` | `comparisons --render-saved-comparisons /abs/spec.json` | [§8](../configs/README.md#8-decoding-comparison-media-cost-and-plots) |
| Cost / `bench` | `bench` | [§8](../configs/README.md#8-decoding-comparison-media-cost-and-plots) |
| Scientific check / `experiment` | `experiments.<selector> --spec /abs/spec.json` | [§9](../configs/README.md#9-experiments), [spec format](experiments/spec_protocol.md#usage) |
| Capture and guide encodes | `corpus.precompute` | [§3](../configs/README.md#3-prepare-the-corpus) |
| Guide render (`argavatar`) | `corpus.build_guidance` | [§3](../configs/README.md#3-prepare-the-corpus) |
| Membership (CPU) | `corpus.subset` | [§3](../configs/README.md#3-prepare-the-corpus) |
| Reference pixels | `media --prepare-training-references` | [§6](../configs/README.md#6-fixed-training-previews) |
| Prepared preview / image | `prepare_inputs preview` / `prepare_inputs supplied-image` | [§6](../configs/README.md#6-fixed-training-previews), [§7](../configs/README.md#7-product-generation-from-a-supplied-image) |
| Training plots (CPU) | `plot_training` | [§8](../configs/README.md#8-decoding-comparison-media-cost-and-plots) |
| Saved metrics and planners (CPU) | `experiments.saved_probe_metrics`, `experiments.prepare_continuation` | [§9](../configs/README.md#9-experiments) |

For example, run a prepared scientific spec directly:

```bash
CUDA_VISIBLE_DEVICES=5 "$PY" -m scripts.onestep_avatar.experiments.stock_parity \
  --spec /abs/stock_spec.json --output /abs/new_stock_check --gpu-id 0
```

The `training_slice_check` FSDP arm needs one-rank Accelerate instead of plain Python:

```bash
CUDA_VISIBLE_DEVICES=5 "$PY" -m accelerate.commands.launch \
  --config_file scripts/onestep_avatar/configs/fsdp.yaml --num_processes 1 \
  --mixed_precision bf16 --main_process_port 29786 \
  -m scripts.onestep_avatar.experiments.training_slice_check \
  --spec /abs/fsdp_spec.json --output /abs/new_slice --gpu-id 0
```

To match a queued `numerical_policy: deterministic` run, set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python. Keep thread counts
and scientific arguments the same when comparing results.

Direct tasks show output in the terminal and stop with Ctrl+C.
To also save terminal output, use `set -o pipefail` and append `2>&1 | tee /abs/task.log`.
Keep that log outside the task output directory, which must start fresh.
Direct output keeps the task's scientific records. Queue receipts, process
ledger entries and external bounded supervision are produced only by queued runs.
Use the queue for agent work, dependencies and acceptance evidence that requires supervision.

The default journal is `state.json` beside the queue job list. `--state` overrides that path.
