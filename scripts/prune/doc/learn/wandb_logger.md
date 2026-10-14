# learn/wandb_logger

## Objective

Log saved distributed training evidence with rank charts and single averaged loss metrics.

## Usage

Run directly from LTX-2 in ltx on CPU:

```bash
CUDA_VISIBLE_DEVICES= python -m scripts.prune.learn.wandb_logger --study /abs/frozen_study.json --arm learned_joint --replicate-seed 42
```

The direct training owner starts this logger when `wandb.enabled=true` in the frozen config.
Arguments select the study, arm and seed. The config supplies project, entity, mode and chart interval.
Outputs are W&B history, a cursor, PID, writer lock and `monitoring/wandb.json` for the primary learned run.
Other arm/seed writers have separate filenames and separate W&B runs. See the [run guide](../usage.md#uniform-f18-ddp-training).

## Data flow

Numeric immutable update receipts and complete monitor JSON enter one CPU writer.
It verifies study identity, rejects incomplete global batches and ignores hidden atomic publisher files.
Training rank zero owns receipts; no rank starts its own W&B run.

## Organization logic

Metric keys start at `train/`, `gates/`, `memory/`, `ranks/` and `validation/`.
There is no `learned_joint/` prefix or arm path in metric tabs. Arm identity is stored in run metadata.
Use `update` as the optimizer axis and `validation/update` as the checkpoint axis.

`train/sigma`, `ranks/update_s` and each memory chart use `wandb.plot.line_series`.
Each chart has one line per rank, labeled rank 0 through rank 3 for four ranks.
Refresh charts at update one, every ten updates by default, and on exit. Every refresh includes all saved points.
Store rank differences in these charts; do not create four separate scalar panels.

`train/loss`, velocity MSE and sparse loss are means over the global consumed draws.
The actual undivided total loss includes any active regional loss. Historical unweighted receipts use velocity plus sparse loss.
Gradient norms and gate statistics describe the synchronized state, so they remain single scalars.
Step duration uses the slowest rank for the global duration scalar.

Validation averages unique actor/view/sigma cases gathered from all ranks.
`validation/trained_sigma_2_gt_latent_mse` is one averaged scalar, never four lines.
Sigma indices are ordered by descending monitored sigma. The third monitored level uses index two.
Reject conflicting duplicate cases and count a repeated matching case once.

## Invariants

No model, VAE, queue or GPU work. Lock one writer per arm and seed.
Stable run IDs and saved cursors support restarting the logger without resetting the run.
Separate arms have separate W&B runs, which avoids metric collisions while keeping tabs free of arm prefixes.
Only completed receipts enter history. Missing rank evidence fails instead of inventing chart lines.

## Gotchas

Charts refresh less often than scalar history to limit table upload overhead; their data still include every optimizer update.
A correct chart does not establish quality or physical compression. Existing experiment logs keep their historical naming.

## Tests

`test_wandb_logger.py` checks four actual W&B line-series labels, total-loss means and unique-case validation means.
It checks constant gate LR, missing rank/batch evidence and hidden temporary-file filtering without contacting W&B.
Use an offline writer smoke check for cursor, lifecycle and chart publication.
