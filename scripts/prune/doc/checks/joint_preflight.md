# checks/joint_preflight

## Objective

Check optimizer scaling and native finite channel removal.

## Usage

The user or agent runs this standalone owner from `LTX-2` in conda environment `ltx`.

```bash
CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n ltx python -m scripts.prune.checks.joint_preflight --phase p0 --output ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261018
```

See the [run guide](../usage.md#joint-clipped-gates-with-re-admission).
Outputs are immutable study, check, gradient, checkpoint or review receipts under the specified root.
P1 can split its 48 blocks across three independent GPU owners. First measure one warmed case:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx python -m scripts.prune.checks.joint_preflight --phase p1 --study ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261018_preflight/configs/frozen_study.json --gpu-id 4 --shard 0 --shards 3 --timing-only
```

Remove `--timing-only` to run this shard. Use shard 1 on GPU 5 and shard 2 on GPU 6.
Do not overlap those owners with native acceptance or the pilot.
Use `--traces /pilot/training/gradients` for the separate centered P0 replay after all 50 pilot updates exist.

## Data flow

P0 replays identical seeded Gaussian sequences across four gradient scales and two Adam epsilons. P1 loads the native model and saved development inputs. It changes one block and one nested removal fraction per observation.

## Organization logic

P0 records raw and executed movement separately. Its noise reference checks both directions and two-window persistence. Optional temporal replay centers each channel over time and resamples five-update blocks. P1 saves exact channel IDs, predictions, paired losses and loss change per removed channel.

## Invariants

Keep saved input hashes, exact schedules and original unit IDs.
Keep native loss reduction in FP32. Preserve the frozen backbone and complete rank RNG.

## Gotchas

Synthetic reference counts are not measured false-removal rates. P1 needs 1152 ablated forwards. Its first warm forward saves a duration estimate. Random finite removal cannot establish learned-mask quality. The pilot binary-mask comparison is a separate required check.

## Tests

Run `tests/test_readmission.py` with CUDA hidden.
Native acceptance and held-out export parity remain independent evidence requirements.
