# evaluate/learned_heads

## Objective

Evaluate frozen merged candidates on exact paired whole-clip inputs.

## Usage

Started by: a direct human or agent CLI. Run from `LTX-2` in conda environment `ltx`.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx python -m scripts.prune.evaluate.learned_heads --study ../expr/refiner_prune/2.5/learned_heads_dev8_20261012_r2/configs/frozen_study.json --split validation --arm learned_joint --replicate-seed 42 --gpu-id 0
```

`--split` selects validation or frozen test cases. `--arm` selects a completed merged candidate.

Outputs: Paired predictions, native velocities, metrics and actor uncertainty.
See the [run guide](../usage.md#learned-head-phases).

## Data flow

A frozen arm, split and replicate select the merged model. Saved original-teacher records provide the paired reference.

## Organization logic

Every selected view uses three saved noise seeds and eight sigmas. Saved native velocity and restored clean output drive separate metrics.
Direction agreement uses the rounded stock clean conversion. Official velocity MSE uses epsilon minus capture. Actor bootstrap averages within actors first.

## Invariants

Evaluation performs no training or selection. Blind inference requires frozen finalists. Missing teacher evidence fails instead of launching a teacher.

## Gotchas

Numeric completion leaves quality unset. Structural, transfer, fusion and reload checks retain their own references and results.

## Tests

Run `test_learned_head_evaluation.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.
