# Matched MobileWan removal evaluation

## Objective

Evaluate the three preselected held-out views at eight exact sigmas and four frozen removal budgets.
Save 96 full clean predictions and generated-frame MSE to recorded capture latents.
This evaluator performs no recovery, fitting, selection or decoding.

## Usage

Run this standalone prune entry script from `LTX-2` in the `ltx` environment.
Complete native gates and freeze all 3,104 train views before launch.
Check GPU occupancy and choose a device with at least 44 GiB free.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m scripts.prune.evaluate.mobilewan_removal \
  --study ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011/configs/study.json \
  --gpu-id 4
```

Use `setsid`, an external timeout and an attributable log for persistent execution.
Outputs are `evaluation/results.json`, `evaluation/records/` and `evaluation/latents/` below the study root.
Shared noise and input manifests remain in `inputs/views/`.
See the [run guide](../usage.md#mobilewan-removal-evaluation).

## Data flow

Verify the audited dataset, complete train freeze, preselected cases and full component hashes.
Open the pinned dev transformer with the deterministic baseline policy and exact empty context bytes.
The shared input owner prepares capture-only evidence and reuses saved epsilon.
The shared reconstructor makes each full-clip D0 modality. This file defines no noising calculation.

Each sigma executes dense and all-one calls before the three removal calls.
Require exact equality between dense and all-one outputs before accepting that sigma.
The hook owner applies original head masks after the native gate and before output projection.
Restore clean capture frame zero and save each full prediction as BF16 on CPU.

## Organization logic

Frozen budgets supply self-attention masks. Construct complete families with all cross-attention heads retained.
Require exact original branch identities, binary masks, one retained head per branch and all four budgets.
Publish tensors without overwriting an existing different tensor.
Save each complete sigma record separately so completed sigmas can resume without model calls.

Resume verifies task identity, tensor hashes, budgets, all-one controls and metrics recomputed from paired tensors.
Final publication requires all three cases, all eight sigmas and all four arms.
The final record keeps individual case observations. It does not replace them with a pooled quality estimate.

## Invariants

Case selection occurs before pruning inspection and cannot change during evaluation.
Train membership alone defines removal. The evaluator never recalculates masks from held-out scores.
Every arm reuses capture, epsilon, context, model, geometry, fps and the exact direct `[sigma,0]` schedule.
Shared source and evaluator hashes bind the saved result to its implementation.

MSE uses FP64 differences, squares and means against recorded capture latents.
Exclude frame zero from every MSE; record its exact equality separately.
The time label is the latent coordinate `latent_frame_index * temporal_scale / fps`.
It does not identify the first decoded pixel in that encoded interval.
The clean prediction wrapper already converts velocity. Do not convert it again.

## Gotchas

A lower latent MSE does not establish better identity, action, motion or perceptual detail.
Three actors and one seed do not support a precise dataset-wide quality interval.
Head-score confidence intervals are not prediction-quality intervals.
Functional masks do not establish reduced transformer latency or storage cost.
Raw outputs require a separate media stage before visual interpretation.

## Tests

`tests/test_mobilewan_removal.py` independently checks GT MSE, conditioned-frame exclusion and frame time labels.
A tiny real transformer executes all four arms at all eight sigmas and verifies all-one equality.
The test resumes without additional calls and rejects altered prediction bytes.
Native removal quality and synchronized videos remain separate evidence gates.
