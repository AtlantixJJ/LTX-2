# evaluate/ffn

## Objective

Measure final binary masks, recovery, decoded capture fidelity and compact execution cost.

## Usage

A human or agent runs this direct CLI from `LTX-2` in `ltx`, on GPUs 4–6.

```bash
conda run --no-capture-output -n ltx python -m scripts.prune.evaluate.ffn --study ../expr/refiner_prune/2.5/learned_ffn_joint_20261018/configs/frozen_study.json --stage infer --split test --arm native --gpu-id 4
```

`--stage infer` supports `native`, `selection`, `final` and `mask_only` arms.
Trained arms require `--checkpoint` with their actual frozen state.
`--stage media` reads native, selection and final predictions and produces synchronized comparisons.
`--stage benchmark` requires a final checkpoint and `--candidate` compact export.
See the [run guide](../usage.md#learned-ffn-masks).

Outputs are per-case tensors and hashes, numerical results, decoded metrics, source videos,
comparison MP4s, posters, rendering receipts and timing brackets under the study root.

## Data flow

The frozen study supplies held-out sources, actual contexts, three seeds and three exact direct schedules.
Inference uses shared native input construction and restores exact clean frame zero.
All arms reuse the same saved epsilon and capture. The first frame is excluded from error reductions.
Saved tensors enter the existing VAE, fixed foreground and comparison-media owners.

## Organization logic

The test split contains 12 views. Three seeds and three sigmas produce 108 cases per arm.
The repeatedly inspected validation panel contains four actors and is development evidence.
Frozen checkpoints retain both masks; disabling adapters measures their selected model's immediate removal effect.
The selection/final comparison changes adapter recovery while keeping the masks fixed.

Media uses the first view of four distinct actors, one saved seed and all three sigmas.
It shows Capture RGB, Capture VAE, Native, Before recovery and After recovery in a wide grid.
All panels use matched source frames, fps and aspect-preserving padding.
Full individual source and output videos are retained.
Person LPIPS uses capture-derived matte coverage and boxes, with a Capture VAE/RGB ceiling.
Frame LPIPS uses the whole frame. Neither proves correct identity or action.
Original matte replay checks recorded crop and scale before interpreting images.

Timing uses sequential model lifetimes and both masked/compact/masked and compact/masked/compact orders.
Each load receives two warmups and seven synchronized timed calls.
The output reports both speed ratios and bracket drift. A speed qualification additionally requires native parity.
The call excludes input preparation, decoding and model-loading time.

## Invariants

Never optimize during evaluation or change a frozen mask.
Reject changed prediction, input, component or checkpoint hashes.
Compare direct schedules within each sigma; do not confuse them with a multistep trajectory.
Use actor-level aggregation and report the one-training-seed limitation.

## Gotchas

Named historical test actors are held-out actors, not automatically a new blind split.
Some read-only caches contain historical paths. They remain input provenance, not alternate prediction producers.
The exact compact parity and decoded acceptance gates remain independent from numerical MSE.

## Tests

CPU tests check input keys and mask/state boundaries. Existing media tests check synchronized panels and fixed regions.
Fresh full-model evidence remains necessary for all scientific claims.
