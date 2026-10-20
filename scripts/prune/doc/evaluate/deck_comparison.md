# Deck comparison

## Objective

Render native dev, old step 3300 and new step 800 for the report-10-19 deck.
This executor does not resume training or change a frozen study.

## Usage

The agent launches directly from LTX-2 in the ltx environment, on authorized free GPUs.
Use one process per free physical GPU. For example:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /home/jianjinx/data2/miniconda3/envs/ltx/bin/python \
  scripts/prune/evaluate/deck_comparison.py.txt \
  --old ../expr/refiner_prune/2.5/progressive_heads_20261017_consolidated \
  --new ../expr/refiner_prune/2.5/progressive_clip01_ste3_reset50_20261018_combined \
  --output ../expr/refiner_prune/report-10-19/data/video_comparison_sigma094 \
  --actor 35 --schedule four --start 0.9403364062309265 --gpu 0
```

`four` selects four intervals at rounded, evenly spaced indices of the official tail.
`tail` uses every interval. Both include the exact selected start and zero.
See the [run guide](../usage.md#deck-comparison).

## Data flow

Saved source and epsilon → native D0 input → checkpoint policy → Euler sampling → VAE decoding.
Each case saves latent and video pins, exact schedules, input hashes, replay controls and decoder settings.
The deck generator reads these saved results and makes the synchronized presentation.

## Organization logic

Build a local manifest row when the selected start is absent from the monitor grid.
Reuse all source and noise bytes. Never rewrite the original manifest.
Load actual adapters and the versioned policy. Keep native adapters disabled.
Reuse archived native/old four-step videos only when the exact schedule and source case match.
Replay a trained policy against its saved direct monitor before sampling.
Release the transformer before loading the decoder.

## Invariants

Use BF16 native execution, deterministic policy, empty text, clean capture frame zero and full bidirectional attention.
Use actual saved binary gates: old removes 867 heads; new removes 292.
Keep actor, view, noise, source frames, geometry, frame rate and decoder seed fixed across methods.
The executor uses `.py.txt` so it does not alter frozen Python inventories.

## Gotchas

These actors are development examples, not the untouched six-actor test split.
The methods have different removal counts and training budgets.
Four-step and full-tail outputs must use the recorded schedules; do not infer them from rounded labels.

## Tests

Require saved-monitor replay within maximum absolute error 0.02.
Verify saved input, checkpoint and output hashes before reuse.
Inspect synchronized media and browser slides at wide and narrow sizes.
