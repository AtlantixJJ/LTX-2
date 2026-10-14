# Partial learned-head preview executor

## Objective

Preview immutable soft checkpoints without changing or restarting the active frozen training code.
This auxiliary Python template has a separate recorded source hash. It is not part of the training source inventory.

## Usage

Use the run's direct CPU preview owner in ltx. It reserves GPU 4 or 5 through the shared ProcessRegistry ledger.
It does not submit avatar queue jobs. The owner runs the template below only after an atomic device claim.
Memory-only polling is insufficient when another owner is about to launch.

```bash
CUDA_VISIBLE_DEVICES=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python scripts/prune/evaluate/partial_heads.py.txt \
  --study <RUN>/configs/frozen_study.json \
  --snapshot <PARTIAL>/analysis/snapshot.json --slot 0
```

Run slot one on physical GPU 5. Each slot processes two of the four fixed validation actors.
Wait until the physical GPU has less than 1 GiB used. Retain a 1.25-GiB allocation across model loading boundaries.
This direct owner does not stop other jobs or submit work to the avatar queue.
See [run guide](../usage.md). The snapshot pins checkpoints 100 and 400 and both video sigmas.

## Data flow

Validate the frozen study. Reconstruct matched inputs through the native whole-clip owner.
Require preview noise to equal the saved monitor noise. Reuse existing highest-sigma monitor predictions.
Generate only missing predictions. Release the transformer before loading the native decoder.
Preserve complete source MP4s and render synchronized five-panel comparisons in three columns.

## Organization logic

Use public native input, forward, adapter, gate, decoder and rendering owners.
Compare native dev, soft checkpoint 100, soft checkpoint 400 and its immediate legal hardening.
Decode GT once per actor. Score complete decoded frames against equally decoded GT, excluding frame zero.
Save a worker receipt only after every declared prediction, control and comparison exists.

## Invariants

Do not modify training config, code, checkpoints, draws or numerical policy.
The requested video sigma is a separate evaluation level; it is not added to the seven training sigmas.
Use full clips, clean c0, seed 2042, exact saved noise, empty text and one direct native call.
Require the single-rank checkpoint-400 replay to match the four-rank monitor within 0.02.
The template's exact bytes are pinned independently in every completed worker receipt.
Report rebuilding reads saved outputs only and cannot call this executor.

## Gotchas

Immediate hardening is a model change, not compact-export parity or binary-refinement quality.
Small soft values do not remove stored weights. Latent and pixel distances do not establish perceptual acceptance.
This pilot has four validation actors and one seed. Other actors receive videos rather than inline figure examples.

## Tests

Check the frozen code identity before and after execution. Check exact noise and clean-frame equality.
Validate all full-length videos, synchronized frame maps, saved hashes and the native replay control.
