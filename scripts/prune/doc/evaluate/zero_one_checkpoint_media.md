# evaluate/zero_one_checkpoint_media.py.txt

## Objective

Render actual zero/one continuation checkpoints against parent step 500 and native dev.

## Usage

From LTX-2 in ltx, on the authorized free reporting GPU:

```bash
CUDA_VISIBLE_DEVICES=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. \
python scripts/prune/evaluate/zero_one_checkpoint_media.py.txt \
  --study ../expr/refiner_prune/2.5/progressive_zero_one_f18_from_u500_20261016/configs/frozen_study.json \
  --stage reference --update 600 \
  --output ../expr/refiner_prune/2.5/progressive_zero_one_f18_from_u500_20261016/partial
```

Run reference once, media per checkpoint, then steps with `--updates N --actors 35 25 84 96`.
The experiment-local report owner serializes GPU-6 stages. See [the run guide](../usage.md#zeroone-continuation).

## Data flow

The reference stage saves native predictions, a saved-monitor replay, capture RGB/VAE media and LPIPS ceiling.
The media stage decodes actual trained monitor tensors and parent step-500 tensors.
The steps stage restores the actual versioned policy for both checkpoint states.
All stages retain the existing numerical, capture/noise/context, native decoder and LPIPS owners.

## Organization logic

Derive this template from the progressive checkpoint media executor. Pin the exact executor bytes in receipts.
Keep the template outside the frozen Python inventory so reports can finish without modifying training code.
The fixed parent checkpoint and current checkpoint receive the same input and exact schedule.
Use one direct step and four Euler steps from sigma 0.9098098278045654, plus eight from 0.9403364062309265.

## Invariants

Render capture VAE, native dev, parent step 500 and continuation step N in a synchronized 2x2 grid.
Use 384-pixel panels, aspect-preserving padding, 30 fps, decoder seed 42 and matched source frames.
Both parent and continuation execute their actual policies; do not globally harden remaining free gates.
Validate monitor replay and Euler [sigma,0] reduction within maximum absolute tolerance 0.02.
Save full reference RGB and its source hash, even though the video grid shows capture VAE.
The report uses text-link tables without embedded images. Posters remain inspection artifacts only.

## Gotchas

Parent-versus-continuation changes include further training and both decision branches.
Do not describe this as an isolated fixed-one causal control. Native execution uses no adapters and all-one gates.
Model and decoder lifetimes are serial on one 48 GB GPU. Never run on training ranks.

## Tests

Check template syntax and frozen code identity. Require full actor/schedule coverage and all execution controls.
Inspect saved video frame maps, labels and image geometry before interpreting quality.
