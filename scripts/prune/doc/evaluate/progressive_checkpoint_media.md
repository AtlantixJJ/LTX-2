# Progressive checkpoint media

## Objective

Render the progressive head-removal study every 100 updates. Compare its actual
mixed gates with the prior r13 soft checkpoint at the same update.

## Usage

Run from LTX-2 in the ltx environment with `CUDA_VISIBLE_DEVICES=6`:

```
python scripts/prune/evaluate/progressive_checkpoint_media.py.txt \
  --study ../expr/refiner_prune/2.5/progressive_head_gate_removal_f18_20261015/configs/frozen_study.json \
  --stage reference --update 100 \
  --output ../expr/refiner_prune/2.5/progressive_head_gate_removal_f18_20261015/partial
```

Then run `--stage media --update N` and `--stage steps --updates N --actors 35 25 84 96`.
The detached report owner runs these stages and the saved-artifact report builder.
See [the run guide](../usage.md) for the training contract.

## Data flow

The frozen study supplies source captures, prompt context, noise and numerical policy.
Reference execution saves native predictions, capture RGB and capture VAE media.
Media execution reads saved training monitors and matched historical soft monitors.
Step execution restores each checkpoint's adapters and gates. It saves one, four and eight
step predictions at the exact schedules from the prior study. Decoding and scoring use
the existing native decoder and AlexNet LPIPS owners.

## Organization logic

This template derives from `learned_head_checkpoint_media.py.txt`. It retains the
reference, media and steps stages. The template is outside the frozen Python inventory.
Each receipt records its executable hash. Report code only reads saved artifacts.

## Invariants

Free heads run sigmoid(alpha). Removed heads run exactly zero. Restore the complete
ProgressivePolicy from the checkpoint before each progressive forward. Clear this policy
for native and prior soft forwards. Never turn remaining free heads to one.
Validate saved prediction hashes and noise tensors. Replay the four-rank monitor within
maximum absolute tolerance 0.02. Validate Euler [sigma,0] against the direct call.
Preserve source frames, fps, crop and matte. The six-panel monitor comparison shows
capture RGB, capture VAE, native, previous progressive, current progressive and prior soft.
The step comparison shows capture VAE, native, progressive and prior soft.

## Gotchas

The first report has no previous checkpoint panel. Label this missing panel explicitly.
A monitor at update N uses the post-commit removed set; update N loss precedes that commit.
The prior soft checkpoint has no progressive policy. Do not apply its legacy legal mask.
The prior study's binary phase starts after update 700, beyond this 500-update study.
Model and decoder stages require one free 48 GB GPU and run serially.

## Tests

Compile the template, lint syntax and imports, and verify the frozen code inventory.
Require replay controls and all twelve monitor cases before publishing each report.
The report validator checks metrics, figures and media references independently.
