# Deck video composition

## Objective

Make one synchronized eight-panel video for each actor from checked saved RGB outputs.

## Usage

Run directly from LTX-2 in ltx. No GPU or model is needed.

```bash
CUDA_VISIBLE_DEVICES= /home/jianjinx/data2/miniconda3/envs/ltx/bin/python \
  scripts/prune/evaluate/deck_composite.py.txt \
  --root ../expr/refiner_prune/report-10-19 --actor 35
```

Repeat for actors 25 and 84. See [the run guide](../usage.md#deck-comparison).
Outputs are `media/videos/actor_<id>.mp4`, a PNG poster, labels and a JSON rendering record.

## Data flow

Execution receipts → hash and frame checks → labeled RGB video grid → MP4, poster and rendering record.

## Organization logic

Require the four-step and full-tail receipts before rendering.
Check capture, noise, context, noisy tokens, clean first frame, geometry and frame rate across both rows.
Check all saved videos against their hashes and require equal frame count and playback rate.
Place four-step outputs above 21-step outputs.
Place capture VAE, native, continuous soft gate step 3300 and window reset hard gate step 800 from left to right.
Use aspect-preserving scaling and white padding in each 384-pixel square panel.
Render static readable titles with Pillow, then overlay synchronized videos with ffmpeg.
Save all source frame IDs, labels, positions, exact schedules and source hashes.

## Invariants

Preserve every source frame and 30 fps. Keep the same columns in both rows.
The start noise level is identical across rows.
The source RGB remains available separately; the displayed reference is explicitly labeled VAE decoding.

## Gotchas

This presentation re-encodes saved RGB videos. It does not define a pixel-metric source.
The continuous soft gate and window reset hard gate methods have different training lengths and removal counts.

## Tests

Reject changed inputs, failed replay controls, changed video bytes and unequal frame coverage.
Inspect poster frames and decoded contact sheets. Check wide and narrow slide layouts.

Panel titles use Continuous soft gate and Window reset hard gate. Fit the font inside each panel.
