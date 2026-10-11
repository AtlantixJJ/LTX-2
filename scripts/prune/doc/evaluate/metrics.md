# `evaluate/metrics.py` — pixel metrics and three-panel video

## Objective

Normalize decoder layouts, calculate pixel metrics and write synchronized comparison panels.

## Data flow

`_as_bchw` converts `[B,C,T,H,W]`, `[F,C,H,W]` or `[F,H,W,C]` to float32 frame tensors.
Metric functions compare matching shapes.
`t3_video` writes the common frame prefix through ffmpeg.

## Organization logic

PSNR is `10*log10(data_range^2/MSE)`. Equal pixels return infinity.
`ssim_global` uses per-frame, per-channel statistics over the whole spatial image.
It averages the SSIM expression across frames and channels.

`t3_video` places source, reference and candidate from left to right.
Panels must share channel count, height and width. At least three channels are required.
It writes the first three channels as RGB, at the supplied fps, with H.264 and CRF 18.
It adds no titles; the matched analyzer owns its title overlay.

Worked example: frame counts 17, 17 and 9 produce nine video frames.
With `data_range=1` and MSE 0.01, PSNR is 20 dB.

## Invariants

- Metric shapes must match.
- Panel geometry and playback rate must match.
- Clamp pixel values before byte conversion.

## Gotchas

Global SSIM is not local-window SSIM.
Layout detection treats a final dimension of 1, 3 or 4 as channels.
ffmpeg can overwrite the output; callers must select a fresh path.
Preserve source videos and label common displayed coverage when lengths differ.

## Tests

`tests/test_metrics.py` checks layout conversion and closed-form PSNR.
Baseline tests check panel order and fps. Inspect actual media after presentation changes.
