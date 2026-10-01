# `evaluate/metrics.py`

## Objective and data flow

Normalize video layouts, compute PSNR/global SSIM, and write synchronized
source/reference/candidate MP4 panels. Avatar probes share the media functions.

## Invariants and verification

PSNR requires matching shapes; identical pixels produce infinity. Global SSIM
is not local-window SSIM. Video panels share frame geometry and playback rate;
coverage is the common frame prefix. `tests/test_metrics.py` covers layout and
closed-form pixel metrics. Inspect generated media after presentation changes.
