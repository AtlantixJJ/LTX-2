# evaluate/learned_head_media

## Objective

Decode matched saved candidates and preserve RGB/VAE comparison roles.

## Usage

Started by: a direct human or agent CLI. Run from `LTX-2` in conda environment `ltx`.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx python -m scripts.prune.evaluate.learned_head_media --study ../expr/refiner_prune/2.5/learned_heads_dev8_20261012_r2/configs/frozen_study.json --split validation --arms energy learned_joint --replicate-seed 42 --gpu-id 0
```

`--arms` declares matched comparison columns. `--split` and `--replicate-seed` select saved evaluation.

Outputs: Synchronized MP4s, posters, region metrics, full references and rendering records.
See the [run guide](../usage.md#learned-head-phases).

## Data flow

Checked saved predictions, original RGB, VAE GT, teacher and fixed GT regions enter the public media renderer.

## Organization logic

Panels share every source frame and playback rate. RGB and VAE roles remain distinct. Full sources and individual videos are preserved.
Pixel errors use fixed foreground coverage. Pinned AlexNet LPIPS uses GT boxes and 256-pixel crops, with invalid small regions excluded.

## Invariants

No student detection chooses a region. Frame zero never enters a metric. Missing face/hand coverage prevents a quality pass.

## Gotchas

LPIPS and motion amount support review; they do not establish identity or correct action. The delivered media keeps missing coverage explicit.

## Tests

Run `test_learned_head_media.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.
