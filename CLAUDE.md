# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

LTX-2 is a DiT-based audio-video foundation model. This repo is a `uv` monorepo with four packages under `packages/`:

* **`ltx-core`** -- model definitions (transformer, video VAE, audio VAE, vocoder, upsampler, Gemma text encoder), diffusion components (schedulers, guiders, patchifiers), conditioning, quantization, and checkpoint loading. Everything else depends on this.
* **`ltx-pipelines`** -- high-level inference pipelines (text/image-to-video, audio-to-video, IC-LoRA, lip dub, retake, etc.) built from `ltx-core` building blocks. See `packages/ltx-pipelines/CLAUDE.md` for the pipeline table, guidance/sigma/LoRA conventions, and shared building blocks (`utils/blocks.py`, `utils/denoisers.py`) -- read it before touching anything under `packages/ltx-pipelines/`.
* **`ltx-trainer`** -- LoRA/full fine-tuning for all conditioning modes (T2V, I2V, video extension/inpainting/outpainting, A2V/V2A, IC-LoRA, etc.), unified through `FlexibleStrategy`. See `packages/ltx-trainer/AGENTS.md` (also linked as `CLAUDE.md`) for the full architecture, config system, and latent-space constants -- read it before touching anything under `packages/ltx-trainer/`.
* **`ltx-kernels`** -- optional compiled CUDA kernels (blockwise FP8/FP6 GEMM, multi-GPU all2all). Excluded from the default `uv` workspace so a plain `uv sync` never requires a CUDA toolchain; opt in explicitly.

There is also a top-level `scripts/` directory of standalone analysis/research scripts (e.g. VAE latent visualization). These are not part of any package -- they `sys.path.append` into `packages/*/src` directly rather than importing an installed package, and each is a self-contained probe rather than a shared library.

## Setup

`uv sync` (optionally `uv sync --group kernels` for the compiled CUDA kernels) builds the workspace, but on this machine the packages are also editably installed into a pre-existing conda env named **`ltx`** (`ltx-core`, `ltx-pipelines`, `ltx-trainer` all point back into this checkout's `packages/*/src`). **Run all Python in this repo -- pipelines, training, and `scripts/` -- through the `ltx` conda env, not `uv run` or base/system Python**:

```bash
conda run -n ltx python -m ltx_pipelines.distilled ...
# or
conda activate ltx && python -m ltx_pipelines.distilled ...
```

`uv sync` is still what actually installs/updates the editable packages and their dependencies into that env when `pyproject.toml`/lockfile changes -- it's a setup step, not a run step.

Model checkpoints are downloaded separately via the Hugging Face CLI (see README.md Quick Start) into `models/` or `checkpoints/` -- both are gitignored, along with weight file extensions (`*.safetensors`, `*.ckpt`, `*.pt`, `*.sft`) and common media extensions.

## Common commands

Run inference (distilled pipeline, fastest):

```bash
conda run -n ltx python -m ltx_pipelines.distilled \
    --distilled-checkpoint-path models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
    --spatial-upsampler-path    models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --gemma-root models/gemma-3-12b \
    --seed 42 --output-path output.mp4 --prompt "..."
```

Other pipeline entry points live in `packages/ltx-pipelines/src/ltx_pipelines/` (see the pipeline table in `packages/ltx-pipelines/CLAUDE.md`); each is runnable the same way via `conda run -n ltx python -m ltx_pipelines.<module>`.

Training:

```bash
cd packages/ltx-trainer
conda run -n ltx python scripts/train.py configs/t2v_lora.yaml                    # single GPU
conda run -n ltx accelerate launch scripts/train.py configs/t2v_lora.yaml         # multi-GPU
```

Ad-hoc analysis scripts under top-level `scripts/`:

```bash
conda run -n ltx python3 scripts/<name>.py
```

Lint / format (repo-wide ruff config in root `pyproject.toml`: `py311` target, 120 line length). `ruff`/`pytest`/`pre-commit` are not installed in the `ltx` conda env -- these still go through `uv run`:

```bash
uv run ruff check .
uv run ruff format .
uv run pre-commit run --all-files
```

Tests (pytest is a root dev dependency; run from within the package under test, e.g. `packages/ltx-trainer`):

```bash
cd packages/ltx-trainer && uv run pytest
```

## Architecture notes

**Data flow**: Video pixels / audio waveform / text prompt -> modality-specific VAE/Gemma encoders -> latents & embeddings -> asymmetric dual-stream transformer (48 shared blocks, 14B video-stream / 5B audio-stream params, bidirectional audio-visual cross-attention with cross-modality AdaLN) -> denoising loop -> VAE decoders back to pixels/mel -> vocoder to waveform. Full diagram and per-component detail in `packages/ltx-core/README.md`.

**Video VAE latent space** (used throughout `ltx-core`/`ltx-pipelines`/`ltx-trainer`): 128 latent channels, 32x spatial compression, 8x temporal compression (`frames % 8 == 1` constraint), width/height divisible by 32. Encoder path: `patchify(patch_size_hw=4)` -> `conv_in` -> `down_blocks` -> `conv_norm_out` (PixelNorm) -> `conv_out` -> `per_channel_statistics.normalize()`. `PerChannelStatistics.normalize`/`un_normalize` are a per-channel affine (mean/std from `mean-of-means`/`std-of-means` buffers in the checkpoint) -- `0.0` in normalized latent space is the per-channel mean, not literal black/empty pixels.

**Model interface is modality-based**: both `ltx-pipelines` and `ltx-trainer` construct `Modality` objects (`ltx_core.model.transformer.modality.Modality`, frozen dataclass -- use `dataclasses.replace()`) for video and audio and call the transformer as `model(video=video, audio=audio, perturbations=None)`. `sigma` is per-batch (used for AdaLN/cross-modality conditioning); `timesteps` is per-token (`sigma * denoise_mask`).

**Version handling**: LTX-2 (19B) vs LTX-2.3 (22B) differences (feature extractor V1/V2, caption projection location, vocoder) are detected automatically from the checkpoint config -- there is a unified API, no manual version branching needed in calling code.

**Memory/streaming**: both inference and training support block streaming (`ltx_core.block_streaming`, `BlockStreamingWrapper`) to run models larger than available GPU memory by keeping only a few transformer blocks resident at a time, plus FP8 quantization backends (`ltx_core.quantization`) and batch-splitting for oversized guidance batches (`BatchSplitAdapter` in `ltx-pipelines`).

When making non-trivial changes inside `packages/ltx-pipelines/` or `packages/ltx-trainer/`, consult and update their package-level `CLAUDE.md`/`AGENTS.md` files -- both explicitly document a maintenance contract (e.g. the pipelines doc must stay in sync with `__init__`/`__call__` signatures and sigma/guidance handling; the trainer doc's config tables must stay in sync with `configs/*.yaml` and `docs/configuration-reference.md`).
