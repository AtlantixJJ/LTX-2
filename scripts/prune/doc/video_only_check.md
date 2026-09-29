# `checks/video_only_check.py`

## Objective

Check that a video-only transformer build matches the audio-video build within bf16 tolerance.

## Data flow

Several real clips from distinct subjects -> paired outputs and gate JSON.

## Organization

Build each configurator once, reuse across subjects, compare at refiner geometry.

## Invariants and gotchas

Dropping audio is accepted only after this checkpoint-specific gate passes; subject diversity matters.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
