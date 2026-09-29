# `core/refine_core.py`

## Objective

Provide the one sliding-window denoising implementation shared by deployment and pruning evaluation.

## Data flow

Pixel window, its real fps, VAE latent, carried frame, prompt, and k2 sigmas -> refined latent and next carryover.

## Organization

WindowGeometry plans strides; tools and state builders preserve the causal keyframe; run_schedule and refine_window own stepping.

## Invariants and gotchas

Index 0 is re-encoded from pixels. Carryover starts at latent index 1. Keep fps, seed, overlap, and schedule aligned with deployment; rerun method_parity after tensor changes.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
