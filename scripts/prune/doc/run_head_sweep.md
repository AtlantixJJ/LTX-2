# `run_head_sweep.sh`

## Objective

Launch one iterative head-sparsity candidate per GPU and evaluate each mask.

## Data flow

Model key, GPU list, sparsity list, calibration cache and parity artifact -> per-target scores and phase1 gate files.

## Organization

Shell launcher checks prerequisites, extracts the emitted score path, and checks requested sparsity.

## Invariants and gotchas

The launcher validates parity, calibration, and baseline via `core.preflight`
before allocating work. Each worker records score/evaluation status, and any
failed worker makes the shell exit nonzero. Outputs and a machine-readable
`sweep_manifest.json` live in one unique run directory. The completion line
appears only after every worker succeeds; it does not by itself establish
quality or speed acceptance.
Set `T2_VIDEO` and `T2_SHA256` together for a hashed external source; set
`SWEEP_ROLLOUT_WINDOWS` for a bounded smoke run. The prerequisite baseline and
method parity must have current source hashes.

## Verification

Check `bash -n scripts/prune/run_head_sweep.sh` plus a controlled sweep run. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
