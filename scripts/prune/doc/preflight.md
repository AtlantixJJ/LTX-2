# `core/preflight.py`

## Objective

Reject missing checkpoints, invalid model choices, and insufficient GPU headroom before expensive loading.

## Data flow

CLI model, sampler, and GPU -> validated RefinerModel; --dump-caps -> caps.json.

## Organization

The fast check inspects paths, metadata, device and memory before a Session is opened.

## Invariants and gotchas

Check nvidia-smi before GPU work. A capability dump is a stable gate artifact
and must include provenance. `--check-sweep-prereqs` binds parity, format-2
calibration, and the unpruned Phase-1 baseline to the current checkpoint,
geometry, and source-code content hashes before a sweep starts.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
