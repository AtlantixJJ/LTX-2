# `core/provenance.py`

## Objective

Identify the checkpoint and execution context on every artifact.

## Data flow

Checkpoint header, size, sampled data regions, git state, host, and device -> compact provenance block.

## Organization

Fingerprint reads the header plus fixed 1 MiB samples, avoiding a full 42 GB hash; run_id adds PID.

## Invariants and gotchas

The fingerprint is an identity cue, not a whole-file integrity proof. Compare
it before reusing model-specific masks or scores. `method_source_hashes()`
pins the deployed script, rollout core, task constants, and Phase-1 path so a
parity or baseline artifact cannot silently survive a code edit.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
