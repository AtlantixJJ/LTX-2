# `checks/method_parity.py`

## Objective

Prove pruning rollout matches the deployed refine script bit for bit.

## Data flow

One real clip, geometry, seed -> reference subprocess latents and harness latents -> method_parity.json.

## Organization

Requires at least two windows so carryover is exercised; compares torch.equal before decoding.

## Invariants and gotchas

Rerun after any tensor-moving change. The report records method-source content
hashes; sweep preflight compares those hashes and the checkpoint fingerprint,
so a stale pass file is rejected.
The reference command uses the current run script's latent-frame CLI flags.
`--video` with `--expected-source-sha256` allows parity checks when the
historical `source.mp4` corpus is unavailable.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
