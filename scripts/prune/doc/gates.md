# `evaluate/gates.py`

## Objective

Turn baseline, candidate, and timing artifacts into a pruning verdict.

## Data flow

Matched baseline/candidate Phase-1 JSONs and their per-window profile envelopes ->
verdict JSON with per-criterion checks, measurements, reasons, and exit status.

## Organization

`verdict` binds checkpoint source, VAE, source-video hash, geometry, seed,
schedule and frame windows. It checks held-out T0 delta, T1 PSNR drop, T2
coverage/drift, T3 file existence, and steady-window speedup on the same GPU.
`--mode short` is a regression check; long form always needs at least 200 windows.

## Invariants and gotchas

Legacy bare timing lists have no run identity and fail closed. The default
tolerances are T0 delta ≤0.05, T1 PSNR drop ≤0.5 dB, T2 slope drop ≤5 dB per
100 windows, and speedup ≥1.4. A missing criterion fails the verdict; CLI
exit status is nonzero.

## Verification

Check [`tests/test_gates.py`](../tests/test_gates.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
