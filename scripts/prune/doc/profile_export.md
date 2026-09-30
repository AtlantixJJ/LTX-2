# `checks/profile_export.py`

## Objective

Locate the cost of a faithful exported checkpoint on a frozen held-out state.

## Data flow

An attributable score mask and exported checkpoint produce a JSON report with
matched denoiser wall times, per-layer profiler ranges, a direct benchmark of
every masked attention branch using selected versus full attention, and the
maximum output difference.

## Organization

The source and export load sequentially. Each model gets a warmup, repeated
unprofiled wall-time samples, then one profiler pass. Python wrappers add
named ranges around projections, preattention, and attention kernels without
changing their tensor operations. The direct benchmark toggles only the
attention execution mode on the exported model and restores it afterward.

## Invariants and gotchas

The score report must match the source checkpoint fingerprint. One held-out
record is a diagnostic workload, not a long-form speed or quality gate.
Profiler range totals can overlap; compare named projections and kernels,
then use unprofiled wall time for end-to-end latency. The direct branch
benchmark records output differences and does not claim an end-to-end speedup.

## Verification

Run with `--model 2.5 --gpu-id N --masks <report> --exported-checkpoint
<checkpoint>`. Inspect `export_profile.json` and confirm `max_abs` is within
the required parity tolerance before using timing results.
