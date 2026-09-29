# `report/summarize_phase0.py`

## Objective

Collect gate, benchmark, and corpus numbers into analysis_summary.json and figures.

## Data flow

Stable artifacts resolved via artifacts.py -> summary tables and charts.

## Organization

Missing files are represented as absent values; report prose should quote emitted scalars.

## Invariants and gotchas

Missing evidence is not a pass. Rebuild summaries after upstream artifacts change; current source has legacy teacher naming.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
