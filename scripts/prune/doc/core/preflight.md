# `core/preflight.py`

## Objective and data flow

Validate model/checkpoint paths, distilled sigma-grid metadata and requested GPU
headroom before loading. `--dump-caps` writes model capabilities and provenance.

## Invariants and verification

Select the requested CUDA device as current for backend launches. GPU memory
availability is checked for that device. Fresh GPU work also starts with
`nvidia-smi`; CPU import checks do not establish model-loading capacity.
