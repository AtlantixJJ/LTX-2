# `core/model_registry.py` — checkpoint and capability resolution

## Objective

Resolve a supported model key into component paths, version, sigma grid and capabilities.

## Data flow

`resolve` accepts `2.3` or `2.5` and optional component overrides.
It builds `ModelPaths`, checks paths, probes VAE scales and transformer metadata, then returns a frozen `RefinerModel`.

## Organization logic

The checkpoint root is `LTX_CHECKPOINTS` when set, otherwise the workspace `checkpoints/` directory.
LTX-2.5 uses split distilled transformer, Gemma4 text encoder and video VAE files.
LTX-2.3 uses a monolithic distilled checkpoint and the registered Gemma3 directory.
New work defaults to 2.5.

`probe_caps` reads the transformer configuration through safetensors metadata.
Defaults follow the native configurator. FFN width is `4*num_heads*head_dim`, matching the upstream implementation.
`resolve` returns the shared `DISTILLED_SIGMA_VALUES` grid. It does not construct a denoising schedule.

Worked example: overriding only the 2.5 transformer keeps the registered text encoder and VAE.
Missing components cause `SystemExit` with paths and a download hint. No download starts automatically.

## Invariants

- Probe head counts and dimensions from the selected checkpoint.
- Keep model keys on artifacts; do not reuse generation-specific head indices.
- Pass probed VAE factors to grid construction and model loading.

## Gotchas

An exported transformer can have per-layer widths. Use
[`checkpoint_mask_widths`](../score/export_pruned.md) for mask validation.
`ModelCaps` alone does not describe those widths.
The shared distilled grid is not a dedicated LTX-2.5 pipeline preset.

## Tests

The CPU suite checks registry interfaces. `tests/test_smoke.py` checks the real 2.5 checkpoint under the GPU gate.
Model-facing changes also need the [native gates](../VALIDATION.md).
