# `core/ltx_adapter.py` — private upstream access

## Objective

Keep private upstream imports and attribute access in one reviewed adapter.

## Data flow

A session supplies checkpoint paths, dtype and device.
The adapter opens the upstream transformer, encoder or decoder context.
The context yields a resident model and frees it when the caller exits.

## Organization logic

- `transformer_ctx` delegates to `DiffusionStage._transformer_ctx`.
- `video_decoder` builds through `VideoDecoder._decoder_builder`.
- `video_encoder` builds through `ImageConditioner._build_encoder`.
- VAE contexts use `torch.no_grad()` and `gpu_model`.

Worked example: leaving `with video_decoder(path, dtype, device)` ends the resident decoder lifetime.

## Invariants

No other pruning module accesses an underscore-prefixed upstream symbol.
The adapter owns model lifetime. Callers own sampling state and schedules.

## Gotchas

The source header records reviewed upstream revision `9a1c49b`.
Review these accesses when upstream interfaces change. The revision does not instruct a submodule pin update.

## Tests

`tests/test_ltx_adapter.py` checks the private-access boundary and recorded revision.
A changed upstream interface also needs a native model load.
