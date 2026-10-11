# `core/geometry.py` — VAE scales and shape conversion

## Objective

Read VAE scale factors and convert latent geometry to pixel geometry.

## Data flow

Read safetensors metadata from the VAE, then the transformer.
Use the first VAE encoder or decoder block list found.
Return the factors and source label: `video_vae`, `transformer` or `default`.

## Organization logic

`probe_scale_factors` delegates scale calculation to `SpatioTemporalScaleFactors.from_model_config`.
If neither checkpoint supplies a block list, it returns the upstream default.

`pixel_frames_for(F)` computes `time*(F-1)+1` for positive latent frame counts.
`latent_shape_for` creates a batch-one pixel shape and converts it back to a latent shape.
It raises if the latent frame count changes.

Worked example: temporal scale 8 maps three latent frames to 17 pixel frames.
Spatial scale 32 maps a 64-pixel dimension to two latent positions.

## Invariants

- Record whether factors were read or defaulted.
- Use the same factors for grid construction and model loading.
- Temporal conversion must preserve the requested latent frame count.

## Gotchas

A `default` result is a fallback, not an encoder measurement.
This module does not select overlap, context or an attention mask.

## Tests

`tests/test_geometry.py` checks pixel-to-latent round trips.
Native input checks must also verify recorded geometry and fps.
