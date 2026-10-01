# `core/geometry.py`

## Objective and data flow

Read checkpoint metadata to probe VAE spatial/temporal scales and their source.
Convert aligned latent counts to pixel/latent shapes with round-trip validation.

## Invariants and verification

A reported default scale is a visible fallback, not a measured encoder layout.
Pixel counts follow `time * (latent_frames - 1) + 1`. No overlap or sliding-window
validation lives here. `tests/test_geometry.py` checks shape round trips.
