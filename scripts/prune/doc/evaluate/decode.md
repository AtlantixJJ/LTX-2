# `evaluate/decode.py` — dense latent decoding

## Objective

Decode dense video latents through a session-owned video VAE.

## Data flow

`decode_latent` receives a session, `[B,C,F,H,W]` latent, decoder and optional generator.
It sends the latent to the session device in BF16.
It concatenates decoder chunks, converts to float32, clamps to `[0,1]`, and returns pixels on CPU.

## Organization logic

The decoder owns pixel generation and chunk layout.
The helper forwards the generator without drawing replacement noise.
Ordinary batch-one calls return channel-last `[F,H,W,C]` pixels.

Worked example: temporal scale 8 maps three latent frames to 17 pixel frames.
The baseline producer checks the decoded count.

## Invariants

- Use the session device and `DTYPE`.
- Pass the intended decoder generator.
- Do not rebuild token records here.

## Gotchas

This helper is not a CLI.
It does not label panels or verify saved-source provenance.
The caller owns the decoder context and validates coverage.

## Tests

`tests/test_whole_clip_baseline.py` checks decoded coverage with a fixture decoder.
A decoder behavior change needs a real dense decode on a free GPU.
