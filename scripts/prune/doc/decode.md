# `evaluate/decode.py`

## Objective and data flow

Decode dense `[B,C,F,H,W]` latents with the session-owned video VAE. Return
channel-last `[F,H,W,C]` float pixels in `[0,1]` on CPU. Avatar probes consume it.

## Invariants and verification

Use session device and dtype. Accept an explicit generator for reproducible VAE
decoding. Token-record reconstruction is not part of this interface. Verify a
real dense decode on a free GPU when changing decoding behavior.
