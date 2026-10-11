# `score/token_sampling.py` — deterministic generated-frame samples

## Objective

Select and identify spatial token samples without including the clean first frame.

## Data flow

Input is token count, latent height and width, stride and sampler name.
`sample_indices` returns an int64 vector on the requested device.
`sampling_record` checks that vector and returns geometry, coverage and hashes.

## Organization logic

Each frame occupies `H*W` consecutive tokens.
`stride` selects spatial offsets `0,stride,2*stride,...` on every generated frame.
`balanced_2d_midpoint_v1` keeps the same budget, `ceil(H*W/stride)`.
It chooses an aspect-dependent row count, divides quotas across rows and selects midpoint columns.

Frame starts begin at `H*W`, excluding clean frame zero.
`index_sha256` encodes each integer as unsigned 64-bit little-endian bytes.
The record identifies spatial offsets and complete token indices.

Worked example: a 32×32 frame at stride 16 has 64 samples.
Stride covers columns 0 and 16 in every row.
Balanced sampling covers an 8×8 midpoint grid at the same budget.
With three latent frames, either sampler returns 128 indices.

## Invariants

- The grid contains complete frames and at least one generated frame.
- Counts and stride are positive integers.
- Recorded indices equal the deterministic selection.
- Hash encoding is `uint64_le_v1`.

## Gotchas

Neither sampler covers the full grid. Record row and column coverage with scores.
Sample changes affect estimates, although the forward still uses all tokens.

## Tests

`tests/test_token_sampling.py` checks budget, uniqueness, coverage, hashing and actual patchifier coordinates.
