# `score/token_sampling.py` — deterministic native token selection

## Interface and data flow

`sample_indices(tokens, height, width, stride, device, sampler="stride")` returns
int64 indices for generated latent frames only. The caller supplies actual latent
H/W, not pixel dimensions or a square-grid guess. The default reproduces the
original flattened stride exactly. `balanced_2d_midpoint_v1` uses the same budget
`ceil(H*W/stride)` on every generated frame, distributing points over unique
midpoint rows and per-row midpoint columns. Row quotas differ by at most one.

`sampling_record` checks indices against the declared algorithm and records
latent geometry, algorithm/version, stride, selected rows/columns, per-frame and
total count, and spatial/full-token SHA256 values. `index_sha256` encodes each
nonnegative int64 index as an unsigned little-endian 64-bit integer; the record
pins that encoding as `uint64_le_v1`.

## Invariants and scope

Frame 0 is excluded; every later frame uses the same spatial positions. Indices
are unique, sorted and within the full native token grid. Counts match the old
stride even on rectangular grids and budgets that are not perfect squares.
Balanced sampling is an opt-in control, not a scoring-quality result. At 32×32,
stride 16 chooses columns 0 and 16; the balanced 64-point control uses an 8×8
midpoint grid. It covers eight columns and eight rows, not all 32 of either.

`test_token_sampling.py` checks exact default compatibility, equal budgets,
rectangular/degenerate grids, deterministic hashes, clean-frame exclusion and
real production patchifier coordinates. Actual ranking and held-out deletion
quality remain GPU experiments.
