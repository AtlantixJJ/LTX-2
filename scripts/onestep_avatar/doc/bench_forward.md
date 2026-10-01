# `bench_forward.py`

## Objective

Measure causal avatar denoising and cache-refresh latency per finalized block.

## Data flow and organization logic

Synthetic tokens at the checkpoint's real geometry populate a steady-state cache.
For each cache depth, warmed synchronized repetitions time `denoise_block` and
`refresh_block` separately. JSON records samples, medians, cache capacity and the
sum of the two medians per block. It makes no comparison against a window renderer.

## Invariants and gotchas

Re-pin cache lengths around repeated refreshes so timed calls see identical cache
coverage. Content is synthetic: this benchmark establishes cost, not quality.
Check `nvidia-smi` and use the `ltx` environment before running.

## Tests

Import/CLI checks verify the interface. Timing needs a free GPU and real model.
