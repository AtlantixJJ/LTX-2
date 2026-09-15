# `bench_forward.py` — cost per finalized chunk

## Objective

Measure the plan's headline compute claim, honestly. The causal scheme moved that number in
both directions at once:

- the denoising forward covers only the **block** (2 latent frames), not a 4-latent-frame
  window — half the query tokens;
- but the cache refresh is a **second** forward, and every block's queries attend over a
  longer key sequence (the pinned sink plus retained context) than a self-contained window.

So the honest unit is **wall clock per finalized chunk of 16 pixel frames**: `k2`'s two window
forwards against the causal path's denoise + refresh, sweeping `--context-latent-frames`.

## Data flow

```
model ─┬─ refine_core window forward × 2      ──▶ k2 baseline seconds/chunk
       └─ causal denoise + refresh, per depth ──▶ causal seconds/chunk, ratio vs k2
```

`k2` is timed **through `refine_core`** — the module every frozen `k2` number came from — so
the comparison is against the real baseline, not a reimplementation of it.

## Organization logic

**It times a steady-state block, never block 0.** Block 0's empty cache would flatter the
causal path by exactly the attention the cache adds.

**A FLOP count does not settle this**, which is why the script exists: the predecessor
benchmark already caught a FLOP-plausible arm (extra reference tokens) being *slower* than
the baseline it was meant to replace — 1.05× `k2` measured, against a 1.09× estimate.

## Status

The estimate to replace: 0.53× `k2` at `context=2`, 0.59× at `context=4`. **Not yet run under
the causal scheme**, so the plan's "roughly half the compute" is currently an estimate.

| | query tokens | key tokens | forwards |
|---|---|---|---|
| `k2` window | 4096 | 4096 | 2 |
| causal, `context=2` | 2048 + 2048 | ≤ 5120 | 2 |
| causal, `context=4` | 2048 + 2048 | ≤ 7168 | 2 |
