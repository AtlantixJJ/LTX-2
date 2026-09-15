# `onestep_core.py` — the deployment rollout

## Objective

The deployed counterpart of `train.py`'s loop: one denoise plus one clean cache refresh per
block, over the clip's **master** latent, at σ₀.

Built **on** `causal_core`, never a copy of it. What differs from training is only the
schedule — one forward instead of `k2`'s two — and that the guide enters as the init rather
than the block being re-encoded from its own pixels.

## Data flow

```
z_g master + σ₀ ─▶ ClipGrid ─▶ plan ─▶ BlockCache.allocate
                                  ▼
              per block: noise_block ─▶ denoise_block ─▶ refresh_block ─▶ evict
                                  ▼
                        RolloutResult(latent, forwards, …)
```

## Organization logic

It is literally the same three calls in the same order the training loop makes. That is the
point: a train/deploy mismatch would have to be an edit to `causal_core`, not a divergence
between two implementations that were supposed to agree.

Since the causal rewrite this path **stopped calling `refine_core` entirely** — the sliding
window with a frozen carryover at latent index 1 is gone, and with it
`make_window_state`/`run_schedule` on this path. `refine_core` remains the `k2` baseline's
implementation and the source of every frozen `k2` number.

## Invariants

- **`RolloutResult.forwards` counts BOTH passes per block** — the denoise and the refresh. A
  compute number quoted from it is therefore comparable with `k2`'s two and is not flattered
  by omitting the refresh.
- **`guide_conditionings` refuses `d0`.** D0 noises the capture latent, and there is no `z_y`
  at inference; a deployable path must not be able to express it.
- A causal rollout writes one latent covering the whole chain, so there is **no per-window
  overlap to stitch** and no seam to get wrong.

## Tests

Covered through `tests/test_causal_core.py` (the shared rollout) plus `scripts/prune`'s
`refine_task` guards, which refuse an off-grid σ₀ or a multi-step schedule.
