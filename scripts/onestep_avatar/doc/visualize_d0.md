# `visualize_d0.py` — the decoded checkpoint probe

## Objective

Make a checkpoint's behaviour visible, as video, at the distilled refiner's operating points.
One MP4 per σ in `PROBE_SIGMAS = (0.909375, 0.725, 0.421875)`, laid out:

```
ground-truth capture │ frozen base │ LoRA checkpoint
```

A fixed chain and fixed seeds make a sequence of checkpoints directly comparable. `--run` +
`--steps` visualizes several checkpoints from one run in a single call (e.g. `--steps 100 500
1000`), reusing one shared frozen-base decode and GT panel across all of them.

## Data flow

```
LoRA checkpoint + corpus masters ─▶ causal_core rollout at each probe σ
                                 ─▶ VAE decode (offline, no FSDP resident)
                                 ─▶ runs/<name>/probes/step_<N>/<σ>.mp4
```

## Organization logic

**It rolls out through `causal_core`**, so the cached context, the mask and the RoPE positions
are the deployed ones. It does **not** call `onestep_core`, which refuses D0 — a probe for a
non-deployable arm cannot go through a deployment-only path.

A causal rollout writes one latent covering the whole chain, so there is no per-window overlap
to stitch and no seam to get wrong — a simplification that fell out of the rewrite.

It is an **offline** probe: no VAE is resident while FSDP training is stepping.

## Invariants

- The chain is asserted `seed_is_clip_start`, so the probe never depends on GT cache priming.
- D0 must be probed in its **own** state (capture-noised), not a guide-noised approximation —
  that is the whole reason this script exists rather than reusing a deployment renderer.
- **`--teacher-forcing` must match how the checkpoint was trained, or the probe is measuring
  the wrong regime.** It threads straight into `causal_core.rollout`'s own `teacher_forcing`
  flag — refresh is fed the D0 guide tokens (which *are* `z_y` for D0, so no separate target
  tensor is needed) instead of the model's own denoised output. Off (the default) is the
  self-forced regime deployment has to use; a run trained with `train.py --teacher-forcing`
  (check its `config.json`) never saw its own errors accumulate in the cache, so a self-forced
  probe of it evaluates an input distribution training never produced.

## Gotchas

- **σ = 0.0 is excluded from `PROBE_SIGMAS`.** `to_velocity` computes `(sample − denoised)/σ`
  and raises "Sigma can't be 0.0", which once crashed this probe into a silent retry loop for
  hours before it was caught.
- The first attempt used the wrong σ grid entirely. The grid is the distilled model's own, not
  a sweep.

## Owed

**A D1 counterpart.** This is D0-only, and the arm comparison needs the same probe for the
guide-conditioned arms.
