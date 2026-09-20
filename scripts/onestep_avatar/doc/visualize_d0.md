# `visualize_d0.py` — the decoded checkpoint probe

> **D0 only** ([G4](known_gaps.md#g4--no-d1-probe)): a D1 counterpart is owed, and this tool
> does not refuse a D1 adapter. It also validates none of the adapter's recorded conditions —
> it always uses the default deployed geometry and all three `PROBE_SIGMAS`, so a probe at a σ
> or a cache depth the adapter was not trained at is off-condition and unlabelled
> ([G3](known_gaps.md#g3--checkpoint-and-artifact-conditions-are-recorded-but-not-enforced)).
> `--teacher-forcing` here goes through the generic rollout, which refreshes from the guide
> ([G2](known_gaps.md#g2--generic-teacher-forced-rollout-refreshes-from-the-guide-not-the-target)).

## Objective

Make a checkpoint's behaviour visible, as video, at the distilled refiner's operating points.
**One rollout per σ in `PROBE_SIGMAS = (0.909375, 0.725, 0.421875)` — three in total** — and
one MP4 each, laid out:

`ground-truth capture | frozen base | LoRA checkpoint`, side by side.

Each rollout covers the **whole clip** (`--span clip`, the default): block 0 through the last
full block, the same sequence deployment produces, so drift accumulated across the AR rollout
is visible instead of being truncated at the training chain's `K` blocks. `--span chain`
restores the older behaviour of covering only the subset chain's own blocks, for a
like-for-like comparison with probes taken before this change.

Every frame carries a burned-in caption — `latent N · rollout step M (block a-b)` — so a
drift seen in the video can be traced to the block that produced it without counting frames by
hand off a 137-frame clip. `--no-frame-labels` turns it off.

A fixed clip and fixed seeds make a sequence of checkpoints directly comparable. `--run` +
`--steps` visualizes several checkpoints from one run in a single call (e.g. `--steps 100 500
1000`), reusing one shared frozen-base decode and GT panel across all of them.

## Data flow

LoRA checkpoint + corpus masters → one `causal_core` rollout per probe σ, over the span
`--span` selects → VAE decode (offline, no FSDP resident) →
`runs/<name>/probes/step_<N>/<σ>.mp4`. Each rollout is `2·len(plan)` forwards: one denoise and
one refresh per block, with no priming, since it starts at block 0.

`_frame_labels` turns the plan into one caption per decoded pixel frame and `_stamp` burns it
into all three panels just before `t3_video` writes them. The manifest records `span`,
`blocks_rolled_out`, `latent_frames_covered` and `frame_labels` alongside the existing fields.

## Organization logic

**It rolls out through `causal_core`**, so the cached context, the mask and the RoPE positions
are the deployed ones. It does **not** call `onestep_core`, which refuses D0 — a probe for a
non-deployable arm cannot go through a deployment-only path.

A causal rollout writes one latent covering the whole span, so there is no per-window overlap
to stitch and no seam to get wrong — a simplification that fell out of the rewrite. The GT
panel is sliced to exactly the frames the rollout covered, so the panels stay frame-aligned at
either span.

It is an **offline** probe: no VAE is resident while FSDP training is stepping.

## Invariants

- **The subset must be a block-chain freeze.** `main` constructs `ChainStore` directly, and
  since S2 of the 2026-09-17 cleanup plan that constructor is where a pre-causal window-chain
  subset is refused (moved out of `train.main`, so both readers share the check) — a pointed
  `SystemExit` instead of `KeyError: 'latent_time_scale'` deep inside geometry setup.
- **The rollout always starts at block 0**, so `prime_cache` is never called and the probe
  never depends on a teacher-forced GT prefix in its cache. At `--span clip` this is automatic;
  at `--span chain` the chain is asserted `seed_is_clip_start` for the same reason.
- **`c0` is the clean first-frame condition in the probe too** — latent frame 0 of the same
  objective's `z_y`, passed as `first_frame_condition`, exactly as
  [`core_algorithm.md` §3](core_algorithm.md#3-the-conditioning-contract) requires of every
  caller of the rollout.
- **The caption must follow the VAE's frame mapping, not a ratio.** Latent frame 0 is one
  pixel frame and every later latent frame is `time_scale` of them, so pixel frame `p` is
  latent `0 if p == 0 else ceil(p / time_scale)`. A wrong caption is worse than none: it sends
  a reader to the wrong block. Pinned by `tests/test_visualize_d0.py`.
- **`_stamp` writes only the caption band.** Rendering a whole frame through PIL would
  round-trip every pixel through uint8 and quantize the image under review; a probe must not
  alter what it is showing. Also pinned by the tests.
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

## Tests

`tests/test_visualize_d0.py` (CPU, no model): `--span clip` covers the whole clip and `--span
chain` only the chain's blocks; captions follow the VAE's frame mapping and cover exactly the
decoded frames; `_stamp` leaves every pixel below the band untouched.

## Owed

**A D1 counterpart.** This is D0-only, and the arm comparison needs the same probe for the
guide-conditioned arms.
