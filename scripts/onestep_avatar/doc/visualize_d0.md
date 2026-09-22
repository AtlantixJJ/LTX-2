# `visualize_d0.py` — the decoded checkpoint probe

> **Both arms, since 2026-09-21.** `--guide-mode d1` noises the guide `z_g` instead of the
> capture, which is what [G4](known_gaps.md#g4--no-d1-probe) was owed; the file keeps its
> `visualize_d0` name. The tool still does not *validate* an adapter's recorded conditions
> against the flags it is given, so a probe at a σ, arm or cache depth the adapter was not
> trained at remains the operator's explicit off-condition experiment
> ([G3](known_gaps.md#g3--checkpoint-and-artifact-conditions-are-recorded-but-not-enforced)).
> `--teacher-forcing` now passes the capture target through explicitly, so a D1 teacher-forced
> probe refreshes from `z_y` and not from the render
> ([G2](known_gaps.md#g2--generic-teacher-forced-rollout-refreshes-from-the-guide-not-the-target)).

## Objective

Make a checkpoint or the frozen base's behaviour visible at explicitly selected distilled
operating points. `--probe-sigmas` defaults to
`PROBE_SIGMAS = (0.909375, 0.725, 0.421875)` and accepts any unique, nonzero values on the
selected model's schedule. Checkpoint mode writes one comparison MP4 per σ, laid out:

`ground-truth capture | frozen base | LoRA checkpoint`, side by side.

**`--schedule` selects the denoising interval decomposition.** Omitted, the probe is the
one-step student (`[probe sigma, 0]`). `--schedule 0.725 0.421875 0` is the two-step causal
teacher arm: the same rollout, the same cached history, one more denoising forward per block.
It takes exactly one `--probe-sigmas` value and refuses a schedule that starts anywhere else,
because a multi-step arm probes the one operating point it starts from. The manifest records
the schedule and the per-block denoise/refresh forward counts separately, so a latency claim
cannot quietly fold the refresh into "one step".

**The arm is one tensor.** `--guide-mode d0` (default) noises the capture master `z_y`;
`--guide-mode d1` noises the guide master `z_g`. The decoded reference panel, the loss-side
target and the clean first-frame condition `c0` are the capture in **both** — the guide's own
frame 0 is a render composite, never the supplied real first frame. `d1` requires the guide
bundle on disk and raises a pointed error naming `GUIDE_COMPOSITING_VERSION` when it is
missing, rather than falling back to the capture.

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

`--base-only` removes the checkpoint requirement. With two or more sigma arms it also writes
`capture | first sigma | second sigma`, plus an uncaptioned capture and uncaptioned individual
base videos. `--block-latent-frames` and `--context-latent-frames` override the deployed
defaults through `causal_core.CausalGeometry`; the manifest records the resolved geometry.

The matched sigma-1 experiment from the 2026-09-20 plan is:

```bash
conda run -n ltx python -m scripts.onestep_avatar.visualize_d0 \
  --subset ../expr/onestep_avatar/windows/t2r2.json \
  --base-only --probe-sigmas 0.909375 1.0 \
  --block-latent-frames 2 --context-latent-frames 15 \
  --teacher-forcing --seed 42 --no-frame-labels \
  --output ../expr/onestep_avatar/runs/base-block-flicker-sigma-20260920/teacher_forced \
  --gpu-id 0
```

Omit `--teacher-forcing` for the corresponding generated-history arms.

## Data flow

LoRA checkpoint or frozen base + corpus masters → one `causal_core` rollout per probe σ, over the span
`--span` selects → VAE decode (offline, no FSDP resident) →
`runs/<name>/probes/step_<N>/<σ>.mp4`. Each rollout is `2·len(plan)` forwards: one denoise and
one refresh per block, with no priming, since it starts at block 0.

`_frame_labels` turns the plan into one caption per decoded pixel frame and `_stamp` burns it
into all three panels just before `t3_video` writes them. The manifest records `span`,
`blocks_rolled_out`, `latent_frames_covered`, conditioning, model paths/fingerprint, timing,
history policy, noise provenance and every output path. Generated latents and the explicit
per-block epsilon tensors are saved as `.pt` artifacts before decoding.
The diffusion VAE receives a fresh generator with the probe seed for every arm, so decoder
noise is identical rather than becoming an unrecorded difference between sigma arms.

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
- **Every sigma arm uses the same epsilon tensors.** The probe creates the tensors once with
  the established `seed + block_index` convention and passes them to `causal_core.rollout`.
  `c0` is restored after mixing, so changing sigma cannot change the first-frame condition.
- **Sigma values belong to the selected checkpoint's schedule.** Zero, duplicates and values
  off the model schedule fail before the transformer is loaded.
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

## Outputs

The output directory contains `manifest.json`, `block_epsilons.pt`, one raw latent and one
uncaptioned MP4 for each frozen-base sigma arm, and `capture.mp4`. Base-only runs with two or
more arms contain a three-panel sigma comparison. Checkpoint runs additionally contain the
existing captioned `capture | base | LoRA` comparisons and raw checkpoint latents.

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
