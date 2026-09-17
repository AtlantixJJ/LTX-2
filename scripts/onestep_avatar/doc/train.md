# `train.py` — the AR LoRA training loop

## Objective

SS1.6's block-causal AR loop as a bespoke training step. One optimizer step per chain,
gradients accumulated over `K` blocks.

**Why not a `ltx-trainer` strategy:** `Trainer._training_step` runs exactly one transformer
forward per step and the strategy interface (`prepare_training_inputs` → forward →
`compute_loss`) does not own the forward, so a `K`-block AR chain cannot be expressed as one.
Only the *step* is ours — model loading, LoRA injection, FSDP preparation and checkpoint
plumbing are all reused from `ltx_trainer`.

## Data flow

```
subset JSON (windows.py) ─┐
corpus masters ───────────┴─▶ ChainStore ─▶ Chain(z_g, z_y, z0_base)
                                              │  lazy per CLIP (~5 MB bf16 each)
                                              ▼
                              clip_grid_for ─▶ ClipGrid   (global RoPE, one keyframe)
                              BlockCache.allocate         (once per run, sized for the
                                                           subset's LONGEST clip)
                                              ▼
            prime_cache (GT; ALWAYS called -- one forward even with nothing to prime)
                                              ▼
       per block:  noise_block ─▶ denoise_block ─▶ full-frame MSE + anchor ─▶ backward
                                              ─▶ refresh_block ─▶ evict
                                              ▼
              metrics_rank<r>.jsonl  +  LoRA safetensors with metadata
```

## The loss rule

The loss is unconditional full-frame token MSE:

`mean((z0_pred.float() - z_y_target.float()) ** 2)`.

There is no alpha, subject, mask, or render/capture-disagreement weighting. `ChainStore` reads
only the continuous capture master (and the guide master for D1), so training does not consume
`capture_mask_crop.mp4` or `argavatar_alpha.mp4`. The anchor, when enabled, is the same
full-frame MSE between `z0_pred` and the frozen-base output on the same noised input.

This deliberately makes every predicted token and channel contribute equally. In particular,
the silhouette-boundary disagreement is now part of the learning signal, rather than a region
with a special loss rule.

## Core chain algorithm

`train_chain` operates on one clip's master latent encodes. `geometry.plan` turns that clip
into ordered causal block spans, and `BlockCache` holds only finalized clean-token K/V from the
past:

```text
grid = ClipGrid.build(z_y.shape, fps, geometry)       # global RoPE positions
guide = patchify(z_g)                                 # D1; D0 uses patchify(z_y)
target = patchify(z_y)

prime_cache(target before first selected block)       # no-grad clean GT prefix
for block in selected consecutive blocks:
    [lo:hi] = grid.token_span(block)
    noisy = (1 - sigma) * guide[lo:hi] + sigma * seeded_noise(block)

    z0 = denoise_block(noisy, cache)                  # reads past K/V; writes none
    mse = mean((z0 - target[lo:hi])^2)                # every token, every channel
    anchor = mean((z0 - base[lo:hi])^2) if enabled else 0
    backward((mse + anchor_weight * anchor) / K)      # release this block's activations

    clean = target[lo:hi] if teacher_forcing else detach(z0)
    refresh_block(clean, cache)                       # no-grad: append this block's K/V
    cache.evict_to_context_plus_frame0()              # sink + rolling recent context
optimizer.step()
```

The ordering is denoise → backward → refresh → evict. The denoise pass cannot write the cache
because it carries gradients; refresh is a separate no-grad clean-latent forward, making its
K/V final and reusable by later blocks. Eviction retains the pinned frame-0 sink and the
configured most-recent context. `prime_cache` is deliberately called even at block 0: its
empty forward keeps every FSDP rank at the same `1 + K + K` transformer-forward count.

## The checkpoint contract

A fixed-σ adapter must not be loadable off-condition. Stamped into the safetensors metadata:
σ₀ (or `"mixed"`), the σ level list, `K`, the schedule, the attention kind, **block and cache
geometry**, the subset hash, the **objective**, guide mode, anchor weight, teacher forcing,
LoRA rank/alpha/target.

Cache depth belongs there for the same reason σ does: an adapter trained with two frames of
cached context is a different function from one trained with sixteen, and nothing downstream
can tell by looking at the weights.

`--save-initial` guarantees both `lora_weights_step_00000.safetensors` and
`lora_weights_step_00001.safetensors`, independent of `--save-every`: step 0 is written
after LoRA injection and FSDP `prepare`, and step 1 immediately after the first optimizer
update. `init_lora_weights=True` zero-inits B, so step 0 should decode bit-identically to the
frozen base. Before writing it, `save_lora` checks the gathered exported `lora_B` weights are
exactly zero; failure refuses to create a misleading baseline artifact. This makes
`visualize_d0.py --run <run> --steps 0 1` a self-contained initialization check.

## Arms and knobs

| Flag | What it selects |
|---|---|
| `--objective {bg,white}` | which pair of bundles is read; must match the subset's |
| `--guide-mode {d0,d1}` | `d1` = D1a (guide as the noised init); `d0` = the GT-renoise capacity check, not deployable |
| `--block-latent-frames` / `--context-latent-frames` | SS1.6's geometry and cache depth |
| `--sigma-levels` | one adapter across several operating points |
| `--teacher-forcing` | ablation: the refresh is fed `z_y[i]` instead of `ẑ₀[i].detach()` |
| `--anchor-weight` | SS1.5 row 2; needs `base_denoised.pt` |
| `--timing` | per-step phase breakdown; see "Reading the timing lines" below |
| `--skip-subset-check` | skip the startup read of each source's master; moves a stale-subset failure to step 0 |

**Teacher vs self forcing differ in exactly one tensor** — what `refresh` is handed. Nothing
else in the loop, and nothing in `causal_core`, knows which regime is in play.

## Invariants

- **Every rank runs the same number of transformer forwards per step**, checked by
  `assert_rank_lockstep` *before* the step's forwards run, while the ranks are still in step
  from the previous optimizer update. The count is `1 prime + K denoise + K refresh`. Under
  FSDP `FULL_SHARD` a forward is a round of all-gathers, so ranks that disagree desynchronise
  and the job **hangs** with no error; the guard converts that into a message naming the
  counts. Checking after the fact cannot work — the mismatched collective is already enqueued
  and the check's own gather joins the pile-up. See `doc/causal_core.md` for the 2026-09-16
  instance that motivated it.
- **The K/V cache is allocated once and sized from `ChainStore.max_latent_frames`**, never
  from the first chain drawn. `BlockCache.allocate` caps capacity at the clip it is sized
  against, so a short first clip would make every later long one overflow `LayerKVCache.write`
  — inside a forward, on one rank, which is an FSDP desync rather than a clean failure.
  `train_chain` re-checks `cache.fits(...)` before the step's first forward.
- **A v1 bundle is refused with a pointed error**, never silently reassembled. A reader that
  quietly reconstructs is a second producer of the tensor the trainer learns from.
- **A stale subset is refused at STARTUP, before the 42 GB checkpoint load.**
  `assert_subset_matches_geometry` runs two checks: every chain's blocks must exist in the
  plan implied by its source's recorded `n_latent_frames` (free), and that recorded count must
  match what the stored master actually holds (one ~6 MB bundle read per distinct source;
  `--skip-subset-check` opts out). The second is the one that fires in practice and the first
  cannot see it — a subset frozen before 2026-09-16 is *internally* consistent, because
  `windows.py` took both the count and the plan from the source video. `train_chain` keeps the
  per-chain check as the backstop for a subset whose recorded count is itself stale.
- **A pre-causal window-chain subset is refused**, not reinterpreted: a window index and a
  block index are different numbers over the same clip.
- **A subset frozen against the other objective is refused.**
- `--guide-mode d0` with `--anchor-weight > 0` is refused: the anchor target is computed on
  the guide-noised input, and d0 noises `z_y`.
- σ = 0.0 is refused as a training level — the noiser adds nothing, so loss and gradient are
  identically zero (a quarter of one run trained on nothing before this was caught).

## Reading the timing lines

Every line is prefixed `timing |` and carries the `[rank N]` prefix `ltx_trainer`'s logging
config already installs, so a straggling rank is visible without correlating timestamps.

**Startup stages are always timed**, on every rank: `Accelerator()`, the prompt cache, the
transformer load, `accelerator.prepare`, step-0 `--save-initial`, and the **first** chain load
(which is also the one that allocates the ~2 GB block cache). That list is exactly the span
that used to produce no output at all — the 2026-09-15 4-GPU launch log ends at "causal
geometry" and the next thing in it is a `ChildFailedError`, with nothing to say which of six
minutes-long phases it died in.

**The per-step breakdown is behind `--timing`**, because it is per *block*: `prime_cache`,
then denoise / backward / refresh for each of the `K` blocks, then load / chain / optimizer
for the step. The **first** step prints it regardless of the flag — it is the step that pays
the corpus read and the allocation, so it is the one worth having unconditionally.

The per-block numbers are honest without an explicit `cuda.synchronize` only because
`float(mse.detach())` already forces one inside the same block. Move that read and the
phases start reporting queue time instead of compute.

## Gotchas

- **A zero-byte stdout log is not a hung run.** `python -u`'s unbuffering does not survive
  `conda run`'s subprocess piping here. Check `metrics_rank<r>.jsonl`, which `train.py`
  writes and flushes directly.
- **Never prefix a log message with `[something]`.** `ltx_trainer` installs a `RichHandler`,
  which reads brackets as console markup and silently **drops** an unknown tag. The timing
  lines shipped as `[timing] ...` first and simply were not in the log — not mangled, gone,
  and no grep for them ever matched. (`ltx_trainer`'s own `[rank N]` survives because its
  format string escapes the bracket.)
- **`sigma_for_rank` is `(rank + step) % len(sigmas)`.** Per-*step* cycling aliased the loss
  curve into a sawtooth; per-*rank*-only never trained levels beyond `world_size`. The current
  form mixes levels within a step *and* walks every rank through every level.
- **Reference-token slicing runs the opposite way from the docs.** `flexible` prepends and
  slices `[:, -target_len:]`; the inference-side item **appends**, so this loop slices
  `[:, :target_len]`. Getting it backwards computes the loss against the model's own input and
  looks like very fast convergence, not like an error.

## Tests

`tests/test_train.py` (stubbed transformer) and
`tests/test_causal_core.py::test_the_real_training_loop_runs_a_chain_against_a_real_transformer`
— the only place the loop, `causal_core` and a real cache meet before a GPU.
