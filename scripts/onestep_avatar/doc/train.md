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
corpus masters ───────────┴─▶ ChainStore ─▶ Chain(z_g, z_y, loss_weights, z0_base)
                                              │  lazy per CLIP (~5 MB bf16 each)
                                              ▼
                              clip_grid_for ─▶ ClipGrid   (global RoPE, one keyframe)
                              BlockCache.allocate         (once per run)
                                              ▼
                     prime_cache (GT, mid-clip chains only)
                                              ▼
       per block:  noise_block ─▶ denoise_block ─▶ masked_mse + anchor ─▶ backward
                                              ─▶ refresh_block ─▶ evict
                                              ▼
              metrics_rank<r>.jsonl  +  LoRA safetensors with metadata
```

## The loss rule (SS1.5)

**Full-frame loss, down-weighted on the silhouette disagreement band** — and nothing else is
weighted at all.

`disagreement_weights(record, band_weight)` computes `1 − (1 − w)·|render_alpha − capture_mask|`.
The band is the **soft** symmetric difference: both grids are area fractions at latent
resolution, so a half-covered boundary cell is half-disputed, not wholly.

This replaced five *subject* masks (`render`/`capture`/`union`/`intersection`/`none`). Each
gave weight zero everywhere outside the subject at time `t`, which is exactly where the ghost
band lives — so a subject-masked loss cannot teach the model to repair the ghost, the region
SS1.2 calls the learning signal. There was nothing to keep.

`--disagreement-weight 1.0` is a plain full-frame loss, and the grids are then not even read.
**Neither are they read under `--guide-mode d0`, at any weight** — the band is render_t ⊖
capture_t, undefined with no render in play, which is d0's whole point (SS1.3: reduces to
ordinary flow-matching on real video). `ChainStore` gates the read on `with_guide`, not just
`band_weight < 1.0`, so d0 works against capture-only precompute the same way it already skips
`z_g` — no paired-precompute artifact (`argavatar_ltx_vae_latent.pt`, `loss_mask_grids.pt`,
`capture_mask_crop.mp4`) is needed to train d0. A non-default `--disagreement-weight` with `d0`
is refused rather than silently ignored, same shape as the anchor-weight guard below.

## The checkpoint contract

A fixed-σ adapter must not be loadable off-condition. Stamped into the safetensors metadata:
σ₀ (or `"mixed"`), the σ level list, `K`, the schedule, the attention kind, **block and cache
geometry**, the subset hash, the **objective**, the disagreement weight, guide mode, anchor
weight, teacher forcing, LoRA rank/alpha/target.

Cache depth belongs there for the same reason σ does: an adapter trained with two frames of
cached context is a different function from one trained with sixteen, and nothing downstream
can tell by looking at the weights.

`--save-initial` writes `lora_weights_step_00000.safetensors` right after LoRA injection and
FSDP `prepare`, before any optimizer step -- the untrained adapter. `init_lora_weights=True`
zero-inits B, so this checkpoint's decode should be bit-identical to the frozen base. Before
writing it, `save_lora` checks the gathered exported `lora_B` weights are exactly zero; failure
refuses to create a misleading step-0 artifact. That is the sanity check it exists for
(`visualize_d0.py --run <run> --steps 0 1` after a one-step run), not something a normal run
needs. Off by default.

## Arms and knobs

| Flag | What it selects |
|---|---|
| `--objective {bg,white}` | which pair of bundles is read; must match the subset's |
| `--guide-mode {d0,d1}` | `d1` = D1a (guide as the noised init); `d0` = the GT-renoise capacity check, not deployable |
| `--disagreement-weight` | SS1.5's band weight, default 0.0 |
| `--block-latent-frames` / `--context-latent-frames` | SS1.6's geometry and cache depth |
| `--sigma-levels` | one adapter across several operating points |
| `--teacher-forcing` | ablation: the refresh is fed `z_y[i]` instead of `ẑ₀[i].detach()` |
| `--anchor-weight` | SS1.5 row 2; needs `base_denoised.pt` |

**Teacher vs self forcing differ in exactly one tensor** — what `refresh` is handed. Nothing
else in the loop, and nothing in `causal_core`, knows which regime is in play.

## Invariants

- **A v1 bundle is refused with a pointed error**, never silently reassembled. A reader that
  quietly reconstructs is a second producer of the tensor the trainer learns from.
- **A pre-causal window-chain subset is refused**, not reinterpreted: a window index and a
  block index are different numbers over the same clip.
- **A subset frozen against the other objective is refused.**
- `--guide-mode d0` with `--anchor-weight > 0` is refused: the anchor target is computed on
  the guide-noised input, and d0 noises `z_y`.
- `--guide-mode d0` with `--disagreement-weight != 0.0` is refused: d0 has no render, so the
  band the weight would apply to doesn't exist.
- σ = 0.0 is refused as a training level — the noiser adds nothing, so loss and gradient are
  identically zero (a quarter of one run trained on nothing before this was caught).

## Gotchas

- **A zero-byte stdout log is not a hung run.** `python -u`'s unbuffering does not survive
  `conda run`'s subprocess piping here. Check `metrics_rank<r>.jsonl`, which `train.py`
  writes and flushes directly.
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
