# `evaluate/whole_clip_baseline.py` — saved whole-clip one-step rollouts

## Objective

Produce the saved native whole-clip rollouts that every pruning step reads:
the unpruned baseline, and each candidate transformer with the same setup. One
bidirectional block covers each complete capture. Latent frame 0 is the clean
capture condition `c0`. Every later frame is generated in one denoising step
from the exact `[sigma, 0]` schedule. The output format is the one
[`data.whole_clip`](../data/whole_clip.md) validates.

It is the only producer of these rollouts and owns no other avatar diagnostic.

## Command

Run from `LTX-2` in the `ltx` environment after checking `nvidia-smi`:

```bash
python -m scripts.prune.evaluate.whole_clip_baseline \
  --view <corpus-view> [--view <corpus-view> ...] \
  --sigmas 0.725 0.909375 1.0 --prompt "" --seed 42 --gpu-id N \
  --output <fresh-output-dir>
```

Add `--transformer <candidate.safetensors>` for a pruned candidate and keep
every other flag unchanged. `--raw-only` skips VAE decoding. Several noise
seeds in one model load need `--seeds S1 S2 ... --raw-only`.

## Data flow

1. **Inputs.** Each `--view` is a corpus view directory. Read the capture and
   guide masters for `--objective` (default `white`) with
   `corpus.dataset.load_training_master`. Capture and guide must have equal
   shape and fps. All views must have the same latent length `T >= 2`.
2. **Geometry.** `whole_clip_block` sets the block to `T - 1` latent frames,
   so `model.causal.deployed_geometry(...).plan(T) == [(0, T)]`. The geometry
   keeps `--context-latent-frames` (default 8) in the manifest, although one
   block has no history. `common.ClipGrid.build` uses the capture's pixel
   size, fps and the session's scale factors.
3. **Noise.** Patchify the capture. `global_epsilon` draws one BF16 tensor with
   `common.epsilon_block(capture_tokens, seed)` over all tokens and slices it
   by the plan. Save it as `<actor>_<view>[_seed<S>]_epsilon.pt` before any
   forward. D0 and D1, and every sigma, reuse this tensor.
4. **Schedule.** The distilled model uses `[sigma, 0]`. Each sigma must lie on
   the session's distilled grid (`checked_sigmas`); the default levels are
   `0.909375 0.725 0.421875`. `--variant dev --steps N` enters the stock
   N-step curve at each sigma instead (`--dev-schedule truncated|rescaled`,
   optional `--dev-denoising-steps K`). Saved-input consumers accept only
   one-step `[sigma, 0]` rows.
5. **Rollout.** For each sigma and arm, call `model.causal.rollout` with no
   cache, `history_mode="recompute"`, the one-block plan and the saved epsilon.
   D0 noises the capture tokens; D1 noises the guide tokens. Both use the
   capture's first latent frame as `c0` and the capture as the target. The
   whole-clip block has no history, no K/V cache and no refresh pass, and
   attention runs without a dense mask. Save the unpatchified BF16 latent as
   `<stem>_sigma_<sigma:.6f>_<suffix>_<arm>.pt`; the suffix is `one_step`, or
   `dev_n<N>[_k<K>]_cfg<c>_stg<s>`.
6. **Media.** Unless `--raw-only`, decode the capture over the covered frames
   and each arm's latent with a fresh generator seeded by `--seed`, then write
   a synchronized `capture VAE | D0 | D1` MP4 with `evaluate.metrics.t3_video`
   at the capture fps. A decoder that returns other than `(T - 1) * 8 + 1`
   frames fails.
7. **Manifest.** `manifest.json` records the settings, geometry, noise scheme,
   prompt text and actual context bytes hash, model/VAE identity and
   fingerprints, guidance, attention backends, per-view capture/guide hashes,
   epsilon hash, block plan, per-arm timing and latent hashes, and one row per
   view and sigma.

Fixed manifest fields record what this producer cannot vary: `whole_clip:
true`, `attention: full_bidirectional`, `history_policy: none_single_block`,
`history_mode: recompute`, `trajectory_only: false`, `teacher_forcing:
false`, `max_blocks: null`, `checkpoint: null` and no adapter condition
problems. The `model.script` stamp names this module.

## Options

Supported options: `--view`, `--objective`, `--sigmas`, `--variant`,
`--transformer`, `--steps`, `--dev-denoising-steps`, `--dev-schedule`,
`--context-latent-frames`, the guidance options `--cfg`, `--stg`,
`--stg-blocks`, `--rescale` and `--negative-prompt`, `--raw-only`, `--seeds`,
`--arms`, `--output`, and the session's `--model`, `--gpu-id`, `--seed`,
`--prompt` and `--prompt-file`.

There are no block, history, teacher-forcing, trajectory or adapter options:
the computation is always one whole-clip block with no history,
`data.whole_clip` refuses multi-step distilled tails, and LoRA adapters need
the avatar training contract checks, which prune must not import.

## Invariants and gotchas

- One global epsilon per view and seed. A matching seed is not proof of
  matching noise; consumers compare the saved tensors.
- `c0` is the clean capture frame in both arms; the guide's frame 0 is never
  a condition. Sigma and per-token timesteps stay float32 inside the model
  owner; BF16 applies to weights and latents.
- The output directory must be new or empty. The command never overwrites a
  saved run.
- This module imports public `corpus.dataset` readers and `model/` owners
  only. It does not import avatar training, run support or experiments.
- Matched baseline and candidate runs differ only in `--transformer`.

## Saved completion verification

`verify_completion(args, root)` is an additive model-free reader for the queued
raw-only white distilled unguided route. It validates exact requested view,
seed, sigma and arm coverage, original producer settings and prompt identity,
source master hashes, one-block geometry, local artifact containment and full
hashes, finite BF16 epsilon and latent shapes, and exact capture `c0`. Every row
for a view uses the same saved epsilon/artifact record. The reader never opens
a model session, decoder or CUDA context and never regenerates CUDA noise.
`evidence_paths` exposes only these checked local artifacts. The avatar queue
wrapper separately pins full input/component bytes and current software before
and after production; prune imports no execution or experiment owners.

## Tests

`tests/test_whole_clip_baseline.py` checks the one-block plan for several
lengths, mixed-length and conflicting-option refusals, dev option validation
and file suffixes, global noise slicing, and a CPU end-to-end run with a tiny
real transformer. That run checks the saved file names, manifest fields that
`data.whole_clip.validate_manifest` reads, exact epsilon reuse across arms and
sigmas, and the `c0` frame of every saved latent. Native reproduction of a saved
baseline (same view, sigmas, prompt, seed and model) runs through the avatar
`prune_baseline` queue experiment.
