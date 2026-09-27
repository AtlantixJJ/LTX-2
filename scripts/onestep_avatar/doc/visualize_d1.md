# `visualize_d1.py` — paired source comparison

## Objective

Write one three-panel MP4 per view and sigma: decoded ground-truth capture, causal rollout noised from the capture master (`D0`), and causal rollout noised from the RGB-render guide master (`D1`). An optional ComfyUI LoRA is loaded into the same transformer for both rollouts. The default levels are the distilled model's 0.421875, 0.725 and 0.909375. The default mode jumps from each selected sigma to zero in one step. `--trajectory-only` instead uses the remaining distilled schedule, producing 2–8-step videos from the matching on-grid start; it skips 0.421875 because that tail already has one step. The full 2.5 grid is `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0]`.

## Data flow

The explicit `--view` directories supply the objective-matched capture and guide master bundles. `train._load_training_master` validates both. All levels and both arms share the same model, geometry, first-frame capture condition, teacher target and epsilon tensor. Noise is drawn once over the complete latent grid, then sliced by each block's global frame indices; a two-versus-four-frame geometry comparison therefore uses identical noise at the same frame. `visualize_d0._run_chain` calls the sole `causal_core.rollout`; this module chooses each arm's noising source, schedule and cache or recomputed-history reference. The transformer is released before VAE decoding. The GT capture is decoded once per view. The output directory must be fresh. It contains raw D0/D1 latents and the global epsilon per view, plus MP4 panels. The manifest records paths and hashes for source bundles, weights, context, noise and latents; model capabilities, geometry, history policy, schedule, dtype and backend provenance are also recorded.

## Organization logic

The explicit view list permits a small diagnostic set without freezing a training subset or changing the split. `--max-blocks 2` stops after block 1 for a cheap pre-eviction real-checkpoint comparison. `--raw-only` omits VAE decoding but keeps raw latents, epsilon and the manifest, including per-arm rollout wall time and peak CUDA allocation. The `visualize_d0` helpers retain one implementation for block planning, rollout and decoding.

## Invariants

- The guide and capture master shape and fps must agree. Missing bundles are errors.
- Both arms reuse exactly the same epsilon blocks; `c0` always comes from the capture.
- The manifest reports `block_causal` attention for cache/recompute and `bidirectional_clean_history_window` for joint. Joint only sees the current noisy block and retained clean history; future generation blocks are absent.
- The intended same-base schedule comparison omits `--checkpoint` in both runs and uses separate fresh output directories. Historical LoRA one-step videos do not satisfy that comparison.
- LoRA fusion occurs once for both arms, so panel differences measure the noising source.
- The output is an offline comparison. A teacher-forced run must pass `--teacher-forcing` explicitly.

## Gotchas

The tool does not yet enforce checkpoint metadata against flags (known gap G3). Choose the objective, sigma, geometry and forcing policy from the checkpoint's `config.json`. A fixed-σ one-step LoRA applied at other σ levels or on a multi-step tail is an off-condition diagnostic. `--history-mode recompute` is an expensive inference reference: it forwards the retained clean history and the current block together at each sigma, with history-token timesteps zero and block-causal attention. It writes no cache. After eviction it also recomputes older hidden states without their former context, so a difference from cached mode is not solely a prompt-AdaLN measurement. `--history-mode joint` uses the same tokens and timesteps but permits bidirectional attention across the clean history and current noisy block. It is an architecture reference, not a cache-equivalence test. Neither mode changes training or deployment.

## Tests

Run `conda run -n ltx python -m scripts.onestep_avatar.visualize_d1 --help` for CLI validation; `tests/test_causal_core.py` checks recomputed-history behavior on a real small transformer. Inspect the manifest, raw latents and MP4s from a real run.
