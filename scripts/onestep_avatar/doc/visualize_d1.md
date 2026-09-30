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
- `--whole-clip` sets the block to `T − 1` latent frames so block 0 is `[0, T)`, and runs it through the explicit-history path with an empty prefix: no `BlockCache` is allocated (the single-block cache run peaked at 43 GB) and no refresh pass runs. Every generated token attends to every other; `c0` stays clean at timestep 0. The manifest records `whole_clip: true`, `attention: full_bidirectional` and `history_policy: none_single_block` so the report never infers the mode from geometry. It refuses `--history-mode`, `--block-latent-frames` and `--max-blocks`, and all `--view`s must share `T`.
- The prompt is chosen with `--prompt` / `--prompt-file` (shared `session.add_prompt_args`) and defaults to `refine_task.REFINE_PROMPT`, so saved runs reproduce. The manifest's `text_context` records the prompt text, `prompt_sha256` (UTF-8 text), `is_default_prompt` and `sha256` (the encoded context bytes). `--prompt ""` is a valid empty-prompt control.
- The output is an offline comparison. A teacher-forced run must pass `--teacher-forcing` explicitly.

## Gotchas

The tool does not yet enforce checkpoint metadata against flags (known gap G3). Choose the objective, sigma, geometry and forcing policy from the checkpoint's `config.json`. A fixed-σ one-step LoRA applied at other σ levels or on a multi-step tail is an off-condition diagnostic. `--history-mode recompute` is an expensive inference reference: it forwards the retained clean history and the current block together at each sigma, with history-token timesteps zero and block-causal attention. It writes no cache. After eviction it also recomputes older hidden states without their former context, so a difference from cached mode is not solely a prompt-AdaLN measurement. `--history-mode joint` uses the same tokens and timesteps but permits bidirectional attention across the clean history and current noisy block. It is an architecture reference, not a cache-equivalence test. Neither mode changes training or deployment.

## Tests

`tests/test_prompt_and_whole_clip.py` (CPU) pins the default prompt, prompt-file reading, flag exclusivity, the one-block whole-clip plan and the flag conflicts. Run `conda run -n ltx python -m scripts.onestep_avatar.visualize_d1 --help` for CLI validation; `tests/test_causal_core.py` checks recomputed-history behavior on a real small transformer. Inspect the manifest, raw latents and MP4s from a real run.

## Dev model and guidance

`--variant dev` loads `ltx-2.5-22b-dev-transformer-bf16.safetensors` beside the distilled file (or `--transformer PATH`) and needs `--steps N`. Each start σ in (0, 1] then runs `causal_core.rescaled_schedule(σ, N)`: the stock pipelines' `LTX2Scheduler().execute(steps=N)` curve (4096-token anchor shift, no latent passed; terminal stretch included), multiplied by σ. Passing the real ~18k-token latent instead over-shifts the schedule and the dev model degenerates — a bug found against `ti2vid_one_stage` on 2026-09-29. Truncating the stock curve would tie step count to σ; rescaling keeps N independent, and σ = 1 reproduces the stock schedule. `--trajectory-only` is refused for dev and `--steps` for distilled.

`--cfg`, `--stg`, `--stg-blocks`, `--rescale` and `--negative-prompt` (default `DEFAULT_NEGATIVE_PROMPT`) build a pipelines `MultiModalGuider`; `causal_core.guided_denoised_from_x0_model` runs its passes **sequentially** (conditional, negative-prompt, STG with video self-attention skipped on the named blocks) and combines them with the guider's own `calculate`, so there is no second guidance formula. Guidance works for either variant and for every history mode. Nothing is detected from checkpoint metadata: the 2.5 dev defaults (30 steps, CFG 3, STG 1 on block 28, rescale 0.7) are unverified, so pass them explicitly. The manifest records `model_variant`, `steps`, `schedule_policy`, the full `guidance` block (values, negative-prompt hash, passes per step) and, per rollout, the exact `schedule` and `forward_passes`. Output files carry `dev_n{N}_cfg{c}_stg{s}` instead of `one_step`/`official`.
