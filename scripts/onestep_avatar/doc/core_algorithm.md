# Core algorithm — shared rules and two modes

Status: **Shared helpers, both mode functions and typed training implemented;
full acceptance and legacy migration incomplete.** Read
[current acceptance](known_gaps.md#current-acceptance-and-next-step) for scope.
The training CLI supports explicit modes. The extracted mode-less loop remains
transitional for callers that still need migration.
Start with the [design index](README.md).
Required means an agreed rule. Current means inspected code. Proposed means planned code.

## 1. Symbols

Use the [README terms](../README.md#terms-used-here).
A video segment contains consecutive encoded frames.
The term `span` in code and option names means that segment.
It is not a separate data type.

| Symbol or term | Meaning |
|---|---|
| `z_y` | Encoded capture video. This is the training target. |
| `z_g` | Encoded guide video on the same frame and image grid. |
| `source` | Data to mix with noise: `z_y` for D0 or `z_g` for D1. |
| `c0` | First-image input with no added noise. Training uses the first selected capture frame. Product generation uses the supplied image. |
| `epsilon` | Random Gaussian noise, with a saved seed or content hash. |
| `sigma` | Noise level: zero means no added noise; one means pure random noise for generated frames. |
| `x_sigma` | Noisy encoded input: `(1-sigma)*source+sigma*epsilon`. |
| `prediction` / `x0` | Predicted encoded frames with no added noise. |
| token | A vector for one encoded frame/image position. The model reads tokens. |
| `K` | Number of causal blocks in one training sample. |
| `A` | Number of accumulated training samples per GPU process and weight update. |
| cache / K/V | Saved attention keys and values computed from past frames. |
| self forcing | Update the cache from generated frames. |
| teacher forcing | Update the cache from capture target frames. |
| priming | Prepare the cache before training a block sequence. |
| refresh | Recalculate cache data from frames with no added noise. |
| eviction | Remove old cache entries. Keep the first-image data. |
| full-frame MSE | Mean squared prediction error over all tokens and channels, including the first frame. |

The VAE produces 128 channels and compresses image axes by 32 and time by 8.
`F` encoded frames cover `(F-1)*8+1` RGB frames.
Encoded frame zero represents one RGB frame; later encoded frames represent eight each.
A video array `[1,128,F,H,W]` becomes tokens `[1,F*H*W,128]` with patch size one.
Frame rate affects model positions. Keep the existing 20-second RoPE position limit.

`bg` uses the capture background. `white` uses a white background.
These settings select input pixels and filenames. They do not select a training mode.

## 2. The block layout

Current `model.causal.CausalGeometry.plan` selects block zero as `[0,1+B)`.
Later blocks contain `B` encoded frames each.
Discard an incomplete final block.
For `B=2`, ranges are `[0,3)`, `[3,5)`, and `[5,7)`.
Keep frame zero and the last `D` past frames in the cache.
Reserve room for the current write before removing old entries.

Typed [bidirectional](model/bidirectional.md) training processes one segment
directly with full attention. It has no cache, prime or refresh call.
Typed [causal](model/causal.md) training uses the block/cache calculations above.
The transitional mode-less engine and old `visualize_d1 --whole-clip` path are
legacy execution paths awaiting removal; they do not define typed mode behavior.
Select the mode explicitly. D0/D1 and dev/distilled remain separate choices.
Joint attention over past frames and a current block is a diagnostic, not a third training mode.

## 3. The conditioning contract

The current rule is `clean_c0_v1`: the first image must be a real model input.

1. Before the first model call, its first-frame tokens equal `c0`. Their token noise level is zero.
2. Other generation tokens follow the noise equation at the active `sigma`.
3. Keep the whole-model `sigma` active. It is distinct from each token's noise level.
4. Keep `c0` unchanged in the prediction and any cache refresh.
5. Later causal blocks read first-image data from the cache, even when `D=0`.
6. Use the correct background, crop, and VAE normalization. Never substitute guide frame zero for `c0`.

For an independent random-start segment, take its first capture encoding as `c0` and reset positions to zero.
At a nonzero start, that encoding represents eight RGB frames.
A supplied-image encoding represents one RGB frame.
This is accepted [G9](known_gaps.md#g9--a-random-window-c0-is-not-a-keyframe-encode).

For training that starts at a later causal block, prime from capture past frames.
Keep their original positions. This applies to both history policies.
Product generation cannot supply those capture past frames.

A keyframe mask, output-only replacement, or a pinned generated frame does not establish this input rule.
Current owners include `model.common.with_clean_prefix`,
`model.common.block_modality`,
both mode `train_sample`/`sample` functions, and `infer.generate(first_image=...)`.

## 4. The per-block algorithm

Current causal training calls prime once.
For each block, it denoises and immediately computes gradients.
It then refreshes the cache and removes old entries, except after the last training block.

Denoising reads the cache without changing it.
Refresh uses `no_grad` at whole-model sigma zero.
Self forcing uses generated frames without gradients.
Teacher forcing uses the explicit capture target.

Use fp32 full-frame MSE, without mask weights.
The unchanged first frame has zero error but stays in the mean's denominator.
Backpropagate `MSE/(K*A)` per causal block.
Typed bidirectional code backpropagates `MSE/A` per segment.
Do not weight blocks by token count or divide the loss again in the engine.
Training does not decode RGB video.

For multi-step generation, `model.sampling.euler_to` owns each update. At a
positive next level it uses the stock Euler step and its dtype conversions.
At zero it returns the model prediction exactly, preserving direct-step
training/generation equality. The stock bf16 reconstructed endpoint can differ;
[G11](known_gaps.md#g11--euler-rounding-differs-from-the-stock-step) records the
measured base video-component endpoint difference and the remaining E1 scope.

See [common](model/common.md), [bidirectional](model/bidirectional.md),
[causal](model/causal.md), and [engine](training/engine.md) for the ordered steps.
The engine updates weights once after the accumulated sample group.

## 5. Worked example — blocks 0, 1, 2

Set `B=2`, `D=1`, and seven encoded frames.

| Block | Input frames | Cached frame data before denoising | Cached frame data after refresh |
|---|---|---|---|
| 0 | `[0,3)`; frame 0 unchanged, frames 1–2 noisy | empty | 0,2 |
| 1 | `[3,5)`, noisy | 0,2 at original positions | 0,4 |
| 2 | `[5,7)`, noisy | 0,4 at original positions | 0,6 if refreshed |

Training skips the last refresh. Current generation keeps it.
The history policy changes the frames used for refresh, not the first-image input.

For training starting at block 1, prime with capture frames 0,2.
Product generation must generate block 0 first.
Those two histories can differ.

[Verification V1–V6](verification.md) gives arithmetic, cache, priming, and call-count checks.
These are design checks, not new implementation results.

## 6. Train / probe / deploy

| Current file or function | Behavior |
|---|---|
| `model.bidirectional.train_sample` | D0/D1; one segment, no cache, direct backward scaled by A. |
| `model.causal.train_sample` | D0/D1; K blocks; capture or generated past frames; one prime; immediate gradients; no last refresh. |
| Both mode `sample` functions | Explicit first-image input and exact schedule; causal capture history requires an explicit target. |
| `evaluate.py` | Explicit mode, checked saved inputs/adapter, recorded diagnostic history and research overrides. |
| `infer` | Product D1 with supplied-image input and generated past frames; version-two pre-weight adapter checks. |

Causal training has `2K` explicit model calls.
Direct causal generation has K denoises and K refreshes.
Direct bidirectional execution has one denoise and no prime/refresh.
Guidance and gradient checkpointing add actual work.

FSDP distributes training across GPU processes. They must agree on calls,
backward calculations, reductions and saves; equal K alone is not sufficient.
Typed FSDP disables root-input casting to preserve float32 modality sigma,
timesteps and positions. Keep the bf16 parameter/reduction policy and fp32
adapter storage. These are actual runtime settings, not merely checkpoint
fields. Original four-rank updates in both modes pass their fixed E4 serial
comparisons. Current complete workflow and final-source checks remain separate.

Typed evaluation and product files use one [adapter checker](training/checkpoints.md).
They call the selected mode's generation function.
D0 is for training/evaluation tests; product generation accepts only D1.
Evaluation can use people excluded from training.

The selected causal computation refreshes clean frames at whole-model sigma
zero. Recomputed history at the active denoise sigma and joint-history paths
are diagnostic computations. Prompt AdaLN and, after eviction, changed old-frame
context make them different from the cache; numerical equality is not assumed.
For a causal adapter, the checker maps `cached_refresh_global_sigma0` to the
explicit cache/refresh request. Changed history/K/V choices require the existing
research override and recorded differences. Product permits only cache/refresh.
Quality and cost before/after eviction still need native E3 measurements.

Required adapter application is frozen bf16 base weights with unmerged fp32 PEFT
adapters, matching training. Current typed evaluation/inference use the shared
`model.adapters` path. [G8](known_gaps.md#g8--bf16-lora-fusion-weakens-the-trained-adapter)
remains open for full E2 views/trained steps and cost verification. Fusion remains
an explicitly changed research condition. Do not silently fall back to it.
Current product acceptance is limited to clip-start inputs and the recorded
frame counts. The pilot uses seven encoded frames in both modes; longer or
random-start-trained product support needs separate validation.

## 7. End-to-end data flow

Use one diagram legend:

| Shape / color | Meaning |
|---|---|
| rectangle / blue | code operation |
| cylinder / grey | saved file or data |
| rounded / green | array in memory or resolved input |
| hexagon / amber | mutable state, such as a cache |
| dashed border/edge | work without gradients or data passed without gradients |
| stadium / purple | final output or consumer |

Show separate mode and history-policy diagrams.
Name actual files or functions in code boxes.
Use defined data names on arrows and input nodes.
Put exact noise levels, positions, and cache contents in adjacent tables.

Capture precompute writes the crop record and capture master.
ARGAvatar writes guide RGB and alpha.
Guide precompute writes the guide master and cropped mask.
The implemented [subset](corpus/subset.md) writes a fixed video list.
Each mode selects frames. Checkpoint code records training settings.
[Evaluation](evaluate.md) records the actual inputs and model calls.
[Media](media.md) decodes saved outputs with existing VAE code.

## Boundaries that remain unverified

[G7](known_gaps.md#g7--cached-history-can-disagree-with-a-causal-prefix):
sigma-zero refresh can differ from recalculation at the active noise level because of prompt AdaLN.
G8: fused bf16 LoRA can differ from training LoRA.
G9: a random-start segment can have a different first-image encoding.
Moving code does not fix these differences.

Documentation size rules are in [README](README.md).
Required checks are in [verification](verification.md).
