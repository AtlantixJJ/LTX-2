# Training choices — mode, input, background, and past frames

This document defines the supported training settings. Commands are in the
[run guide](../configs/README.md). See the [README terms](../README.md#terms-used-here)
before the model designs.

Raw results, checked render records and numeric logs are the evidence for
report code under `expr/`. A missing file fails before any model work. The
training loss is full-frame MSE. A report metric must not give a different
loss this name.

## Attention mode

Training supports two explicit modes, selected with `--mode`.

- **Bidirectional** processes one segment together. It has no cache or priming.
- **Causal** processes blocks in order with stored past-frame data.

Joint-history and recalculated-history paths are evaluation diagnostics, not
additional training modes. The pruning whole-clip producer is
`scripts.prune.evaluate.whole_clip_baseline`.

## Arm

Code uses `arm` or `guide_mode` for the D0/D1 input choice.

| Choice | D0 | D1 |
|---|---|---|
| Data mixed with noise | capture encoding `z_y` | guide encoding `z_g` |
| Training target | capture encoding `z_y` | capture encoding `z_y` |
| First-image input | capture or supplied-image encoding | the same input, never guide frame zero |
| Required files | capture master | capture and checked guide masters |
| Purpose | capacity test | guide-to-capture training |
| Product generation | unavailable because capture is absent | supported when conditions pass |

Both use full-frame MSE; see [core_algorithm](core_algorithm.md).
A lower D0 loss does not establish better D1 video quality.
At sigma one, D0/D1 agree only with the same model, adapter, noise, first image and history.
Do not compare separately trained adapters as this equality control.

## Objective

Code uses `objective` for the background choice.

| Choice | `bg` | `white` |
|---|---|---|
| Capture pixels | original capture | capture matted to white |
| Guide pixels | render over capture frame zero | render on white |
| Filename | no suffix | `_white` suffix |
| First-image input | bg capture encoding | white capture encoding |

The guide pixels are composited as `R_white+(1-alpha)*(B_frame0-white)`
(compositing version 2). Do not blend encoded data instead of this pixel
operation. Both backgrounds use the same code. Before D1 training, check the
producer records of each selected video. The check runs for each selected
input.

## Causal history

Self forcing updates the cache from generated frames without gradients.
Teacher forcing (`--teacher-forcing`) updates it from the explicit capture
target. Both keep the same first-image input. `model.causal.sample` requires
`teacher_tokens` for teacher forcing; a guide cannot supply teacher history.

Causal training and the product refresh the cache from clean frames at
whole-model sigma zero. Recomputation at the active sigma and joint history are
diagnostics. They can differ, because of prompt AdaLN and the old-frame context
after eviction. A causal adapter needs an explicit research override for a
changed `history_mode` or `kv_source`. The product uses `cache`/`refresh` with
generated history and allows no override.

When training starts at a later block, priming uses capture past frames with
either policy. Product generation starts at frame zero and has no capture past
frames. If you evaluate a teacher-trained adapter with generated history, its
training conditions change. Record that change. Bidirectional mode has no
history policy.

## Weights, noise and schedules

Dev or distilled selects the base weights, not the mode. The code checks the
base weight identity and the supported noise levels before the adapter
settings. Distilled weights accept only their nine grid levels.

Training uses one fixed sigma (`--sigma0`) or several direct levels
(`--sigma-levels`). Each update draws one uniform sigma for each GPU process.
All accumulated samples on that process use the same draw.

- `--noise-policy fresh` draws new noise for each visit.
- `--noise-policy fixed_per_chain` keeps one noise draw for each sample for the
  whole run. It is a debug control.

Seeds stay fixed. A random-start segment resets the positions and uses its
first selected capture encoding as `c0`. A later latent frame encodes eight
pixel frames, so it differs from a single-image keyframe (see
[known gaps](known_gaps.md#random-start-first-image-distribution)). To evaluate
a directly trained adapter with more steps, record the changed conditions
explicitly.

## Adapter application

Evaluation and the product use the same unmerged fp32 PEFT adapter function
as training, with frozen bf16 base weights. The fused bf16 method is an
explicit evaluation diagnostic, and the record names it. A memory failure never
selects a fused fallback.

Masks and alpha are data-quality records, not loss weights.
[Known gaps](known_gaps.md) records current limits; the
[acceptance gates](verification.md#acceptance-gates) define native evidence.
