# Training choices — mode, input, background, and past frames

This document defines the supported training settings. Commands are in the
[run guide](../configs/README.md). See the [README terms](../README.md#terms-used-here)
before the model designs.

Raw results, checked render records and numeric logs supply evidence to report
code under `expr/`. Missing artifacts fail before any model work. Full-frame MSE
is the objective; report metrics must not relabel a different loss.

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

Guide pixel compositing is `R_white+(1-alpha)*(B_frame0-white)` (compositing
version 2). Do not blend encoded data to replace this pixel operation.
Both backgrounds use the same code. Check each selected video's producer
records before D1 training; readiness is checked per selected input.

## Causal history

Self forcing updates the cache from generated frames without gradients.
Teacher forcing (`--teacher-forcing`) updates it from the explicit capture
target. Both keep the same first-image input. `model.causal.sample` requires
`teacher_tokens` for teacher forcing; a guide cannot supply teacher history.

Clean cache refresh at whole-model sigma zero is the selected causal training
and product computation. Active-sigma recomputation and joint-history execution
are diagnostics; they can differ because of prompt AdaLN and old-frame context
after eviction. A causal adapter needs an explicit research override for a
changed `history_mode` or `kv_source`. Product uses cache/refresh with
generated history and permits no override.

When training starts at a later block, priming uses capture past frames under
either policy. Product generation starts at frame zero and has no capture past
frames. Generated-history evaluation of a teacher-trained adapter changes its
training conditions; record that change. Bidirectional mode has no history policy.

## Weights, noise and schedules

Dev/distilled selects base weights, not mode. Base weight identity and supported
noise levels are checked before adapter settings; distilled weights accept only
their nine grid levels.

Training uses a fixed sigma (`--sigma0`) or multiple direct levels
(`--sigma-levels`). One uniform sigma draw is made per update and GPU process;
accumulated samples on that process share the draw.

`--noise-policy fresh` draws new noise per visit; `fixed_per_chain` keeps one
noise draw per sample for the whole run as a debug control. Seeds stay fixed.
A random-start segment resets positions and uses its first selected capture
encoding as `c0`; a later latent frame encodes eight pixel frames and differs
from a single-image keyframe (see [known gaps](known_gaps.md#random-start-first-image-distribution)).
More-step evaluation of a directly trained adapter needs an explicit record of
changed conditions.

## Adapter application

Evaluation and product use the same unmerged fp32 PEFT adapter function as
training, against frozen bf16 base weights. The fused bf16 method is an explicit
evaluation diagnostic that records its application method. A memory failure
never selects a fused fallback.

Masks and alpha are data-quality records, not loss weights.
[Known gaps](known_gaps.md) records current limits; the
[acceptance gates](verification.md#acceptance-gates) define native evidence.
