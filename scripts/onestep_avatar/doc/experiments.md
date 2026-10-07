# Training choices — mode, input, background, and past frames

These definitions apply to the implemented typed modes. Legacy execution
paths remain until their callers and native replacement gates are checked.

Historical D0 completion reports keep their original attribution and metrics.
The removed fixed-run report builder expected a training grid containing sigma
zero, separate frozen-base videos, and union-masked losses. These conditions do
not describe current full-frame training. Do not rebuild a current result with
that historical contract or relabel a historical masked loss. Current raw
results, checked render records and numeric logs supply evidence to report code
under `expr/`; missing artifacts must fail rather than start model jobs.
Current commands are in [configs/README](../configs/README.md).
See [README terms](../README.md#terms-used-here) before the model designs.

## Attention mode

Training supports explicit bidirectional and causal modes.
A transitional mode-less loop still uses causal block sequences.
The old loop's one-block complete-video path and `--whole-clip` visualizer
are legacy behavior. Typed bidirectional execution has no cache or priming.
The `--mode` switch is implemented. Native acceptance remains pending.

Bidirectional mode processes one segment together.
Causal mode processes blocks in order with stored past-frame data.
Joint-history and recalculated-history paths are diagnostic comparisons.
They are not additional training modes.

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
At sigma one, D0/D1 agree only with the same model, adapter, noise, first image, and history.
Do not compare separately trained adapters as this equality control.

## Objective

Code uses `objective` for the background choice.

| Choice | `bg` | `white` |
|---|---|---|
| Capture pixels | original capture | capture matted to white |
| Guide pixels | render over capture frame zero | render on white |
| Filename | no suffix | `_white` suffix |
| First-image input | bg capture encoding | white capture encoding |

Version-two pixel compositing is `R_white+(1-alpha)*(B_frame0-white)`.
Do not blend encoded data to replace this pixel operation.
Both backgrounds use the same code.
Check each selected video's current producer records before D1 training.
Old inventory counts are not a live readiness check.

## Causal history

Self forcing updates the cache from generated frames without gradients.
Teacher forcing updates it from the explicit capture target.
Both keep the same first-image input.

Current `model.causal.sample` requires `teacher_tokens` for teacher forcing.
It no longer substitutes guide frames. G2 is fixed and verified.

Clean cache refresh at whole-model sigma zero is the selected causal training
and product computation. Active-sigma recomputation and joint-history execution
are diagnostics; they can differ because of prompt AdaLN and old-frame context
after eviction. This is a different continuation calculation, not a numerical
optimization of recomputation. A causal adapter requires an explicit research
override for changed `history_mode` or `kv_source`. Product uses cache/refresh
with generated history and permits no override. Native quality/cost evidence
before and after eviction remains open.

When training starts at a later block, priming uses capture past frames under either policy.
Product generation starts at frame zero and has no capture past frames.
Generated-history evaluation of a teacher-trained adapter changes its training conditions.
Record that change.
Bidirectional mode has no history policy.

## Weights, noise and schedules

Dev/distilled selects base weights, not mode.
Check base weight identity and supported noise levels before adapter settings.

Current fixed-sigma and multiple-level direct training are supported.
Since 2026-10-05, one uniform sigma draw is made per update/GPU process.
Accumulated samples on that process share the draw.
Earlier rotation-based results keep their historical attribution.

Fresh noise and fixed-per-sequence noise are different settings.
Keep their current seeds.
Random-start segment selection resets positions and uses the first selected capture encoding.
G9 remains visible.
More-step evaluation of a directly trained adapter needs an explicit record of changed conditions.

## Implementation boundaries

Current code supports D0/D1, both backgrounds, both causal history policies, clean first-image input,
and existing evaluation/generation paths.
Explicit mode files, a fixed video list independent of frame selection,
shared typed training, and one adapter checker are implemented. Their native
acceptance, remaining input preparation and legacy ownership cleanup are open.

Required ordinary adapter application uses the same unmerged fp32 PEFT adapter
function as training, against frozen bf16 base weights. Current ordinary
typed evaluation/inference use the shared unmerged loader; G8 remains open
pending native matched effect and cost measurement. Keep historical fused results labeled
with that original method. Memory failure does not authorize a fused fallback.

Remove disabled anchor plumbing during cleanup.
Masks and alpha are data-quality records, not loss weights.
Old sliding-window designs and dropped choices are not active defaults.
[Known gaps](known_gaps.md) records fixed defects and unresolved limits.
Documentation adds no new model-test results.
