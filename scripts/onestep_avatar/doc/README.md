# `onestep_avatar` — current design index

## Review reading order

Read [architecture](architecture.md) for ownership and dependencies, then
[core algorithm](core_algorithm.md) for shapes, conditioning, and mode calculations.
[Training choices](training_choices.md) defines supported settings;
[verification](verification.md) defines worked checks and the acceptance gates;
[known gaps](known_gaps.md) defines current limits.
The [run guide](../configs/README.md) owns runnable commands.

## Documentation rule

A production file over 100 physical lines has a mirrored design doc at
`doc/<relative-source-path>.md`. Count blanks, comments, and headers. A file
with at most 100 lines explains objective, ordered logic, input/output, failure,
and invariants in its header. Empty markers and tests need no module document.
Do not split or pad source to alter this rule.

Large-file docs use Objective, Data flow, Organization logic, Invariants,
Gotchas, and Tests. Explain actual decisions, calculations, state changes, and
output/failure meaning; a symbol list cannot replace the core logic.
Give a worked input and expected result. For visual output, state selected data,
positions, labels, time mapping, and readability. Use small left-to-right Mermaid
flows with the [shared legend](core_algorithm.md#7-end-to-end-data-flow), then
render and inspect them. Diagrams cannot certify numerical or distributed behavior.

## Current source owners

The index lists present sources only. Corpus owns data production and checked
membership; model owns numerical kernels; training owns the optimizer and typed
contracts; execution owns the one queue and bounded supervision; ordinary root
owners supply evaluation, product, media, preparation, plots, and cost.
Experiments orchestrate scientific comparisons through those public owners.
Ordinary imports and profiles exclude experiments. Report code under `expr/`
reads saved inputs and cannot run models, decoding, or queue work.

| Source | Design | Physical lines |
|---|---|---:|
| `__init__.py` | source header | 13 |
| `bench.py` | [bench.md](bench.md) | 577 |
| `comparisons.py` | [comparisons.md](comparisons.md) | 501 |
| `corpus/build_guidance.py` | [corpus/build_guidance.md](corpus/build_guidance.md) | 778 |
| `corpus/dataset.py` | [corpus/dataset.md](corpus/dataset.md) | 397 |
| `corpus/geometry.py` | [corpus/geometry.md](corpus/geometry.md) | 111 |
| `corpus/mask_video.py` | [corpus/mask_video.md](corpus/mask_video.md) | 181 |
| `corpus/motion.py` | [corpus/motion.md](corpus/motion.md) | 187 |
| `corpus/precompute.py` | [corpus/precompute.md](corpus/precompute.md) | 1390 |
| `corpus/qa.py` | source header | 29 |
| `corpus/subset.py` | [corpus/subset.md](corpus/subset.md) | 400 |
| `decode_saved.py` | [decode_saved.md](decode_saved.md) | 192 |
| `evaluate.py` | [evaluate.md](evaluate.md) | 740 |
| `execution/devices.py` | source header | 59 |
| `execution/process_registry.py` | [execution/process_registry.md](execution/process_registry.md) | 358 |
| `execution/queue.py` | [execution/queue.md](execution/queue.md) | 2056 |
| `execution/queue_launch.py` | [execution/queue_launch.md](execution/queue_launch.md) | 1060 |
| `execution/queue_protocol.py` | source header | 11 |
| `execution/runtime.py` | [execution/runtime.md](execution/runtime.md) | 173 |
| `execution/software.py` | [execution/software.md](execution/software.md) | 195 |
| `execution/supervision.py` | [execution/supervision.md](execution/supervision.md) | 525 |
| `experiments/adapter_effect_check.py` | [experiments/adapter_effect_check.md](experiments/adapter_effect_check.md) | 957 |
| `experiments/capacity_review.py` | [experiments/capacity_review.md](experiments/capacity_review.md) | 256 |
| `experiments/causality.py` | [experiments/causality.md](experiments/causality.md) | 764 |
| `experiments/conditioning_check.py` | [experiments/conditioning_check.md](experiments/conditioning_check.md) | 595 |
| `experiments/continuation_check.py` | [experiments/continuation_check.md](experiments/continuation_check.md) | 917 |
| `experiments/continuation_identity_check.py` | [experiments/continuation_identity_check.md](experiments/continuation_identity_check.md) | 474 |
| `experiments/fusion_parity.py` | [experiments/fusion_parity.md](experiments/fusion_parity.md) | 330 |
| `experiments/past_influence_check.py` | [experiments/past_influence_check.md](experiments/past_influence_check.md) | 584 |
| `experiments/prune_baseline.py` | [experiments/prune_baseline.md](experiments/prune_baseline.md) | 201 |
| `experiments/prune_reproduction.py` | [experiments/prune_reproduction.md](experiments/prune_reproduction.md) | 417 |
| `experiments/prepare_continuation.py` | [experiments/prepare_continuation.md](experiments/prepare_continuation.md) | 521 |
| `experiments/reference_inputs.py` | [experiments/reference_inputs.md](experiments/reference_inputs.md) | 149 |
| `experiments/saved_probe_metrics.py` | [experiments/saved_probe_metrics.md](experiments/saved_probe_metrics.md) | 141 |
| `experiments/short_causal_check.py` | [experiments/short_causal_check.md](experiments/short_causal_check.md) | 573 |
| `experiments/sigma_sweep.py` | [experiments/sigma_sweep.md](experiments/sigma_sweep.md) | 329 |
| `experiments/sigma_sweep_results.py` | source header | 70 |
| `experiments/spec_protocol.py` | [experiments/spec_protocol.md](experiments/spec_protocol.md) | 139 |
| `experiments/stats.py` | [experiments/stats.md](experiments/stats.md) | 470 |
| `experiments/stock_parity.py` | [experiments/stock_parity.md](experiments/stock_parity.md) | 654 |
| `experiments/training_slice_check.py` | [experiments/training_slice_check.md](experiments/training_slice_check.md) | 462 |
| `experiments/training_update_check.py` | [experiments/training_update_check.md](experiments/training_update_check.md) | 703 |
| `hashing.py` | source header | 34 |
| `infer.py` | [infer.md](infer.md) | 696 |
| `media.py` | [media.md](media.md) | 714 |
| `metrics.py` | [metrics.md](metrics.md) | 151 |
| `model/adapters.py` | [model/adapters.md](model/adapters.md) | 145 |
| `model/backbone.py` | source header | 68 |
| `model/bidirectional.py` | [model/bidirectional.md](model/bidirectional.md) | 141 |
| `model/causal.py` | [model/causal.md](model/causal.md) | 899 |
| `model/common.py` | [model/common.md](model/common.md) | 326 |
| `model/sampling.py` | [model/sampling.md](model/sampling.md) | 150 |
| `plot_training.py` | [plot_training.md](plot_training.md) | 481 |
| `prepare_inputs.py` | [prepare_inputs.md](prepare_inputs.md) | 381 |
| `previews.py` | [previews.md](previews.md) | 606 |
| `train.py` | source header | 38 |
| `training/checkpoints.py` | [training/checkpoints.md](training/checkpoints.md) | 584 |
| `training/config.py` | [training/config.md](training/config.md) | 406 |
| `training/consumer_trace.py` | [training/consumer_trace.md](training/consumer_trace.md) | 492 |
| `training/engine.py` | [training/engine.md](training/engine.md) | 962 |
| `training/numerics.py` | source header | 79 |
| `training/resources.py` | [training/resources.md](training/resources.md) | 234 |
| `training/runtime.py` | [training/runtime.md](training/runtime.md) | 132 |
| `training/startup.py` | source header | 63 |
| `training/update_state.py` | source header | 93 |

## Core logic review map

- [Common model operations](model/common.md): tokens, exact clean `c0`, noise, full-frame loss.
- [Causal computation](model/causal.md): block zero, block one, mid-clip start, cache retention and refresh.
- [Bidirectional computation](model/bidirectional.md): whole segment, clean first image, target and reduction.
- [Adapter application](model/adapters.md): frozen bf16 base, fp32 PEFT tensors, unmerged operation order.
- [Training engine](training/engine.md): rank/sample exposure, update order, distributed reduction, exports.
- [Queue](execution/queue.md): direct occupancy, owned process identity, bound startup/phases and completion.
- [Media](media.md): aligned RGB references, labeled panels, full and narrow reading formats.

## Artifact and evidence boundaries

The same content hash, schedule, geometry, noise, image, context, and source identity
must be checked at preflight, model boundary, and saved-only completion as applicable.
A changed file fails; readers do not reconstruct missing scientific artifacts.
The queue alone launches native work and owns supervision. Child-local runtime
receipts measure declared phase scope; they do not replace continuous supervision.
Saved completion verifies identity and structure without model handles.
Report rebuilds are saved-only and fail before writes when cited evidence is missing.

The [verification contract](verification.md) describes controls and acceptance gates;
no entry in this index claims native acceptance.
