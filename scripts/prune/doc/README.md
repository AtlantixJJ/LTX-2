# Pruning documentation and file map

Start with the [overview](overview.md) and [glossary](glossary.md).
The [algorithm](core_algorithm.md) explains calculations, [usage](usage.md) owns commands and preset descriptions,
and [verification](VALIDATION.md) owns evidence requirements.
[Architecture](ARCHITECTURE.md) maps changes to owners and checks. [Reporting](reporting.md) defines scientific presentation.

## Documentation rules

Prune uses one shared doc set for humans and agents, with [README](../README.md) and [working rules](../CLAUDE.md) as entry files.
It explicitly differs from the avatar package's mirrored-page rule. Module size does not decide documentation ownership.

- Put local behavior, data flow, non-obvious invariants and failure reasons in the source module docstring.
- Keep separate pages for cross-module contracts: native input, width/runtime export, depth export, head phases, joint selection and the DDP owner.
- Update the owning docstring and shared contract in the same change when behavior changes.
- Give every source one file-map row and every CLI an entry row. Link config values instead of copying them into module pages.
- Keep commands and one descriptive preset copy in usage; keep scientific evidence requirements in verification.
- Define terms before use and explain why non-obvious rules exist. Use active, simplified technical English.
- Aim for instructions of 20 words, descriptions of 25 words and paragraphs of six sentences; preserve complete explanations.
- Start contract pages with Objective, then Usage for CLIs, Data flow, Organization logic, Invariants, Gotchas and Tests.
  Fold details into those sections; Tests is last and names real files and the behavior each checks.
- Use numbered steps and worked examples for algorithms. Render and inspect changed Mermaid diagrams using the [legend](core_algorithm.md#diagram-legend).
- Keep run IDs, results, machine allocations and status in the [workspace handoffs](../../../../plans/README.md) and saved experiment records.
  Usage uses input placeholders rather than dated study paths.

The contract test checks references and selected code/config facts as well as structure.
The [comprehension review](VALIDATION.md#comprehension-review) checks whether readers can use the explanations.
The AGENTS symlink points to the same working rules as CLAUDE.

## Current source owners

Descriptions identify the source's job; read its module docstring for local logic and invariants.
Empty package markers retain their own rows. No physical line-count index is maintained.

| Source | Computes or owns | Shared contract |
|---|---|---|
| [`checks/__init__.py`](../checks/__init__.py) | Whole-clip functional-mask versus checkpoint-export parity. | Source docstring |
| [`checks/ddp_native.py`](../checks/ddp_native.py) | Run bounded native acceptance for uniform temporal-crop head training. | Source docstring |
| [`checks/export_parity.py`](../checks/export_parity.py) | Compare an exported checkpoint with its declared functional D0 intervention. | Source docstring |
| [`checks/joint_export.py`](../checks/joint_export.py) | Isolate learned export transitions on an identical saved native case. | Source docstring |
| [`checks/joint_faults.py`](../checks/joint_faults.py) | Localize export rounding differences on one captured native attention input. | Source docstring |
| [`checks/joint_preflight.py`](../checks/joint_preflight.py) | Measure optimizer sensitivity and native one-block channel-ablation costs. | Source docstring |
| [`checks/joint_review.py`](../checks/joint_review.py) | Review complete saved optimizer, channel-ablation and export-isolation evidence. | Source docstring |
| [`checks/joint_split_parity.py`](../checks/joint_split_parity.py) | Verify compact split bundles against their matched masked computation. | Source docstring |
| [`checks/learned_heads.py`](../checks/learned_heads.py) | Validate recovered head envelopes and independently saved numerical checks. | Source docstring |
| [`core/__init__.py`](../core/__init__.py) | Model bootstrap, geometry, provenance and artifact paths. | Source docstring |
| [`core/artifacts.py`](../core/artifacts.py) | Locate output roots and allocate attributable run directories. | Source docstring |
| [`core/geometry.py`](../core/geometry.py) | Probe VAE temporal/spatial factors and convert latent shapes to pixel shapes. | Source docstring |
| [`core/ltx_adapter.py`](../core/ltx_adapter.py) | Isolate access to private upstream model, transformer and VAE interfaces. | Source docstring |
| [`core/model_registry.py`](../core/model_registry.py) | Resolve checkpoint components, schedules and probed capabilities by release. | Source docstring |
| [`core/native_manifest.py`](../core/native_manifest.py) | Build shared task, visibility, dtype and guidance fields for saved manifests. | Source docstring |
| [`core/numerics.py`](../core/numerics.py) | Deterministic numerical policy for every native pruning model process. | Source docstring |
| [`core/preflight.py`](../core/preflight.py) | Validate model components and free device capacity before loading weights. | Source docstring |
| [`core/provenance.py`](../core/provenance.py) | Hash saved inputs and record attributable checkpoint and process identities. | Source docstring |
| [`core/records.py`](../core/records.py) | Content digests and atomic publication of immutable evidence records. | Source docstring |
| [`core/session.py`](../core/session.py) | Open selected components and own transformer, text-context and VAE lifetimes. | Source docstring |
| [`core/waits.py`](../core/waits.py) | Bounded polling with one immutable receipt for every completed or failed wait. | Source docstring |
| [`data/__init__.py`](../data/__init__.py) | Saved whole-clip bidirectional inputs and text-conditioning cache. | Source docstring |
| [`data/compact_study.py`](../data/compact_study.py) | Prepare aligned compact weights and a new fixed-width adapter-recovery study. | Source docstring |
| [`data/corpus_pins.py`](../data/corpus_pins.py) | Pinned DNARendering membership and the official LTX-2.5 dev schedule. | Source docstring |
| [`data/ffn_study.py`](../data/ffn_study.py) | Freeze an FFN-mask study from verified source inputs and paired draws. | Source docstring |
| [`data/foreground.py`](../data/foreground.py) | Verify cropped person mattes and reviewed source-pixel face/hand tracks. | Source docstring |
| [`data/joint_study.py`](../data/joint_study.py) | Freeze joint clipped-selection studies from verified paired input authorities. | Source docstring |
| [`data/learned_head_study.py`](../data/learned_head_study.py) | Prepare actor-disjoint learning, validation and test inputs on CPU. | Source docstring |
| [`data/prompt_cache.py`](../data/prompt_cache.py) | Cache video text conditioning by model key and prompt hash. | Source docstring |
| [`data/study_io.py`](../data/study_io.py) | Frozen-study identity, source identity and pinned-file verification. | Source docstring |
| [`data/training_windows.py`](../data/training_windows.py) | Continuous F18 training windows from validated full-source input bundles. | Source docstring |
| [`data/whole_clip.py`](../data/whole_clip.py) | Validate saved D0 evidence and reconstruct its exact full-video model input. | [Contract](data/whole_clip.md) |
| [`evaluate/__init__.py`](../evaluate/__init__.py) | Whole-clip bidirectional comparison and benchmarking; shared media helpers. | Source docstring |
| [`evaluate/bench_whole_clip_d0.py`](../evaluate/bench_whole_clip_d0.py) | Measure matched baseline and candidate full-video D0 forward costs. | Source docstring |
| [`evaluate/compact.py`](../evaluate/compact.py) | Measure fixed compact recovery against its native and joint references. | Source docstring |
| [`evaluate/decode.py`](../evaluate/decode.py) | Decode dense latents through the session-owned video VAE. | Source docstring |
| [`evaluate/ffn.py`](../evaluate/ffn.py) | Evaluate saved binary head/FFN checkpoints on matched full-source D0 cases. | Source docstring |
| [`evaluate/historical_clip.py`](../evaluate/historical_clip.py) | Replay a saved head-only endpoint before new fixed-checkpoint evaluation. | Source docstring |
| [`evaluate/joint_final.py`](../evaluate/joint_final.py) | Evaluate completed joint selection and recovery checkpoints on saved inputs. | Source docstring |
| [`evaluate/joint_media.py`](../evaluate/joint_media.py) | Decode fixed joint endpoints and references into matched comparison media. | Source docstring |
| [`evaluate/joint_monitor_media.py`](../evaluate/joint_monitor_media.py) | Decode matched native, joint, head-only and budget-control monitor predictions. | Source docstring |
| [`evaluate/learned_head_media.py`](../evaluate/learned_head_media.py) | Decode frozen learned-head evidence and render synchronized comparisons. | Source docstring |
| [`evaluate/learned_heads.py`](../evaluate/learned_heads.py) | Evaluate merged head candidates and their independent native checks. | Source docstring |
| [`evaluate/media_common.py`](../evaluate/media_common.py) | Shared decoded-media helpers for pruning comparisons. | Source docstring |
| [`evaluate/metrics.py`](../evaluate/metrics.py) | Render synchronized capture, native and candidate comparison videos. | Source docstring |
| [`evaluate/timing.py`](../evaluate/timing.py) | Warmed, synchronized transformer timing and ABA/BAB bracket statistics. | Source docstring |
| [`evaluate/whole_clip_baseline.py`](../evaluate/whole_clip_baseline.py) | Produce saved full-video D0/D1 one-step rollout evidence. | [Contract](data/whole_clip.md) |
| [`evaluate/whole_clip_d0.py`](../evaluate/whole_clip_d0.py) | Compare saved D0 rollouts or execute functional head/FFN ablations. | Source docstring |
| [`learn/adapters.py`](../learn/adapters.py) | Attach FP32 low-rank updates to frozen BF16 projection weights. | Source docstring |
| [`learn/checkpoints.py`](../learn/checkpoints.py) | Resumable learning state: rank RNG, head-gate checkpoints and versioned state receipts. | Source docstring |
| [`learn/compact.py`](../learn/compact.py) | Run native acceptance and fixed compact adapter recovery with the DDP owner. | [Contract](learn/ddp_owner.md) |
| [`learn/compact_training.py`](../learn/compact_training.py) | Train fresh FP32 adapters on a fixed, physically compact BF16 backbone. | Source docstring |
| [`learn/ddp.py`](../learn/ddp.py) | Full BF16 replicas with DDP-managed FP32 adapters and head gates. | Source docstring |
| [`learn/ddp_owner.py`](../learn/ddp_owner.py) | Share native acceptance, monitoring and distributed training across mask families. | [Contract](learn/ddp_owner.md) |
| [`learn/ffn.py`](../learn/ffn.py) | Run bounded native acceptance and learned FFN/head-mask training. | [Contract](learn/ddp_owner.md) |
| [`learn/ffn_gates.py`](../learn/ffn_gates.py) | Learn individual FFN choices with binary forwards and aligned retained counts. | Source docstring |
| [`learn/ffn_training.py`](../learn/ffn_training.py) | Update aligned FFN masks, optional head gates and low-rank adapters. | Source docstring |
| [`learn/gates.py`](../learn/gates.py) | Learn scalar self-attention head gates and select executable block widths. | Source docstring |
| [`learn/heads.py`](../learn/heads.py) | Run bounded head selection and compact adapter-recovery phases. | [Contract](learn/heads.md) |
| [`learn/joint.py`](../learn/joint.py) | Run joint clipped selection with allocation and alignment reviews. | [Contract](learn/readmission.md) |
| [`learn/joint_training.py`](../learn/joint_training.py) | Update clipped head/channel scores and adapters and save exact joint state. | [Contract](learn/readmission.md) |
| [`learn/losses.py`](../learn/losses.py) | Official FP32 velocity reductions, excluding the supplied clean frame. | Source docstring |
| [`learn/protocol.py`](../learn/protocol.py) | Frozen batch semantics and phase budgets for serial and uniform-F18 DDP head learning. | Source docstring |
| [`learn/readmission.py`](../learn/readmission.py) | V3 family-local clipped gates and bounded joint selection. | [Contract](learn/readmission.md) |
| [`learn/receipt_logger.py`](../learn/receipt_logger.py) | Follow saved learned FFN or V3 joint receipts on CPU and publish one W&B run. | Source docstring |
| [`learn/run_full.py`](../learn/run_full.py) | Run the frozen GT head recipe through bounded child phases and CPU logging. | Source docstring |
| [`learn/step.py`](../learn/step.py) | Register the native model and hook-owned gates inside one trainable step. | Source docstring |
| [`learn/teacher_references.py`](../learn/teacher_references.py) | Store immutable original-model references with bounded disk and CPU caches. | Source docstring |
| [`learn/visualization.py`](../learn/visualization.py) | Save matched noisy, capture and predicted frames from an actual training draw. | Source docstring |
| [`learn/wandb_logger.py`](../learn/wandb_logger.py) | Publish one CPU W&B run per arm from already saved optimizer and monitor receipts. | Source docstring |
| [`score/__init__.py`](../score/__init__.py) | Native structural scoring, mask validation and checkpoint export. | Source docstring |
| [`score/estimators.py`](../score/estimators.py) | Task-independent unit scores and deterministic structural mask allocation. | Source docstring |
| [`score/export_depth.py`](../score/export_depth.py) | Stream intact-block deletion into a physically shorter checkpoint. | [Contract](score/export_depth.md) |
| [`score/export_joint.py`](../score/export_joint.py) | Export terminal joint masks, merged adapters and a recovered envelope. | Source docstring |
| [`score/export_joint_split.py`](../score/export_joint_split.py) | Export sliced base weights with separate trained low-rank adapters. | Source docstring |
| [`score/export_learned_ffn.py`](../score/export_learned_ffn.py) | Merge learned head/FFN adapters and export original-ID binary selections. | Source docstring |
| [`score/export_learned_heads.py`](../score/export_learned_heads.py) | Export selected head widths and merge fixed-width recovered adapters. | Source docstring |
| [`score/export_pruned.py`](../score/export_pruned.py) | Export structural width masks as self-describing safetensors checkpoints. | [Contract](score/export_pruned.md) |
| [`score/hooks.py`](../score/hooks.py) | Observe or mask video branches at their executed output-projection inputs. | Source docstring |
| [`score/safetensors_stream.py`](../score/safetensors_stream.py) | Publish transformed safetensors with bounded host-memory use. | Source docstring |
| [`score/token_sampling.py`](../score/token_sampling.py) | Select deterministic spatial query tokens from generated latent frames. | Source docstring |
| [`score/whole_clip_d0_scores.py`](../score/whole_clip_d0_scores.py) | Rank native head and FFN activations and publish a content-pinned width mask. | Source docstring |

## Entry scripts

Run from `LTX-2` in `ltx` as `python -m scripts.prune.<module> --help` to inspect all arguments without launching a model.
Use the linked workflow for prerequisites and command order. Parsers remain the authority for flags.

| Module | Source and parser | Workflow |
|---|---|---|
| `checks.ddp_native` | [Source](../checks/ddp_native.py) | [Recipe](usage.md#uniform-f18-ddp-training) |
| `checks.export_parity` | [Source](../checks/export_parity.py) | [Recipe](usage.md#width-export-and-parity) |
| `checks.joint_export` | [Source](../checks/joint_export.py) | [Recipe](usage.md#joint-clipped-gates-with-re-admission) |
| `checks.joint_faults` | [Source](../checks/joint_faults.py) | [Recipe](usage.md#joint-clipped-gates-with-re-admission) |
| `checks.joint_preflight` | [Source](../checks/joint_preflight.py) | [Recipe](usage.md#joint-clipped-gates-with-re-admission) |
| `checks.joint_review` | [Source](../checks/joint_review.py) | [Recipe](usage.md#joint-clipped-gates-with-re-admission) |
| `checks.joint_split_parity` | [Source](../checks/joint_split_parity.py) | [Recipe](usage.md#joint-clipped-gates-with-re-admission) |
| `checks.learned_heads` | [Source](../checks/learned_heads.py) | [Recipe](usage.md#joint-clipped-gates-with-re-admission) |
| `core.preflight` | [Source](../core/preflight.py) | [Recipe](usage.md#preflight-and-prompt-cache) |
| `data.compact_study` | [Source](../data/compact_study.py) | [Recipe](usage.md#compact-a-lora-recovery) |
| `data.ffn_study` | [Source](../data/ffn_study.py) | [Recipe](usage.md#learned-ffn-masks) |
| `data.joint_study` | [Source](../data/joint_study.py) | [Recipe](usage.md#joint-clipped-gates-with-re-admission) |
| `data.learned_head_study` | [Source](../data/learned_head_study.py) | [Recipe](usage.md#learned-head-phases) |
| `data.prompt_cache` | [Source](../data/prompt_cache.py) | [Recipe](usage.md#preflight-and-prompt-cache) |
| `evaluate.bench_whole_clip_d0` | [Source](../evaluate/bench_whole_clip_d0.py) | [Recipe](usage.md#held-out-quality-and-timing) |
| `evaluate.compact` | [Source](../evaluate/compact.py) | [Recipe](usage.md#compact-a-lora-recovery) |
| `evaluate.ffn` | [Source](../evaluate/ffn.py) | [Recipe](usage.md#held-out-quality-and-timing) |
| `evaluate.historical_clip` | [Source](../evaluate/historical_clip.py) | [Recipe](usage.md#completed-joint-v3-evaluation) |
| `evaluate.joint_final` | [Source](../evaluate/joint_final.py) | [Recipe](usage.md#completed-joint-v3-evaluation) |
| `evaluate.joint_media` | [Source](../evaluate/joint_media.py) | [Recipe](usage.md#completed-joint-v3-evaluation) |
| `evaluate.joint_monitor_media` | [Source](../evaluate/joint_monitor_media.py) | [Recipe](usage.md#completed-joint-v3-evaluation) |
| `evaluate.learned_head_media` | [Source](../evaluate/learned_head_media.py) | [Recipe](usage.md#held-out-quality-and-timing) |
| `evaluate.learned_heads` | [Source](../evaluate/learned_heads.py) | [Recipe](usage.md#held-out-quality-and-timing) |
| `evaluate.whole_clip_baseline` | [Source](../evaluate/whole_clip_baseline.py) | [Recipe](usage.md#save-the-baseline) |
| `evaluate.whole_clip_d0` | [Source](../evaluate/whole_clip_d0.py) | [Recipe](usage.md#held-out-quality-and-timing) |
| `learn.compact` | [Source](../learn/compact.py) | [Recipe](usage.md#compact-a-lora-recovery) |
| `learn.ffn` | [Source](../learn/ffn.py) | [Recipe](usage.md#learned-ffn-masks) |
| `learn.heads` | [Source](../learn/heads.py) | [Recipe](usage.md#learned-head-phases) |
| `learn.joint` | [Source](../learn/joint.py) | [Recipe](usage.md#joint-clipped-gates-with-re-admission) |
| `learn.receipt_logger` | [Source](../learn/receipt_logger.py) | [Recipe](usage.md#training-visualization) |
| `learn.run_full` | [Source](../learn/run_full.py) | [Recipe](usage.md#uniform-f18-ddp-training) |
| `learn.wandb_logger` | [Source](../learn/wandb_logger.py) | [Recipe](usage.md#training-visualization) |
| `score.export_depth` | [Source](../score/export_depth.py) | [Recipe](usage.md#depth-export) |
| `score.export_joint` | [Source](../score/export_joint.py) | [Recipe](usage.md#width-export-and-parity) |
| `score.export_joint_split` | [Source](../score/export_joint_split.py) | [Recipe](usage.md#width-export-and-parity) |
| `score.export_learned_ffn` | [Source](../score/export_learned_ffn.py) | [Recipe](usage.md#width-export-and-parity) |
| `score.export_learned_heads` | [Source](../score/export_learned_heads.py) | [Recipe](usage.md#width-export-and-parity) |
| `score.export_pruned` | [Source](../score/export_pruned.py) | [Recipe](usage.md#width-export-and-parity) |
| `score.whole_clip_d0_scores` | [Source](../score/whole_clip_d0_scores.py) | [Recipe](usage.md#calibration-and-functional-ablation) |

## Stable references and preserved material

[METHODS](METHODS.md), [known limits](known_gaps.md) and the [baseline pointer](evaluate/whole_clip_baseline.md)
keep external links working; they point into current owners instead of duplicating facts.
All retired module pages and the starting source state have an [exact-byte snapshot](../../../../expr/prune_doc_system_20261022/before/).
The [migration record](../../../../expr/prune_doc_system_20261022/migration.json) maps retired pages to source owners and records their hashes.
This archive supports review and attribution; active launch instructions live in usage.
