# `prune` — current design index

## Reading order

1. [Architecture](ARCHITECTURE.md): source owners and allowed dependencies.
2. [Core algorithm](core_algorithm.md): shapes, conditioning, scoring and numerical rules.
3. [Verification](VALIDATION.md): CPU checks and native acceptance gates.
4. [Known gaps](known_gaps.md): current support and evidence limits.
5. [Run guide](usage.md): direct commands and the queued baseline route.
6. [Report and wording guide](reporting.md): question structure, technical language, media and saved-report verification.

The [literature and method agenda](METHODS.md) supplies research context.
It does not replace implementation contracts or current native evidence.

The learned FFN extension has separate [preparation](data/ffn_study.md),
[training](learn/ffn.md), [numerical state](learn/ffn_training.md) and
[export](score/export_learned_ffn.md) owners.
The small `learn/ffn_gates.py` and CPU `learn/receipt_logger.py` modules explain their logic in their headers.

The joint V3 extension has [family rules](learn/readmission.md), [numerical state](learn/joint_training.md),
[training](learn/joint.md), [preparation](data/joint_study.md), [preflight checks](checks/joint_preflight.md),
[export isolation](checks/joint_export.md), [fault diagnosis](checks/joint_faults.md),
[native split replay](checks/joint_split_parity.md) and [terminal split export](score/export_joint_split.md).
The [merged export](score/export_joint.md) requires separate qualification.
Completed V3 studies use [held-out inference and deployment](evaluate/joint_final.md)
and [matched decoded comparisons](evaluate/joint_media.md). These owners do not resume training.

## Documentation rules

Use the [avatar documentation reference](../../onestep_avatar/doc/README.md#documentation-rules).
A production module over 100 physical lines requires `doc/<relative-source-path>.md`.
Count blank lines, comments and headers. A module with 100 lines or fewer uses a header.
Keep existing pruning design pages for small modules as supplemental references.
Empty package markers and tests need no design page.
Do not split or pad source files to change the rule.

A module design has Objective, Data flow, Organization logic, Invariants, Gotchas and Tests.
An entry-script design adds Usage directly after Objective.
Explain calculations, decision order, state changes and failures.
Give a worked input and expected result. A symbol list alone is insufficient.

Each Usage section states the launch owner, conda environment, a complete example, main arguments and output files.
Link to the matching run-guide section.
Include a CPU `--dry-run` only when the actual entrypoint supports it.
Prune's direct entrypoints do not expose `--dry-run`; the avatar experiment and queue do.

Use simplified technical English, about 80% to ASD-STE100:

- Keep one topic per sentence and at most six sentences per paragraph.
- Keep instructions to 20 words and descriptions to 25 words, where practical.
- Use active voice, present tense and one term per concept.
- Use the imperative for instructions. Keep the articles `a`, `an` and `the`.
- Do not stack more than three nouns.
- Define terms before use. Put ordered steps in lists.
- Preserve exact code names, paths, numbers and equations.
- Use “must” for a rule and “can” for an option.

Use small left-to-right Mermaid flows with the [shared legend](core_algorithm.md#diagram-legend).
Render and inspect diagrams before delivery.
For visual outputs, explain data, panel positions, labels, coverage and readability.

Update the module design or header when objective, data flow, organization or invariants change.
Docs describe current behavior. Historical runs and conclusions stay in saved experiment records.

## Report contract

Use the [report and wording guide](reporting.md) for reports, figure captions and video labels.
It follows the workspace report rules and the avatar documentation and media contracts.
Organize by question, define Terms, give the Short answer first, and vary one factor per experimental comparison.
Use PNG/PDF/data JSON figures and synchronized RGB/VAE comparison evidence.
Report generators read saved artifacts only. Keep prose in narrative config and layout in the generator.
Rebuild, validate and inspect the report, including narrow layouts, after changes.

## Current source owners

| Source | Design | Physical lines |
|---|---|---:|
| `checks/__init__.py` | inline header | 1 |
| `checks/ddp_native.py` | [checks/ddp_native.md](checks/ddp_native.md) | 178 |
| `checks/export_parity.py` | [checks/export_parity.md](checks/export_parity.md) | 185 |
| `checks/joint_export.py` | [checks/joint_export.md](checks/joint_export.md) | 236 |
| `checks/joint_faults.py` | [checks/joint_faults.md](checks/joint_faults.md) | 167 |
| `checks/joint_preflight.py` | [checks/joint_preflight.md](checks/joint_preflight.md) | 256 |
| `checks/joint_review.py` | [checks/joint_review.md](checks/joint_review.md) | 97 |
| `checks/joint_split_parity.py` | [checks/joint_split_parity.md](checks/joint_split_parity.md) | 191 |
| `checks/learned_heads.py` | [checks/learned_heads.md](checks/learned_heads.md) | 228 |
| `core/__init__.py` | inline header | 1 |
| `core/artifacts.py` | [core/artifacts.md](core/artifacts.md) | 40 |
| `core/geometry.py` | [core/geometry.md](core/geometry.md) | 50 |
| `core/ltx_adapter.py` | [core/ltx_adapter.md](core/ltx_adapter.md) | 59 |
| `core/model_registry.py` | [core/model_registry.md](core/model_registry.md) | 179 |
| `core/native_manifest.py` | inline header | 35 |
| `core/numerics.py` | inline header | 23 |
| `core/preflight.py` | [core/preflight.md](core/preflight.md) | 136 |
| `core/provenance.py` | [core/provenance.md](core/provenance.md) | 154 |
| `core/records.py` | [core/records.md](core/records.md) | 136 |
| `core/session.py` | [core/session.md](core/session.md) | 159 |
| `core/waits.py` | inline header | 45 |
| `data/__init__.py` | inline header | 1 |
| `data/corpus_pins.py` | inline header | 56 |
| `data/ffn_study.py` | [data/ffn_study.md](data/ffn_study.md) | 148 |
| `data/foreground.py` | [data/foreground.md](data/foreground.md) | 120 |
| `data/joint_study.py` | [data/joint_study.md](data/joint_study.md) | 157 |
| `data/learned_head_study.py` | [data/learned_head_study.md](data/learned_head_study.md) | 564 |
| `data/prompt_cache.py` | [data/prompt_cache.md](data/prompt_cache.md) | 111 |
| `data/study_io.py` | inline header | 47 |
| `data/training_windows.py` | [data/training_windows.md](data/training_windows.md) | 155 |
| `data/whole_clip.py` | [data/whole_clip.md](data/whole_clip.md) | 490 |
| `evaluate/__init__.py` | inline header | 1 |
| `evaluate/bench_whole_clip_d0.py` | [evaluate/bench_whole_clip_d0.md](evaluate/bench_whole_clip_d0.md) | 129 |
| `evaluate/decode.py` | [evaluate/decode.md](evaluate/decode.md) | 21 |
| `evaluate/ffn.py` | [evaluate/ffn.md](evaluate/ffn.md) | 381 |
| `evaluate/joint_final.py` | [evaluate/joint_final.md](evaluate/joint_final.md) | 245 |
| `evaluate/joint_media.py` | [evaluate/joint_media.md](evaluate/joint_media.md) | 156 |
| `evaluate/learned_head_media.py` | [evaluate/learned_head_media.md](evaluate/learned_head_media.md) | 231 |
| `evaluate/learned_heads.py` | [evaluate/learned_heads.md](evaluate/learned_heads.md) | 339 |
| `evaluate/media_common.py` | [evaluate/media_common.md](evaluate/media_common.md) | 148 |
| `evaluate/metrics.py` | [evaluate/metrics.md](evaluate/metrics.md) | 76 |
| `evaluate/timing.py` | [evaluate/timing.md](evaluate/timing.md) | 110 |
| `evaluate/whole_clip_baseline.py` | [evaluate/whole_clip_baseline.md](evaluate/whole_clip_baseline.md) | 685 |
| `evaluate/whole_clip_d0.py` | [evaluate/whole_clip_d0.md](evaluate/whole_clip_d0.md) | 309 |
| `learn/adapters.py` | [learn/adapters.md](learn/adapters.md) | 162 |
| `learn/checkpoints.py` | [learn/checkpoints.md](learn/checkpoints.md) | 209 |
| `learn/ddp.py` | [learn/ddp.md](learn/ddp.md) | 79 |
| `learn/ddp_owner.py` | [learn/ddp_owner.md](learn/ddp_owner.md) | 513 |
| `learn/ffn.py` | [learn/ffn.md](learn/ffn.md) | 95 |
| `learn/ffn_gates.py` | inline header | 79 |
| `learn/ffn_training.py` | [learn/ffn_training.md](learn/ffn_training.md) | 213 |
| `learn/gates.py` | [learn/gates.md](learn/gates.md) | 123 |
| `learn/heads.py` | [learn/heads.md](learn/heads.md) | 1336 |
| `learn/joint.py` | [learn/joint.md](learn/joint.md) | 228 |
| `learn/joint_training.py` | [learn/joint_training.md](learn/joint_training.md) | 338 |
| `learn/losses.py` | [learn/losses.md](learn/losses.md) | 153 |
| `learn/protocol.py` | inline header | 69 |
| `learn/readmission.py` | [learn/readmission.md](learn/readmission.md) | 362 |
| `learn/receipt_logger.py` | [learn/receipt_logger.md](learn/receipt_logger.md) | 96 |
| `learn/run_full.py` | [learn/run_full.md](learn/run_full.md) | 228 |
| `learn/step.py` | inline header | 42 |
| `learn/teacher_references.py` | [learn/teacher_references.md](learn/teacher_references.md) | 105 |
| `learn/wandb_logger.py` | [learn/wandb_logger.md](learn/wandb_logger.md) | 268 |
| `score/__init__.py` | inline header | 1 |
| `score/estimators.py` | [score/estimators.md](score/estimators.md) | 81 |
| `score/export_depth.py` | [score/export_depth.md](score/export_depth.md) | 524 |
| `score/export_joint.py` | [score/export_joint.md](score/export_joint.md) | 127 |
| `score/export_joint_split.py` | [score/export_joint_split.md](score/export_joint_split.md) | 227 |
| `score/export_learned_ffn.py` | [score/export_learned_ffn.md](score/export_learned_ffn.md) | 328 |
| `score/export_learned_heads.py` | [score/export_learned_heads.md](score/export_learned_heads.md) | 236 |
| `score/export_pruned.py` | [score/export_pruned.md](score/export_pruned.md) | 254 |
| `score/hooks.py` | [score/hooks.md](score/hooks.md) | 286 |
| `score/safetensors_stream.py` | [score/safetensors_stream.md](score/safetensors_stream.md) | 168 |
| `score/token_sampling.py` | [score/token_sampling.md](score/token_sampling.md) | 97 |
| `score/whole_clip_d0_scores.py` | [score/whole_clip_d0_scores.md](score/whole_clip_d0_scores.md) | 209 |

## Entry scripts

Run from `LTX-2` in `ltx`. Read each owner's Usage section before launching.

| Module | Design |
|---|---|
| [`checks.ddp_native`](checks/ddp_native.md) | [Usage](checks/ddp_native.md#usage) |
| [`checks.export_parity`](checks/export_parity.md) | [Usage](checks/export_parity.md#usage) |
| [`checks.joint_export`](checks/joint_export.md) | [Usage](checks/joint_export.md#usage) |
| [`checks.joint_faults`](checks/joint_faults.md) | [Usage](checks/joint_faults.md#usage) |
| [`checks.joint_preflight`](checks/joint_preflight.md) | [Usage](checks/joint_preflight.md#usage) |
| [`checks.joint_review`](checks/joint_review.md) | [Usage](checks/joint_review.md#usage) |
| [`checks.joint_split_parity`](checks/joint_split_parity.md) | [Usage](checks/joint_split_parity.md#usage) |
| [`checks.learned_heads`](checks/learned_heads.md) | [Usage](checks/learned_heads.md#usage) |
| [`core.preflight`](core/preflight.md) | [Usage](core/preflight.md#usage) |
| [`data.ffn_study`](data/ffn_study.md) | [Usage](data/ffn_study.md#usage) |
| [`data.joint_study`](data/joint_study.md) | [Usage](data/joint_study.md#usage) |
| [`data.learned_head_study`](data/learned_head_study.md) | [Usage](data/learned_head_study.md#usage) |
| [`data.prompt_cache`](data/prompt_cache.md) | [Usage](data/prompt_cache.md#usage) |
| [`evaluate.bench_whole_clip_d0`](evaluate/bench_whole_clip_d0.md) | [Usage](evaluate/bench_whole_clip_d0.md#usage) |
| [`evaluate.ffn`](evaluate/ffn.md) | [Usage](evaluate/ffn.md#usage) |
| [`evaluate.joint_final`](evaluate/joint_final.md) | [Usage](evaluate/joint_final.md#usage) |
| [`evaluate.joint_media`](evaluate/joint_media.md) | [Usage](evaluate/joint_media.md#usage) |
| [`evaluate.learned_head_media`](evaluate/learned_head_media.md) | [Usage](evaluate/learned_head_media.md#usage) |
| [`evaluate.learned_heads`](evaluate/learned_heads.md) | [Usage](evaluate/learned_heads.md#usage) |
| [`evaluate.whole_clip_baseline`](evaluate/whole_clip_baseline.md) | [Usage](evaluate/whole_clip_baseline.md#usage) |
| [`evaluate.whole_clip_d0`](evaluate/whole_clip_d0.md) | [Usage](evaluate/whole_clip_d0.md#usage) |
| [`learn.ffn`](learn/ffn.md) | [Usage](learn/ffn.md#usage) |
| [`learn.heads`](learn/heads.md) | [Usage](learn/heads.md#usage) |
| [`learn.joint`](learn/joint.md) | [Usage](learn/joint.md#usage) |
| [`learn.receipt_logger`](learn/receipt_logger.md) | [Usage](learn/receipt_logger.md#usage) |
| [`learn.run_full`](learn/run_full.md) | [Usage](learn/run_full.md#usage) |
| [`learn.wandb_logger`](learn/wandb_logger.md) | [Usage](learn/wandb_logger.md#usage) |
| [`score.export_depth`](score/export_depth.md) | [Usage](score/export_depth.md#usage) |
| [`score.export_joint`](score/export_joint.md) | [Usage](score/export_joint.md#usage) |
| [`score.export_joint_split`](score/export_joint_split.md) | [Usage](score/export_joint_split.md#usage) |
| [`score.export_learned_ffn`](score/export_learned_ffn.md) | [Usage](score/export_learned_ffn.md#usage) |
| [`score.export_learned_heads`](score/export_learned_heads.md) | [Usage](score/export_learned_heads.md#usage) |
| [`score.export_pruned`](score/export_pruned.md) | [Usage](score/export_pruned.md#usage) |
| [`score.whole_clip_d0_scores`](score/whole_clip_d0_scores.md) | [Usage](score/whole_clip_d0_scores.md#usage) |
## Core logic review map

- [Saved inputs](data/whole_clip.md): manifest content, exact reconstruction, actor split and export identity.
- [Session](core/session.md): BF16 dtype, prompt selection and resident model lifetime.
- [Hooks](score/hooks.md): executed mask boundaries and complete families.
- [Estimators](score/estimators.md) and [sampling](score/token_sampling.md): score meaning and spatial coverage.
- [Width export](score/export_pruned.md): Q/K normalization, positional identity and storage/execution modes.
- [Depth export](score/export_depth.md): retained order, streaming copy and parameter denominators.
- [Comparison](evaluate/whole_clip_d0.md): generated-frame metrics and labeled VAE panels.
- [Timing](evaluate/bench_whole_clip_d0.md): saved-output guard, synchronized repetitions and drift.

## Artifact and evidence boundaries

Full manifest and tensor hashes bind scientific inputs.
Sampled checkpoint fingerprints identify checkpoints but do not prove full-file integrity.
The queued baseline wrapper adds complete component and software pins.

Saved verification opens no model and does not regenerate CUDA noise.
Report rebuilds read saved results; they must not launch missing runs.
Passing schema checks, CPU tests or export parity alone does not establish decoded quality or speed.
