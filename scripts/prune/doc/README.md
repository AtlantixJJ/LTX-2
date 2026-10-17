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
| `checks/__init__.py` | package marker | 1 |
| `checks/export_parity.py` | [design](checks/export_parity.md) | 147 |
| `checks/four_gpu_feasibility.py` | [design](checks/four_gpu_feasibility.md) | 85 |
| `checks/fsdp_native.py` | [design](checks/fsdp_native.md) | 263 |
| `checks/ddp_f28.py.txt` | [bounded diagnostic](checks/ddp_f28.md) | 203 |
| `checks/fsdp_small.py` | [design](checks/fsdp_small.md) | 162 |
| `checks/learned_heads.py` | [design](checks/learned_heads.md) | 227 |
| `checks/mobilewan_native.py` | [design](checks/mobilewan_native.md) | 118 |
| `checks/ddp_native.py` | [design](checks/ddp_native.md) | 307 |
| `core/__init__.py` | package marker | 1 |
| `core/artifacts.py` | [design](core/artifacts.md) | 40 |
| `core/geometry.py` | [design](core/geometry.md) | 72 |
| `core/ltx_adapter.py` | [design](core/ltx_adapter.md) | 59 |
| `core/model_registry.py` | [design](core/model_registry.md) | 179 |
| `core/preflight.py` | [design](core/preflight.md) | 136 |
| `core/provenance.py` | [design](core/provenance.md) | 119 |
| `core/session.py` | [design](core/session.md) | 117 |
| `data/continue_eligible.py` | [design](data/continue_eligible.md) | 208 |
| `data/progressive_head_study.py` | [design](data/progressive_head_study.md) | 151 |
| `data/__init__.py` | package marker | 1 |
| `data/foreground.py` | [design](data/foreground.md) | 140 |
| `data/head_score_dataset.py` | [design](data/head_score_dataset.md) | 139 |
| `data/learned_head_study.py` | [design](data/learned_head_study.md) | 551 |
| `data/prompt_cache.py` | [design](data/prompt_cache.md) | 111 |
| `data/reuse_head_scores.py` | [design](data/reuse_head_scores.md) | 137 |
| `data/resume_learned_heads.py` | [design](data/resume_learned_heads.md) | 211 |
| `data/extend_learned_heads.py` | [design](data/extend_learned_heads.md) | 226 |
| `data/whole_clip.py` | [design](data/whole_clip.md) | 353 |
| `data/training_windows.py` | [design](data/training_windows.md) | 104 |
| `evaluate/__init__.py` | package marker | 1 |
| `evaluate/bench_whole_clip_d0.py` | [design](evaluate/bench_whole_clip_d0.md) | 193 |
| `evaluate/decode.py` | [design](evaluate/decode.md) | 23 |
| `evaluate/learned_head_media.py` | [design](evaluate/learned_head_media.md) | 290 |
| `evaluate/learned_head_checkpoint_media.py.txt` | [pinned checkpoint media](evaluate/learned_head_checkpoint_media.md) | 721 |
| `evaluate/eligible_checkpoint_media.py.txt` | [design](evaluate/eligible_checkpoint_media.md) | 706 |
| `evaluate/zero_one_checkpoint_media.py.txt` | [design](evaluate/zero_one_checkpoint_media.md) | 706 |
| `evaluate/progressive_checkpoint_media.py.txt` | [progressive checkpoint media](evaluate/progressive_checkpoint_media.md) | 725 |
| `evaluate/learned_heads.py` | [design](evaluate/learned_heads.md) | 338 |
| `evaluate/metrics.py` | [design](evaluate/metrics.md) | 103 |
| `evaluate/mobilewan_media.py` | [design](evaluate/mobilewan_media.md) | 218 |
| `evaluate/mobilewan_removal.py` | [design](evaluate/mobilewan_removal.md) | 215 |
| `evaluate/whole_clip_baseline.py` | [design](evaluate/whole_clip_baseline.md) | 707 |
| `evaluate/whole_clip_d0.py` | [design](evaluate/whole_clip_d0.md) | 245 |
| `learn/eligible.py` | [design](learn/eligible.md) | 195 |
| `learn/zero_one.py` | [design](learn/zero_one.md) | 180 |
| `data/continue_zero_one.py` | [design](data/continue_zero_one.md) | 187 |
| `learn/progressive.py` | [design](learn/progressive.md) | 261 |
| `learn/adapters.py` | [design](learn/adapters.md) | 127 |
| `learn/checkpoints.py` | [design](learn/checkpoints.md) | 235 |
| `learn/distributed.py` | [design](learn/distributed.md) | 153 |
| `learn/gates.py` | [design](learn/gates.md) | 141 |
| `learn/heads.py` | [design](learn/heads.md) | 1531 |
| `learn/losses.py` | [design](learn/losses.md) | 155 |
| `learn/placement.py` | [design](learn/placement.md) | 107 |
| `learn/run_full.py` | [design](learn/run_full.md) | 280 |
| `learn/teacher_references.py` | [design](learn/teacher_references.md) | 105 |
| `learn/ddp.py` | [design](learn/ddp.md) | 51 |
| `learn/protocol.py` | logic in module header | 77 |
| `learn/step.py` | logic in module header | 44 |
| `learn/wandb_logger.py` | [design](learn/wandb_logger.md) | 289 |
| `score/__init__.py` | package marker | 1 |
| `score/estimators.py` | [design](score/estimators.md) | 48 |
| `score/export_depth.py` | [design](score/export_depth.md) | 431 |
| `score/export_learned_heads.py` | [design](score/export_learned_heads.md) | 269 |
| `score/export_pruned.py` | [design](score/export_pruned.md) | 205 |
| `score/ffn_reconstruction.py` | [design](score/ffn_reconstruction.md) | 261 |
| `score/hooks.py` | [design](score/hooks.md) | 264 |
| `score/mobilewan_aggregate.py` | [design](score/mobilewan_aggregate.md) | 241 |
| `score/mobilewan_dataset.py` | [design](score/mobilewan_dataset.md) | 212 |
| `score/mobilewan_heads.py` | [design](score/mobilewan_heads.md) | 127 |
| `score/mobilewan_records.py` | [design](score/mobilewan_records.md) | 134 |
| `score/token_sampling.py` | [design](score/token_sampling.md) | 79 |
| `score/whole_clip_d0_scores.py` | [design](score/whole_clip_d0_scores.md) | 164 |

`scripts/` and `scripts/prune/` remain PEP 420 namespace packages.
There is no root `__init__.py` to add.

## Entry scripts

Run modules from `LTX-2` in `ltx`. Each link goes to its Usage section.

| Module | Purpose | Started by |
|---|---|---|
| [`core.preflight`](core/preflight.md#usage) | Check components and GPU memory | direct CLI |
| [`data.prompt_cache`](data/prompt_cache.md#usage) | Build or verify default text context | direct CLI |
| [`evaluate.whole_clip_baseline`](evaluate/whole_clip_baseline.md#usage) | Save source or candidate rollouts | human CLI; agent baseline through avatar queue |
| [`score.whole_clip_d0_scores`](score/whole_clip_d0_scores.md#usage) | Rank heads and FFN channels | direct CLI |
| [`score.export_pruned`](score/export_pruned.md#usage) | Export width masks | direct CLI; CUDA preflight, CPU surgery |
| [`score.export_depth`](score/export_depth.md#usage) | Export retained blocks | direct CPU CLI |
| [`checks.export_parity`](checks/export_parity.md#usage) | Check intervention/export agreement | direct CLI |
| [`evaluate.whole_clip_d0`](evaluate/whole_clip_d0.md#usage) | Compare saved outputs or functional masks | CPU saved analysis; GPU ablation |
| [`evaluate.bench_whole_clip_d0`](evaluate/bench_whole_clip_d0.md#usage) | Measure matched forward cost | direct CLI |
| [`checks.fsdp_small`](checks/fsdp_small.md#usage) | Check four-rank gradients and exact resume | direct torchrun |
| [`checks.ddp_native`](checks/ddp_native.md#usage) | Accept uniform-F18 DDP crops, gradients, memory and resume | direct torchrun |
| [`checks.fsdp_native`](checks/fsdp_native.md#usage) | Accept full-clip FSDP parity and memory | direct torchrun |
| [`checks/ddp_f28.py.txt`](checks/ddp_f28.md#usage) | Test one full F18 or F28 clip per GPU with two-rank DDP | direct torchrun file |
| [`evaluate/learned_head_checkpoint_media.py.txt`](evaluate/learned_head_checkpoint_media.md#usage) | Decode and compare saved monitor predictions of a live study | direct file on one free GPU |
| [`evaluate/progressive_checkpoint_media.py.txt`](evaluate/progressive_checkpoint_media.md#usage) | Compare actual progressive gates with the prior soft study | direct file on GPU 6 |
| [`data.reuse_head_scores`](data/reuse_head_scores.md#usage) | Import compatible historical raw scores | direct CPU CLI |
| [`data.resume_learned_heads`](data/resume_learned_heads.md#usage) | Preserve a stopped checkpoint under the amended sparsity loss | direct CPU CLI |
| [`data.extend_learned_heads`](data/extend_learned_heads.md#usage) | Continue a completed soft checkpoint under a longer phase budget | direct CPU CLI |
| [`learn.wandb_logger`](learn/wandb_logger.md#usage) | Log saved rank charts and globally averaged metrics | direct CPU writer |
| [`learn.run_full`](learn/run_full.md#usage) | Run and monitor the full paired GT pilot | direct detached owner |

The avatar wrapper owns supervision and queue receipts.
These entry scripts retain their broader direct interfaces.
See [launch rules](usage.md#launch-and-output-rules).

## Core logic review map

- [Saved inputs](data/whole_clip.md): manifest content, exact reconstruction, actor split and export identity.
- [Session](core/session.md): BF16 dtype, prompt selection and resident model lifetime.
- [Hooks](score/hooks.md): executed mask boundaries and complete families.
- [Estimators](score/estimators.md) and [sampling](score/token_sampling.md): score meaning and spatial coverage.
- [Width export](score/export_pruned.md): Q/K normalization, positional identity and storage/execution modes.
- [Depth export](score/export_depth.md): retained order, streaming copy and parameter denominators.
- [Local fitting](score/ffn_reconstruction.md): bounded ridge solve and calibration-cache limits.
- [Comparison](evaluate/whole_clip_d0.md): generated-frame metrics and labeled VAE panels.
- [Timing](evaluate/bench_whole_clip_d0.md): saved-output guard, synchronized repetitions and drift.

## Artifact and evidence boundaries

Full manifest and tensor hashes bind scientific inputs.
Sampled checkpoint fingerprints identify checkpoints but do not prove full-file integrity.
The queued baseline wrapper adds complete component and software pins.

Saved verification opens no model and does not regenerate CUDA noise.
Report rebuilds read saved results; they must not launch missing runs.
Passing schema checks, CPU tests or export parity alone does not establish decoded quality or speed.

The progressive [preparation owner](data/progressive_head_study.md#usage) verifies r13 inputs and checkpoint zero.
Its [policy owner](learn/progressive.md) defines irreversible mixed gates. The direct run ends after 500 updates and saved monitors.

The [zero/one preparation owner](data/continue_zero_one.md) migrates checkpoint 500 without resetting optimizer/RNG state.
The [zero/one policy](learn/zero_one.md) adds permanent high-regret ones and supports empty free populations.
