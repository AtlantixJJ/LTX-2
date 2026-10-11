# `prune` — current design index

## Reading order

1. [Architecture](ARCHITECTURE.md): source owners and allowed dependencies.
2. [Core algorithm](core_algorithm.md): shapes, conditioning, scoring and numerical rules.
3. [Verification](VALIDATION.md): CPU checks and native acceptance gates.
4. [Known gaps](known_gaps.md): current support and evidence limits.
5. [Run guide](usage.md): direct commands and the queued baseline route.

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
- Aim for 20 words per instruction and 25 words per description.
- Use active voice, present tense and one term per concept.
- Define terms before use. Put ordered steps in lists.
- Preserve exact code names, paths, numbers and equations.
- Use “must” for a rule and “can” for an option.

Use small left-to-right Mermaid flows with the [shared legend](core_algorithm.md#diagram-legend).
Render and inspect diagrams before delivery.
For visual outputs, explain data, panel positions, labels, coverage and readability.

Update the module design or header when objective, data flow, organization or invariants change.
Docs describe current behavior. Historical runs and conclusions stay in saved experiment records.

## Current source owners

| Source | Design | Physical lines |
|---|---|---:|
| `checks/__init__.py` | package marker | 1 |
| `checks/export_parity.py` | [design](checks/export_parity.md) | 147 |
| `core/__init__.py` | package marker | 1 |
| `core/artifacts.py` | [design](core/artifacts.md) (small-module supplement) | 40 |
| `core/geometry.py` | [design](core/geometry.md) (small-module supplement) | 72 |
| `core/ltx_adapter.py` | [design](core/ltx_adapter.md) (small-module supplement) | 42 |
| `core/model_registry.py` | [design](core/model_registry.md) | 179 |
| `core/preflight.py` | [design](core/preflight.md) | 136 |
| `core/provenance.py` | [design](core/provenance.md) | 119 |
| `core/session.py` | [design](core/session.md) | 117 |
| `data/__init__.py` | package marker | 1 |
| `data/prompt_cache.py` | [design](data/prompt_cache.md) | 111 |
| `data/whole_clip.py` | [design](data/whole_clip.md) | 280 |
| `evaluate/__init__.py` | package marker | 1 |
| `evaluate/bench_whole_clip_d0.py` | [design](evaluate/bench_whole_clip_d0.md) | 140 |
| `evaluate/decode.py` | [design](evaluate/decode.md) (small-module supplement) | 23 |
| `evaluate/metrics.py` | [design](evaluate/metrics.md) | 103 |
| `evaluate/whole_clip_baseline.py` | [design](evaluate/whole_clip_baseline.md) | 570 |
| `evaluate/whole_clip_d0.py` | [design](evaluate/whole_clip_d0.md) | 245 |
| `score/__init__.py` | package marker | 1 |
| `score/estimators.py` | [design](score/estimators.md) (small-module supplement) | 48 |
| `score/export_depth.py` | [design](score/export_depth.md) | 431 |
| `score/export_pruned.py` | [design](score/export_pruned.md) | 205 |
| `score/ffn_reconstruction.py` | [design](score/ffn_reconstruction.md) | 261 |
| `score/hooks.py` | [design](score/hooks.md) | 229 |
| `score/token_sampling.py` | [design](score/token_sampling.md) (small-module supplement) | 79 |
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
