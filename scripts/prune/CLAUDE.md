# `scripts/prune` working rules

Pruning has one task: a full-clip bidirectional D0 forward at each `[sigma, 0]`
schedule, with the capture's first latent frame clean. Calibration, ablation,
export parity, output quality and timing use that same input contract.
Learned-head optimizer inputs can use the frozen `uniform_f18` training amendment:
continuous 18-latent crops with a clean window first latent. Full-source evidence stays explicit.

Run from LTX-2 in the `ltx` conda environment. Check `nvidia-smi` before loading
models. Keep `scripts/` and `scripts/prune/` as PEP 420 namespace packages.

Read the [design index](doc/README.md), [architecture](doc/ARCHITECTURE.md),
[core algorithm](doc/core_algorithm.md), [verification](doc/VALIDATION.md)
and [known gaps](doc/known_gaps.md) before a model-facing change.
Use the [run guide](doc/usage.md) before writing a launch command.
The registry supports 2.3 and 2.5; new work defaults to 2.5.
Check each entrypoint's narrower support before using another key.

## Owners and invariants

- `data.whole_clip` owns saved input validation and exact reconstruction.
  Preserve capture, noise, prompt context, fps, geometry, conditioning and sigmas.
- `core.session` owns BF16 dtype, prompt selection and model/decoder lifetime.
  It defines no sampling schedule or window geometry. Callers own their schedules.
- `core.artifacts` owns output roots and attributable run directories.
- `score.hooks` accepts only complete native D0 provenance and executable binary
  mask families. Export, parity, evaluation and timing have no cross-task bypass.
- Calibration actors cannot appear in held-out validation, including other views.
  Pin manifest content, source/noise/output hashes and actual context bytes.
- `compact_faithful` restores original execution widths from compact stored
  parameters; distinguish storage savings from reduced GEMM work in `compact`.
- `core.ltx_adapter` owns private upstream access. Avatar consumers also use model,
  prompt and media utilities: check their imports before changing shared interfaces.

## Documentation contract

Follow the [avatar reference](../onestep_avatar/doc/README.md#documentation-rules).
A production module over 100 physical lines requires
`doc/<relative-source-path>.md`. Smaller modules explain their logic in the
header. Keep existing small-module designs as supplemental references.
Empty package markers and tests need no design page.

Use Objective, Data flow, Organization logic, Invariants, Gotchas and Tests.
Entry scripts add Usage after Objective: launch owner, environment, complete
command, main arguments, output files and matching run-guide link.
Give worked calculations and failure meanings. Write in simplified technical
English, about 80% to ASD-STE100. Use small left-to-right Mermaid diagrams
with the [shared legend](doc/core_algorithm.md#diagram-legend); render and inspect them.

Update designs or headers with changes to objective, data flow, organization
or invariants. Describe current behavior only. Keep history in saved records.
The source inventory must match existing files and physical line counts.
Do not add an avatar queue requirement to an unrelated standalone prune CLI.

## Report and wording contract

Read the [report and wording guide](doc/reporting.md) before writing reports, captions or comparison labels.
It applies the workspace report structure and the `onestep_avatar` wording and media rules.
Use simplified technical English, about 80% ASD-STE100, with one defined term per concept.
Keep instructions to 20 words, descriptions to 25 words and paragraphs to six sentences, where practical.
Keep articles, use active voice and avoid stacks of more than three nouns.

Organize reports by question. Open with the problem, Terms, shared setup and numbered findings.
Use plain question headings of 15 words or fewer. Each comparison changes one factor.
Give its Short answer first, then setup, variants, coverage, results, figures and synchronized media.
Use no experiment-summary tables. Save scientific figures as PNG, PDF and plotted-data JSON.
Prove matched source, context and noise. Define metrics and uncertainty, including their limits.
Put comparison roles and changed values on videos; keep common settings in captions.

Model work and decoding stay in this package. Report code under `expr/` reads saved artifacts only.
Keep prose in narrative config and layout in the report generator. Rebuild and validate every report change.
Inspect wide and narrow rendered layouts. Check labels, synchronization and aspect ratios.

## Verification

Run focused CPU tests and Ruff. Changes to native input construction or scoring
require fresh saved-baseline forward checks. Export changes require functional-mask
versus export parity on a held-out capture, using the documented tolerance.
Inspect synchronized VAE comparisons and measure benchmark drift before claiming
quality or speed. Fresh experiment results belong under ignored `expr/` paths.

The CPU gate runs the exact suite with `CUDA_VISIBLE_DEVICES=`. Collection
marks only tests already labeled `gpu` skipped when CUDA is unavailable or
hidden. With real CUDA those tests still run. Metadata and tiny-model CPU
fixtures retain their own checks; a marker does not disable their imports.

## Joint V3 selection

`learn.ddp_owner` owns DDP runtime, native acceptance, saved inputs, monitors and the update loop.
`learn.joint` supplies V3 family hooks; `learn.readmission` owns the policy and `learn.joint_training` owns numerical state.
`data.joint_study` freezes the study. Its code identity changes after any package source edit.
`checks.joint_preflight`, `checks.joint_export`, `checks.joint_faults`, `checks.joint_review` and
`checks.joint_split_parity` retain independent acceptance receipts.
`score.export_joint` merges adapters. `score.export_joint_split` preserves trained adapter arithmetic.
`evaluate.joint_final` and `evaluate.joint_media` evaluate fixed completed checkpoints without optimizer updates.
The CPU `learn.receipt_logger` follows FFN and V3 receipts.
Live studies retain isolated snapshots. Do not restart, resume or alter them during package maintenance.

## Learned FFN masks

`data.ffn_study` freezes a separate versioned study from verified existing sources.
`learn.ffn` trains individual binary channel masks alongside existing head gates, with one DDP gradient average.
Its aligned preset retains 14,848 channels per video FFN. It uses rank-16 FFN up/down adapters.
The joint arm preserves rank-128 attention V/output adapters and the established sigmoid/binary head recipe.
Selection ends at 1000; fixed-mask adapter recovery ends at 1300.
Tests use physical GPUs 4–6. Formal acceptance and training use physical GPUs 0–3.
The three-rank test receipt cannot replace exact four-rank acceptance at formal startup.
Use `score.export_learned_ffn` for the new combined recovered family; historical head-only readers remain strict.
Native parity, decoded quality and timing retain independent evidence gates.

## Learned heads

`learn/` owns FP32 scalar gates, V/output adapters, official velocity loss, immutable teacher references and bounded model phases.
`data.learned_head_study` freezes disjoint actors, complete clips, paired draws and actual source/context/component bytes.
`data.foreground` orchestrates public corpus crop and mask owners; reviewed GT tracks supply semantic regions.
Recovered exports use `whole_clip_d0_learned_heads_v1` and candidate family `width_recovered`.
Validate all consumed draws, original head IDs, optimizer transfer and merged tensor inventory.
Preserve full original Q/K normalization. Retained pruning gates are exactly one in compact recovery.
Keep structural, transfer, fusion, merged reload, decoded quality and speed evidence separate.
Standalone prune phases keep their direct launch ownership and explicit dev transformer override.
New learned-head training uses `configs/learned_heads_dev10_ddp_gt_uniform_f18_rank128.json` by default.
Use full model replicas, native block gradient checkpointing and one F18 draw per DDP rank.
Four ranks consume four draws. Preserve raw-logit loss and the 100-update weight ramp.
The current preset uses gate LR 0.01 and sparsity coefficient 0.01, with effective weight 0.001 to 0.01.
Its ten-level grid includes all five official 30-step dev sigmas in [0.9, 0.95].
W&B uses one CPU writer per arm. Rank sigma/memory charts have one line per rank;
total loss and validation errors are averaged scalars. Metric tabs omit `learned_joint/`.
Do not restart a user-stopped run while updating code or documentation.
