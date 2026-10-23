# Pruning architecture and change impact

The package reconstructs saved video inputs, chooses retained transformer units and measures exported execution.
Shared owners keep input calculations, model lifetime and distributed update order consistent across methods.
Read the [glossary](glossary.md), [overview](overview.md) and [file map](README.md) for names and entrypoints.

## Ownership and dependencies

| Area | Shared responsibility |
|---|---|
| `core/` | Registry, VAE factors, model lifetime, numerical policy, provenance and publication |
| `data/` | Saved-input reconstruction, frozen studies, actor splits, draws and crop coordinates |
| `score/` | Observations, scores, masks, coupled width/depth export and bounded checkpoint writing |
| `learn/` | Gate policies, adapters, optimizer state, distributed execution, monitoring and logging |
| `evaluate/` | Baseline production, frozen predictions, decoding, metrics and synchronized timing |
| `checks/` | Native acceptance, structural validation, export parity and diagnostics |

`evaluate.whole_clip_baseline` produces saved rollout evidence.
`data.whole_clip` owns native-input validation and reconstruction.
`core.session.open_model` runs preflight and resolves text context; `core.session.open_session` forwards parsed arguments to it.
`core.session.open_frozen` uses explicitly pinned learned-study components.
Sessions own transformer and VAE lifetime. Callers supply schedules, geometry and selected views.

Imports follow core → data → score → learn → evaluate/checks, with explicit exceptions in `tests/test_import_boundaries.py`.
Recovered input validation imports export/check validators lazily to avoid cycles.
Private upstream access stays in `core.ltx_adapter`; public upstream classes can be imported normally.
Prune uses public avatar corpus/model helpers. Avatar consumers also use prune registry, session, prompt and media utilities.
Check those consumers before changing a shared interface.

## Training and saved state

The head-only route uses `learn.heads` with `learn.ddp`.
The shared `learn.ddp_owner` serves FFN, joint clipped-selection and compact-recovery families.
Family hooks choose numerical state and policy; they do not create a second update loop.
The [DDP contract](learn/ddp_owner.md) fixes update, review, checkpoint and monitor order.

`data.study_io.code_identity` binds package, transformer, pipeline-utility and shared avatar source bytes.
`core.records` publishes immutable evidence and separately replaces mutable status records.
`learn.checkpoints` binds state payloads, rank RNG, draw cursors and input chains.
Any production source edit requires a fresh frozen identity for new training; saved studies use their isolated snapshots.

## Export and reports

`score.export_pruned` owns coupled width surgery; `score.export_depth` owns block renaming.
`score.safetensors_stream` holds one transformed tensor and a bounded copy buffer, keeping full checkpoint payloads off the host heap.
The [width/runtime contract](score/export_pruned.md) defines metadata consumed by ltx-core.
Learned exporters add adapters and versioned envelopes without changing the shared native-input owner.

Model execution and decoding live here. Report code under `expr/` reads saved artifacts only, as defined in [reporting](reporting.md).
The avatar queue owns its restricted `prune_baseline` wrapper and its additional completion records.
Other pruning CLIs keep direct launch ownership; see the [run guide](usage.md).

## Change-impact map

| If you change… | Update… | Run… |
|---|---|---|
| `data.whole_clip` reconstruction or validation | Its docstring, [input contract](data/whole_clip.md), [input algorithm](core_algorithm.md#input-reconstruction) | `tests/test_input_loading.py`, `tests/test_whole_clip_d0.py`, `tests/test_whole_clip_baseline.py`, `tests/test_training_windows.py`; fresh saved-baseline forwards after numerical changes |
| Token sampling or score meaning | Source docstrings, [scoring algorithm](core_algorithm.md#training-free-scores-and-masks) | `tests/test_token_sampling.py`, `tests/test_estimators.py`, `tests/test_projected_energy.py`, `tests/test_hooks.py`; native calibration replay |
| Width surgery or transformer metadata | Source docstrings, [runtime contract](score/export_pruned.md) | `tests/test_export_pruned.py`, `tests/test_safetensors_stream.py`, `tests/test_joint_exports.py`; native export parity |
| Block removal or mappings | Source docstrings, [depth contract](score/export_depth.md) | `tests/test_export_depth.py`; native depth parity |
| Head phases, gates or loss | Source docstrings, [head phases](learn/heads.md), [algorithm](core_algorithm.md) | `tests/test_learned_gates.py`, `tests/test_head_losses.py`, `tests/test_head_learning_state.py`, `tests/test_ddp_training.py`; matching native acceptance |
| Joint event rules or state | Source docstrings, [selection contract](learn/readmission.md), [event algorithm](core_algorithm.md#joint-clipped-selection-with-return) | `tests/test_readmission.py`, `tests/test_joint_monitor_gradients.py`, `tests/test_ddp_owner_families.py`; matching native acceptance |
| DDP loop or family hooks | Source docstrings, [DDP contract](learn/ddp_owner.md) | `tests/test_ddp_owner_loop.py`, `tests/test_ddp_owner_families.py`, `tests/test_ffn_learning.py`, `tests/test_readmission.py`, `tests/test_import_boundaries.py`; matched native gradient/state replay |
| Frozen device policy | Config before freezing, [launch guide](usage.md#launch-and-output-rules) | `tests/test_device_policy.py`; new native acceptance for the selected ranks |
| Session or adapter interface | Source docstring, callers in prune and avatar, relevant shared contract | `tests/test_session.py`, `tests/test_ltx_adapter.py`, affected avatar-interface tests |
| Reports or media | [Reporting rules](reporting.md), narrative config and generator | Saved-artifact rebuild and validation; inspect wide/narrow layouts and synchronized videos |
| Docs, source inventory or entrypoints | [File map](README.md), relevant shared guide and docstring | `tests/test_doc_contract.py`; render and inspect changed diagrams |

The [verification guide](VALIDATION.md) defines what each native check must establish.
Maintenance runs CPU checks; model-facing acceptance runs only within an authorized experiment.
