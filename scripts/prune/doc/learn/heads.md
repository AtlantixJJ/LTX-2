# Learned head phases

## Objective

Train attention-head choices and low-rank recovery weights against saved capture velocity targets.
The phase owner keeps one model resident and publishes resumable state plus matched monitoring evidence.
The [glossary](../glossary.md) defines gates, original IDs, receipts and DDP.

## Usage

For a fresh default GT study, use the [full-run recipe](../usage.md#uniform-f18-ddp-training).
It launches `learn.run_full`, which checks acceptance and runs the phases in order.
A direct bounded diagnostic from `LTX-2` in `ltx` is:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.prune.learn.heads \
  --study <frozen_study.json> --phase feasibility --gpu-id <configured_free_gpu>
```

`--phase` selects the operation; `--arm`, `--replicate-seed` and `--removed-heads` select a frozen experiment choice.
`--max-updates` bounds a disposable smoke run. Use DDP launch settings from the run guide for optimizer phases.
Outputs include checkpoints, immutable update/phase receipts, monitor tensors, input identities and gate traces under the study root.

## Data flow

The frozen study binds components, actor splits and all draws.
`data.whole_clip.load_case` verifies and deserializes each capture/noise buffer once.
`data.training_windows` crops optimizer inputs when configured; monitoring uses complete sources.
`core.ltx_adapter.velocity_forward` supplies native velocity for the [official loss](../core_algorithm.md#learned-velocity-and-binary-selection).
The unchanged native model is an evaluation reference, not the velocity-loss target.

## Organization logic

| Phase or recovery arm | Inputs | Decision or calculation | Saved outputs |
|---|---|---|---|
| `feasibility` | Frozen components, both full-source lengths, all configured sigmas | Check velocity conversion, all-one gates, zero adapters, initialized gates and bounded backwards | Native predictions, gradients, headroom and feasibility receipt |
| `execution_probes` | Frozen source and declared test layouts | Compare dense, functional and compact execution; bracket timing | Layout-specific predictions, export differences and timing receipts |
| `energy_calibration` | Frozen learning subset and sampled query tokens | Rank heads by `score.estimators.projected_energy` | Pinned `calibration/ranking.json` with actual coverage |
| `teacher_references` | Selected split, exact inputs and unchanged model | Produce immutable teacher predictions for a teacher-reference study | Versioned bounded teacher cache |
| `selection`, soft part | Paired draws, initialized head logits and adapters | Train sigmoid gates with the configured sparsity penalty | Updated FP32 logits/adapters, optimizer state, trace and input chain |
| `selection`, binary part | Soft checkpoint and allowed block counts | Choose an exact legal binary budget each update; use a sigmoid surrogate gradient | Binary masks in original IDs, adapters and resumable state |
| `compact_recovery` | Fixed selected IDs, compact base and selection adapters | Slice adapters and Adam moments by original IDs; train adapters with retained gate amplitude one | Fixed-width recovered checkpoint |
| `compact_recovery`, arm `learned_sequential` | Selected IDs and that arm's declared recovery budget | Spend the extra recovery budget with fixed choices | Separately attributable continuation state |
| `compact_recovery`, a configured fresh arm | Same selected IDs and a fresh source adapter initialization | Discard selection adapters/moments, then slice and recover | Independent fresh-recovery control |

`learn.protocol.phase_budget` derives global boundaries from the frozen config.
The [preset section](../usage.md#preset-values) owns update counts, ranks, learning rates and monitor settings.
Soft-only, random and energy arms use their declared selection paths; they share the same input and adapter-state contracts.

In default `monitor_reference=gt_capture` mode, monitors compare native predictions with GT capture without a teacher cache.
`learn.run_full` runs matching DDP acceptance, selection, compact-base export, compact recovery and final merge.
An energy arm requires an already prepared ranking record.
Foreground or regional arms require reviewed GT region preparation; a study that defers those inputs cannot run them.

Each DDP rank consumes its own frozen crop and computes one complete loss.
DDP averages parameter gradients once. A decision is fixed for the whole update; each forward gets a fresh surrogate graph.
Monitoring gathers unique full-source cases across ranks, then rank zero publishes one JSON result.

When logits train, `gate_trace/NNNN.pt` records post-step logits, total averaged pre-clip gradients and analytic sparsity gradients.
Subtract the sparsity row from the total-gradient row to obtain the velocity component.
The shared preview owner saves the actual first rank-zero training forward and preserves RNG during decoding.
See [preview and logging settings](../usage.md#training-visualization).

## Invariants

- Freeze backbone weights so recovery changes only the declared adapters and selection state.
- Keep original head IDs through slicing, checkpointing and optimizer transfer; position and moment identity depend on them.
- Preserve source/noise pins and consumed-window hashes so resume replays the same draws.
- Restore complete optimizer and rank RNG state before another update; use a new frozen study after a loss change.

## Gotchas

The initialized sigmoid of a positive logit is less than one; compare it separately from all-one-gate parity.
A timeout saves completed progress; the full owner resumes only its bounded timeout segments with increased progress.
Teacher-reference studies require their immutable reference evidence; GT mode follows its own configured prerequisites.
Use [verification](../VALIDATION.md) for structural, native, decoded and speed qualification.

## Tests

`tests/test_learned_gates.py` compares the legal selector with brute force and checks binary surrogate gradients.
`tests/test_head_losses.py` checks the official loss and analytic sparsity gradient.
`tests/test_head_learning_state.py` checks serialized resume and exact optimizer state.

`tests/test_head_adapters.py` checks original-ID parameter and Adam-moment transfer.
`tests/test_ddp_training.py` compares distributed gradients with serial four-draw accumulation.
`tests/test_gt_reference.py` and `tests/test_training_windows.py` check full-source GT monitoring and crop identities.
`tests/test_bench_learned_heads.py` checks probe budgets, nonfinite timing refusal and drift decisions.
