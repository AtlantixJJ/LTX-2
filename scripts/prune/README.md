# Whole-clip bidirectional pruning

This package calibrates and evaluates attention-head, FFN-channel and intact-block pruning.
The native task is one full-video bidirectional D0 forward at each exact `[sigma,0]` schedule.
Capture latent frame zero is clean. Later frames are generated.

Start with the [run guide](doc/usage.md) for complete commands.
Read the [design index](doc/README.md), [architecture](doc/ARCHITECTURE.md),
[algorithm](doc/core_algorithm.md), [verification](doc/VALIDATION.md)
and [known gaps](doc/known_gaps.md) for implementation contracts.
The [method review](doc/METHODS.md) supplies research context and proposed comparisons.

## Workflow

Run from `LTX-2` in the `ltx` conda environment.
Use LTX-2.5 for new work. Check `nvidia-smi` before model loading.
Keep fresh outputs under `../expr/refiner_prune/2.5/`.

1. [Save a baseline](doc/usage.md#save-the-baseline) with
   `python -m scripts.prune.evaluate.whole_clip_baseline`.
   Fix source bundles, prompt, epsilon, seed, fps, geometry and exact sigmas.
2. [Calibrate and ablate](doc/usage.md#calibration-and-functional-ablation) with
   `python -m scripts.prune.score.whole_clip_d0_scores` and
   `python -m scripts.prune.evaluate.whole_clip_d0 --functional-mask`.
   Calibration must reproduce the saved native baseline.
3. [Export width masks](doc/usage.md#width-export-and-parity) with
   `python -m scripts.prune.score.export_pruned`.
   `masked_full` is the default control. `compact_faithful` compresses storage while restoring original execution widths.
   `compact` reduces supported executed widths. `sparse` selects active heads with full-width projections.
4. [Export retained blocks](doc/usage.md#depth-export), when testing depth, with
   `python -m scripts.prune.score.export_depth`.
   This CPU exporter uses a separate native depth artifact.
5. [Check export parity](doc/usage.md#width-export-and-parity) with
   `python -m scripts.prune.checks.export_parity`.
   Compare the export with its in-memory intervention. Default maximum absolute tolerance is 0.02.
6. [Judge quality and cost](doc/usage.md#held-out-quality-and-timing) with
   `python -m scripts.prune.evaluate.whole_clip_d0` and
   `python -m scripts.prune.evaluate.bench_whole_clip_d0`.
   Inspect synchronized VAE panels and timing drift separately.

The shared [preflight and prompt-cache commands](doc/usage.md#preflight-and-prompt-cache)
are `python -m scripts.prune.core.preflight` and
`python -m scripts.prune.data.prompt_cache`.

## Native reproduction and evidence

Agents run the restricted raw baseline through the existing avatar `prune_baseline` queue experiment.
Humans can use the queue or direct producer.
See the [queued reproduction recipe](doc/usage.md#queued-baseline-reproduction).

The original producer manifest and tensors remain in `<job-output>/baseline/`.
The queue wrapper owns separate full component/software pins and completion records.
Prune retains its broader standalone interfaces.

Calibration actors cannot serve as held-out actors through another clip, Part, view or alias.
The paired baseline must retain pinned calibration rows.
The saved analyzer visits every row; judge and report held-out rows separately.

A smaller checkpoint does not establish faster execution.
Acceptance requires matched inputs, export parity, held-out numerical and decoded quality,
and measured same-device cost beyond drift.
See [working rules](CLAUDE.md).

## Focused checks

```bash
conda activate ltx
CUDA_VISIBLE_DEVICES='' python -m pytest scripts/prune/tests -q -m 'not gpu'
ruff check --select F,I scripts/prune
```

The [verification guide](doc/VALIDATION.md#cpu-gate) explains test coverage and native gates.
