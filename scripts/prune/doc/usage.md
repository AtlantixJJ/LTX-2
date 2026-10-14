# Pruning run guide

## Launch and output rules

Run from the `LTX-2` root in the `ltx` conda environment.
Run entry scripts as `python -m scripts.prune.<area>.<module>`.
Use LTX-2.5 for new work.

The commands below describe the standalone pruning workflow.
Humans can run these direct CLIs.
Agents use the avatar queue for the restricted native baseline reproduction.
That route has its own [recipe](#queued-baseline-reproduction).
Other prune CLIs have no queue wrapper defined here.

Replace the two view paths with existing corpus views.
Choose different actors for calibration and held-out validation.
All selected views in one producer run must have the same latent length.
Choose a fresh run root and a free GPU after checking occupancy.

```bash
cd LTX-2
conda activate ltx
nvidia-smi

PRUNE_GPU=0
PRUNE_CALIB=/abs/corpus/Part_1/0008_01/views/view00_cam51
PRUNE_HELDOUT=/abs/corpus/Part_1/0264_01/views/view00_cam51
PRUNE_RUN=../expr/refiner_prune/2.5/prune_doc_example
```

These view names illustrate paths. They do not declare an actor split.
Check saved experiment membership before selecting real actors.

Keep prompt, seed, source bundles, epsilon, fps, geometry, schedule and numerical policy fixed.
Store weights, tensors, media and logs under ignored `expr/` paths.
Some writers can replace files. Use fresh paths even when an entrypoint does not enforce this.

## Preflight and prompt cache

```bash
python -m scripts.prune.core.preflight \
  --model 2.5 --gpu-id "$PRUNE_GPU" --min-free-gb 44 --dump-caps
python -m scripts.prune.data.prompt_cache \
  --model 2.5 --gpu-id "$PRUNE_GPU" --verify
```

Preflight reports decimal GB, while the scorer and benchmark require 44 GiB.
The second command verifies `DEFAULT_PROMPT`.
The workflow below explicitly uses empty text, which has a different cache.

A direct baseline builds its selected context if missing.
The queued wrapper requires that selected cache to exist before the run.
Use the public API to prepare and verify empty text:

```bash
python - "$PRUNE_GPU" <<'PY'
import sys
import torch
from scripts.prune.core import preflight, session
from scripts.prune.data import prompt_cache

gpu_id = int(sys.argv[1])
model = preflight.check("2.5", gpu_id=gpu_id)
device = torch.device(f"cuda:{gpu_id}")
prompt_cache.get_or_build(model, "", session.DTYPE, device)
report = prompt_cache.verify(model, "", session.DTYPE, device)
print(report)
if not report["bit_exact"]:
    raise SystemExit("empty-text context verification failed")
PY
```

Verification re-encodes and replaces the cache. Preserve the old cache before investigating a mismatch.
See [preflight](core/preflight.md#usage) and [prompt cache](data/prompt_cache.md#usage).

## Save the baseline

Save calibration and held-out inputs in one native manifest for mask and parity checks:

```bash
python -m scripts.prune.evaluate.whole_clip_baseline \
  --model 2.5 --variant distilled --objective white \
  --view "$PRUNE_CALIB" --view "$PRUNE_HELDOUT" \
  --sigmas 0.725 0.909375 1.0 --context-latent-frames 8 \
  --prompt "" --seed 42 --gpu-id "$PRUNE_GPU" \
  --output "$PRUNE_RUN/baseline"
```

The exact direct schedules are `[0.725,0]`, `[0.909375,0]` and `[1,0]`.
Defaults are CFG 1, STG 0 and rescale 0.
The output contains a manifest, saved epsilon, latent outputs and decoded capture/D0/D1 video.

`--raw-only --arms d0` saves only raw D0 evidence.
It cannot support a decoded comparison without separate media.
Use a separate run directory for each seed consumed by native calibration.
See [producer usage](evaluate/whole_clip_baseline.md#usage).

## Calibration and functional ablation

Calibrate only on the calibration actor:

```bash
python -m scripts.prune.score.whole_clip_d0_scores \
  --baseline "$PRUNE_RUN/baseline" --view "$PRUNE_CALIB" \
  --sigmas 0.725 0.909375 1.0 \
  --head-fraction 0.10 --ffn-fraction 0.10 \
  --sampler stride --spatial-stride 16 --gpu-id "$PRUNE_GPU" \
  --output "$PRUNE_RUN/mask.json"

python -m scripts.prune.evaluate.whole_clip_d0 \
  --baseline "$PRUNE_RUN/baseline" --functional-mask "$PRUNE_RUN/mask.json" \
  --view "$PRUNE_HELDOUT" --sigmas 0.725 0.909375 1.0 \
  --gpu-id "$PRUNE_GPU" --output "$PRUNE_RUN/ablation.json"
```

The scorer first reproduces every calibration baseline.
The functional analyzer checks the unmasked held-out baseline, then measures head-only, FFN-only and combined masks.
These are numerical ablations. They do not establish export fidelity or speed.
See [scorer](score/whole_clip_d0_scores.md#usage) and [analyzer](evaluate/whole_clip_d0.md#usage).

## Width export and parity

```bash
python -m scripts.prune.score.export_pruned \
  --model 2.5 --masks "$PRUNE_RUN/mask.json" --mode compact_faithful \
  --output "$PRUNE_RUN/candidate.safetensors"

python -m scripts.prune.checks.export_parity \
  --baseline "$PRUNE_RUN/baseline" --masks "$PRUNE_RUN/mask.json" \
  --exported-checkpoint "$PRUNE_RUN/candidate.safetensors" \
  --view "$PRUNE_HELDOUT" --sigmas 0.725 0.909375 1.0 \
  --gpu-id "$PRUNE_GPU" --max-abs 0.02
```

Width export performs CPU tensor surgery, but its current CLI requires CUDA preflight.
`masked_full` is the default full-width control.
`compact_faithful` reduces storage and restores original execution geometry.
Use `compact` to test supported reduced execution widths with separate parity.
`sparse` selects active attention heads while keeping full-width projections.

Parity writes an indexed `d0-export-parity` run and prints its result path.
Failure returns a nonzero exit status.
Use an all-ones native artifact for the no-prune width control.
See [export](score/export_pruned.md#usage) and [parity](checks/export_parity.md#usage).

## Depth export

The CPU depth exporter can make a no-prune architecture control:

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.prune.score.export_depth \
  --baseline "$PRUNE_RUN/baseline" --calibration-views "$PRUNE_CALIB" \
  --sigmas 0.725 0.909375 1.0 --remove-blocks \
  --artifact "$PRUNE_RUN/depth_control.json" \
  --output "$PRUNE_RUN/depth_control.safetensors" \
  --purpose no_prune_control --quality-status not_evaluated

python -m scripts.prune.checks.export_parity \
  --baseline "$PRUNE_RUN/baseline" --depth-artifact "$PRUNE_RUN/depth_control.json" \
  --exported-checkpoint "$PRUNE_RUN/depth_control.safetensors" \
  --view "$PRUNE_HELDOUT" --sigmas 0.725 0.909375 1.0 \
  --gpu-id "$PRUNE_GPU" --max-abs 0.02
```

A removal candidate must supply an explicitly selected `--remove-blocks` list and a suitable purpose.
Its source order and original/compact mappings remain pinned.
No selection method is implied by this control.
See [depth export](score/export_depth.md#usage).

## Held-out quality and timing

Generate the candidate with the same calibration and held-out views as the baseline.
The distribution validator requires the pinned calibration rows in the selected baseline.
The analyzer reports every row; use the held-out rows for held-out quality claims.

```bash
python -m scripts.prune.evaluate.whole_clip_baseline \
  --model 2.5 --variant distilled --objective white \
  --view "$PRUNE_CALIB" --view "$PRUNE_HELDOUT" \
  --sigmas 0.725 0.909375 1.0 --context-latent-frames 8 \
  --prompt "" --seed 42 --gpu-id "$PRUNE_GPU" \
  --transformer "$PRUNE_RUN/candidate.safetensors" \
  --output "$PRUNE_RUN/candidate"

python -m scripts.prune.evaluate.whole_clip_d0 \
  --baseline "$PRUNE_RUN/baseline" --candidate "$PRUNE_RUN/candidate" \
  --output "$PRUNE_RUN/comparison"

python -m scripts.prune.evaluate.bench_whole_clip_d0 \
  --baseline "$PRUNE_RUN/baseline" --candidate "$PRUNE_RUN/candidate" \
  --view "$PRUNE_HELDOUT" --sigma 0.909375 --gpu-id "$PRUNE_GPU" \
  --warmup 2 --repeats 5 --order ABA --output "$PRUNE_RUN/benchmark_aba.json"
```

The paired analyzer checks actual saved epsilon equality.
It writes metrics and synchronized capture VAE, baseline D0 VAE and candidate D0 VAE panels.
Inspect those videos for identity, action and motion.
The numerical metrics support that judgment; they do not replace it.

Repeat timing with `--order BAB` and a new output file when gains approach drift.
Keep device, geometry and numerical policy fixed.
See [analyzer](evaluate/whole_clip_d0.md#usage),
[benchmark](evaluate/bench_whole_clip_d0.md#usage) and [acceptance](VALIDATION.md).

## Queued baseline reproduction

Agents use the existing avatar queue for the native `prune_baseline` experiment.
Start with its [run guide](../../onestep_avatar/configs/README.md#pruning-baseline-prune_baseline),
[protocol design](../../onestep_avatar/doc/experiments/prune_baseline.md)
and [queue usage](../../onestep_avatar/doc/usage.md).

The scientific spec permits one absolute view, one seed, white distilled unguided raw output and context eight.
It keeps the requested sigmas and supports D0, optionally D1.
The selected prompt cache must exist first.

The experiment supports a CPU spec check:

```bash
conda run -n ltx python -m scripts.onestep_avatar.experiments.prune_baseline \
  --spec /abs/prune_baseline_spec.json --output /abs/fresh_prune_job --dry-run
```

That check parses the spec without loading or hashing model components.
Validate the prepared job list with the queue's `--dry-run`.
Start the queue owner with `setsid`, following its documented process-ledger command.
The prune producer's original files stay under `<job-output>/baseline/`.
Job-root manifests and acceptance records belong to the wrapper.

For drift diagnosis, use the existing
[pruning reproduction controls](../../onestep_avatar/configs/README.md#pruning-reproduction-controls-prune_reproduction).
Do not substitute new sigmas or change scientific flags to make a comparison pass.

## CPU verification and stopping

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest scripts/prune/tests -q -m 'not gpu'
ruff check --select F,I scripts/prune
```

If Ruff is unavailable in `ltx`, follow the repository's dev-tool setup.
Run the tests with the `ltx` interpreter.

Direct CLIs run in the foreground. Interrupt your own command to stop it.
Keep partial outputs for diagnosis and use a new directory for another producer attempt.
For managed jobs, use the avatar queue's documented owned-process stop path.

## MobileWan capture audit

Run the complete source audit on CPU before model scoring:

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.prune.data.head_score_dataset \
  --membership ../expr/onestep_avatar/runs/d0_bidirectional_sigma0.9/membership_all.json \
  --output ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011
```

The audit verifies all supplied rows and chooses three held-out actors deterministically.
It writes source, coverage and failure records. It does not launch inference.
See [audit design](data/head_score_dataset.md) and [numerical helper](score/mobilewan_heads.md).
The native dataset runner and removal evaluator are not yet implemented.

The baseline producer also supports direct dev schedules with `--variant dev --dev-schedule direct`.
Use `--raw-only --arms d0` to load capture without a guide.
See [direct dev design](evaluate/whole_clip_baseline.md#explicit-direct-dev-and-capture-only-d0).
The restricted agent queue wrapper supports unmodified direct dev D0 at sigma <=0.95.

## MobileWan native check

After the queued dev baseline completes, check its stock primitive, score hook and all-one parity:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m scripts.prune.checks.mobilewan_native \
  --baseline ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011/validation/native_18/baseline \
  --gpu-id 4 \
  --output ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011/validation/native_check_18.json
```

Repeat for the 28-frame latent baseline on a free GPU.
The checker executes three independent native calls per sigma, using the same saved inputs.
See [checker design](checks/mobilewan_native.md).

## MobileWan dataset worker

After both native checks pass, run one resident standalone prune worker per free GPU.
Use non-overlapping deterministic shards of the complete train membership:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m scripts.prune.score.mobilewan_dataset \
  --study ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011/configs/study.json \
  --split train --shard 0 --shards 3 --gpu-id 4 --wall-seconds 86400
```

For persistent work, use `setsid`, an external timeout and a per-worker log.
Set distinct shard indices. Recheck free GPUs and live process identities before any restart.
Use `--max-views 1` for the first native worker smoke check.
The worker saves each sigma independently and reuses complete matching observations on resume.
Held-out workers require frozen train-only masks.
See [worker design](score/mobilewan_dataset.md).

## MobileWan aggregation

After all train views complete, freeze their pooled train-only masks:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' python -m scripts.prune.score.mobilewan_aggregate \
  --study ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011/configs/study.json --freeze-train
```

After held-out scoring completes, omit `--freeze-train` to publish final statistics.
Use `--verify` to recompute the saved final result and validate its confidence intervals from all raw records.
Missing membership observations fail before final publication.
See [aggregate design](score/mobilewan_aggregate.md).

## MobileWan removal evaluation

After complete train scoring and mask freezing, evaluate all three preselected held-out cases:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m scripts.prune.evaluate.mobilewan_removal \
  --study ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011/configs/study.json \
  --gpu-id 4
```

The evaluator saves full GT and prediction tensors, matched identities and generated-frame MSE.
It validates all-one equality before each sigma's removal calls.
Complete saved sigmas resume with tensor and metric verification.
It does not decode or generate the final report.
See [evaluation design](evaluate/mobilewan_removal.md).

## MobileWan comparison media

After complete removal evaluation, decode saved evidence and render all 24 synchronized comparisons:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m scripts.prune.evaluate.mobilewan_media \
  --study ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011/configs/study.json \
  --gpu-id 4
```

Each video has six panels in two rows by three columns, with the matched recorded white-matte capture.
Full source and individual decoded videos remain available.
Report code reads these saved artifacts and must not run this model-facing stage.
See [media design](evaluate/mobilewan_media.md).

## Learned head phases

New training uses the **uniform-F18 rank128 DDP** preset described below.
The full-source FSDP and older phase commands later in this guide are reproduction paths.
Code or documentation updates do not restart user-stopped training.

## Uniform F18 DDP training

Use `configs/learned_heads_dev7_ddp_gt_uniform_f18_rank128.json` for subsequent learned-head runs.
It is also the default config when `data.learned_head_study --config` is omitted.
Four ranks on GPUs 0–3 each store a full BF16 model and consume one independent F18 draw per update.
Native non-reentrant block gradient checkpointing stays enabled. Adapters and gates are FP32.
DDP averages their gradients once; no manual second gradient average runs.

Keep F18 sources intact. Crop longer sources to one continuous 18-latent span using the draw's noise seed.
Capture and saved epsilon share bounds. The first latent of the window is clean c0; positions start locally at zero.
This yields 137 decoder frames at the original 1024×1024 geometry and fps.
It does not re-encode raw RGB. Full-source manifests and per-draw crop hashes both remain saved.
Calibration, export parity and GT quality evaluation still use full sources.

Keep rank/alpha 128, adapter LR 0.0001, gate LR 0.1 and `mean(relu(alpha+3))`.
Use `lambda(t)=0.1*(0.1+0.9*min(t/100,1))`: weight 0.0109 at update one and 0.1 from update 100.
Keep seven sigmas, the cycle `[0,6,1,5,2,4,3]` and 700-soft/300-binary/500-compact updates.
Four draws per update cover each sigma four times in seven updates. Draw cursors and checkpoint records use this batch size.
Use a new empty output directory and freeze the current plan/code/config before any authorized launch.

```bash
conda activate ltx
CUDA_VISIBLE_DEVICES= CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.prune.data.learned_head_study \
  --config scripts/prune/configs/learned_heads_dev7_ddp_gt_uniform_f18_rank128.json \
  --plan ../plans/2026-10-12-learned-head-pruning-experiment.md \
  --output ../expr/refiner_prune/2.5/learned_heads_ddp_uniform_f18_new

CUDA_VISIBLE_DEVICES= CUBLAS_WORKSPACE_CONFIG=:4096:8 setsid python -u -m scripts.prune.learn.run_full \
  --study ../expr/refiner_prune/2.5/learned_heads_ddp_uniform_f18_new/configs/frozen_study.json \
  --arms learned_joint \
  > ../expr/refiner_prune/2.5/learned_heads_ddp_uniform_f18_new/logs/full.log 2>&1 < /dev/null &
```

These are launch instructions for a future authorized run; the stopped study stays stopped.
The owner checks occupancy, runs matching `checks.ddp_native` acceptance, then starts formal DDP training.
Native acceptance covers crop semantics, full F18 and cropped F28 sources, gradients, memory and exact resume.
The older two-rank F18 memory result is supporting evidence, not a replacement for the new four-rank acceptance.
No teacher cache, fresh head scoring, speed test or avatar queue is required.

W&B logging is enabled in this preset. One CPU writer reads rank-zero published receipts.
Use one line-series chart per rank-varying value: `train/sigma`, `ranks/update_s` and the memory charts.
Four ranks give four lines labeled rank 0 through rank 3 in each chart.
`train/loss`, velocity MSE, sparse loss and `validation/trained_sigma_2_gt_latent_mse` each log a single mean.
Validation averages unique cases gathered from all ranks. The third monitored sigma uses index two.
Metric tabs are `train`, `gates`, `memory`, `ranks` and `validation`, without a `learned_joint/` prefix.
Chart refresh defaults to ten updates and includes every saved point; global scalar metrics log each update.
Different arms use separate W&B runs with identical metric names. See [logger design](learn/wandb_logger.md).

## Learned head reproduction commands

Run all commands from `LTX-2` in conda environment `ltx`.
The user-authorized device pool for this experiment is physical GPUs 0–3.
Check occupancy before loading a model. Keep one model process per GPU.
Preparation and checkpoint export use CPU only.
Set `export CUBLAS_WORKSPACE_CONFIG=:4096:8` before GPU commands.
For isolation, set `CUDA_VISIBLE_DEVICES=<physical GPU>` and use logical `--gpu-id 0`.
The corrected active revision is `_r2`; keep the original frozen study intact.
Prepare into a new empty output directory when code changes.

```bash
conda run --no-capture-output -n ltx python -m scripts.prune.data.learned_head_study \
  --config scripts/prune/configs/learned_heads_dev8.json \
  --plan ../plans/2026-10-12-learned-head-pruning-experiment.md \
  --output ../expr/refiner_prune/2.5/learned_heads_dev8_new
```

Preparation checks complete clips, source bytes, canonical crops and foreground replay.
It pins an immutable plan snapshot, source components, actual empty context, actor splits and paired draws.
`--seen-actors` supplies further audited exclusions.
`--region-tracks` supplies reviewed source-pixel face and hand boxes with pinned producer identity.
It does not guess the corpus keypoint layout.

Use the resulting `configs/frozen_study.json` for each following module.
The commands below use an example shell variable for that path.

```bash
STUDY_PATH=../expr/refiner_prune/2.5/learned_heads_dev8_20261012_r2/configs/frozen_study.json
conda run --no-capture-output -n ltx python -m scripts.prune.learn.heads --study "$STUDY_PATH" --phase feasibility --gpu-id 0
conda run --no-capture-output -n ltx python -m scripts.prune.learn.heads --study "$STUDY_PATH" --phase execution_probes --gpu-id 0
conda run --no-capture-output -n ltx python -m scripts.prune.learn.heads --study "$STUDY_PATH" --phase energy_calibration --gpu-id 0
conda run --no-capture-output -n ltx python -m scripts.prune.learn.heads --study "$STUDY_PATH" --phase teacher_references --split validation --gpu-id 0
conda run --no-capture-output -n ltx python -m scripts.prune.learn.heads --study "$STUDY_PATH" --phase selection --arm learned_joint --replicate-seed 42 --removed-heads 152 --gpu-id 0
conda run --no-capture-output -n ltx python -m scripts.prune.score.export_learned_heads --study "$STUDY_PATH" --arm learned_joint --replicate-seed 42 --stage compact_base
conda run --no-capture-output -n ltx python -m scripts.prune.learn.heads --study "$STUDY_PATH" --phase compact_recovery --arm learned_joint --replicate-seed 42 --removed-heads 152 --gpu-id 0
conda run --no-capture-output -n ltx python -m scripts.prune.score.export_learned_heads --study "$STUDY_PATH" --arm learned_joint --replicate-seed 42 --stage final_merged
conda run --no-capture-output -n ltx python -m scripts.prune.checks.learned_heads --study "$STUDY_PATH" --arm learned_joint --replicate-seed 42 --gpu-id 0
conda run --no-capture-output -n ltx python -m scripts.prune.evaluate.learned_heads --study "$STUDY_PATH" --split validation --arm learned_joint --replicate-seed 42 --gpu-id 0
conda run --no-capture-output -n ltx python -m scripts.prune.evaluate.learned_head_media --study "$STUDY_PATH" --split validation --arms energy learned_joint --replicate-seed 42 --gpu-id 0
```

Run the same selection/recovery sequence for `energy`, `random` and `learned_soft_only`.
Each paired seed uses the same frozen ordinary draws and adapter initialization.
Use a separate frozen study directory for each removal budget.
Training paths are arm/seed scoped; checkpoint identity rejects changing the budget in an existing run.
`learned_balanced` uses its separately frozen balanced draws.
`learned_person` requires canonical person coverage.
`learned_small_regions` additionally requires reviewed face and hand tracks.
`learned_no_sparsity` keeps the main schedule with zero sparsity coefficient.
`learned_sequential` keeps zero adapter delta during selection, then receives 1,500 recovery adapter updates.
Fresh controls run only `compact_recovery`: `learned_fresh`, `energy_fresh`, `energy_counts_fresh` and `dense_fresh`.
They use frozen primary masks and fresh full-source adapter initialization, with fresh optimizer state.

No speed training starts without both feasibility and execution-probe evidence.
`--quality-research` explicitly selects a quality pilot when timing has not established a speed saving.
`--max-updates N` bounds a smoke run and saves a resumable checkpoint.
`--branch-prefix PATH` requires the identical 700-soft prefix and records its source hash.
Resume checks code, config, components, consumed noise, optimizer state and original IDs.
Changed scientific settings need a new frozen study and output path.

For the user-directed loss amendment, stop the source workers at a checkpoint boundary.
Run the CPU [`data.resume_learned_heads`](data/resume_learned_heads.md#usage) owner to freeze the raw-logit loss in a new root.
It retains source update receipts, both optimizers and all rank RNG states. It does not reset the global update count.
The continuation uses `mean(relu(alpha+3))` and `lambda(t)=min(t/300,1)` during the remaining soft phase.
Binary refinement and compact recovery keep the sparsity penalty disabled.

Use `setsid` and an external wall-time bound for persistent model work:

```bash
setsid timeout --signal=TERM --kill-after=120s 12h \
  conda run --no-capture-output -n ltx python -m scripts.prune.learn.heads \
  --study "$STUDY_PATH" --phase selection --arm learned_joint \
  --replicate-seed 42 --removed-heads 152 --gpu-id 0 > ../expr/refiner_prune/2.5/learned_heads_dev8_20261012_r2/logs/selection.log 2>&1 < /dev/null &
```

Teacher records live under `teacher/references`. Training saves small adapter/gate checkpoints and immutable update receipts.
Exports, native checks, paired predictions, metrics and media retain separate paths.
A completed numeric evaluation is not a decoded quality pass.
Blind teacher generation and inference require `provenance/finalists_frozen.json` after validation selection.

Set the cuBLAS policy in the launcher environment before importing any native module:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Some upstream imports initialize CUDA while the module loads.
The CLI checks the existing policy and refuses to change it after CUDA initialization.
Use this export before every GPU command above, including checks and media.
For a persistent launch, `CUBLAS_WORKSPACE_CONFIG=:4096:8 setsid timeout ...` also passes the policy to its children.

## Four-GPU native memory test

The user-authorized four-device diagnostic uses all physical GPUs 0–3 in one process.
Freeze `configs/learned_heads_dev8_four_gpu.json` into a new directory before execution.
Run [checks.four_gpu_feasibility](checks/four_gpu_feasibility.md#usage) with the prior r2 study as the numerical reference.
The test preserves full clips and scientific settings; single-GPU timing stays a separate gate.

## Sustained four-GPU learning

A frozen study with `native_placement` automatically applies four-GPU block placement to learning and recovery.
Use `CUDA_VISIBLE_DEVICES=0,1,2,3` and `--gpu-id 0` for these phases.
Keep the same cuBLAS policy. Each twelve-hour segment saves its last completed update before bounded exit.
Resume with the identical command and frozen source. Do not edit production sources during a live study.
Single-GPU teacher, energy and speed phases can run independently before learning occupies all four devices.

## Four-rank FSDP with GT references

The frozen `learned_heads_dev8_fsdp_gt.json` preset uses four-rank FSDP and exactly two unique clips per update.
Run directly from LTX-2 in ltx. This path does not use the avatar queue.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc_per_node=4 -m scripts.prune.checks.fsdp_native --study "$STUDY_PATH"
CUDA_VISIBLE_DEVICES=0,1,2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc_per_node=4 -m scripts.prune.learn.heads --study "$STUDY_PATH" --phase selection --arm learned_joint --replicate-seed 42 --removed-heads 152 --gpu-id 0
```

The native acceptance check saves unsharded controls and checks both full clip lengths, gradients, resume and per-rank memory.
Training requires that acceptance. It does not require teacher references or speed-test receipts in GT mode.
Use the same direct command with `--phase compact_recovery` after exporting the compact base.
Resume an interrupted segment with the identical frozen study and command.
`data.reuse_head_scores` imports compatible historical raw observations for the available current learning views on CPU.
It declares missing coverage and pins all consumed evidence. No held-out actor enters the energy ranking.
GT numeric evaluation and decoded media use the saved capture references without a teacher cache.

For the complete paired pilot, start one detached owner for the frozen study:

```bash
conda activate ltx
STUDY_PATH="$PWD/../expr/refiner_prune/2.5/learned_heads_fsdp_gt_20261013_r6/configs/frozen_study.json"
RUN_ROOT="$(dirname "$(dirname "$STUDY_PATH")")"
CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 setsid python -u -m scripts.prune.learn.run_full --study "$STUDY_PATH" > "$RUN_ROOT/logs/full.log" 2>&1 < /dev/null &
```

The owner runs CPU score reuse, native acceptance, selection, compact export, recovery and final merge for each arm.
Defaults are `learned_joint energy`, seed 42 and 152 removed heads. Each arm has 1,000 selection and 500 recovery updates.
Compact export saves missing structural input manifests after selection, without model forwards or fresh calibration.
The owner resumes bounded segments only when saved progress increases; other failures stop the assigned run.
Read `monitoring/status.json` for live loss, gradients, memory and checkpoint paths. It updates every thirty seconds.
`monitoring/owner.pid` records the owner. SIGTERM to that PID stops only this run's children.
GT numeric validation and decoded review follow the completed pair as separate gates.

## Fresh seven-level GT training

Use `configs/learned_heads_dev7_fsdp_gt_logit_l1.json` for a fresh raw-logit-loss experiment.
It omits sigma 0.1 and resets every trainable tensor, optimizer and training counter.
The first 700 updates use soft gates, then 300 use binary gates with no sparsity penalty.
Read each exact sigma from the frozen study. Historical energy rankings retain their declared scoring scope.

```bash
conda activate ltx
CUDA_VISIBLE_DEVICES= CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.prune.data.learned_head_study --config scripts/prune/configs/learned_heads_dev7_fsdp_gt_logit_l1.json --plan ../plans/2026-10-12-learned-head-pruning-experiment.md --output ../expr/refiner_prune/2.5/learned_heads_fsdp_gt_20261014_r8_fresh_logit_l1_dev7
CUDA_VISIBLE_DEVICES=0,1,2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u -m scripts.prune.learn.run_full --study ../expr/refiner_prune/2.5/learned_heads_fsdp_gt_20261014_r8_fresh_logit_l1_dev7/configs/frozen_study.json
```

The supervisor can reuse explicitly attributed native execution acceptance for unchanged model inputs at retained noise levels.
Such reuse does not establish fresh-loss quality. Verify loss coefficients, gradients and four-rank memory on initial formal updates.

## Rank-128 seven-level training

Use `configs/learned_heads_dev7_fsdp_gt_logit_l1_rank128.json` for rank-128 V/output adapters with scaling one.
Keep full clips, the seven sigmas, loss warmup and soft/binary boundaries unchanged.
A rank change starts fresh tensors and optimizers; a rank-16 checkpoint cannot resume rank-128 tensors directly.
The same direct supervisor command accepts the separate frozen rank-128 study.
Verify actual rank-128 memory and gradients on the first full-length native updates.

```bash
conda activate ltx
CUDA_VISIBLE_DEVICES=0,1,2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u -m scripts.prune.learn.run_full --study ../expr/refiner_prune/2.5/learned_heads_fsdp_gt_20261014_r9_fresh_logit_l1_dev7_rank128/configs/frozen_study.json
```

For the lower sparsity coefficient and higher gate LR recipe, use
`configs/learned_heads_dev7_fsdp_gt_logit_l1_rank128_lambda01_gatelr01.json`.
It uses `lambda(t)=0.1*(0.1+0.9*min(t/100,1))` and gate LR 0.1 from the first update.
The effective sparsity weight starts at 0.01 at t=0, is 0.0109 on update one, and reaches 0.1 at update 100.
The loss remains `mean(relu(alpha+3))`; its sparsity gradient is zero at or below minus three.
Adapter LR stays 0.0001 with its 50-update warmup. Rank, inputs, draw cycle and phase boundaries stay unchanged.
Start a separate fresh study with new trainable tensors, empty optimizer states and a new W&B run.
The coefficient, ramp and gate LR change together, so this compares recipes rather than isolating the coefficient.

The confirmed 100-update-ramp run has a separate frozen study. Launch or resume its direct owner with:

```bash
conda activate ltx
CUDA_VISIBLE_DEVICES=0,1,2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u -m scripts.prune.learn.run_full --study ../expr/refiner_prune/2.5/learned_heads_fsdp_gt_20261014_r11_fresh_rank128_lambda01_gatelr01_ramp100/configs/frozen_study.json
```

Read its `monitoring/status.json` and `monitoring/fresh_initial_verification.json` for actual startup evidence.
