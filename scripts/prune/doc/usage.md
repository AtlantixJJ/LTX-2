# Pruning run guide

Run the training-free workflow or freeze a fresh learned-gate study with the current source identity.
Use the actual reviewed specification for `--plan`. Existing frozen studies retain their isolated source trees.
Completed-study recipes are preserved in the [cleanup archive](../../../../expr/prune_package_cleanup_20261020/originals/doc/usage.md).
Those recipes require their original snapshots. They are not launch instructions for this tree.

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

## Joint clipped gates with re-admission

Use [joint study preparation](data/joint_study.md#usage) to freeze r3 inputs and all 6800 paired draws.
The [V3 owner](learn/joint.md#usage) tests on physical GPUs 4–6 and requires separate four-rank acceptance on 0–3.
The [optimizer and contribution checks](checks/joint_preflight.md#usage) and [export isolation](checks/joint_export.md#usage) retain independent receipts.
The [V3 exporter](score/export_joint.md#usage) requires terminal alignment and the declared recovery endpoint.
Historical V2 and learned-FFN studies remain separate.
A three-rank pilot uses `--phase pilot --testing --max-updates 50`; it never initializes A.
Final training uses `--phase formal` on GPUs 0–3 after preflight acceptance.
Do not start model work on an occupied device.

### Joint export fault isolation and split bundles

The [operation diagnostic](checks/joint_faults.md#usage) locates compact attention differences on identical inputs.
The [split bundle exporter](score/export_joint_split.md#usage) preserves the two adapter GEMMs instead of merging BF16 weights.
Load bundles through `score.export_joint_split.open_bundle`; the base checkpoint alone omits the adapters.
The [native matrix](checks/joint_split_parity.md#usage) checks every representation against the masked trained computation.
The native gate projection preserves its original output width. V, attention, output and FFN up paths remain compact.
The default split bundle preserves original FFN down reduction coordinates. Its actual cost must include that full-width GEMM.
Timing requires a separate qualified benchmark. A merged BF16 export failure remains visible.

## Completed joint V3 evaluation

Use [joint_final](evaluate/joint_final.md#usage) for fixed selection/final held-out predictions and final split-bundle parity/timing.
Use [joint_media](evaluate/joint_media.md#usage) for 36 synchronized held-out comparisons and fixed-region metrics.
Run these direct CLIs in ltx on free GPUs 4–6. They preserve historical studies and record separate execution identities.
Keep concurrent training on its independently accepted four-GPU group. Never share a GPU with another model worker.

The [historical endpoint reader](evaluate/historical_clip.md#usage) scores the saved head-only update-600 checkpoint without resuming training.
It first replays all 12 saved development predictions; held-out scoring requires parity at the unchanged 0.02 limit.

## Learned FFN masks

Read the [study preparation](data/ffn_study.md), [training entrypoint](learn/ffn.md)
and [export design](score/export_learned_ffn.md) for complete commands and arguments.
Run CPU preparation first. The explicit preset supports FFN-only and joint head/FFN arms.
The first formal preset learns joint masks with 1000 selection updates and 300 fixed-mask adapter updates.
It retains 14,848 FFN channels per block and 1384 video self-attention heads overall.

Use physical GPUs 4–6 for the three-rank disposable native check.
Use physical GPUs 0–3 for the formal owner, which first runs its own four-rank native acceptance.
The device lists and rank counts are enforced. Test acceptance does not substitute for formal acceptance.
Resume formal work with `--phase train`, using the unchanged frozen study and canonical checkpoints.
Checkpoints and monitors retain source, noise, crop and original-ID evidence.

Start the CPU W&B reader with `python -m scripts.prune.learn.receipt_logger --study <frozen_study.json>`.
It follows saved optimizer and monitor receipts and never opens a CUDA model.
Read `monitoring/status.json`, `monitoring/owner.json`, `monitoring/failure.json` and the launch log for local status.
Final compact export requires the exact completed recovery checkpoint.
`--probe` exports remain disposable and never qualify as final candidates.

## Learned head phases

Head-only training remains supported with the uniform-F18 DDP preset below.
Fresh study preparation requires `--evidence-list` with an explicit pinned list of prior inspected evidence.
Use completed immutable evidence; the preparer no longer scans experiment trees to choose actor exclusions.
Use the [head owner](learn/heads.md), [checkpoint checks](checks/learned_heads.md),
[paired evaluation](evaluate/learned_heads.md) and [media owner](evaluate/learned_head_media.md) for bounded phase commands.
Code maintenance never restarts a stopped study.

## Uniform F18 DDP training

Use `configs/learned_heads_dev10_ddp_gt_uniform_f18_rank128.json` for subsequent learned-head runs.
It is also the default config when `data.learned_head_study --config` is omitted.
Four ranks on GPUs 0–3 each store a full BF16 model and consume one independent F18 draw per update.
Native non-reentrant block gradient checkpointing stays enabled. Adapters and gates are FP32.
DDP averages their gradients once; no manual second gradient average runs.

Keep F18 sources intact. Crop longer sources to one continuous 18-latent span using the draw's noise seed.
Capture and saved epsilon share bounds. The first latent of the window is clean c0; positions start locally at zero.
This yields 137 decoder frames at the original 1024×1024 geometry and fps.
It does not re-encode raw RGB. Full-source manifests and per-draw crop hashes both remain saved.
Calibration, export parity and GT quality evaluation still use full sources.

Keep rank/alpha 128, adapter LR 0.0001, gate LR 0.01 and `mean(relu(alpha+3))`.
Use `lambda(t)=0.01*(0.1+0.9*min(t/100,1))`: weight 0.00109 at update one and 0.01 from update 100.
Use ten sigmas, the cycle `[0,9,1,8,2,7,3,6,4,5]` and 700-soft/300-binary/500-compact updates.
The grid adds all missing official 30-step dev levels in [0.9,0.95] to the previous seven levels.
Four draws per update cover each sigma twice in five updates. Draw cursors and checkpoint records use this batch size.
Validation keeps the previous three levels through `monitor_sigma_indices=[0,7,9]`.
Use a new empty output directory and freeze the current plan/code/config before any authorized launch.

```bash
conda activate ltx
CUDA_VISIBLE_DEVICES= CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.prune.data.learned_head_study \
  --config scripts/prune/configs/learned_heads_dev10_ddp_gt_uniform_f18_rank128.json \
  --plan /path/to/reviewed-study-plan.md \
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

## Training visualization

Head, FFN and joint training save a visualization every 100 optimizer updates by default.
Set `"visualization_interval": 25` in the config before freezing a new study to save every 25 updates.
Set `"visualization_interval": 0` to disable it.
Resumed runs use the global optimizer update number.

Each preview shows the actual noisy latent, GT latent and predicted x0, decoded with the selected video VAE.
The prediction comes from the training forward before its optimizer update. All panels use rank zero's first draw.
The comparison keeps the complete decoded window and original fps, including the clean first frame.

Head outputs: `training/<arm>/replicate_<seed>/visualizations/update_NNNN/`.
FFN and joint outputs: `training/visualizations/update_NNNN/`.
Each directory contains `comparison.mp4`, `poster.png`, `rendering.json` and the three individual latent tensors and decoded videos.
Decoding uses model-aware overlapping tiles while the transformer remains resident, then unloads the VAE and restores RNG.
See the [preview design](learn/visualization.md) for memory and evidence limits.

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
