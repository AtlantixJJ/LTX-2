# Pruning run guide

Run the training-free workflow or freeze a fresh learned-gate study with the current source identity.
Use the actual reviewed specification for `--plan`. Existing frozen studies retain their isolated source trees.
The [glossary](glossary.md) defines terms. The [file map](README.md#entry-scripts) lists every CLI and its source.

## Launch and output rules

Run from the `LTX-2` root in the `ltx` conda environment.
Run entry scripts as `python -m scripts.prune.<area>.<module>`.
Use LTX-2.5 for new work.

The commands below describe the standalone pruning workflow.
Humans can run these direct CLIs.
Agents use the avatar queue for the restricted native baseline reproduction.
That route has its own [recipe](#queued-baseline-reproduction).
Other prune CLIs have no queue wrapper defined here.

Replace angle-bracket placeholders before running a command. They identify required local inputs, not shell redirections.
Replace the two view paths with existing corpus views.
Choose different actors for calibration and held-out validation.
All selected views in one producer run must have the same latent length.

Choose a fresh run root and a free GPU after checking occupancy.

For learned FFN/joint studies, freeze disjoint `gpu_ids` and `test_gpu_ids` in config; `world_size` remains four for the draw contract.
Use those exact lists in torchrun. Single-device joint diagnostics default to the first frozen test device and require it to be available.
Run those single-device diagnostics with `CUDA_VISIBLE_DEVICES` unset; otherwise logical indices can select the wrong physical GPU.
Use `env -u CUDA_VISIBLE_DEVICES python ...` when the calling shell has a mask set.

Preparers validate lists on CPU; native startup requires 44 GiB free on each selected device.
The dedicated compact-recovery schema retains its own stricter device assignment; read its frozen config.

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

The generic `--min-free-gb` flag uses decimal GB.
The scorer, functional ablation and baseline benchmark enforce 44 GiB before loading a model.
Joint diagnostics and shared DDP startup use the same native-capacity guard.
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
See [preflight](../core/preflight.py) and [prompt cache](../data/prompt_cache.py).

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
See the [producer source](../evaluate/whole_clip_baseline.py) and [input contract](data/whole_clip.md).

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

The scorer reproduces each calibration baseline. Functional mode checks finite replay of the unmasked held-out baseline,
then measures head-only, FFN-only and combined masks. Apply the [evidence gates](VALIDATION.md) before making quality or speed claims.
See [scorer](../score/whole_clip_d0_scores.py) and [analyzer](../evaluate/whole_clip_d0.py).

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
See [export](score/export_pruned.md#usage) and [parity](../checks/export_parity.py).

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
The [verification guide](VALIDATION.md#g4-assess-held-out-decoded-quality) defines numerical and visual evidence.

Repeat timing with `--order BAB` and a new output file when gains approach drift.
Keep device, geometry and numerical policy fixed.
See [analyzer](../evaluate/whole_clip_d0.py),
[benchmark](../evaluate/bench_whole_clip_d0.py) and [acceptance](VALIDATION.md).

## Joint clipped gates with re-admission

Freeze a fresh study from a verified parent FFN study and its extended paired-draw authority.
Use exact arm names: `joint_clip_reset_v3`, `heads_clip_reset_v3` or `budget_joint_v3`.
The [selection algorithm](core_algorithm.md#joint-clipped-selection-with-return) explains their differences.
The preparer inherits the parent's device lists unless `--device-config <new_device_config.json>` supplies new groups.
That JSON must contain exactly `gpu_ids`, `test_gpu_ids` and `world_size`.
Its values become part of the new frozen config and identity; the parent remains unchanged.
The budget arm additionally requires `--budget-receipt` from the joint reference endpoint.

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.prune.data.joint_study \
  --source-study <parent_frozen_study.json> --continuation-study <extended_draw_study.json> \
  --plan <reviewed_plan.md> --arm joint_clip_reset_v3 --output <fresh_study_root>
```

The preparer writes `<fresh_study_root>/configs/frozen_study.json`, paired draws and a pinned plan copy.
Set `PRUNE_STUDY` to that exact config path; the study-root directory is recorded inside the file.

For disposable tests, set the following variables from the frozen config, retaining device order:

```bash
PRUNE_STUDY="<frozen_study.json>"
PRUNE_TEST_DEVICES="<comma_separated_test_gpu_ids>"
PRUNE_TEST_RANKS="<length_of_test_gpu_ids>"
PRUNE_TRAIN_DEVICES="<comma_separated_gpu_ids>"
PRUNE_TRAIN_RANKS="<world_size>"

CUDA_VISIBLE_DEVICES="$PRUNE_TEST_DEVICES" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  torchrun --standalone --nproc-per-node="$PRUNE_TEST_RANKS" \
  -m scripts.prune.learn.joint --study "$PRUNE_STUDY" --phase native --testing

CUDA_VISIBLE_DEVICES="$PRUNE_TEST_DEVICES" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  torchrun --standalone --nproc-per-node="$PRUNE_TEST_RANKS" \
  -m scripts.prune.learn.joint --study "$PRUNE_STUDY" --phase pilot --testing --max-updates 50
```

The test pilot is bounded and cannot substitute for acceptance on the formal device group.
The shared [DDP owner](learn/ddp_owner.md) documents how test-rank draws use the formal stride.

### Prerequisites and formal launch

The legacy phase names `p0` and `p1` mean optimizer replay and native channel ablations, as defined in the [glossary](glossary.md).
Read complete parser options with `--help`; each shard must use a different output/case partition.

```bash
python -m scripts.prune.checks.joint_preflight --phase p0 --output <pilot_root> \
  --traces <pilot_gradient_trace_directory>
python -m scripts.prune.checks.joint_preflight --phase p1 --study "$PRUNE_STUDY" \
  --gpu-id <configured_free_test_gpu> --shard 0 --shards 1
python -m scripts.prune.checks.joint_export --study "$PRUNE_STUDY" --output <export_isolation_output>
python -m scripts.prune.checks.joint_review --study "$PRUNE_STUDY" --pilot <pilot_root> \
  --p1 <complete_channel_ablation_directory> --p2 <export_isolation_output> --output <prerequisite_receipt.json>
```

Freeze a new **formal** study with `data.joint_study --prerequisites <prerequisite_receipt.json>` after reviewing complete evidence.
The replay requires at least 50 complete native channel-gradient traces; an independent-noise replay alone leaves correlation unresolved.
Keep pilot evidence and final formal inputs separate. The formal owner requires pinned, passed prerequisites.
Use its new `frozen_study.json` as `PRUNE_STUDY`, then run:

```bash
CUDA_VISIBLE_DEVICES="$PRUNE_TRAIN_DEVICES" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  torchrun --standalone --nproc-per-node="$PRUNE_TRAIN_RANKS" \
  -m scripts.prune.learn.joint --study "$PRUNE_STUDY" --phase formal
```

`--phase formal` runs independent formal-rank native acceptance before training.
`--phase train` resumes an authorized unchanged frozen study after its matching acceptance exists.
Pending `allocation_review` and `alignment_review` triggers stop continuation; see the [review contract](learn/readmission.md#reviews-belong-to-the-training-owner).
Code maintenance does not request a launch or a restart.

### Joint export fault isolation and split bundles

```bash
python -m scripts.prune.checks.joint_faults --study "$PRUNE_STUDY" --output <operation_diagnostic_output>
python -m scripts.prune.checks.joint_split_parity --study "$PRUNE_STUDY" --output <split_parity_output>
```

These single-device diagnostics use the frozen test list and reject insufficient free capacity.
`score.export_joint` writes merged final weights after terminal alignment and recovery.
`score.export_joint_split` writes compact base weights, adapters and an envelope.
Load that envelope with `score.export_joint_split.open_bundle`, including both payloads.
The [runtime contract](score/export_pruned.md#learned-exports-and-split-bundles) explains FFN reduction geometry and fusion rounding.

## Completed joint V3 evaluation

Read fixed completed checkpoints without creating an optimizer:

```bash
python -m scripts.prune.evaluate.joint_final --help
python -m scripts.prune.evaluate.joint_media --help
python -m scripts.prune.evaluate.joint_monitor_media --help
python -m scripts.prune.evaluate.historical_clip --help
```

`joint_final` saves fixed selection/final predictions and runs deployment parity/timing stages.
`joint_media` decodes the matched held-out reference set.

`joint_monitor_media` pairs existing development predictions; its legacy `--study-a`, `--study-b` and `--study-c` flags identify
joint, head-only and budget-control studies respectively. Missing matched predictions are a named gap.
`historical_clip` replays a saved head-only monitor before scoring its fixed endpoint.
Source docstrings and parsers in the [entry map](README.md#entry-scripts) own stage-specific arguments.
These schema-specific evaluators can retain stricter device rules; inspect their parser and the study before choosing a free device.

## Learned FFN masks

Prepare on CPU from a verified head study and an explicit config:

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.prune.data.ffn_study \
  --source-study <parent_frozen_study.json> \
  --config scripts/prune/configs/learned_ffn_dev10_ddp_joint.json \
  --plan <reviewed_plan.md> --output <fresh_study_root>
```

Use the formal/test torchrun variables above with `scripts.prune.learn.ffn --phase native --testing` for disposable acceptance,
and `--phase formal` for separate formal acceptance followed by training.
The preset supports `ffn_only` and `joint_heads_ffn` arms; each frozen study selects one.
Resume only the unchanged authorized study with `--phase train`.
`score.export_learned_ffn` requires its exact final recovery checkpoint; `--probe` selects a disposable export.

## Learned head phases

Fresh preparation requires an explicit `--evidence-list` of inspected prior evidence; preparation does not scan experiment trees.
The [phase contract](learn/heads.md) lists what each phase consumes and produces.
Default GT training defers regional preparation; choose a separately prepared study for foreground/region-weighted arms.

## Preset values

The JSON files own exact values. This section is the one descriptive copy of head/FFN preset settings.
The **frozen study config**, including explicit overrides, controls a particular run.

| Setting | [Head preset](../configs/learned_heads_dev10_ddp_gt_uniform_f18_rank128.json) | [FFN preset](../configs/learned_ffn_dev10_ddp_joint.json) |
|---|---|---|
| Update budget | `soft_updates=700`, `binary_updates=300`, `compact_updates=500` | `selection_updates=1000`, `recovery_updates=300`, `head_soft_updates=700` |
| Adapter ranks | `rank=128`, `lora_alpha=128` | `head_rank=128`, `ffn_rank=16` |
| Learning rates | `gate_lr=0.01`, `adapter_lr=0.0001` | `gate_lr=0.01`, `adapter_lr=0.0001` |
| Head budget | `removed_heads` choices `[152,352,508]` out of `blocks*heads=48*32` | `removed_heads=152` |
| Allowed head block widths | `allowed_counts=[16,20,24,28,32]` | Inherited head config |
| FFN retained width | Execution probes only | `ffn_kept=14848` of `ffn_width=16384`, `ffn_alignment=128` |
| FFN ramp | Not applicable | `ffn_warmup=50`, `ffn_ramp_end=300` |
| Optimizer inputs | `uniform_f18=true`, one draw per rank | Inherited `uniform_f18=true`, one draw per rank |
| Head sparsity | `sparsity_penalty=relu_logits`, `sparsity_logit_offset=3`, `lambda_max=0.01`, `lambda_warmup=100`, `lambda_start_multiplier=0.1` | Inherited head penalty; no FFN penalty |
| Monitoring | `monitor_reference=gt_capture`, `monitor_sigma_indices=[0,7,9]` | Inherited grid and monitor indices |

The head preset's ten exact `sigmas` live in JSON.
The update cycle is `sigma_indices_cycle=[0,9,1,8,2,7,3,6,4,5]`; four draws cover each sigma twice in five updates.
Five levels are official dev levels in `[0.9,0.95]`.
The head sparsity coefficient is `lambda(t)=0.01*(0.1+0.9*min(t/100,1))`, giving 0.00109 at update one and 0.01 from update 100.
For the 152-head removal choice, 1,384 of 1,536 heads remain.
V3 event constants and the budget cost have their sole descriptive home in the [joint algorithm](core_algorithm.md#joint-clipped-selection-with-return).

## Uniform F18 DDP training

Use a new output directory and the current reviewed plan and source identity.
Keep F18 sources intact; crop longer sources to a continuous 18-latent window with a separate RNG keyed by draw noise seed.
Capture and epsilon share bounds, positions start locally at zero and the window's first latent is clean.
These optimizer crops decode to 137 frames; monitoring, calibration and export parity retain complete sources.
Both full-source file pins and consumed-window hashes are saved.

```bash
CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.prune.data.learned_head_study \
  --config scripts/prune/configs/learned_heads_dev10_ddp_gt_uniform_f18_rank128.json \
  --evidence-list <inspected_evidence.json> --plan <reviewed_plan.md> --output <fresh_study_root>

CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -u -m scripts.prune.learn.run_full --study <frozen_study.json> --arms learned_joint
```

Here too, `<frozen_study.json>` means `<fresh_study_root>/configs/frozen_study.json`.
The full owner checks its configured devices, matching `checks.ddp_native` receipt and method before optimizer phases.
It runs separate native acceptance when missing. GT mode requires neither teacher caching nor a speed test.
For the energy control, prepare `calibration/ranking.json` explicitly before passing `--arms learned_joint energy`.
Long jobs can use `setsid` and redirected logs after creating the study's log directory; the direct owner manages only its own children.

## Training visualization

`visualization_interval` defaults to 100 updates; zero disables previews and another positive value changes their interval.
Head, FFN and joint previews use rank zero's actual first training draw, before the optimizer update.
Noisy input, GT and predicted x0 use the same crop, VAE, fps and full window coverage.
Model-aware overlapping decoder tiles bound memory; decoding restores RNG and synchronizes ranks.

Head outputs live under `training/<arm>/replicate_<seed>/visualizations/update_NNNN/`.
FFN/joint outputs use `training/visualizations/update_NNNN/`.
Each directory includes `comparison.mp4`, `poster.png`, `rendering.json`, individual latents and decoded videos.

When `wandb.enabled=true`, use one CPU writer per arm.
Head logging uses `learn.wandb_logger`; FFN/joint logging uses `learn.receipt_logger --study <frozen_study.json>`.
Rank sigma/time/memory charts have one line per rank; losses and unique-case validation errors are averaged scalars.
Metric tabs are `train`, `gates`, `memory`, `ranks` and `validation`, with separate runs rather than arm-name prefixes.
In `validation/trained_sigma_2_gt_latent_mse`, suffix `_2` means position two in `monitor_sigma_indices`,
which is grid index 9 in the default preset, sigma 0.48016369342803955.
Chart refresh follows `wandb.chart_interval`; saved scalar/update coverage stays complete.

Read the study's `monitoring/status.json`, `monitoring/owner.json`, failure records and launch log for local progress.
Immutable evidence records belong beside the checkpoints; status files can change.

## Queued baseline reproduction

Agents use the existing avatar queue for its restricted native `prune_baseline` experiment.
Read its [run guide](../../onestep_avatar/configs/README.md#pruning-baseline-prune_baseline),
[protocol](../../onestep_avatar/doc/experiments/prune_baseline.md) and [queue usage](../../onestep_avatar/doc/usage.md).

The spec selects one absolute view, one seed, white distilled unguided raw output and context eight.
It preserves requested sigmas and supports D0, optionally D1. Build the selected prompt cache first.
Check the spec and job list before the queue owner launch:

```bash
python -m scripts.onestep_avatar.experiments.prune_baseline \
  --spec <prune_baseline_spec.json> --output <fresh_prune_job> --dry-run
```

This parses the spec without model loading or component hashing.
Start the queue owner with `setsid` and the documented shared process ledger.
Original prune producer files remain under `<job-output>/baseline/`; wrapper records stay at the job root.
Use the [reproduction controls](../../onestep_avatar/configs/README.md#pruning-reproduction-controls-prune_reproduction) for drift diagnosis.

## Compact A LoRA recovery

`compact_A_lora` is the recorded schema/CLI name for fresh adapter recovery on an aligned joint checkpoint.
It fuses the source adapters into the base, restores aligned widths and trains fresh zero-output adapters with fresh Adam state.
It preserves the paired draw prefix, full-source input identities and original retained unit order.

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.prune.data.compact_study \
  --source-study <completed_joint_study.json> --plan <reviewed_compact_plan.md> \
  --output <fresh_compact_study_root> --updates 500
python -m scripts.prune.learn.compact --help
python -m scripts.prune.evaluate.compact --help
```

Freeze isolated numerical source before preparing a new study.
The dedicated compact schema and its source docstrings define alignment, acceptance, recovery and evaluation stages.
Use its frozen device policy and independent formal-rank acceptance.
The [workspace handoffs](../../../../plans/README.md) hold completed results and stopped-run status.

## CPU verification and stopping

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest scripts/prune/tests -q -m 'not gpu'
ruff check --select F,I scripts/prune
```

Direct CLIs run in the foreground; interrupt your own command to stop it.
Keep partial outputs for diagnosis and use a new root for another producer attempt.
Managed avatar jobs use the queue's owned-process stop path.
The [verification guide](VALIDATION.md) and [change-impact map](ARCHITECTURE.md#change-impact-map) define further checks.
