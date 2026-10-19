# Native head-gate collection owner

## Objective

`score/native_head_gates.py` observes the fixed dev checkpoint on retained training draws.
It owns CPU preparation, three standalone GPU workers and validated milestone publication.
It does not train or change any existing pruning run.

## Usage

Agents and humans launch this standalone prune diagnostic directly, in `ltx`, from `LTX-2`.
Check physical GPUs 4–6 before model loading. The avatar queue does not own this diagnostic.

```bash
cd LTX-2
conda activate ltx
nvidia-smi
python -m scripts.prune.score.native_head_gates --phase prepare \
  --source ../expr/refiner_prune/2.5/learned_ffn_joint_20261018_r3/configs/frozen_study.json \
  --combined ../expr/refiner_prune/2.5/progressive_clip01_ste3_reset50_20261018_combined \
  --plan ../plans/2026-10-18-native-head-gate-statistics-vs-pruning.md \
  --output ../expr/refiner_prune/2.5/native_head_gate_stats_20261018 --steps 1300
mkdir -p ../expr/refiner_prune/2.5/native_head_gate_stats_20261018/logs
setsid python -m scripts.prune.score.native_head_gates --phase run \
  --study ../expr/refiner_prune/2.5/native_head_gate_stats_20261018/configs/frozen_study.json \
  > ../expr/refiner_prune/2.5/native_head_gate_stats_20261018/logs/owner.log 2>&1 < /dev/null &
```

Create the run's `logs/` directory before shell redirection.
`--phase prepare` is CPU-only; it verifies inputs and never opens a CUDA model.
`--phase run` starts one worker per allowed GPU and supervises only its own children.
`--phase worker --gpu-id 4` is an internal/resume worker route. Use the supervisor for complete runs.
`--phase report --study <path>` rebuilds saved Markdown without model loading.
`--steps` accepts 100-step multiples up to the verified 1300-step endpoint during preparation.

Outputs are `workers/`, `draws/`, `milestones/`, `provenance/`, `monitoring/` and `REPORT.md` under the new run root.
See the [run guide](../usage.md#native-head-gate-statistics).

## Data flow

Preparation pins the original training stream and selects source sigma indices `[1,4,7,9]`.
Workers receive retained global draw IDs according to `draw_index % 3`.
The three physical workers preserve the original four-scheduled-record training step convention.
The input owner reuses exact saved epsilon and F18 crop rules.
Native forwards produce sufficient statistics and immutable per-draw input receipts.
Each worker writes a milestone shard. The supervisor merges only complete retained source prefixes.

## Organization logic

The session uses explicit dev components and the saved context, without a text-encoder fallback.
Each worker checks observed versus ordinary native output before sustained collection.
All workers wait for matching E0 acceptance on all three GPUs before collection starts.
The acceptance draw has the requested noise level near 0.9 and a nonzero-start F18 crop.
Full weight hashes and input hashes must stay unchanged; forward outputs must be bit-identical.
Parameter-version checks guard every later draw.

Each 100-source-step window includes 400 scheduled records and 160 native observations.
The supervisor requires all 160 retained global IDs, no duplicates, and 40 draws per selected sigma.
Worker arrival order cannot move a later record into an earlier milestone.
Only complete milestones reach the saved-statistics analysis/report owner.
A report sketch appears immediately after worker launch, before the first milestone.

## Invariants

- Physical GPU IDs are restricted to 4, 5 and 6.
- The model stays fixed, in inference mode, with no optimizer, adapters or pruning masks.
- Existing training receipts must match the reconstructed crop and saved input manifest.
- Checkpoints atomically bind unfinished arrays to consumed draw receipts.
- Immutable completed worker shards retain content hashes.
- An exact NPZ can finish a missing sidecar on resume, without model replay.
- Resume rejects changed draw IDs or a conflicting sidecar; it never replaces completed evidence.
- Model code covered by the frozen identity cannot change during execution.

## Gotchas

The supervisor writes current progress but advances scientific steps only after a complete prefix.
A worker can be ahead of that prefix; its local step is not the report's validated milestone.
Workers checkpoint unfinished state every 20 local observations and on Python exceptions.
A hard interruption can require replay from the last complete worker checkpoint; per-draw receipts reject changed replay.
Worker failure or 30 minutes without recorded progress stops this owner's remaining children and preserves errors.
It never terminates unrelated jobs.

## Tests

`tests/test_native_head_gates.py` covers arithmetic transparency, accumulation, resume and report boundaries.
The full native acceptance runs separately on each authorized GPU and produces saved receipts.
CPU checks do not qualify native correctness, completion or convergence.
