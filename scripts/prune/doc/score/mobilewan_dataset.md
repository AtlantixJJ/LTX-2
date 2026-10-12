# Resident MobileWan dataset worker

## Objective

Score every complete capture in one deterministic membership shard without reloading the transformer per view.
Each view reuses one saved epsilon across the eight exact task sigmas.

## Usage

Launch this standalone prune worker from `LTX-2` in the `ltx` environment.
The native checks at both lengths must already have passed.
Check GPU occupancy before launch. Use a process timeout and an attributable log for persistent work.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m scripts.prune.score.mobilewan_dataset \
  --study ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011/configs/study.json \
  --split train --shard 0 --shards 3 --gpu-id 4 --wall-seconds 86400
```

`--split` selects train or held-out membership. `--shard` is zero-based within `--shards`.
`--max-views` bounds a smoke run without changing membership or the full-clip span.
`--wall-seconds` includes component hashing and model loading. External timeout must also bound stalled calls.
Outputs are `inputs/views/`, `scores/per_sigma/`, `scores/per_view/` and `provenance/worker_*.json` below the study root.
See the [run guide](../usage.md#mobilewan-dataset-worker).

## Data flow

Validate both saved native gates before loading a model.
Each gate must pin the current projected-energy source hash; a score change requires fresh checks.
Recheck the complete audited manifest and every model component's full file hash.
Open the selected dev transformer with empty text and verify the actual context bytes.
Discover the self-attention head widths from the loaded model.
Sort membership by relative view ID and assign every view by index modulo shard count.

The shared `data.whole_clip.prepare_score_inputs` saves capture-only input evidence and one epsilon per view.
The shared `build_input` constructs every D0 modality. The worker defines no second noising or conditioning calculation.
The existing hook owner observes the executed gated output-projection input.
The projected-energy helper computes every eligible head over the complete generated-token set.

## Organization logic

A completed view record must match source, noise, context, transformer, software and sampler identities before resume.
A partial view retains each completed sigma independently.
A missing observation runs one native call. An existing observation runs none.
Publish the complete view record only after all eight observations pass.
Reject incomplete block coverage, nonfinite predictions and changed software.
A worker failure records the active view, error and elapsed time before re-raising the exception.

The model remains resident across assigned views. Its weights are frozen and it runs without gradients.
The worker applies the deterministic baseline policy and bounds allocated memory at 48,000,000,000 bytes.
It checks the wall bound before each sigma. Saved observations survive a process failure.

## Invariants

The membership manifest must retain the exact audited hash.
Only train views can run before frozen train masks exist.
The held-out phase requires `masks/frozen_train.json` with train-only identity, complete train record hashes and the requested budgets.
No source failure silently removes a view. The worker fails with attributable evidence.
Complete coverage is checked across all workers, not inferred from one completed shard.

## Gotchas

With three shards, sorted indices 0,3,6 belong to shard zero; indices 1,4,7 belong to shard one.
Every view still uses all its frames and all eight sigmas.
A longer view has more query tokens but remains one equal-weight observation in dataset statistics.
Changing code invalidates resume identities. Preserve earlier records and diagnose the change before replacing artifacts.
The worker does not publish final statistics, choose masks, inspect held-out quality or build a report.

## Tests

`tests/test_mobilewan_dataset.py` verifies disjoint complete shard coverage and shared-input noise reuse.
A tiny real model produces all eight sigma records, then resumes exactly one deliberately missing observation.
Native two-length checks and the first real worker run remain independent acceptance evidence.
