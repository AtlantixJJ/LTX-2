# data/learned_head_study

## Objective

Freeze source identity, disjoint actors and paired optimization draws.

## Usage

Started by: a direct human or agent CLI. Run from `LTX-2` in conda environment `ltx`.

```bash
conda run --no-capture-output -n ltx python -m scripts.prune.data.learned_head_study --config scripts/prune/configs/learned_heads_dev8.json --plan ../plans/2026-10-12-learned-head-pruning-experiment.md --output ../expr/refiner_prune/2.5/learned_heads_dev8_20261012_r2
```

`--seen-actors` adds audited exclusions. `--region-tracks` accepts reviewed GT box records.

Outputs: Frozen study, source pins, regions, buckets and paired draws.
See the [run guide](../usage.md#learned-head-phases).

## Data flow

Pinned membership supplies views. GT-only replay supplies motion and area. Splits, draws, component hashes and actual context bytes form the study.

## Organization logic

Validation uses twenty hashed train actors. Learning uses actor rounds until 256 views are selected. Later rounds prefer an unused clip length.
Blind selection excludes known inspected actors and actors cited in earlier evidence. Each seed has 3,000 named noise draws and an eight-sigma cycle.

## Invariants

Complete clips, crops, source hashes and actor identities must agree. Both lengths must exist. Changed code or input content prevents resume.

## Gotchas

Balanced sampling requires six nonempty buckets. Small-region arms require separately reviewed tracks. Plan status changes preserve the frozen plan snapshot.

## Tests

Run `test_learned_head_study.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.

## Preparation workers and report tools

Four independent CPU workers audit the selected views.
Each completed source audit pins its reader code and file content before reusing GT statistics.
Changed code or source bytes reject the cache; no selected view disappears after a failure.
Preparation installs saved-only report readers from `configs/learned_heads_report/`.
The narrative preserves all seventeen questions and labels unrun comparisons explicitly.
Missing declared report evidence raises an error and launches no model or decoder.
