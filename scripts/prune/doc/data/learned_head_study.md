# data/learned_head_study

## Objective

Freeze source identity, disjoint actors and paired optimization draws.

## Usage

Historical study plans were consolidated in the [workspace history](../../../../../plans/history/README.md).
For fresh preparation, replace `/path/to/reviewed-study-plan.md` with the actual reviewed specification.
Retired reference pointers are not plan inputs, and retired source folders must not be recreated from these examples.

Started by: a direct human or agent CLI. Run from `LTX-2` in conda environment `ltx`.

```bash
CUDA_VISIBLE_DEVICES= python -m scripts.prune.data.learned_head_study --config scripts/prune/configs/learned_heads_dev10_ddp_gt_uniform_f18_rank128.json --plan /path/to/reviewed-study-plan.md --evidence-list /path/to/seen-evidence.json --output ../expr/refiner_prune/2.5/learned_heads_ddp_uniform_f18_new
```

`--seen-actors` adds audited exclusions. `--region-tracks` accepts reviewed GT box records.
`--evidence-list` is required for fresh preparation. It declares prior evidence explicitly, instead of scanning experiment trees.
The default config is the ten-level uniform-F18 rank128 DDP preset.
`--uniform-f18` explicitly adds the crop option to another compatible preset and saves its effective config.

Outputs: Frozen study, source pins, regions, buckets and paired draws.
See the [run guide](../usage.md#learned-head-phases).

## Data flow

Pinned membership supplies views. GT-only replay supplies motion and area. Splits, draws, component hashes and actual context bytes form the study.

## Organization logic

Validation uses twenty hashed train actors. Learning uses actor rounds until 256 views are selected. Later rounds prefer an unused clip length.
Blind selection excludes known inspected actors and actors cited in earlier evidence.
The JSON declaration has `format="prune_seen_evidence_v1"` and a `files` list of `{path, sha256}` records.
Relative paths resolve beside the declaration. Markdown actor mentions and JSON `rows`/`cases` supply exclusions.
The declaration and every declared evidence file enter the frozen file pins. Duplicate or unsupported paths fail.
An explicitly empty list declares no additional prior files; existing known/seen actor exclusions still apply.
Choose completed immutable evidence. A later change to any declared input invalidates that new study.
Each seed has `1500*global_unique_clips` named draws: 6,000 for four-rank DDP, 3,000 for legacy two-draw studies.
The frozen config selects exact, distinct, descending levels from the official 30-step dev scheduler, with no latent argument.
Only positive values at most 0.95 are accepted. Historical seven/eight-level configs remain supported.
The sigma cycle contains each selected index exactly once. Four consecutive draws form a four-rank DDP update.
For seven levels, `[0,6,1,5,2,4,3]` repeats. Seven DDP updates cover every level four times.
The current ten-level grid adds 0.9403364062309265, 0.9309937953948975 and 0.9098098278045654 to the earlier seven levels.
It therefore covers all five official dev levels in [0.9, 0.95].
Cycle `[0,9,1,8,2,7,3,6,4,5]` gives every level twice in five four-draw updates.
`monitor_sigma_indices=[0,7,9]` preserves the previous three validation levels despite the expanded training grid.
Legacy FSDP studies retain two draws per update and twice-per-level coverage in seven updates.
Training, input manifests, native checks and evaluation read this frozen grid through `study_sigmas`.
Historical head-score records retain their separate eight-level grid.

## Invariants

Complete clips, crops, source hashes and actor identities must agree. Both lengths must exist. Changed code or input content prevents resume.
DDP requires `uniform_f18=true`, accumulation one and one distinct draw index per rank.
The study pins full source geometry; optimizer draw receipts separately pin the selected F18 window.

## Gotchas

Balanced sampling requires six nonempty buckets. Small-region arms require separately reviewed tracks. Plan status changes preserve the frozen plan snapshot.

## Tests

Run `test_learned_head_study.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.
`test_declared_evidence.py` verifies exclusions, ignored undeclared files and refusal of changed content.

## Preparation workers and report tools

Four independent CPU workers audit the selected views.
Each completed source audit pins its reader code and file content before reusing GT statistics.
Changed code or source bytes reject the cache; no selected view disappears after a failure.
Preparation installs saved-only report readers from `configs/learned_heads_report/`.
Report terms use the configured adapter rank, including rank 128.
The narrative preserves all seventeen questions and labels unrun comparisons explicitly.
Missing declared report evidence raises an error and launches no model or decoder.

## Deferred foreground preparation

The unweighted GT preset uses `foreground_preparation=deferred`.
It still verifies capture, RGB and metadata hashes, complete latent geometry, fps and canonical crop records.
It does not decode or replay mattes, compute regional tracks or motion/size buckets on the training critical path.
Mark this scope in every source audit. Generate only ordinary paired draws.
Regional and balanced training reject this preset. Their later studies require complete foreground preparation.
