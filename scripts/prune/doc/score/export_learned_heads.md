# score/export_learned_heads

## Objective

Export a frozen compact base and merge the recovered adapter.

## Usage

Started by: a direct human or agent CLI. Run from `LTX-2` in conda environment `ltx`.

```bash
conda run --no-capture-output -n ltx python -m scripts.prune.score.export_learned_heads --study ../expr/refiner_prune/2.5/learned_heads_dev8_20261012_r2/configs/frozen_study.json --arm learned_joint --replicate-seed 42 --stage compact_base
```

`--stage` selects compact_base or final_merged. Export performs CPU surgery.
`--input-gpu-id 0` saves missing seed-42 structural input manifests before surgery. It loads no transformer and runs no scoring.
The full GT supervisor uses this option after selection. The CUDA device only creates native noise and input metadata.

Outputs: Compact checkpoint, geometry wrappers, masks and recovered envelope.
See the [run guide](../usage.md#learned-head-phases).

## Data flow

The final binary selection supplies original head IDs. Existing coupled export reduces V, native gate and output widths.

## Organization logic

Separate seed-42 legacy wrappers retain each clip geometry. The recovered envelope pins every consumed draw receipt and the full frozen study.
GT training can reuse historical energy without creating these wrappers' inputs.
Save only missing inputs with the shared native input producer after selection. Preserve the complete learning-view scope and original seed-42 contract.
Fusion computes W0 plus B@A in FP32, then rounds once. The checkpoint pins its envelope; the envelope pins the merged tensor inventory.

## Invariants

Q/K projections and normalization remain full width. No selector runs during export. Retained pruning amplitudes are one.

## Gotchas

Structural transfer, fusion and reload are different numerical checks. An exported checkpoint remains unqualified until native and quality evidence passes.

## Tests

Run `test_export_learned_heads.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.
The lazy-input test verifies missing-view selection, saved native settings and refusal to open a transformer.
