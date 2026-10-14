# data/reuse_head_scores

## Objective

Import compatible saved raw head energies without a new GPU calibration pass.

## Usage

Run directly from LTX-2 in ltx with CUDA hidden:

```bash
CUDA_VISIBLE_DEVICES= python -m scripts.prune.data.reuse_head_scores --study /abs/frozen_study.json --previous /abs/mobilewan_head_scores_dev8_20261011
```

Output: calibration/ranking.json. See the prune run guide.

## Data flow

Frozen learning membership and saved original-model observations enter the importer.
Only complete records belonging to the current learning views contribute.

## Organization logic

Validate dev components, actual context bytes, native score controls and saved capture/noise provenance.
Validate all eight sigma rows and all 48 by 32 head values for each consumed record.
Average raw energy over sigmas, then equally over available views, before block normalization.
Pin every consumed record and input file. Declare missing views and the original requested pool.

## Invariants

No held-out actor contributes. A partial pool never claims full learning-view coverage.
No global historical ranking is substituted for the current learning subset.
The importer runs on CPU and generates no new model observations.

## Gotchas

Historical raw-byte context hashes differ from shape/dtype-aware hashes. Compare actual tensor bytes.
Sparse per-sigma fragments do not count as a complete view.

## Tests

Check native record validation, held-out exclusion, raw-before-normalized pooling and deliberate missing coverage.
