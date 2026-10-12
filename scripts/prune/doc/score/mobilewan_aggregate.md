# MobileWan final statistics and train mask freeze

## Objective

Publish final head statistics only after complete supplied membership coverage.
Freeze removal masks from pooled train scores before any held-out scoring.

## Usage

Run on CPU from `LTX-2` in the `ltx` environment.
After all train records complete, freeze their masks:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' python -m scripts.prune.score.mobilewan_aggregate \
  --study ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011/configs/study.json --freeze-train
```

After held-out scoring completes, run the same command without `--freeze-train` to publish `head_scores.json`.
Add `--verify` to recompute the final statistics from all saved raw records and check the saved final file.
Output masks are under `masks/`. Verification writes `validation/head_scores_validation.json`.
See the [run guide](../usage.md#mobilewan-aggregation).

## Data flow

The aggregate owner rechecks the audited manifest hash and inspects checkpoint projection shapes on CPU.
It reads every required complete view record, input manifest, saved epsilon and per-sigma observation.
View, actor, split, capture, noise, context, transformer, software and sampler identities must match.
The complete view energies must equal each saved sigma observation.
Missing records fail. The code does not fill missing model runs.

Train selection averages raw energies across sigmas and equally weighted views before normalization.
It uses the global allocator with stable original block/head identities.
Each functional artifact retains all text cross-attention heads.
The frozen record pins every train view record and exact integer removal allocation.

## Organization logic

`--freeze-train` requires all 3,104 train views and 24,832 observations.
A new freeze fails if any held-out observation already exists.
An existing freeze is verified and reused; it is never recomputed after held-out inspection.

Final publication requires all 3,360 views and 26,880 observations.
Statistics cover train, held-out and all cohorts, pooled and at each exact sigma.
Each block uses the shared actor-cluster bootstrap helper with 2,000 replicates and seed 42.
Each cohort's actor draws remain paired across every block and sigma.
Layer summaries use mean raw head energy, not the normalized layer mean of one.

Final schema version one uses `kind: mobilewan_head_scores` and `status: complete`.
`cohorts.<cohort>.layers.<original block>` contains raw and normalized means, sample standard deviations and pointwise intervals.
Matrix row zero is pooled. Rows one through eight follow `sigmas` exactly.
`head_indices` maps every column to its original head index.
Separate ascending rankings exist for pooled and per-sigma descriptive scores.
`train_selection` carries the frozen allocation and train-only selection provenance.

## Invariants

The schema distinguishes membership JSON bytes, CSV bytes and semantic membership identity.
No partial run publishes a final result.
Every original head, cohort, sigma, count, interval and rank must be present and finite.
Normalized means and standard deviations must agree with raw values and the fixed block denominator.
Rank order must agree with the normalized scores and stable block/head ties.
The final validator recomputes frozen allocation from complete train energies.
`--verify` also recomputes all statistics and bootstrap intervals from the raw records.

## Gotchas

A confidence interval describes a mean under actor resampling, not individual view ranges or simultaneous head coverage.
One saved seed does not estimate noise-seed variability.
All-view and held-out rankings are descriptive and never select removal masks.
Raw records from changed model, source or sampler identities cannot be pooled.
Checkpoint shape inspection loads no model tensors and needs no GPU.

## Tests

`tests/test_mobilewan_aggregate.py` checks pooled equal-view statistics and paired actor bootstrap identities.
It verifies that changing held-out values leaves train allocation unchanged.
It rejects missing intervals, nonfinite values, incomplete counts, changed head indices and wrong rank order.
The real dataset freeze and final publication remain dependent on complete GPU scoring.
