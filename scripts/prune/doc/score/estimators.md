# `score/estimators.py` — score calculation and mask allocation

## Objective

Keep score mathematics independent of model forwards and input construction.

## Data flow

The caller supplies activation statistics, projection weights or explicit token indices.
The functions return one score or binary mask value per pruning unit.

## Organization logic

`rms_projection_scores(S, P, n)` computes `sqrt(S/n)*P`.
The native scorer averages squared activation means equally across clip/sigma cases.

`exact_local_head_energy` projects each sampled head through its output-weight columns.
It returns the RMS projected vector norm and keeps within-head covariance.
The default scorer does not use this helper.

`fractional_masks` removes the lowest `round(width*fraction)` scores per branch.
It keeps at least one unit. Stable sorting resolves ties by source order.

Worked example: scores `[1,1,3,4]` and fraction 0.5 produce `[0,0,1,1]`.
A zero fraction returns all ones.

## Invariants

- Allocation scores are finite, nonempty and one-dimensional.
- Fractions lie in `[0,1)`.
- The caller selects prediction tokens and excludes clean frame zero.

## Gotchas

The RMS proxy omits cancellation and downstream effects.
Exact local energy also does not measure a full-model ablation.
A new default estimator needs held-out comparison at fixed executable widths.

## Tests

`tests/test_estimators.py` checks score scaling, local energy, stable ties and malformed scores.
