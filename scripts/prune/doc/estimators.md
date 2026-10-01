# `score/estimators.py` — native-token unit score primitives

## Objective

Keep reusable scoring mathematics independent of model and input construction. The caller specifies native prediction tokens and owns model forwards and task loss.

## Data flow

`rms_projection_scores` turns accumulated sampled activation square means and output-projection norms into the current cheap D0 proxy. `exact_local_head_energy` computes each head's actual projected local output RMS over an explicit token index set, retaining within-head covariance; it is a potential ranking comparison, not a downstream ablation. `fractional_masks` removes the lowest scores per branch with stable ties and one retained unit minimum. The D0 scorer uses the RMS proxy and allocator; it does not silently enable exact local energy, Taylor gradients or reconstruction.

## Invariants and checks

The clean conditioning frame is excluded by the D0 caller's token indices. The caller owns token selection and model forwards. `test_estimators.py` checks local energy, score scaling, stable ties and malformed scores. A new estimator needs held-out D0 comparison before it becomes the default.
