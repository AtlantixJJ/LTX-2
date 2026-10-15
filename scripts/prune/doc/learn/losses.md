# learn/losses

## Objective

Compute the official velocity target and generated-token loss.

## Data flow

Saved epsilon minus capture forms the FP32 target. Native velocity enters squared error. The exact clean-prefix token count removes c0.

## Organization logic

The official term averages all generated channels and tokens. A region term divides by its actual coverage sum and channel count.
Valid hands average separately. Each microbatch divides the complete phase loss by two. Sparsity applies only in the soft phase.
The frozen `sparsity_penalty` selects `mean_sigmoid` or `relu_logits`.
For `relu_logits`, use the FP32 mean of `relu(alpha + sparsity_logit_offset)` on raw logits.
With offset three, the penalty and its logit gradient are zero at alpha at most minus three.
For alpha above minus three, the complete-update gradient is `lambda(t) / number_of_heads`.
At alpha three, the raw penalty is six. At alpha minus two, it is one. At alpha minus four, it is zero.
The shared `sparsity_coefficient` owner computes
`lambda_max * (lambda_start_multiplier + (1-lambda_start_multiplier)*min(update/lambda_warmup,1))`.
Defaults are start multiplier zero and 300 warmup updates, preserving the historical ramp.
With maximum 0.1, start multiplier 0.1 and warmup 100, the effective weight rises from 0.01 at t=0 to 0.1 at t=100.
The first optimizer update uses t=1 and weight 0.0109. Updates at or after 100 use 0.1.
Binary, fixed and compact phases use zero sparsity weight. Resume keeps the global update count.
Loss reduction, analytic gradients and update receipts use the same coefficient owner.
`sparsity_gradient` supplies the analytic, unclipped gradient used to separate the logged loss contributions.

## Invariants

Targets come from saved noise and capture. Never recover training velocity from BF16 clean outputs. Empty person coverage fails.
Raw-logit sparsity requires the raw logits with the same shape as the gates. Never recover logits from rounded sigmoid values.
The ReLU stops sparsity pressure; it does not clamp the logits or prevent the velocity loss from moving them below minus three.
Reject nonfinite coefficients, nonpositive ramp durations and start multipliers outside [0,1].

## Gotchas

A perfect target prediction gives zero loss. Teacher predictions can have nonzero loss. Missing small regions remain explicit.

## Tests

Run `test_head_losses.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.

## Progressive gate removal

Progressive sparsity multiplies each raw-logit ReLU and its analytic derivative by the free mask.
The mean still divides by the original full matrix size, 1,536, after heads become terminal.
