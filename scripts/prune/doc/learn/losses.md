# learn/losses

## Objective

Compute the official velocity target and generated-token loss.

## Data flow

Saved epsilon minus capture forms the FP32 target. Native velocity enters squared error. The exact clean-prefix token count removes c0.

## Organization logic

The official term averages all generated channels and tokens. A region term divides by its actual coverage sum and channel count.
Valid hands average separately. Each microbatch divides the complete phase loss by two. Sparsity applies only in the soft phase.

## Invariants

Targets come from saved noise and capture. Never recover training velocity from BF16 clean outputs. Empty person coverage fails.

## Gotchas

A perfect target prediction gives zero loss. Teacher predictions can have nonzero loss. Missing small regions remain explicit.

## Tests

Run `test_head_losses.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.
