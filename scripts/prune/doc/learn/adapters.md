# learn/adapters

## Objective

Adapt frozen V and output projections, then retain original head identities.

## Data flow

A named CPU generator initializes A. B starts at zero. BF16 projection arithmetic produces the residual update.

## Organization logic

Each projection owns FP32 A and B tensors. The base projection stays frozen.
For heads [0,2] and dimension two, retained channels are [0,1,4,5]. V slices B rows; output slices A columns. Adam moments use the same slices.

## Invariants

Rank remains 16. Scaling remains one. Fresh recovery initializes full-source tensors before slicing. Optimizer step counters survive ordinary transfer.

## Gotchas

Merging changes BF16 operation order. Algebraic fusion does not establish native output parity.

## Tests

Run `test_head_adapters.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.
