# learn/adapters

## Objective

Adapt frozen V and output projections, then retain original head identities.
The FFN extension also adapts the native up and down projections.

## Data flow

A named CPU generator initializes A. B starts at zero. BF16 projection arithmetic produces the residual update.

## Organization logic

Each projection owns FP32 A and B tensors. The base projection stays frozen.
`attach_ffn` uses a separate seed-plus-20000 generator and preserves attached attention adapters.
FFN up transfer slices B rows. FFN down transfer slices A columns at the same original channel IDs.
`transfer_ffn_optimizer` applies those slices to moments and preserves the optimizer step counter.
For heads [0,2] and dimension two, retained channels are [0,1,4,5]. V slices B rows; output slices A columns. Adam moments use the same slices.

## Invariants

The frozen study supplies the adapter rank. The default API rank is 16; the high-capacity preset uses 128.
Scaling remains one. Fresh recovery initializes full-source tensors at the frozen rank before slicing.
Native checks and evaluation use the same rank when resetting source state. Optimizer step counters survive ordinary transfer.

## Gotchas

Merging changes BF16 operation order. Algebraic fusion does not establish native output parity.

## Tests

Run `test_head_adapters.py` and `test_ffn_learning.py` with CUDA hidden.
Native BF16 checks remain separate scientific gates.
