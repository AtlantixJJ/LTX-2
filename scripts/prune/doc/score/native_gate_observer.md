# Native gate observer

## Objective

`score/native_gate_observer.py` measures native per-token gates without changing the attention output.
It observes video self-attention only. It does not install pruning masks or adapters.
See the [study plan](../../../../../plans/2026-10-18-native-head-gate-statistics-vs-pruning.md) for the scientific questions.

## Data flow

A forward hook reads `to_gate_logits` output in each native video `attn1` module.
It evaluates the same native expression, `2.0 * sigmoid(logits)`, before any dtype conversion.
It transfers detached FP32 values to CPU; their BF16 values remain exact after that conversion.
It reduces moments in FP64 and counts in int64. No token activation survives the callback.
`finish` stacks blocks in original ID order and requires complete coverage.

## Organization logic

`moments` returns sums, squared sums, extrema for independent merging.
`summarize` separates the clean first frame from generated tokens and computes a 2,048-bin generated-token histogram.
The last bin includes gate value two. Quantile bin width is `1/1024` native gate units.
Threshold counts use strict comparisons below 0.1/0.5 and above 1.5/1.9.
Frame summaries use local F18 indices, including the separate clean frame at index zero.
`NativeGateObserver` owns hooks, one-draw lifecycle and coverage checks.

## Invariants

- Each complete draw observes exactly 48 modules and 32 original heads per module.
- The supported gate callable is the unchanged native `PytorchGatedAttention`.
- The hook returns `None`; it preserves the projection's output object.
- Gates are evaluated in native dtype. FP32 sigmoid before casting is a different measurement.
- Output tokens and attention backend remain unchanged.

For 18 frames and 1,024 tokens per frame, each generated-head histogram contains 17,408 observations.
Its clean-frame moments contain 1,024 observations. Mixing the two populations is an error.

## Gotchas

A projection hook cannot observe a model with absent native gates or compact head identities.
The observer rejects such a model. It also rejects duplicate callbacks and wrong batch/token shapes.
Native gates modulate attention outputs; they are not a causal removal score.

## Tests

`tests/test_native_head_gates.py` checks independent NumPy histograms, moments and frame exclusion.
A real tiny BF16 native attention fixture checks bit-identical outputs and matching observer/native sigmoid values.
The native owner separately checks full-model equality on each authorized GPU.
