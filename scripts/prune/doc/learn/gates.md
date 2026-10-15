# learn/gates

## Objective

Learn scalar head gates at the existing projection boundary.

## Data flow

FP32 logits enter sigmoid gates. A legal binary decision enters the same boundary during refinement.

## Organization logic

The dynamic program keeps one best score for each cumulative retained count. It sorts tied heads by original identity.
Each update freezes its binary decision. Each microbatch builds a new sigmoid graph. The provider returns that graph during recomputation.

## Invariants

All logits start at exactly +3. Allowed counts and the total budget must be feasible. Native gates and text heads stay intact.

## Gotchas

Sigmoid(3) is approximately 0.952574. This initial network differs from the teacher. An all-one override provides the native control.

## Tests

Run `test_learned_gates.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.

## Progressive gate removal

The progressive policy composes free sigmoid gates with exact removed zeros during the soft phase.
Snapshots measure actual mixed amplitudes and include free/removed counts. Legacy gate phases retain their existing semantics.
