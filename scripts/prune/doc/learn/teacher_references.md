# learn/teacher_references

## Objective

Produce and validate shared immutable original-teacher evidence.

## Data flow

Components, exact input, noise, context, geometry, code and runtime policy form a key. Student state is absent.

## Organization logic

The producer saves native BF16 velocity and stock clean output once. Readers verify file and tensor hashes.
A 32 GiB disk cap protects pinned evidence. A 2 GiB CPU cache evicts only reproducible hot entries.

## Invariants

The teacher has no recovery adapter or pruning attenuation. A missing reference needs its own declared teacher phase.

## Gotchas

Official velocity training does not require teacher forwards for every training draw.

## Tests

Run `test_teacher_references.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.
