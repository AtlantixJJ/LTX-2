# `data/chunk_states.py`

## Objective

Persist exact noisy calibration states and teacher targets for all scorers.

## Data flow

Patchified LatentState, x0 target, and ChunkStateMeta -> atomic format-2 .pt record plus index.

## Organization

make_state delegates to refine_core; load_record reconstructs LatentState; chunk_token_mask identifies emitted AR tokens.

## Invariants and gotchas

Format 1 is refused because its tensors used wrong geometry/fps; the error
points to `data.source_target --build-calibration`. The chunk mask excludes
the index-0 keyframe even though denoise_mask includes it.

## Verification

Check [`tests/test_chunk_states.py`](../tests/test_chunk_states.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
