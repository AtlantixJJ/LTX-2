# `score/hooks.py`

## Objective

Attach functional head/FFN masks and collect intermediate activations without modifying weights.

## Data flow

Transformer attention/FFN modules and optional masks -> removable forward hooks.

## Organization

MaskAttachments is a context manager that removes all hooks on exit.

## Invariants and gotchas

Attention masks sit before to_out[0]; FFN masks sit before net[2]. Match mask
width to actual module width and never leak hooks between comparisons.
`read_mask_artifact` checks model key, checkpoint fingerprint, complete mask
families, exact widths, binary finite values, and nonempty branches before use. Active D0 consumers also require `candidate_format=whole_clip_d0_mask_v1`, task, bidirectional attention, clean-frame conditioning, calibration views, sigmas and baseline-manifest provenance. Historical readers opt out explicitly.
`require_native_heldout_scope` rejects use of a calibration view as held-out
validation and rejects sigma levels absent from the mask's calibration list.

## Verification

Check [`tests/test_hooks.py`](../tests/test_hooks.py). Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
