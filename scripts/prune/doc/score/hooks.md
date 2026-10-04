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
Native readers also call `data.whole_clip.validate_native_provenance`: seed, VAE, manifest content hash, guidance, geometry, dtype, text context and calibration inputs are mandatory and bound to the pinned calibration manifest. Parity and ablation additionally bind the selected baseline distribution.
`require_native_heldout_scope` rejects a calibration actor across views, clips,
Parts and symlink aliases using the bare DNARender identity in
`data.whole_clip.actor_identity`. A different clip directory does not establish
a different subject. It also rejects identical capture content under another
name using the selected baseline's capture hash, and rejects sigma levels absent
from calibration.

## Verification

Check [`tests/test_hooks.py`](../../tests/test_hooks.py). Run `python -m pytest scripts/prune/tests -q -m 'not gpu'` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For model-facing changes, run the native checks in [VALIDATION](../VALIDATION.md).
