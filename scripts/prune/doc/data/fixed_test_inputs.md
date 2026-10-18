# data/fixed_test_inputs

## Objective

Freeze one complete test split and its exact saved noise before any paired model evaluation.
The six actors have two full F28 views each. Three direct schedules give 36 cases per checkpoint and method.

## Usage

The standalone owner runs from `LTX-2` in `ltx` on free GPU six:

```bash
conda run -n ltx python -m scripts.prune.data.fixed_test_inputs \
  --study ../expr/refiner_prune/2.5/progressive_clip01_ste3_reset50_20261017_from_r13u0/configs/frozen_study.json \
  --gpu-id 6
```

Run this after study preparation. Check `nvidia-smi` before running it.
The GPU constructs native input tokens and epsilon. This command opens no transformer or decoder and runs no model forward.
Outputs are native input bundles, saved empty context, `comparisons/fixed_test_manifest.json`, and its immutable pin receipt.
The receipt pins both the manifest file bytes and its canonical content digest.
See the [run guide](../usage.md#clipped-scores-with-survivor-resets).

## Data flow

Read the verified study, source split, and actual empty-context tensor.
The shared native input owner verifies capture, RGB, metadata, geometry, and encoding records.
It saves epsilon for seed 2042 once per view and reuses those bytes for all schedules.
The manifest pins capture/GT, RGB, clip metadata, matte, crop, context, component, noise, software, and numerical identities.

## Organization logic

Require actors 51, 800, 801, 178, 791, and 195 with two recorded F28 views each.
Reject duplicate views, F18 crops, unexpected geometry, or learning-actor overlap.
Use exact direct schedules at 0.9489720463752747, 0.7663628458976746, and 0.48016369342803955, each ending at zero.
Save full-source token and clean-prefix hashes. Pin native input manifests and actual epsilon tensor hashes.

Each case ID binds the source pin, actor, view, schedule, and noise seed.
Publish the complete manifest atomically, then publish its immutable pin.
Repeat preparation reuses the same epsilon and rejects any changed completed artifact.
`validate_manifest` requires the previously pinned content digest before checking membership or files.
This prevents a self-consistent replacement manifest from changing the noise or target silently.

## Invariants

Both methods read these same saved inputs at every checkpoint.
The metric is FP32 capture-latent MSE after excluding the clean first latent frame.
Aggregate the two views and three sigmas equally within each actor, then average the six actor means equally.
The manifest declares no training, selection, or stopping use.
Repeated scoring does not make this split an untouched final test.

## Gotchas

The shared input bundle can contain the full training sigma grid; the frozen comparison uses only the three specified schedules.
Matching a seed alone cannot replace the saved epsilon hashes.
The historical four-actor development panel is a different input identity and cannot populate this manifest.
Saving inputs does not establish that a checkpoint has predictions for all 36 cases.

## Tests

`test_fixed_test_inputs.py` uses tiny saved fixtures without model execution.
It checks immutable replay, changed noise bytes, missing cases, changed GT/context/schedules, and development-cache rejection.
It rejects F18 inputs and training actors. Native input reconstruction remains a separate execution check.
