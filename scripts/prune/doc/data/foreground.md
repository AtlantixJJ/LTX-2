# data/foreground

## Objective

Check canonical person mattes and reviewed GT region coordinates.

## Data flow

The recorded crop and source matte replay the canonical mask. The public corpus pooler produces transient FP32-promoted weights.

## Organization logic

Replay reads full-resolution source frames one at a time. It uses the public crop owner and the producer interpolation.
Reviewed source-pixel boxes expand by 25 percent on each side. Crop mapping, clipping and matte intersection precede pooling.

## Invariants

Time and spatial scales are 8 and 32. Exclude c0 from loss. A GT box must meet the 32-pixel minimum. Keypoint indices are never guessed.

## Gotchas

The pooled weights approximate VAE regions. Invalid frames receive no auxiliary loss. Absent regions cannot pass required quality checks.

## Tests

Run `test_foreground.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.

## Missing canonical masks

`repair_canonical_mask` creates only missing canonical capture mattes.
It verifies the crop against `CaptureManifest`, then calls the corpus producer's public `read_cropped_capture_masks` reader.
The lossless corpus writer preserves those pixels and the source fps.
It leaves existing masks intact. Preparation verifies the resulting canonical pixels against the source matte before freeze.
This repair requires no guide render, checkpoint load or alternate crop calculation.
