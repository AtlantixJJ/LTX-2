# evaluate/media_common

## Objective

Share fixed-region pixel metrics, source RGB reconstruction and saved-media verification.
Latent decoding and model lifetime remain with each evaluation owner.

## Data flow

`source_pixels` checks the recorded root and replays the capture crop and white matte through the public avatar media owner.
`region_lpips` receives matched FCHW pixels, fixed GT weights and optional reviewed boxes and validity flags.
`pixel_metrics` receives the same matched pixels and weights.
`saved_video`, `preserve_source` and `verify_comparison` publish or verify complete output artifacts.

## Organization logic

Exclude frame zero from pixel and LPIPS metrics.
LPIPS uses valid GT coverage, boxes at least 32 pixels wide and high, white masking and 256-square interpolation.
It permits only bounded interpolation roundoff, then clamps to the metric's RGB range.
Missing valid regions return `missing_coverage` with no score.
Pixel MSE divides weighted squared error by three times region mass.
PSNR uses RGB values divided by 255; zero error has no finite PSNR value.
Motion amount measures absolute changes across generated frames using the union of adjacent weights.
It measures amount, not correctness, of motion.

Source preservation copies into a temporary file, verifies both source and copy, then publishes through a hard link.
A concurrent destination must match. Saved comparisons require actual video dimensions, frame count, fps, PNG dimensions and hashes.

Worked example: identical generated pixels produce MSE zero and PSNR `None`.
Empty required pixel coverage raises an error; absent LPIPS coverage returns a named missing result.

## Invariants

Use matched pixel geometry and fixed GT coverage. Do not derive boxes from generated images.
Preserve source aspect ratio, frame ordering and fps in the calling renderer.
Verify complete saved artifacts before publishing the rendering record.

## Gotchas

A finite pixel metric cannot establish identity or correct motion.
Source RGB differs from capture VAE decoding; labels must identify which reference is shown.
Metric and media helpers do not load a transformer or decode latents.

## Tests

`tests/test_media_common.py` checks source preservation, pixel metrics and rendering validation.
Learned-head media tests verify region geometry and missing coverage.
Saved native panels require separate reproduction evidence.
