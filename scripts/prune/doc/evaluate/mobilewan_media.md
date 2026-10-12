# MobileWan comparison media

## Objective

Decode the saved removal evidence and produce 24 synchronized six-panel videos with linked poster images.
Retain each full source RGB and white-matte video, cropped capture RGB and every individual decoded video.
This stage runs no transformer and creates no scientific predictions.

## Usage

Run from `LTX-2` in the `ltx` environment after complete removal evaluation:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m scripts.prune.evaluate.mobilewan_media \
  --study ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011/configs/study.json \
  --gpu-id 4
```

Check GPU occupancy before launch. Use `setsid`, an external timeout and a per-run log for persistent execution.
Outputs are `media/index.json` and one directory per selected view hash.
Each view directory retains references and eight official-step comparison directories.
See the [run guide](../usage.md#mobilewan-comparison-media).

## Data flow

Validate all 96 saved predictions, their exact sigma coverage, paired controls, input identities and file hashes on CPU.
Verify the study and frozen train-mask identities before opening a decoder.
Open the pinned LTX-2.5 video VAE through the public decoder session.

The existing capture replay checks the recorded crop, white objective, fps and source fingerprint.
It applies the original producer's crop and matte. This stage defines no replacement preprocessing.
Compare saved GT bytes with the recorded capture encoding.
Decode GT and each prediction with a fresh generator seeded at 42.

## Organization logic

Keep the full source RGB and matte files with content hashes.
Check the RGB against audited membership. Check the matte against the recorded producer fingerprint during public capture replay.
Atomic copies publish only complete bytes and reject different existing copies.
Recheck both source hashes after replay. A changed source cannot become valid comparison evidence.
Save cropped capture RGB and decoded GT once per case.
Save all four full decoded arms at every sigma.
Each GT video records its saved latent hash. Each arm video records its prediction hash and requested removal fraction.

Use the existing RGB panel renderer with six explicit roles in two rows by three columns.
The order is capture RGB, capture VAE, dense D0, then 10%, 23% and 33% removal.
Every moving panel has the same complete source-frame map and recorded playback rate.
The renderer preserves aspect ratios with padding and records placement, pixel identities and readable labels.
Each comparison receives a poster image and a rendering record.
Probe each actual comparison MP4 for complete frame count, fps and recorded dimensions.
Read the full poster PNG and check its dimensions. Recheck both output hashes after these probes.
Save these verified fields in the rendering record before publishing the index.
Publish the complete index only after all 24 comparisons exist.

## Invariants

The media stage cannot regenerate missing predictions or change selected cases.
Exact sigma and original scheduler step are saved as common settings, not substituted by rounded input values.
The paired GT, capture and predictions must use the same shape and clean frame.
Panel titles distinguish original RGB from VAE decoding.
Metric scores stay in scientific figures; video titles state roles and removal percentages.
All source frames remain available in individual videos.
The matte hash pins the file used for replay; the original encode record supplies its historical size/time fingerprint.

## Gotchas

A visually different crop or matte can mimic conditioning damage.
Reject changed producer fingerprints before interpreting alignment.
The full source RGB has its original background; the comparison RGB uses the recorded white-matte objective.
A comparison video supports inspection. It does not by itself establish a perceptual-quality conclusion.
The final report must inspect current videos and cite actual evidence.

## Tests

`tests/test_mobilewan_media.py` checks full shared frame coverage, panel order, frame times and readable rendering.
It uses differing aspect ratios and checks the recorded aspect-preserving padding policy.
Incomplete saved evidence fails before decoder creation.
Atomic reference tests check full-byte preservation, changed-source rejection and interrupted-copy cleanup.
A CPU video fixture probes an actual encoded MP4 and PNG, then rejects incorrect declared frame coverage.
Native VAE videos and wide/narrow poster inspection remain separate gates.
