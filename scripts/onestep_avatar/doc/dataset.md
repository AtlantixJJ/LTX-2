# `dataset.py` — find and load encoded video data

Status: path, filename, and crop-record functions are **Current**.
`load_training_master` is the shared checked reader extracted from `train.py`.
`ClipStore` reads the new fixed-video list. Engine integration remains pending.
Source already exceeds 100 lines.

## Objective

Own corpus paths, background-dependent filenames, version constants, crop-record access, and atomic writes.
Add one checked video reader for both modes.
Evaluation and pruning must not import training helpers to read data.

## Data flow

```mermaid
flowchart LR
  M[("video list and encoded masters")] --> L["ClipStore.load"]
  L --> V["check shapes and frame rate"] --> C(["encoded frames and input records"])
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef disk fill:#eceff3,stroke:#6b7280,color:#1f2937;
  classDef out fill:#ece0f8,stroke:#7048a0,color:#26123f;
  class L,V proc;
  class M disk;
  class C out;
```

This diagram is Proposed.
Current `load_master` reads schema-two continuous masters.
`dataset.load_training_master` checks arrays and frame rate.
`ChainStore` checks capture/guide pairs.

## Organization logic

Import `WORKSPACE_ROOT` from the package marker. Build the default corpus path
from that constant; the reader's location never determines the workspace root.
Filename and data helpers retain their existing artifact paths after a move.

Current functions follow these rules:

- `bg` uses filenames without a suffix. `white` uses `_white` filenames.
- Alpha and cropped masks do not depend on background choice. Reject unknown choices.
- `GUIDE_COMPOSITING_VERSION=2` records the pixel-compositing rule. Missing or older values fail the current-guide check.
- `CaptureManifest` reads the saved crop box. It does not calculate a replacement crop.
- `capture_master_latent_frames` reads the actual encoded frame count.
- `ClipRef` resolves video/view paths and person IDs for train/evaluation groups.
- `atomic_write` writes to a temporary sibling file and replaces the destination only after success.
  On failure, remove the temporary file and keep the old destination.

`load_training_master(path)` checks a nonempty floating array `[C,F,H,W]` and finite positive frame rate.
Keep the current loader calculations.
`ClipStore.load(source_id, require_guide)` returns `EncodedVideo`: video/person/group IDs,
capture, optional guide, frame rate, encoding records, and content hashes.
It loads one video when needed. It does not select frame ranges or history policies.

The constructor checks fixed-list identity and groups. `verify(require_guide)`
loads and checks every selected video before model loading. Each read verifies the
encoded-file hash, shape, frame rate, background, and recorded encoding fields.
D1 also verifies current render bytes, sidecar identity, compositing version,
and matching capture/guide crop and VAE records. D0 never opens guide artifacts.
Keep subset validation imports lazy so filename access does not load model code.

For D1, check matching shapes, frame rate, background, crop, and VAE records.
Check file hashes and metadata before model loading.
A valid file schema alone does not prove correct encoding.
Fail on missing producer output.
D0 loading does not read guide data.

### Checked-reader procedure

The proposed `ClipStore` receives a verified video-list entry and the chosen background.
Resolve its relative view path against the requested corpus root.
Read capture encoding through `load_training_master`: a nonempty floating `[C,F,H,W]`
array and finite positive frame rate.
Match its content hash, dimensions, crop identity, VAE identity, and encode records to the entry.
For D1, resolve/read the guide and require the same geometry, frame coverage, frame rate,
crop, background, VAE convention, and current compositing/encode records.
For D0, do not open the guide or fail on its absence.

Return the unsliced capture, optional guide, frame rate, IDs/group, and checked identities.
Keep their encoded values/dtype unchanged. The mode selects frames and builds tokens later.
Loading a video never selects its training start, resets positions, primes history, or rebuilds a missing producer artifact.
If a checked entry changes on disk, report the path and failed field rather than skipping it.

Keep filename/path functions usable in `argavatar` without heavy model imports.
Use lazy tensor imports where needed.
Never import the ARGAvatar renderer or training CLI here.

## Invariants

- Each filename and crop rule has one owner.
- Frame rate is required because it affects model positions.
- Train/evaluation groups come from the verified fixed video list.
- Masters remain continuous encodings. Do not encode each segment again.
- Loading does not allocate history caches or select causal blocks.

## Gotchas

Keep old alpha `.npy` reading until its usage and conversion checks pass.
Use person IDs that prevent the same person from entering both data groups.
A guide file can exist and still have outdated encode records.
Equal shape alone is insufficient for D1.

G9 is caused by frame selection, not this reader.
Do not replace a stored frame with a new image encoding silently.

## Tests

Current tests include `tests/test_dataset.py`, `test_geometry.py`, `test_mask_video.py`,
and master/pair checks in `tests/test_train.py`.
After extraction, keep useful array/frame-rate and atomic-write tests here.
Check that this file imports no training code.
[V8](verification.md) checks that conversion preserves videos and master hashes.
Worked check: capture is `[128,17,8,8]` at 30 fps and guide is `[128,16,8,8]` at 30 fps.
D1 fails on frame coverage; D0 can load capture without opening guide.
Returning the 17-frame master does not itself select a causal block or a random-start segment.
