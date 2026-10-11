# `experiments/saved_probe_metrics.py` — model-free saved-probe measurements

## Objective

Compute the short and long saved-probe measurements and their report row
names. The fixed frame and block list belongs to this study. It is separate
from the general metrics and from training's full-frame loss. The module reads
evidence, measures it and publishes JSON. It never generates missing outputs,
and it never opens a transformer, text encoder or VAE.

## Usage

Run the module directly on CPU, from `LTX-2` in the `ltx` environment. It needs
no queue job. See
[run guide §9, other studies](../../configs/README.md#other-studies).

```bash
cd LTX-2
"$PY" -m scripts.onestep_avatar.experiments.saved_probe_metrics \
  --saved-metrics <PROBE_DIR> [<PROBE_DIR> ...] [--long-metrics]
```

| Argument | Meaning |
|---|---|
| `--saved-metrics DIR ...` | Required. Saved probe directories, each with its original `manifest.json`. |
| `--long-metrics` | Use long mode (any odd frame count) instead of short mode (17 frames). |

Output in each directory: `metrics.json`, or `metrics_long.json` in long mode.
The raw encodings and media do not change. Ordinary evaluation accepts neither
option.

## Data flow

```mermaid
flowchart LR
  M[("saved probe manifest and masters")] --> P["deduplicate view and seed"]
  P --> H["check encoding file SHA"] --> T("aligned output, capture and guide")
  T --> C["short or long metrics"] --> J[("atomic metrics JSON")]
  J --> R(["saved-result report reader"])
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef disk fill:#eceff3,stroke:#6b7280,color:#1f2937;
  classDef tensor fill:#dff2e0,stroke:#43834a,color:#16341b;
  classDef out fill:#ece0f8,stroke:#7048a0,color:#26123f;
  class P,H,C proc;
  class M,J disk;
  class T tensor;
  class R out;
```

The original `manifest.json` of each directory names the masters and the saved
encodings.

## Organization logic

### Evidence and rows

1. Visit the manifest videos in their original order.
2. Read each seed from the video files. If it is missing, use the manifest
   seed.
3. Process only the first occurrence of each `(view, seed)` pair.
4. Load the capture and the guide with the public master reader.
5. For every encoding, check the recorded file SHA before the
   `weights_only=True` load.
6. Require one `[1,C,F,H,W]` tensor. Align the masters to its first F frames.
   Call the one metric function.

Missing, changed or invalid evidence fails without a call to a producer.

Each row keeps the probe's three-level view label and the seed. Short rows add
sigma, arm, the latent file SHA and the recorded epsilon SHA. Both summaries
keep the probe, checkpoint and rows. Short mode also keeps the off-condition,
the model variant and the schedule of the first video. The JSON is written
atomically with `allow_nan=False` after all rows succeed. A successful
recomputation can replace existing metrics. A failure leaves the earlier JSON
unchanged.

### Exact calculations

The tensors must be finite and equal `[C,F,H,W]`, with odd `F >= 3` and spatial
dimensions of at least two. They are converted to fp32. Short mode requires
F=17. Long mode uses the actual odd F. The generated blocks are `[1,3)`,
`[3,5)`, ..., `[F-2,F)`.

Both modes report `c0_exact` and the per-block MSE of the output against the
capture. Long mode adds the frame count, the per-block MSE of the guide against
the capture and the output/capture spatial-detail ratios. It omits the short
aggregate scores on purpose.

Short-mode definitions:

- **Capture and guide MSE** average the squared differences against the
  capture over `[1,F)`. `c0` counts only in the exact-equality report, unlike
  training's full-frame loss.
- **Detail** is the mean absolute vertical difference plus the mean absolute
  horizontal difference, on `[1,F)`.
- **Motion** averages the absolute changes between successive generated
  frames, 1→2 through 15→16. It excludes c0→1.
- **Motion and detail ratios** divide by the capture measurements.
- **Seams.** First calculate squared transitions averaged over channel and
  space. Across-block transitions enter frames 3, 5, ..., 15. Within-block
  transitions enter frames 2, 4, ..., 16. `seam_ratio` divides the across mean
  by the within mean. `capture_seam_ratio` does the same for the capture.
  Neither includes c0→1.

Every zero denominator raises an error before publication.

### Worked checks

Use `[C,F,H,W]=[1,17,2,2]` and capture values `capture[0,f,h,w]=f+h+w`. The
output equals the capture, and the guide equals the capture plus two.

- The generated capture MSEs are zero and the guide MSE is four. `c0` is exact.
- The temporal differences are one, and the directional spatial differences are
  one.
- The motion and detail ratios, the guide detail ratio and both seam ratios are
  one.
- A change of only the output `c0` changes `c0_exact`. The generated-frame MSE,
  motion, detail and seams do not change.

With the same pattern and five frames in long mode, there are two blocks. The
output MSE is `[0,0]`, the guide MSE is `[4,4]`, the detail ratio is `[1,1]` and
`latent_frames` is five. A spatially constant capture has zero detail and
cannot publish a ratio. Duplicate view and seed videos give only the rows of
the first video's encoding.

## Invariants

- Keep the generated-frame exclusions, the blocks, the fp32 arithmetic and the
  row names. Do not substitute the training loss.
- Check the output file SHA before loading. The shared corpus owns master
  reading.
- A complete successful measurement comes before atomic publication. A failure
  cannot replace an earlier complete file with partial results.
- Report code reads measurements. This owner starts no job.

## Gotchas

- Motion measures how much content changes, not whether the action is right.
- Detail and seams do not prove identity or quality.
- This reader pins the output file bytes and forwards the epsilon SHA as
  metadata. That field alone does not check the original noise file again or
  prove matched inputs for a new comparison. Scientific interpretation still
  needs matched source and video evidence.

## Tests

`tests/experiments/test_saved_metrics.py` checks short and long results for an
identical capture, scores for a different guide, file-SHA refusal, CLI
ownership, preservation of earlier metrics on failure and zero-denominator
refusal. Ordinary CLI and boundary checks refuse study flags in ordinary
evaluation before writes. These prove the saved calculations, not native
generation or quality acceptance.
