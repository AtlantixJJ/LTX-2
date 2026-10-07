# `stock_parity.py` — compare the stock video sampling path

Status: Implemented and CPU checked; native acceptance remains open.

## Objective

Run the actual stock video state builder, factory denoiser and Euler loop against
the bidirectional sampler. Use one resident native video-only transformer and
fixed inputs. Preserve a repeated stock control and intermediate calls so a
final difference can be traced instead of hidden by a loose tolerance.

## Data flow

```mermaid
flowchart LR
  I[("supplied image and fixed text record")] --> P["prepare"]
  P --> N("fixed image, text and noise")
  N --> S["euler_denoising_loop"]
  S --> O[("stock tokens and call traces")]
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef disk fill:#eceff3,stroke:#6b7280,color:#1f2937;
  classDef tensor fill:#e0f4e8,stroke:#3b8061,color:#123524;
  class P,S proc;
  class I,O disk;
  class N tensor;
```

The bidirectional branch reads the same in-memory image, text and noise and
writes its own tokens and traces. Each branch uses the same native x0 model.
The decoder reads saved encodings only after the transformer has been released.

## Organization logic

Before GPU work, reject an existing output directory, invalid steps or frame
counts, malformed image metadata and changed fixed text files. Read the actual
one-image master through the shared dataset reader. Require one RGB frame,
`input_role=supplied_image`, native channel count and the current VAE fingerprint.
Read the fixed preview record as historical input evidence: validate its saved
software integrity, its text file hash and actual tensor hash. Do not restamp
the original record. Recover the literal prompt from its evaluation arguments.
Resolve the real dev checkpoint and construct the exact schedule with
`LTX2Scheduler().execute(steps=N)` without a latent argument. At least two
intervals are required here; do not substitute the one-step special case.

Snapshot current source/runtime and input/weight hashes. Publish a predeclared
protocol before native model calls. The repeated stock control must be exactly
equal. Positive-interval inputs and predictions must be exactly equal when all
modality fields match. The terminal endpoint comparison is reported separately:
the package returns the prediction, whereas stock reconstructs with rounded
velocity. No guessed final-output tolerance makes a discrepancy pass.

Build the native grid for the requested encoded frames and image spatial shape.
Draw one bf16 token noise tensor with the native Gaussian noiser on an empty
state. Replay the same seed through `create_noised_state`, with
`VideoConditionByLatentIndex(image,strength=1,latent_idx=0)`. Assert that the
assembled state is exactly the saved noise with the image tokens restored.
This is pure-noise D0 sampling; there is no guide mixing.

Run stock, repeated stock and bidirectional sampling in one resident model
session. Stock uses `FactoryGuidedDenoiser` with cfg one, STG zero, modality
scale one and no rescale, `BatchSplitAdapter(max_batch_size=1)`, the native
`EulerDiffusionStep` and `euler_denoising_loop`. The custom branch uses the
ordinary bidirectional sampler with float32 global sigma, matching stock.
Save each call's input, prediction, sigma, token timesteps, positions and image
marks. Record actual forward counts, elapsed time and peak CUDA memory. Trace
copying is included in elapsed time; these timings are diagnostic, not a
deployment benchmark. Release the model before decoding all three encodings
with identical fresh decoder seeds. Compare raw tensors and decoded float
pixels numerically. Save synchronized readable media through the shared renderer.
Use 448-square panels for this producer. Select the compact layout with the exact
titles before rendering. A one-column result is 464 pixels wide and has at least
16-pixel text at its natural width; it must not rely on browser upscaling.
Write the final record last, after rechecking input/software hashes.

## Invariants

- One transformer load supplies all compared paths; no adapter or fused weights.
- The first image, prompt context, noise, grid, fps and schedule stay fixed.
- A repeated stock disagreement is a failed control, never a tolerance estimate.
- The actual stock loop runs; the module does not reconstruct stock arithmetic.
- Historical text preparation remains attributable to its original software.
- Intermediate exact equality is distinct from terminal endpoint agreement.

## Gotchas

The outer `TI2VidOneStagePipeline` always instantiates audio. This check uses its
public stock video sampling components with audio absent and the same video-only
loader as the renderer. It does not certify joint audio-video execution or the
outer pipeline's RGB/text preparation. The stock pipeline source is included in
the manifest to bind the reviewed calls. A seven-frame trained product pilot
is separate from this 17-frame base diagnostic. Ordinary product global-sigma
precision and adapter effect require their own checks; this matched stock check
explicitly uses float32 global sigma. Native E1 acceptance remains open until
raw, decoded and intermediate evidence is inspected.

## Tests

CPU controls use the real stock builder/denoiser/loop and a deterministic x0
model. They require identical initial noise and c0, repeated output equality,
matched calls before the terminal step, and an explicitly measured terminal
difference. Reject tampered text, video-prefix image bundles, invalid schedule
requests and changed software before publication. Native acceptance uses 17
encoded frames, 129 decoded RGB frames and the same saved input hashes.
