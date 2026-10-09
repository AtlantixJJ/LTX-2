# `metrics.py` — measure encoded and RGB outputs

## Objective

Compute reusable measurements from supplied tensors. Preserve the exact fp32
arithmetic, clean-first-frame denominator, subject mask and LPIPS batching.
No function loads transformer, text-encoder or VAE weights.

## Data flow

```mermaid
flowchart LR
  T("matched output and reference tensors") --> M["encoded_metrics or RGB measurement"]
  M --> R(["metric values"])
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef tensor fill:#dcfce7,stroke:#4d9863,color:#143d23;
  classDef output fill:#eee0ff,stroke:#8156b4,color:#39205b;
  class M proc;
  class T tensor;
  class R output;
```

## Organization logic

`encoded_metrics` requires equal B,C,F,H,W tensors. Convert both to fp32,
subtract, square, then average every element. Keep c0 in the denominator.
Average B,C,H,W separately to return one MSE per encoded frame. RGB checks
require matching finite nonempty F,3,H,W floating pixels in [0,1]. RGB PSNR
is `-10*log10(MSE)`; exact matches return `None` PSNR plus an explicit match
flag instead of infinity. Subject scores average only an aligned, nonempty
boolean mask after averaging the three colour channels.

Subject masks read the public lossless mask codec. Keep the requested prefix,
scale uint8 by 255, dilate with a five-cell max pool, resize by nearest-neighbour
and threshold at .5. Transition scores use the union of adjacent foreground
masks and mean absolute RGB change; clip an empty denominator to one.

LPIPS inputs map [0,1] to [-1,1] in each explicit batch. Require one finite score
per frame. Frame scores retain each score; scalar distance sums native batch
scores and divides by the actual total frame count. The supplied model owns
its weights. Neither helper creates a VAE or model session.

Worked check: prediction has four scalar elements, target has zeros, and
prediction values are [0,0,1,1]. Encoded MSE is .5 including the clean first
frame. For equal RGB pixels, MSE is zero, PSNR is None and exact-match is true.

## Invariants

Array shape and scope decide the denominator. Keep float conversion before
subtraction. No hidden first-frame exclusion or study boundary selection.

## Gotchas

A zero PSNR denominator has an explicit status. A supplied subject mask
measures only its pixels. LPIPS is a supporting perceptual measurement.

## Tests

Retain the numerical fixtures in test_evaluate.py and test_rgb_measurements.py,
including malformed RGB, mask alignment, frame scores and batch weighting.
Caller fixtures in ordinary evaluation and comparisons use this one owner.


### Subject RGB and perceptual measurements

`subject_mask` reads the optional saved lossless capture mask for RGB QA.
Absent files return no mask. Require positive requested dimensions/count and
enough uint8 F,H,W frames. Take the requested prefix, divide by 255, apply
5x5 maximum pooling at stride one with two-pixel padding, resize with nearest
neighbors to the actual RGB height/width, then threshold strictly above 0.5.
This preserves the old renderer's two-cell dilation; it is a QA rule, not a
training mask or new crop. A single interior foreground pixel selects 25 cells
before resizing and 100 after a twofold resize in each direction.

`rgb_metrics` also returns full-frame MSE and PSNR over all F,3,H,W values.
Compute squared error once. Per-frame scores average C,H,W; the full-frame
score averages F,C,H,W before the logarithm. Do not average per-frame PSNR.
Zero aggregate MSE returns null PSNR and `all_exact_match=true`.

`subject_rgb_metrics` accepts aligned floating F,3,H,W inputs in [0,1] and a
boolean F,H,W subject mask. Check all shapes, values and nonempty selected
pixels before measuring. Average squared error across RGB channels, then across
selected pixels/frames. Return MSE, PSNR and exact-match status. Zero MSE has
no finite PSNR (`null` plus exact-match), rather than the old renderer's 99-dB
sentinel. The mask is supplied checked evidence; this function invents no crop,
dilation or source alignment. A uniform 0.5 difference gives MSE 0.25 and
PSNR 6.0206 dB regardless of selected area.

`lpips_frame_scores` returns ordered Python floats, one per aligned frame,
using an already-loaded perceptual model. Its default batch is sixteen,
matching the historical full-resolution analyzer; it does not exclude c0 or
average frames. The caller explicitly selects any excluded frames. One shared
batch iterator owns RGB validation, [-1,1] conversion, model calls and output
checks for this helper and `lpips_distance`. Neither loads or downloads weights.
Require an integer positive batch, a tensor output with one finite score per
frame and preserved order across the final short batch. The helper's per-frame
list must match the historical FHWC-to-FCHW scoring path exactly on controlled
pixels. That parity verifies measurement plumbing, not native VAE or identity.

`lpips_distance` takes an already-loaded perceptual model and the same checked
RGB inputs. Reject empty/nonfinite/out-of-range inputs or nonpositive batch
size before calling the model. Convert each frame batch to [-1,1] on the stated
device. Sum one model score per frame, then divide by total frame count. Require
finite scores and exactly one score per frame, including a final short batch.
It opens no model session and downloads no weights. Preserve its original
per-batch tensor sums on the model device,
rather than changing scalar rounding through a Python per-frame reduction.
Record model identity at the caller. LPIPS measures perceptual difference, not correctness of motion or
identity. Both helpers stay outside report code; historical rendering caller
migration remains pending until layout/decoder execution also moves.

