# evaluate/joint_final

## Objective

Measure completed V3 masks with saved full-clip inputs. Keep training provenance immutable.

## Usage

Run from LTX-2 in conda environment ltx. Use a free physical GPU from 4–6.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.prune.evaluate.joint_final --study ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261019_A/configs/frozen_study.json --stage infer --arm final --gpu-id 5
```

Use `--arm selection` for the aligned selection endpoint. Use `--stage deployment` for final export, parity and timing.
The deployment stage waits for the matching final inference receipts. See the [run guide](../usage.md).

## Data flow

The completed study supplies its original split, context, component paths, exact schedules and seeds.
The checkpoint supplies fixed binary masks and split FP32 adapters. Native full-clip owners construct all inputs.
Saved predictions and per-case receipts enter the report. This module never optimizes or resumes training.

```mermaid
flowchart LR
  S("Saved capture and noise") -->|"fixed selection or final masks"| M["Native model and split adapters"]
  M --> P("Prediction") --> Q(["Paired error and export parity"])
```

## Organization logic

Inference covers 108 test cases: six actors, two views, three seeds and three direct schedules.
Selection and final use the same aligned masks. Their adapters differ only through the recorded recovery.
The owner verifies checkpoint bytes and records the current evaluation code separately from training code.
Native input manifest hashes must equal the saved r3 native baseline for every case.

The final split bundle retains original FFN down reduction coordinates and separate adapter GEMMs.
Parity covers the first view of all six test actors, seed 2042 and all three schedules: 18 comparisons.
Maximum latent difference must not exceed 0.02. A failure stops timing qualification.
Timing uses one held-out source and both masked/bundle/masked and bundle/masked/bundle orders.
Each load uses two warmups and seven timed calls. Report bracket drift and both speed ratios.

## Invariants

Preserve every checkpoint, original unit ID and clean first latent frame.
Never merge adapters for a masked-versus-bundle comparison. Preserve native Q/K normalization.
Reject changed predictions and inputs. A completed numerical result does not establish perceptual quality.
Deployment timing includes the full-width FFN down projection and split adapters.

## Gotchas

The six test actors were scored in earlier studies. They are held-out from training, but are not a new blind population.
One timing source does not establish a hardware-wide or length-wide speed result.
Historical training code can differ from the evaluation owner. Both identities remain explicit.

## Tests

Run existing mask/adapter and V3 state tests. Check the independent final input hashes and all 18 native parity cases.
The full native run supplies the scientific acceptance evidence.
