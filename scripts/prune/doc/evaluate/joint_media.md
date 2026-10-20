# evaluate/joint_media

## Objective

Compare A, its selection endpoint and r3 with matched native and capture videos.

## Usage

Run from LTX-2 in ltx. Assign one free physical GPU from 4–6.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.prune.evaluate.joint_media --study ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261019_A/configs/frozen_study.json --gpu-id 4
```

The owner waits for completed selection/final inference and an idle assigned device.
It never starts or stops training. See the [run guide](../usage.md).
Use `--shard-index I --shard-count N` to assign disjoint source-view indices modulo N.
Each shard writes its own result index. An external CPU reader verifies all 36 cases before publishing the combined index.

## Data flow

Saved predictions and capture latents enter the pinned VAE. Recorded RGB and replayed mattes supply fixed references and regions.
Hash checks prove matched input and noise across native, selection, final and r3.

```mermaid
flowchart LR
  S("Saved predictions and capture") -->|"native, A selection, A final, r3"| D["Pinned VAE and fixed regions"]
  D --> P("Synchronized RGB panels") --> Q(["Person LPIPS and visual review"])
```

## Organization logic

All 12 test views use seed 2042 and three exact direct schedules: 36 decoded cases.
Metrics exclude the supplied first frame. Person LPIPS uses capture-derived foreground boxes and coverage.
Whole-frame LPIPS and pixel errors supplement it. Capture VAE versus recorded RGB gives the reconstruction ceiling.
Six synchronized panels show Capture RGB, Capture VAE, native, A before recovery, A after recovery and r3.
The shared renderer preserves source frame order, fps and aspect ratios in two rows of three.
Original videos remain available beside the comparison and poster.

## Invariants

Never infer a region from generated pixels. Reject changed prediction bytes or mismatched saved noise.
Keep RGB and VAE roles explicit. Do not treat LPIPS as proof of identity or action correctness.
Do not change model weights, masks, schedules or the frozen checkpoint.

## Gotchas

Earlier r3 work already inspected these test actors. Describe them as held-out from training, not newly blind actors.
Report finite sample size and actor clustering. A video preview is presentation evidence, not a statistical acceptance gate.
The numerical inference and media owners have separate resumable output receipts.

## Tests

Use existing region-metric and comparison-renderer controls. Require all 36 case receipts and verified frame geometry.
Inspect the delivered comparison videos and the wide/narrow report layouts.
