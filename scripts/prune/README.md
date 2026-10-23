# Pruning LTX video transformers

This package removes attention heads, feed-forward channels or complete transformer blocks from LTX models.
It measures the result on saved video inputs, exports runnable checkpoints and checks quality and execution cost.
Use LTX-2.5 for new work. The checkpoint registry also retains LTX-2.3 for reproduction.

Start with the [overview](doc/overview.md) and [glossary](doc/glossary.md).
The [algorithm](doc/core_algorithm.md) explains the calculations, including joint selection with removal and return.

## What gets pruned

Training-free scores can remove video self-attention heads, video text cross-attention heads and video feed-forward channels.
Learned methods select video self-attention heads and, in joint methods, feed-forward channels.
Depth export removes an explicitly supplied list of complete blocks.

Sessions load the video-only transformer. Audio and audio-video attention are outside width selection;
width exports preserve their stored tensors. Deleting a complete block also deletes its audio tensors.
Q/K projections and their full original normalization remain intact during width export.
Residual feature width stays unchanged.

## Choose a method

- **Training-free scores** rank sampled activations and turn low scores into fixed binary masks.
  Use this route for calibration and controlled functional ablations without optimization.
- **Learned gates** train head scores and low-rank adapters, then select fixed widths and recover with adapters.
  The binary-channel variant also learns aligned feed-forward masks.
- **Clipped selection with return**, called **V3**, trains both families together.
  It periodically removes units, allows eligible units to return, and ends with aligned widths and fixed-mask recovery.
  `joint_clip_reset_v3` is the current joint selection method; its head-only and budget-matched controls have explicit arm names.

Implemented methods and completed studies have different kinds of evidence.
Read a study's report through the [workspace handoffs](../../../plans/README.md) for its measured results.
The package guides describe code behavior, with no run status embedded in recipes.

## Shared input and output

Every native pruning forward reconstructs a saved capture and saved noise at one exact `[sigma, 0]` denoising step.
The first capture latent frame stays clean because it is the image condition; other frames receive noise.
Attention sees the complete video in both directions. Optimizer inputs can use the explicitly configured continuous temporal crop.
The [input contract](doc/data/whole_clip.md) gives the identities, coordinates and reconstruction order.

All four width export modes preserve the declared kept units:

| Mode | Stored weights | Executed computation |
|---|---|---|
| `masked_full` | Full widths | Compute all heads/channels, then zero inactive outputs before projection |
| `sparse` | Full widths | Select active attention heads; projections and feed-forward widths stay full |
| `compact` | Sliced V/output and feed-forward weights; full Q/K | Execute retained widths where supported |
| `compact_faithful` | Compact weights | Pad back to original execution widths to preserve reduction geometry |

The [export contract](doc/score/export_pruned.md) connects these modes to the transformer runtime.
Learned exports can merge adapters or preserve their separate matrix multiplications in a split bundle.
The bundle loader must load both base weights and adapters.

## Reading paths

- To **understand the method**, read [overview](doc/overview.md), [algorithm](doc/core_algorithm.md) and [evidence gates](doc/VALIDATION.md).
- To **run a workflow**, read the [run guide](doc/usage.md). It owns commands, device selection and preset values.
- To **change code**, read [working rules](CLAUDE.md), the [change-impact map](doc/ARCHITECTURE.md#change-impact-map),
  and the relevant source docstring in the [file map](doc/README.md#current-source-owners).
- To **write a report**, use the [report guide](doc/reporting.md). Results and saved media belong under ignored `expr/` paths.

Claim acceptable quality only after the declared held-out numerical and decoded checks pass.
Claim faster execution only from warmed, synchronized measurements on the same device, with gains exceeding drift.
The [verification guide](doc/VALIDATION.md) owns the precise requirements and test commands.
