# Whole-clip pruning verification

## CPU gate

Run from `LTX-2` with the `ltx` interpreter:

```bash
conda activate ltx
CUDA_VISIBLE_DEVICES='' python -m pytest scripts/prune/tests -q -m 'not gpu'
ruff check --select F,I scripts/prune
```

The collection hook skips only tests already marked `gpu` when CUDA is hidden.
CPU fixtures retain import, metadata, shape and small-model checks.
A marker does not disable imports. Ruff follows the repository's dev-tool setup when unavailable in `ltx`.

For documentation changes, check source coverage, required headings, local links and documented CLI options.
Render and inspect changed Mermaid diagrams.
Documentation verification does not require expensive GPU work.

## G1: native baseline reproduction

Reconstruct the exact capture, VAE, fps, geometry, context bytes, seed and saved epsilon.
Use one bidirectional block and the exact `[sigma,0]` schedule.
Inspect the actual float32 sigma and token timesteps, BF16 latents and clean capture frame zero.

Calibration forwards must reproduce their saved D0 outputs.
The scorer rejects nonfinite maximum differences and values above 0.02.
Use the existing avatar `prune_baseline` queue experiment for native reproduction.
Saved noise and D0 tensors must be compared directly across old/new runs.
Equal seeds alone do not prove equal noise.

The saved completion reader checks structure, coverage, hashes, finite BF16 tensors and exact c0.
It opens no model and cannot regenerate device-specific CUDA noise.
The wrapper separately checks full component and software pins.
See [producer verification](evaluate/whole_clip_baseline.md#saved-completion-verification)
and [queued baseline](usage.md#queued-baseline-reproduction).

## G2: artifact identity and held-out scope

Require `whole_clip_d0_mask_v1` for width or `whole_clip_d0_depth_v1` for intact-block removal.
Check native task, attention, conditioning, model/VAE identity and full calibration-manifest content.
Check source, noise and context identities.

Exclude calibration actors across clips, Parts, views and aliases.
Also reject a held-out capture reused from calibration.
Held-out sigmas must occur in calibration.

The selected baseline must retain the pinned calibration rows for distribution validation.
The saved quality analyzer visits all rows. Identify held-out rows separately in the report.
See [native input contract](data/whole_clip.md) and [mask checks](score/hooks.md).

## G3: export parity

Use identical saved inputs for the in-memory intervention and exported model.
Width reference is the functional mask.
Depth reference is the ordered retained block list.
Load one transformer at a time.

The default gate requires matching shapes and maximum absolute latent difference at most 0.02.
Report relative L2 and finite values as additional evidence.
Include a no-prune export control.
A changed normalization or fitted weight requires its own reference.

Depth architecture verification checks inventory, config, mappings and accounting.
It does not compare payload bytes or establish numerical parity.
The native parity gate remains separate.
See [export parity](checks/export_parity.md).

## G4: held-out decoded quality

Report generated-frame direction relative L2/cosine, capture latent MSE and prediction disagreement.
Exclude clean frame zero.
State sample size, actor split, seeds and exact schedules.

Inspect synchronized capture VAE, baseline D0 VAE and candidate D0 VAE panels.
Judge identity, detail, pose, motion and temporal consistency.
Identify VAE decoding explicitly. Include matched capture RGB when a report assesses original capture fidelity.
Verify preprocessing coordinates and source hashes before interpreting alignment.

Latent metrics measure numerical agreement. They do not prove correct action or useful motion.
Predeclare acceptance thresholds before blind tests.
See [comparison logic](evaluate/whole_clip_d0.md).

## G5: measured cost

Measure repeated warmed and synchronized forwards on the same GPU.
Keep input, software, precision and numerical policy fixed.
Use ABA and BAB brackets when gains are close to measured drift.
The benchmark requires at least one warmup and three timed calls.

A speed claim must exceed drift and timing spread.
Report checkpoint bytes, resident-video parameters and peak allocated memory separately.
A compact file or mask is not a speed result.
See [benchmark logic](evaluate/bench_whole_clip_d0.md).

## Worked evidence checks

- Rehashing a malformed tensor does not make its shape, dtype, finiteness or c0 valid.
- A depth export can match retained source blocks while losing quality against the full teacher.
- A local ridge fit can reduce projection MSE without improving held-out model output.
- A benchmark can improve its median within the bracket drift. That is insufficient speed evidence.

## Test ownership and limits

The [module index](README.md) links each production owner and its Tests section.
Depth tests use real small video/AV models, strict reload, cache comparisons and backward gradients.
Reconstruction tests compare dual ridge with an independent primal solve.
Saved-baseline tests exercise a tiny real transformer and model-free completion refusals.

Those CPU results establish focused correctness.
Native BF16 baseline, export, quality and timing evidence are independent acceptance gates.
Do not label a candidate accepted while a required gate remains unrun.
