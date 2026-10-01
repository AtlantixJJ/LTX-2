# `scripts/prune` working rules

Pruning has one task: a full-clip bidirectional D0 forward at each `[sigma, 0]`
schedule, with the capture's first latent frame clean. Calibration, ablation,
export parity, output quality and timing use that same input contract.

Run from LTX-2 in the `ltx` conda environment. Check `nvidia-smi` before loading
models. Keep `scripts/` and `scripts/prune/` as PEP 420 namespace packages.

## Owners and invariants

- `data.whole_clip` owns saved input validation and exact reconstruction.
  Preserve capture, noise, prompt context, fps, geometry, conditioning and sigmas.
- `core.session` owns BF16 dtype, prompt selection and model/decoder lifetime.
  It defines no sampling schedule or window geometry. Callers own their schedules.
- `core.artifacts` owns output roots and attributable run directories.
- `score.hooks` accepts only complete native D0 provenance and executable binary
  mask families. Export, parity, evaluation and timing have no cross-task bypass.
- Calibration actors cannot appear in held-out validation, including other views.
  Pin manifest content, source/noise/output hashes and actual context bytes.
- `compact_faithful` restores original execution widths from compact stored
  parameters; distinguish storage savings from reduced GEMM work in `compact`.
- `core.ltx_adapter` owns private upstream access. Avatar consumers also use model,
  prompt and media utilities: check their imports before changing shared interfaces.
- Every production module has `doc/<area>/<module>.md`, mirroring its source path; update its interface, data flow
  and invariants with the code.

## Verification

Run focused CPU tests and Ruff. Changes to native input construction or scoring
require fresh saved-baseline forward checks. Export changes require functional-mask
versus export parity on a held-out capture, using the documented tolerance.
Inspect synchronized VAE comparisons and measure benchmark drift before claiming
quality or speed. Fresh experiment results belong under ignored `expr/` paths.
