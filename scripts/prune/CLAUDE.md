# `scripts/prune` working rules

Pruning has one task: a full-clip bidirectional D0 forward at each `[sigma, 0]`
schedule, with the capture's first latent frame clean. Calibration, ablation,
export parity, output quality and timing use that same input contract.

Run from LTX-2 in the `ltx` conda environment. Check `nvidia-smi` before loading
models. Keep `scripts/` and `scripts/prune/` as PEP 420 namespace packages.

Read the [design index](doc/README.md), [architecture](doc/ARCHITECTURE.md),
[core algorithm](doc/core_algorithm.md), [verification](doc/VALIDATION.md)
and [known gaps](doc/known_gaps.md) before a model-facing change.
Use the [run guide](doc/usage.md) before writing a launch command.
The registry supports 2.3 and 2.5; new work defaults to 2.5.
Check each entrypoint's narrower support before using another key.

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

## Documentation contract

Follow the [avatar reference](../onestep_avatar/doc/README.md#documentation-rules).
A production module over 100 physical lines requires
`doc/<relative-source-path>.md`. Smaller modules explain their logic in the
header. Keep existing small-module designs as supplemental references.
Empty package markers and tests need no design page.

Use Objective, Data flow, Organization logic, Invariants, Gotchas and Tests.
Entry scripts add Usage after Objective: launch owner, environment, complete
command, main arguments, output files and matching run-guide link.
Give worked calculations and failure meanings. Write in simplified technical
English, about 80% to ASD-STE100. Use small left-to-right Mermaid diagrams
with the [shared legend](doc/core_algorithm.md#diagram-legend); render and inspect them.

Update designs or headers with changes to objective, data flow, organization
or invariants. Describe current behavior only. Keep history in saved records.
The source inventory must match existing files and physical line counts.
Do not add an avatar queue requirement to an unrelated standalone prune CLI.

## Verification

Run focused CPU tests and Ruff. Changes to native input construction or scoring
require fresh saved-baseline forward checks. Export changes require functional-mask
versus export parity on a held-out capture, using the documented tolerance.
Inspect synchronized VAE comparisons and measure benchmark drift before claiming
quality or speed. Fresh experiment results belong under ignored `expr/` paths.

The CPU gate runs the exact suite with `CUDA_VISIBLE_DEVICES=`. Collection
marks only tests already labeled `gpu` skipped when CUDA is unavailable or
hidden. With real CUDA those tests still run. Metadata and tiny-model CPU
fixtures retain their own checks; a marker does not disable their imports.
