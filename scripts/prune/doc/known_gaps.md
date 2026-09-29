# Review findings and current status

The [2026-09-29 improvement plan](../../../../plans/2026-09-29-refiner-prune-review-and-next-steps.md)
records the original findings. The gate verdict now requires matched provenance,
quality, coverage, artifacts, and measured speed. The launcher validates its
prerequisites and collects worker failures. The score mask loader validates
checkpoint identity and mask shape. A frame-aligned 3209-frame source permits
200-window evaluation. Format-1 recovery points at `data/source_target.py`.

## Remaining performance gap

The p05 compact export did not match the functional mask in bf16. Slicing
projection matrices changes GEMM reduction shapes and numerical results. Sparse
export retains full projections, skips masked attention heads, and matched the
functional mask exactly on a held-out record and two rollout windows. It
therefore has a faithful checkpoint path, but full-width GEMMs remain. The
matched 200-window p05 run failed acceptance: speedup was 0.960× against 1.4×,
and T1 PSNR fell 0.686 dB against a 0.5 dB limit. T0, drift, coverage,
artifacts, and input matching passed. Peak allocated GPU memory in the
two-window parity check was 24.82 GiB for the functional mask and 24.83 GiB
for sparse export. Profile attention selection/scatter and projection cost
before further sparsity work, and revisit mask selection to recover quality.
