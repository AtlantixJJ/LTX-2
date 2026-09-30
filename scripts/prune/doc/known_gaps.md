# Review findings and current status

**Historical k2 review.** These 200-window figures are not the native whole-video
D0 pruning decision. Use the [current D0 findings](../../../../expr/refiner_prune/2.5/FINDINGS.md)
and [validation guide](VALIDATION.md) for the active task; [HISTORY](HISTORY.md)
keeps the relationship between the two studies explicit.

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

## Follow-up profile and execution choice

The per-branch benchmark found full attention with the same output mask faster
than selected-head attention in all 14 pruned branches. A full-width export
also carried identity RoPE head lists, adding a gather in every attention
layer; omitting them cut measured preattention time from about 41 ms to 26 ms
on the held-out record. The resulting `masked_full` checkpoint matches the
functional mask exactly and takes 544.1 ms versus 543.6 ms for the masked
source on that record. Two-window rollout parity is also exact. Its projections
and FFN GEMMs remain full width, so this execution choice addresses overhead
but does not satisfy the 1.4× target. See
[`checks/profile_export.py`](profile_export.md) and the profile artifact linked
from the workspace plan.
