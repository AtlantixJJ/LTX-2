# `score/export_pruned.py`

## Objective

Export score masks as a self-describing safetensors checkpoint and preserve
the numerical meaning of the functional mask.

## Data flow

Source checkpoint and binary masks produce a full-width sparse checkpoint by
default. `--mode compact` slices V, output, gate, and FFN tensors; it also
accepts fitted FFN projections. Both modes update per-layer metadata.

## Organization

Sparse mode records active attention head IDs and FFN channel IDs. It leaves
Q/K/V/output/FFN GEMM shapes unchanged, skips masked heads in attention, and
applies the mask before the original output projection. Compact mode preserves
retained RoPE head identities. Both record pruning provenance in metadata.

## Invariants and gotchas

The CLI validates mask provenance, lengths, and fitted FFN shapes using
safetensors headers before materializing weights. It records the mask SHA256,
source fingerprint, and peak RSS. Export still materializes the checkpoint in
RAM; use `checks.export_parity` before deployment. Compact export changes bf16
GEMM reduction shapes, and the p05 compact experiment failed two-window
parity. Sparse mode avoids that source of difference but may offer little
speedup because projection GEMMs remain full width.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
