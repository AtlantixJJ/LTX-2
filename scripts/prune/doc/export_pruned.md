# `score/export_pruned.py`

## Objective

Export score masks as a self-describing safetensors checkpoint and preserve
the numerical meaning of the functional mask.

## Data flow

A native D0 mask with complete task provenance and a source checkpoint produce a full-width `masked_full`
checkpoint by default. Older k2 masks require `--historical-k2-mask`; the export metadata records which task produced the mask. `--mode sparse` selects retained heads before the
attention kernel, and `--mode compact` slices V, output, gate, and FFN tensors;
compact mode also accepts fitted FFN projections. All modes update metadata.

## Organization

Full-width modes record active attention head IDs and FFN channel IDs. They
leave Q/K/V/output/FFN GEMM shapes unchanged and apply the mask before the
original output projection. `sparse` skips masked heads in attention;
`masked_full` runs full attention. Compact mode preserves retained RoPE head
identities. All modes record pruning provenance in metadata.

## Invariants and gotchas

The CLI validates mask provenance, lengths, and fitted FFN shapes using
safetensors headers before materializing weights. It records the mask SHA256,
source fingerprint, and peak RSS. Export still materializes the checkpoint in
RAM; use `checks.export_parity` before deployment. Compact export changes bf16
GEMM reduction shapes, and the p05 compact experiment failed two-window
parity. The p05 direct benchmark found all 14 pruned attention branches faster
with full attention than with head selection. Full-width exports therefore
default to `masked_full`, and omit identity RoPE head lists to avoid an
unnecessary frequency gather in every layer. This preserves exact parity but
cannot remove projection GEMM work.
An FFN-only compact export omits identity RoPE indices in unpruned attention
branches. The measured 10% compact FFN candidate used less GPU memory, but
changed bf16 results beyond the export-parity tolerance and is not deployable.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
