# `evaluate/phase1_gates.py`

## Objective

Establish the unpruned reference and evaluate functionally pruned candidates.

## Data flow

Frozen states and source clips -> T0/T1/T2/T3 JSON, review figures, and a
matching per-window timing profile. An external long source requires
`--t2-video` and its expected SHA256.

## Organization

T0 reads records; T1 decodes; T2 encodes windows and rolls out through refine_core; T3 renders comparisons.

## Invariants and gotchas

The reference level is nonzero against VAE source x0. T2 uses deployed 25/9
geometry and real fps; method_parity verifies the tensor path. The source hash
and exact native-frame window spans travel with T2, so a long-form verdict can
reject mismatched or insufficient coverage. Source pixels are read in bounded
CPU chunks; model windows are still VAE-encoded one at a time.
Decoded pixels use the deployed stitch rule: window 0 contributes all 25
frames; later windows contribute only their 16-frame non-overlapping tail.
T2's finalized spans follow that mapping.

## Verification

Check the package CPU suite and the relevant phase gate. Run `python -m pytest scripts/prune/tests -q` from the LTX-2 root in the `ltx` conda environment for the CPU suite. For any change that can alter rollout tensors, rerun `python -m scripts.prune.checks.method_parity --model 2.5 --gpu-id N --windows 3` on a free GPU.
