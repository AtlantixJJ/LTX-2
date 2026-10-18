# data/clipped_reset_study

## Objective

Prepare one bounded clipped-reset pilot from the verified r13 checkpoint zero.
Preserve initialized adapters, adapter Adam initialization, rank RNG, source inputs, and the ordered draw prefix.

## Usage

The direct CPU owner runs from `LTX-2` in the `ltx` conda environment:

```bash
conda run -n ltx python -m scripts.prune.data.clipped_reset_study \
  --previous ../expr/refiner_prune/2.5/learned_heads_ddp_uniform_f18_20261014_r13_rank128_gate001_sparse001_dev10/configs/frozen_study.json \
  --removal ../expr/refiner_prune/2.5/progressive_head_gate_removal_f18_20261015/configs/frozen_study.json \
  --continuation ../expr/refiner_prune/2.5/progressive_zero_one_f18_from_u500_20261016/configs/frozen_study.json \
  --plan ../plans/2026-10-16-progressive-pruning-recovery-and-scale-preserving-export.md \
  --output ../expr/refiner_prune/2.5/progressive_clip01_ste3_reset50_20261017_from_r13u0
```

The five arguments select the pinned source studies, plan snapshot, and a fresh output root.
Outputs include the frozen study/config, 3,200 draws, checkpoint pairs, initialized checkpoint zero, and preparation receipt.
See the [run guide](../usage.md#clipped-scores-with-survivor-resets).
Finish model-code changes before freezing a formal study. Later source changes invalidate its code identity.

## Data flow

Verify the source identity, plan snapshot, every source file, all components, and the explicit checkpoint-zero SHA-256.
The source must be untrained, have 192 finite adapter tensors with zero B matrices, and have empty optimizer state.
Freeze the first 3,200 source draws and all 17 exact-step references with the checkpoint-pair owner.

## Organization logic

`pilot_config` preserves the source numerical policy and specifies the single seed-42 arm on GPUs zero through three.
It sets raw-score initialization to one, disables regional loss, and installs the versioned clipped-reset policy.
The selection cap is 500. Recovery lasts exactly 300 further global updates, with a maximum endpoint of 800.
The fixed wall-time allowance is 72 hours for training, acceptance, comparison inference, media, validation, and export.
This allowance is a bound, not a runtime measurement or speed result.

`initialize` replaces alpha and creates a fresh, empty gate Adam optimizer.
It checks a hash of all other state, including adapters, adapter Adam, rank RNG, and draw counters.
It then installs an all-alive clipped-reset policy without importing historical zero or fixed-one decisions.
Publication binds the unchanged state to the new study and draw-file identities.
The final reload verifies the published checkpoint and complete state hash.

## Invariants

Use the exact r13-zero source and immutable previous studies. Do not initialize from Source2000 or another trained checkpoint.
Keep all source rank assignments and the first 3,200 ordinary draws.
An early terminal event uses the immediately following recovery draws; it cannot skip to draw 2,000.
Never import a native acceptance receipt. Preparation cannot establish execution or prediction equality.

## Gotchas

The source sigmoid gate starts at sigmoid(3); the new initial execution is exactly one.
Require native all-one equality rather than equality to the old soft checkpoint.
Preparation completes expensive source checks before creating the new output root.
An incomplete publication remains evidence for diagnosis; use a fresh root after resolving it.

## Tests

`test_clipped_preparation.py` checks restricted replacement, immutable source payloads, zero-B rejection, and frozen phase budgets.
The independent pair preflight rehashes every historical reference.
Fresh four-rank acceptance, fixed-test scoring, and full pilot execution remain separate gates.
