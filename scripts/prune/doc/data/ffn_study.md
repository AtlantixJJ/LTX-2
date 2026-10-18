# data/ffn_study

## Objective

Freeze an FFN experiment with verified existing corpus bytes and a new code identity.

## Usage

A human or agent runs this CPU preparation directly, from `LTX-2` in `ltx`.

```bash
conda run --no-capture-output -n ltx python -m scripts.prune.data.ffn_study --source-study ../expr/refiner_prune/2.5/progressive_clip01_ste3_reset50_continue_nostrong_20261018_from_u500/configs/frozen_study.json --plan ../plans/2026-10-18-learned-ffn-gate-mask-pruning.md --output ../expr/refiner_prune/2.5/learned_ffn_joint_20261018
```

`--config` selects the explicit joint preset by default. `--output` must name a fresh experiment directory.
Outputs are the frozen study, effective config, immutable plan copy and ordered draws.
See the [run guide](../usage.md#learned-ffn-masks).

## Data flow

Verify the parent identity and every source, context, component and plan file hash.
Copy actor-disjoint membership, exact sigmas and source metadata.
Generate seed-42 draws with the existing draw owner. Check their saved parent prefix.
Pin the current code, software, new config, plan copy and draw file.

## Organization logic

The new format is `ffn_mask_study_v1`. It is separate from historical head study formats.
Remove the parent's progressive policy before applying the FFN settings.
The parent provides inputs; its checkpoints and native acceptance do not qualify this new learner.
Runtime readers verify the new identity and code. The primary rank also verifies all original input bytes.
Consumed draws receive their own exact source, noise and F18 window checks.

## Invariants

Use four formal ranks on physical GPUs 0–3. Use physical GPUs 4–6 for disposable tests.
Keep source actor separation, actual context and exact saved schedules.
Never change the original frozen study or its files.

## Gotchas

Input verification can read large shared model files. It performs no model forward.
Changing production code requires a new study; changing the live plan does not change its immutable saved copy.
For 1,300 updates, four formal ranks consume 5,200 draws.

## Tests

The FFN tests check count schedules and training semantics. Preparation and CLI checks run before a new launch.
Native backward, compact parity and quality remain separate checks.
