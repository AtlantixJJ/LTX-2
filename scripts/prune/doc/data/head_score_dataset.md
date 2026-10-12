# MobileWan capture dataset audit

## Objective

Verify every supplied capture view before scoring. Preserve the complete clip and actor split.

## Usage

Launch directly on CPU from `LTX-2` in the `ltx` conda environment:

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.prune.data.head_score_dataset \
  --membership ../expr/onestep_avatar/runs/d0_bidirectional_sigma0.9/membership_all.json \
  --output ../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011
```

The output is `inputs/dataset_manifest.json`, `provenance/coverage.jsonl` and `provenance/failures.json`.
The CLI does not load models, allocate GPUs or score heads.
See the [run guide](../usage.md#mobilewan-capture-audit).

## Data flow

Full JSON and CSV hashes must match the task pins before reading sources.
The public membership loader validates its semantic hash, unique rows and actor-disjoint splits.
The public `ClipStore` checks capture hashes, recorded encoding fields, geometry and fps.
It receives `require_guide=False`; a missing guide does not invalidate D0.

This audit also checks RGB and clip metadata hashes and the metadata actor identity.
It resolves filesystem aliases and rejects duplicate resolved view directories.
It verifies finite BF16 masters, both full clip lengths and recorded pixel spans.
It records the stride-one generated-token count and index hash for every view.

## Organization logic

Write a status line after each source. Flush it so ongoing progress can be inspected.
Collect all row failures instead of silently omitting invalid sources.
Write the failure ledger even when a row fails.
Publish the final manifest atomically only after all 3,360 rows pass.
The selected cases use SHA-256 of `seed42:<actor>` and `seed42:<relative_dir>`.
Choose actors first, then one view per actor. Input order has no effect.

`official_schedule` constructs the original scheduler on CPU with no latent argument.
It requires 30 default steps and exact task sigmas at original indices 8,11,14,17,20,23,26,29.
It retains the unchanged eligible tail and source identity.
This helper supplies schedule provenance; the audit CLI does not yet publish a complete study config.

## Invariants

Membership file hashes differ from the membership semantic hash. Record each separately.
Missing inputs produce failures, never reduced final coverage.
Both 18-frame and 28-frame latent clips retain all their generated tokens.
Three fixed cases must come from three distinct held-out actors.

## Gotchas

For 18 latent frames, generated-token coverage is `17*32*32=17408`.
For 28 latent frames, coverage is `27*32*32=27648`.
The conditioned first frame contributes no score tokens.
An audit proves source validity at audit time. Workers must recheck pins on resume.
The full-study runner, model checks and report remain separate work.

## Tests

`tests/test_head_score_dataset.py` checks changed membership rejection, input-order-independent cases and exact tail preservation.
The real audit supplies full-dataset source evidence and records failures per row.
