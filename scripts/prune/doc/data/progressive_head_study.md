# data/progressive_head_study

## Objective

Freeze a new progressive study from the verified r13 study and its saved checkpoint zero.

## Usage

Historical study plans were consolidated in the [workspace history](../../../../../plans/history/README.md).
For fresh preparation, replace `/path/to/reviewed-study-plan.md` with the actual reviewed specification.
Retired reference pointers are not plan inputs, and retired source folders must not be recreated from these examples.

The direct CPU owner runs from LTX-2 in the ltx conda environment:

```bash
conda run -n ltx python -m scripts.prune.data.progressive_head_study \
  --previous ../expr/refiner_prune/2.5/learned_heads_ddp_uniform_f18_20261014_r13_rank128_gate001_sparse001_dev10/configs/frozen_study.json \
  --config scripts/prune/configs/learned_heads_progressive_removal_f18.json \
  --plan /path/to/reviewed-study-plan.md \
  --output ../expr/refiner_prune/2.5/progressive_head_gate_removal_f18_20261015
```

The arguments identify source study, explicit overrides, plan snapshot and fresh output root.
Outputs include `configs/frozen_study.json`, preparation evidence, 2,000 draws and the progressive checkpoint zero.
See the [run guide](../usage.md#progressive-head-gate-removal).

## Data flow

Verify the source study digest, saved plan, every input file and each full model component hash.
Load checkpoint zero through the checkpoint owner. Keep its initial tensors, optimizers and all rank RNG states.

## Organization logic

Merge explicit overrides into the source config. Reject changes to source numerical or input settings.
Copy the first 2,000 ordinary draws from seed 42. Keep all ten sigmas and their exact draw order.
Replace only config, draw, code and plan pins. Reuse the source corpus, splits and context pins.
Bind checkpoint zero to the new training identity and add an empty configurable removal policy.
Hash the remaining state payload before and after publication. Any changed optimizer, RNG or tensor fails preparation.
For example, update zero retains cursor zero; update 500 consumes exactly 2,000 original draws.

## Invariants

Require a fresh output, source update zero, logits exactly three and four ranks on GPUs 0–3.
No native acceptance receipt is imported. The new study requires fresh bounded native acceptance.
The preparer never starts training or loads a model onto a GPU.

## Gotchas

Current code has a new frozen identity. The source study remains immutable and is verified against its recorded bytes.
Copying checkpoint zero preserves CUDA RNG device IDs. This experiment uses the same four logical ranks.

## Tests

The preparer verifies full source bytes, initial-state identity and exact draw prefix before returning success.
CPU policy tests and the fresh native acceptance cover execution separately.
