# data/resume_learned_heads

## Objective

Resume an incomplete soft checkpoint under the requested raw-logit sparsity loss.
Keep the source run immutable and record the loss boundary in a new frozen study.

## Usage

Historical study plans were consolidated in the [workspace history](../../../../../plans/history/README.md).
For fresh preparation, replace `/path/to/reviewed-study-plan.md` with the actual reviewed specification.
Retired reference pointers are not plan inputs, and retired source folders must not be recreated from these examples.

Launch directly on CPU from `LTX-2` in conda environment `ltx`, after stopping all source workers.

```bash
CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n ltx python -m scripts.prune.data.resume_learned_heads \
  --previous ../expr/refiner_prune/2.5/learned_heads_fsdp_gt_20261013_r6/configs/frozen_study.json \
  --checkpoint ../expr/refiner_prune/2.5/learned_heads_fsdp_gt_20261013_r6/training/learned_joint/replicate_42/checkpoints/update_0546.pt \
  --plan /path/to/reviewed-study-plan.md \
  --output ../expr/refiner_prune/2.5/learned_heads_fsdp_gt_20261013_r7_logit_l1
```

The output must be new. `--previous` and `--checkpoint` identify the source authority and state.
`--plan` supplies the current protocol snapshot. Output includes the new study, checkpoint, lineage and continuation receipt.
Continue with the direct four-rank supervisor from the [run guide](../usage.md#four-rank-fsdp-with-gt-references).

## Data flow

The source study, checkpoint, paired draws and consumed input receipts enter the CPU validator.
The validator preserves components, actor splits, schedules, source bytes and draw order.
Only `lambda_max=1`, `sparsity_penalty=relu_logits` and offset three amend the config.

## Organization logic

Verify the source study digest and frozen plan, then restrict changed production files to the loss amendment owners.
Verify the source training identity, incomplete soft phase and every consumed input/noise hash.
Freeze the amended config and plan. Rebind the checkpoint identity; preserve every state payload bit.
Retain the original prefix receipts and their study identity in lineage for final complete-budget validation.
Reuse accepted FSDP execution evidence with explicit source hashes and scope. Resumed updates check the amended loss on native execution.

A source checkpoint at update 546 remains update 546 with draw cursor 1092.
The next update is 547, and `lambda=1` because the existing 300-update warmup has elapsed.
There is no optimizer reset, RNG reset, fresh gate initialization or new corpus preparation.

## Invariants

Preserve gate logits, adapters, optimizer moments, masks, draw cursor, phase and all rank RNG states.
The new checkpoint payload fingerprint must equal the source fingerprint after excluding identity and lineage.
Never modify source receipts or claim the historical prefix used the amended loss.
Unrelated numerical code changes, changed epsilon bytes, another checkpoint identity or a used output path fail.

## Gotchas

ReLU stops sparsity pressure at logits at most minus three; it does not hard-clamp logits.
Native execution evidence from the source does not establish quality of the amended loss.
The resumed learned arm has an explicitly mixed loss history. It is not a fresh run using the new loss throughout.

## Tests

Run `test_resume_learned_heads.py`, `test_head_losses.py` and `test_head_learning_state.py` with CUDA hidden.
Check payload equality, optimizer steps, draw continuation and changed-noise/code rejection.
Inspect resumed native update receipts for the requested coefficient, finite gradients and unchanged memory headroom.
