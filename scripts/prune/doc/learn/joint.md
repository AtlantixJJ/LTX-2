# learn/joint

## Objective

Run V3 native acceptance and training.

## Usage

The user or agent runs this standalone owner from `LTX-2` in conda environment `ltx`.

```bash
CUDA_VISIBLE_DEVICES=4,5,6 CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx torchrun --standalone --nproc_per_node=3 -m scripts.prune.learn.joint --study ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261018_preflight/configs/frozen_study.json --phase native --testing
```

See the [run guide](../usage.md#joint-clipped-gates-with-re-admission).
Outputs are immutable study, check, gradient, checkpoint or review receipts under the specified root.
Use `--phase pilot --testing --max-updates 50` only in a disposable preflight study.
Its three ranks consume the first three draws of each four-draw group; it does not initialize A.

Final training uses the pinned P0/P1/P2 receipt and fresh four-rank native acceptance:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx torchrun --standalone --nproc_per_node=4 -m scripts.prune.learn.joint --study ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261019_A/configs/frozen_study.json --phase formal
```

The V3 entrypoint supplies its family hooks to `learn.ddp_owner`.
It accepts the frozen `joint_clip_reset_v3`, `heads_clip_reset_v3` and `budget_joint_v3` arms.
The historical `learn.ffn` CLI does not dispatch these studies.

## Data flow

The owner loads the frozen study and exact physical GPU list. Each rank loads one native model. Native checks compare all-one execution, mixed-gradient behavior, event resume and fixed-mask recovery.

The [shared DDP owner](ddp_owner.md) owns the runtime, input reuse, native acceptance, monitor and update loop.
This module supplies V3 family hooks and its numerical state owner.
The frozen world size determines the draw stride, including smaller disposable pilots.

## Organization logic

Training saves immutable gradient, trace, event and checkpoint files. Monitor calls restore RNG and use saved source noise. G1 pauses at update 100 when declared allocation or quality limits fail. G2 scores before and after alignment, saves the budget for C and requires prerequisite acceptance before recovery.

## Invariants

Keep saved input hashes, exact schedules and original unit IDs.
Keep native loss reduction in FP32. Preserve the frozen backbone and complete rank RNG.

## Gotchas

A root `reviews/G*.json` trigger remains pending on restart. The frozen owner has no separate approval flag. An external reviewed-continuation supervisor can move the original record into `reviews/resolved/` only after saving and verifying its decision. It retains the original bytes and records a path/hash resolution index. The report reads both locations. Resume the unchanged study with `--phase train`; archive the old mutable terminal-owner receipt before restarting CPU readers. Formal training verifies pinned preflight receipts before any optimizer update. Final training requires separate four-rank native acceptance. The test command uses only physical GPUs 4–6. The owner never stops unrelated processes.

## Tests

Run `tests/test_readmission.py` with CUDA hidden.
Native acceptance and held-out export parity remain independent evidence requirements.
