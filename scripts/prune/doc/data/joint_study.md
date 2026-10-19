# data/joint_study

## Objective

Freeze a paired V3 study.

## Usage

The user or agent runs this standalone owner from `LTX-2` in conda environment `ltx`.

```bash
CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n ltx python -m scripts.prune.data.joint_study --source-study ../expr/refiner_prune/2.5/learned_ffn_joint_20261018_r3/configs/frozen_study.json --continuation-study ../expr/refiner_prune/2.5/progressive_clip01_ste3_reset50_continue_nostrong_20261018_from_u500/configs/frozen_study.json --plan ../plans/2026-10-18-joint-head-ffn-clip-reset-readmission.md --output ../expr/refiner_prune/2.5/joint_clip_reset_readmission_20261018_preflight
```

See the [run guide](../usage.md#joint-clipped-gates-with-re-admission).
Outputs are immutable study, check, gradient, checkpoint or review receipts under the specified root.

## Data flow

Read the r3 frozen study and clipped continuation. Verify their identities and input files. Copy the r3 settings and replace only declared V3 method settings.

## Organization logic

The complete 5200-draw r3 prefix must equal the continuation prefix. The new file stores 6800 ordered draws. Exact monitor schedules must equal the three declared float32 values. Source components and saved context retain their hashes. C also pins the A selection-end budget receipt.

## Invariants

Keep saved input hashes, exact schedules and original unit IDs.
Keep native loss reduction in FP32. Preserve the frozen backbone and complete rank RNG.

## Gotchas

New code needs a new frozen root. Do not rewrite historical frozen studies. Test evidence cannot replace four-rank acceptance. Preparation reads full component hashes and can take several minutes.
Use `--prerequisites /absolute/path/to/accepted_review.json` for a formal training study.
The preparer pins that receipt and its source files. Training refuses an unresolved or missing receipt.


## Tests

Run `tests/test_readmission.py` with CUDA hidden.
Native acceptance and held-out export parity remain independent evidence requirements.
