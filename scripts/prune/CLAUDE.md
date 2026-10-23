# Pruning working rules

Run from `LTX-2` in conda environment `ltx`; its editable packages match this source tree.
Use LTX-2.5 for new work; other registry entries exist for reproduction.
Read the [overview](doc/overview.md) and [glossary](doc/glossary.md) for shared terms.
Current results and run status live in [workspace handoffs](../../../plans/README.md).

## Rules and reasons

- Check free devices before model loading; another worker's allocation is not yours to reclaim.
- Use the frozen config's device lists; rank order determines draw assignment and checkpoint identity.
- Preserve stopped and live studies and their source snapshots; maintenance does not authorize new model work.
- Keep the workspace submodule pin unchanged; editing this worktree has separate authorization from updating that pin.
- Keep source, noise, context, geometry and schedules pinned; comparisons require the same actual inputs.
- Exclude calibration actors from held-out data across views and aliases; another view is still the same actor.
- Use the input and session owners; duplicate reconstruction or loading logic creates inconsistent forwards.
- Keep private upstream access in `core.ltx_adapter`; this isolates changes to upstream interfaces.
- Keep `scripts/` and `scripts/prune/` as namespace packages; consumers import from both workspaces.
- Put model work and decoding in this package; report rebuilds must depend only on saved evidence.
- Use the avatar queue for its restricted baseline wrapper; other prune CLIs retain direct launch ownership.

## Read before changing

| Change | Read |
|---|---|
| Inputs or numerical behavior | [Input contract](doc/data/whole_clip.md), [algorithm](doc/core_algorithm.md) |
| Scoring or learned selection | [Algorithm](doc/core_algorithm.md), [selection state](doc/learn/readmission.md) |
| Exports or transformer config | [Width/runtime contract](doc/score/export_pruned.md), [depth contract](doc/score/export_depth.md) |
| Distributed training | [DDP contract](doc/learn/ddp_owner.md), [head phases](doc/learn/heads.md) |
| Any launch command | [Run guide](doc/usage.md) |
| Documentation or reports | [Documentation rules](doc/README.md#documentation-rules), [report rules](doc/reporting.md) |

Prune explicitly uses concept pages plus module docstrings, independently of the avatar mirrored-page contract.
Update the owning docstring and shared guide in the same change when behavior changes.
Define terms before use and explain non-obvious rules; short sentences must still explain the method.
Use the [change-impact map](doc/ARCHITECTURE.md#change-impact-map) to choose additional checks.

## Verification

```bash
conda activate ltx
CUDA_VISIBLE_DEVICES='' python -m pytest scripts/prune/tests -q -m 'not gpu'
ruff check --select F,I scripts/prune
```

The [verification guide](doc/VALIDATION.md) defines native checks required after numerical or export changes.
Documentation work requires CPU checks and rendered-diagram inspection, with no experiment launch.
