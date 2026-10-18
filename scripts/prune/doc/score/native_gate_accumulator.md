# Native gate accumulation

## Objective

`score/native_gate_accumulator.py` stores mergeable sufficient statistics and exact draw identities.
It is model-free and imports neither torch nor a model owner.
The [collector](native_head_gates.md) assigns four selected source noise indices to four accumulator groups.

## Data flow

One observer result contains all original blocks and heads for one retained draw.
The accumulator adds that result to its sigma group and saves the draw receipt in the same state.
Worker shards contain one 100-source-step window. The supervisor merges them into window and cumulative states.

## Organization logic

Sums, squared sums and counts add. Minima and maxima use their corresponding reductions.
Histograms use int64. Moments use FP64. No individual token values are saved.
Each draw must contain the same token count, so token weighting equals equal-draw weighting.
Different token counts fail rather than silently changing the estimand.
`save_arrays` writes an NPZ with arrays and JSON metadata in one atomic replacement.
This binds statistics and receipts to the same cursor after a crash.
`load` validates study identity, unique draw IDs and per-sigma count conservation.

`require_indices` checks the complete retained source-index set.
`require_prefix` is the corresponding helper for an unfiltered contiguous list.
Skipped noise levels create intentional source-index gaps; they do not count as missing retained data.

For a 100-step source window there are 400 scheduled records, 160 retained records,
and 40 samples in each selected sigma group. A cumulative milestone at step 200 contains 320 retained records.
Worker completion order does not define either prefix.

## Invariants

- A draw ID cannot occur twice, including across merged workers.
- Every head's histogram sum equals its group's generated-token count.
- Clean-frame token counts remain separate.
- A resumed state has the same study identity and the same completed draw set.
- Merge order is fixed by physical worker order; each worker processes ascending original draw IDs.

## Gotchas

Empty groups use infinite initial extrema but zero counts. Only complete, balanced milestones enter analysis.
Small partial worker states can be checkpointed without inventing observations for empty groups.
Distribution distance compares normalized histograms on fixed bins; it has native gate units.
FP64 sums are deterministic under the recorded reduction order, not arbitrary floating-point reassociation.

## Tests

The focused tests check exact save/resume, sigma isolation, shuffled-worker coverage, duplicate refusal,
independent histogram counts, constant weighting and known distribution distances.
