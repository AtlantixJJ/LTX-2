# Saved native-gate milestone audit

## Objective

`score/native_gate_audit.py` independently checks a completed native-gate source prefix.
It reads saved files on CPU. It does not open a model or change milestone evidence.

## Usage

Run directly from `LTX-2` in the `ltx` environment after the collection owner publishes a milestone.
The avatar queue does not own this saved-file check.

```bash
conda activate ltx
python -m scripts.prune.score.native_gate_audit \
  --root ../expr/refiner_prune/2.5/native_head_gate_stats_20261018_exec_isolated \
  --step 100
```

`--root` selects the frozen study. `--step` selects a complete 100-step milestone.
The command writes `monitoring/audits/step_0100.json` and prints the checks.
A failed check raises a named error. It does not publish a passing receipt.
See the [native-gate run guide](../usage.md#native-head-gate-statistics).

## Data flow

Read the frozen study, source draw list and completed milestone receipt.
Verify archive hashes before loading statistics.
Check every retained draw against its source view, actor, seed, exact sigma and logical rank.
Check the deterministic F18 crop, input manifest hash and recorded noise/context identities.
When a draw claims a historical training match, compare the saved training receipt again.

```mermaid
flowchart LR
  I("saved draws and worker shards") --> C["check identities and counts"]
  C --> M["rebuild exact milestone"] --> R(["audit receipt"])
  classDef tensor fill:#dcf5df,stroke:#43854b,color:#173b1c;
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f;
  classDef out fill:#ece0f8,stroke:#7048a0,color:#26123f;
  class I tensor;
  class C,M proc;
  class R out;
```

## Organization logic

The audit derives retained indices from the original ten-level stream.
It compares these indices with the configured retained and skipped prefixes.
It loads each worker shard and checks its physical GPU assignment.
It rebuilds the window in the owner's fixed GPU order and compares every array exactly.
It adds that window to the verified predecessor and requires exact cumulative equality.

For example, step 100 must contain 400 scheduled records, 160 observations and 40 draws per selected sigma.
With 1,024 tokens per frame, each head has 696,320 generated tokens per selected sigma.
Step 200 must double those counts and equal the preceding cumulative state plus the new window.

The histogram mean must agree with the exact gate mean within half one histogram bin.
Per-frame generated sums must agree with generated-token sums.
Threshold counts must stay within their token population and preserve threshold order.
The full statistic inventory keeps integer counts and FP64 moments.

## Invariants

- No torch, model, decoder, queue or avatar module is imported.
- Completed milestones and source receipts remain unchanged.
- A selected draw appears once, in its original source position.
- The worker provenance list identifies each draw's verified shard and physical GPU.
- All crops have the same generated-token count across workers.
- Passing this audit does not replace E0 or prove causal pruning importance.

## Gotchas

A process PID or a completion label alone is not evidence of an exact prefix.
The archive hash, receipt chain, source membership and deterministic merge must all agree.
The audit checks saved input pins and training receipts; it does not regenerate CUDA noise.
It does not independently replay each token observation or measure output quality.
Run from the live package or a separate audit snapshot, without changing the running collection snapshot.

## Tests

`tests/test_native_gate_audit.py` constructs a small complete four-sigma prefix without model execution.
Controls alter sigma groups, crop starts, epsilon pins, worker ownership and sufficient statistics.
An isolated import check rejects accidental model-stack imports.
The native run's audit receipts remain separate evidence from these synthetic tests.
