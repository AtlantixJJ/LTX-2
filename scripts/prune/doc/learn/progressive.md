# learn/progressive

## Objective

Own configurable global head removals, regret windows, frozen Adam coordinates and serialized policy history.

## Data flow

Save checkpoint zero and each post-update FP32 logit matrix. An event uses `hardening_interval+1` snapshots.

```mermaid
flowchart LR
  A("Post-update logits") --> B["Filter by decrease and regret"]
  B --> C["Rank globally and select percentage"]
  C --> D(["Removed IDs, masks and event receipt"])
  classDef proc fill:#dbe7ff,stroke:#3b5ea8,color:#10203f
  classDef tensor fill:#dcf5df,stroke:#43854b,color:#173b1c
  classDef out fill:#ece0f8,stroke:#7048a0,color:#26123f
  class A tensor
  class B,C proc
  class D out
```

## Organization logic

Validate a positive integer `hardening_interval`, `removal_fraction` in `(0,1]`, and `regret_threshold` in `[0,1]`.
Compute FP64 travel T as the sum of absolute logit increments. Decrease D is the start logit minus the end logit.
For positive travel, regret is `1-abs(D)/T`. Constant heads have undefined regret, stored as null in event JSON.
Filter free heads by positive decrease and regret at most the configured threshold. Equality passes the threshold.
Rank globally by `(regret, -decrease, block_id, head_id)`. Never apply a gate-magnitude threshold.
Select up to `ceil(removal_fraction*original_head_count)` eligible heads. Record shortfalls without relaxing the filter.
The fraction always uses the original matrix size. For 1,536 heads and fraction 0.01, each event requests 16 removals.
Ten events can remove 160 heads. There is no cumulative cap, block quota or width divisibility constraint.
A trajectory from three to two has travel one, decrease one and regret zero. It passes even if the decrease is tiny.
An increasing trajectory fails the decrease filter. A constant trajectory fails the positive-travel filter.

## Invariants

Compose `free*sigmoid(alpha)` at original native widths. Removed heads execute exactly zero; all unselected heads stay soft.
Commit after backward and the optimizer step. Decisions affect the next update and the current post-event monitor.
Mask removed gradients, clear their Adam moments and restore their stored logits after each optimizer step.
Train LoRA normally. Functional removal does not establish compute savings.
Serialize the hyperparameters, cursor, window, free mask, frozen logits and complete event history with every checkpoint.
Resume rejects changed hyperparameters. Loading rechecks threshold filtering, rankings, selected IDs, counts and irreversibility.
All ranks compare the full policy hash with tensor all-gather at each event.
Publish event receipts before canonical checkpoints. Receipts include candidate scores, global ranks and original selected IDs.

## Gotchas

Regret measures reversal, rather than pruning damage or importance. Small positive decreases are valid evidence under this policy.
Store undefined constant-head regret as null, rather than zero or nonfinite JSON.
The terminal mask affects the velocity path and free-logit sparsity. The sparsity denominator remains the original 1,536 heads.

## Tests

`test_progressive_gates.py` covers configurable intervals/fractions/thresholds, threshold equality, stable sorting, shortfalls and arbitrary counts.
It also checks soft/zero gradients, frozen Adam coordinates, serialized ranking validation and exact replay across an event.
The bounded native check verifies three removed heads with checkpointed backward and exact resumed updates.
