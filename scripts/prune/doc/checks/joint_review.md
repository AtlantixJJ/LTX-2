# checks/joint_review

## Objective

Review saved V3 prerequisites.

## Usage

Run this standalone CPU reader from `LTX-2` in the `ltx` environment.

```bash
python -m scripts.prune.checks.joint_review --study /abs/frozen_study.json --pilot /abs/pilot --p1 /abs/p1 --p2 /abs/p2 --output /abs/new_prerequisites.json
```

See the [run guide](../usage.md#joint-clipped-gates-with-re-admission).
The study path selects saved evidence. The reader opens no transformer or decoder.

## Data flow

Read all 1,152 P1 ablations and require exact actor, sigma, block and count coverage.
Compare early and late median cost for each metric. Mixed results remain unresolved.
Require 12 cases in each pilot monitor, a completed P0 replay and passed P2 parity.
Publish the check booleans and pinned source receipts; missing evidence raises an error.

## Organization logic

Validate saved evidence before publishing the result. Preserve original study identities and exact metric names.
Missing evidence fails or remains pending; it never triggers model work.

## Invariants

Keep saved scientific receipts immutable. Do not regenerate missing predictions.

## Gotchas

A receipt reader cannot establish native execution or qualify a candidate's speed.

## Tests

Run the prune CPU suite and the documentation contract checks.
