# learn/receipt_logger

## Objective

Follow FFN and V3 receipts on CPU.

## Usage

Run this standalone CPU reader from `LTX-2` in the `ltx` environment.

```bash
CUDA_VISIBLE_DEVICES= python -m scripts.prune.learn.receipt_logger --study /abs/frozen_study.json
```

See the [run guide](../usage.md#joint-clipped-gates-with-re-admission).
The study path selects saved evidence. The reader opens no transformer or decoder.

## Data flow

The frozen study format selects FFN or V3 mask metrics.
Read saved optimizer and monitor receipts and log each global update once.
Rank sigma and memory retain one labeled series per rank. Loss and validation use global means.
A mutable cursor records published steps. Stop after a terminal owner receipt or failure.
W&B must be enabled in the frozen config. This reader never loads a model.

## Organization logic

Validate saved evidence before publishing the result. Preserve original study identities and exact metric names.
Missing evidence fails or remains pending; it never triggers model work.

## Invariants

Keep saved scientific receipts immutable. Do not regenerate missing predictions.

## Gotchas

A receipt reader cannot establish native execution or qualify a candidate's speed.

## Tests

Run the prune CPU suite and the documentation contract checks.
