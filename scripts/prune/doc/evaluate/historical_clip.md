# evaluate/historical_clip

## Objective

Score the historical head-only selection endpoint without resuming its stopped training.

## Usage

Run this direct CLI from `LTX-2` in conda environment `ltx`, on a free physical GPU from 4–5.

```bash
python -m scripts.prune.evaluate.historical_clip --study /abs/completed_A/configs/frozen_study.json --historical-study /abs/historical/configs/frozen_study.json --checkpoint /abs/historical/training/learned_progressive/replicate_42/checkpoints/update_0600.pt --monitor /abs/historical/checks/learned_progressive/replicate_42/monitor_0600.json --output /abs/fresh/historical_clip_600 --gpu-id 5
```

Outputs include `identity.json`, `monitor_parity.json` and 108 held-out case receipts under `test/`.
Use `--shard-index I --shard-count N` with distinct output roots for disjoint modulo-N case lists.
Every shard first replays all 12 historical monitors. The CPU coordinator must verify the complete 108-case union.
See the [run guide](../usage.md#completed-joint-v3-evaluation).

## Data flow

The completed A study supplies the shared held-out split and verified inputs.
The historical checkpoint supplies fixed binary head gates and its original attention adapters.
The shared inference owner saves predictions and metrics with source and noise hashes.

## Organization logic

Check full checkpoint bytes, adapter tensors, gate tensors, policy identity and the post-reset state.
Require the same split, components, schedule and prompt context as the completed A study.
Replay all 12 historical monitor predictions before held-out scoring.
Monitor replay reads the original historical input root and verifies the exact recorded manifest hash.
Equivalent tensors in a newer manifest do not replace that original provenance record.
Saved dense predictions use `[1,C,F,H,W]`; convert to `[1,F*H*W,C]` before comparison.
The conversion preserves each original frame, height, width and channel coordinate. Reject other shapes and nonfinite values.
Save every discrepancy and stop scoring if any maximum error exceeds 0.02.
Score six actors, two views, three seeds and three exact schedules only after replay passes.

## Invariants

Do not build an optimizer or call a historical training owner.
Do not change a checkpoint, binary gate, training receipt or input.
The historical endpoint is update 600 with 2,400 consumed draws.
Keep original attention dimensions and separate trained adapter arithmetic.

## Gotchas

The historical checkpoint has a soft phase label although its post-reset gates are binary.
Check the actual gates and progressive policy instead of inferring their values from the phase label.
These actors were previously inspected; they are held out from training, not a new blind population.
Monitor replay establishes execution agreement, not decoded quality or export qualification.

## Tests

`tests/test_historical_clip.py` rejects changed bytes, altered identities, fractional gates and invalid endpoint state.
The independent native replay remains required before scientific scoring.
