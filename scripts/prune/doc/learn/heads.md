# learn/heads

## Objective

Run bounded resident-model phases for the learned-head experiment.

## Usage

Started by: a direct human or agent CLI. Run from `LTX-2` in conda environment `ltx`.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx python -m scripts.prune.learn.heads --study ../expr/refiner_prune/2.5/learned_heads_dev8_20261012_r2/configs/frozen_study.json --phase feasibility --gpu-id 0
```

`--phase` selects one bounded phase. `--arm`, `--replicate-seed` and `--removed-heads` select the frozen arm. `--branch-prefix` reuses a verified 700-update prefix. `--max-updates` bounds a smoke run.

Outputs: Phase receipts, learning state, observations, monitoring predictions and timing probes.
See the [run guide](../usage.md#learned-head-phases).

## Data flow

The frozen study selects explicit dev components and draws. The shared input owner constructs every complete native D0 input.

## Organization logic

Feasibility checks native velocity conversion, all-one gates and zero adapters. Zero-adapter parity runs under `no_grad`; only optimizer forwards enable autograd. Ten backward updates test each complete clip length.
Execution probes export three 152-head layouts and four FFN widths. ABA/BAB timing repeats twenty times when drift exceeds two percent.
Selection uses 700 soft and 300 binary updates. Soft-only uses 1,000 soft updates. Fixed energy and random arms ramp for 300 updates.
Recovery slices adapters and Adam moments at original head IDs. Fresh controls reset adapters and moments. Sequential recovery uses 1,500 adapter updates.

## Invariants

The caller enables autograd locally. Base weights stay frozen. Non-reentrant block checkpointing is enabled. There is one legal decision per optimizer update.

## Gotchas

A failed memory or execution gate prevents sustained speed training. Timeout retains progress. Regional training rejects unavailable reviewed tracks. Monitoring never creates teacher substitutes.

## Tests

Run `test_learned_gates.py, test_head_learning_state.py, test_bench_learned_heads.py` with CUDA hidden. Native BF16 checks remain separate scientific gates.

## Numerical and execution records

Feasibility saves native velocity, ordinary clean predictions and initialized sigmoid-gate predictions at each exact sigma.
Its ten optimizer updates per length use two sequential backward graphs with unchanged logits per update.
Execution probes preserve functional-mask predictions separately from dense and compact outputs.
They record both the functional/export difference and the intentional dense/pruned difference.
Fresh and sequential arms retain their extra-compute lineage.
Monitoring uses precomputed original-teacher references on four validation actors at sigma indices 0, 4 and 7.
It saves trained and hardened predictions with the same adapter weights at each selection checkpoint.
A timeout checkpoints the latest completed update; completed evidence remains immutable.

## Native placement diagnostics

The four-GPU diagnostic wraps the shared Session with twelve consecutive blocks per device.
Feasibility compares every native prediction against saved single-GPU references before optimization.
It publishes each completed backward update and per-device peak allocated/reserved bytes and free headroom.
It rejects zero gate/adapter gradients and changes to frozen parameter version counters.
Failures retain current clip length, update and sigma in the diagnostic receipt.
