# checks/ddp_native

## Objective

Accept native uniform-F18 DDP inputs, gradients, memory and resume before sustained training.

## Usage

Run directly from LTX-2 in ltx after checking GPU availability:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc_per_node=4 -m scripts.prune.checks.ddp_native --study /abs/frozen_study.json
```

The direct owner selects this entrypoint for the DDP preset. No avatar queue is used.
Outputs include `checks/ddp_native/update_0001.pt` and the matching `checks/feasibility.json` after all checks pass.
See the [run guide](../usage.md#uniform-f18-ddp-training).

## Data flow

A frozen study supplies components, learning sources, exact sigmas and crop semantics.
Alternate ranks use an F18 source or an F28 source cropped to F18. Longer-source controls require a nonzero start.
Use the production crop, model, loss, DDP and checkpoint owners.

## Organization logic

Verify exact local c0 and 18-latent geometry. Compare native output with zero adapters and all-one gates bit-for-bit.
Run three soft updates and one binary update, including Adam allocation and later bucket handling.
After update one, save all rank RNG states. Repeat update two from that checkpoint and require identical trained state and losses.
Check finite positive adapter/gate gradients, unchanged frozen parameters and two-GiB free headroom on every rank.
All ranks must agree on the trained-state hash after each update.
Only rank zero publishes acceptance, tied to the frozen study, method and actual world size.

## Invariants

This is bounded acceptance, not the formal trajectory. Trainable state from this check never enters formal training.
A failed crop, parity, memory, gradient or replay check leaves no passed feasibility receipt.
Full-source calibration and quality evaluation retain their own input contract.

## Gotchas

This check samples startup sigmas and inputs. It does not establish final quality, full-source validation memory or throughput.
The earlier two-rank F18 memory result is supporting evidence; it cannot substitute for this study's native four-rank acceptance.

## Tests

CPU DDP and crop tests verify independent averaging and replay. Run this native acceptance only when GPU execution is authorized.

## Progressive gate removal

For a progressive study, compare all-free native outputs bit-for-bit against ordinary soft gates.
A disposable three-update policy requests exactly three removals. It uses a three-head fraction and regret threshold one.
Its first synthetic snapshot makes exactly three heads decline; other heads increase.
Save after policy cursor two, then replay cursor three and the following update with checkpointed backward.
Require exact parameters, losses and policy. Check zero removed gradients, frozen logits and cleared Adam moments.
The three-head fixture proves that functional widths need not be multiples of four. It does not change the formal hyperparameters.
The native checkpoint filename remains `update_0001.pt`; its progressive cursor is two.
No disposable state advances the formal checkpoint-zero trajectory.

For zero/one continuation, load parent and migrated step-500 states and require bit-identical native predictions.
The disposable checkpointed-backward fixture selects three zeros and two ones at the same event.
Require exact execution, frozen gradients/moments and uninterrupted/resumed equality across the event.
Record `fixed_one_execution`, `arbitrary_fixed_one_count` and `parent_migration_bit_identical` in fresh acceptance.
