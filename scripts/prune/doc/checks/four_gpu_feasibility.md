# checks/four_gpu_feasibility

## Objective

Test complete native forward and backward with four-device block placement.

## Usage

Run directly from `LTX-2`, in conda environment `ltx`. This is a bounded diagnostic, not sustained training.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run --no-capture-output -n ltx python -m scripts.prune.checks.four_gpu_feasibility --study ../expr/refiner_prune/2.5/learned_heads_dev8_20261012_r3_four_gpu/configs/frozen_study.json --reference-study ../expr/refiner_prune/2.5/learned_heads_dev8_20261012_r2/configs/frozen_study.json
```

The frozen `native_placement` declares four logical and physical devices. Use exactly four visible GPUs in order.
Outputs include placement provenance, individual update receipts, progress, failure context and final feasibility results.

## Data flow

Validate both frozen study identities. Require identical components, actors, text context, software and scientific settings.
Load the native model through the shared Session. Place twelve consecutive blocks on each GPU.
Keep preprocessing, output projection, loss and logits on GPU zero.
Transport complete native `TransformerArgs` with differentiable copies. Return the last block output to GPU zero.
Compare every native control against the saved single-GPU reference before backward.

## Organization logic

The placement context adds pre-hooks on blocks and a final output hook. These hooks also run during checkpoint recomputation.
The native velocity owner and official loss owner remain unchanged. Rank-16 adapters follow their block parameters.
The shared feasibility phase runs ten optimizer updates per complete clip length and publishes each completed update.

## Invariants

Full clips, sigmas, noise, c0, precision, target, adapter rank and gate rules stay unchanged.
The reference comparison is bit-exact. Audio and causal caches are rejected.
Every device must retain two GiB of measured headroom. Frozen parameters must retain their version counters.
This test does not change the single-GPU inference timing protocol or enable sustained training.

## Gotchas

The context removes hooks on failure. The enclosing Session disposes all parameters on their current devices.
Do not move the complete trained model back to one GPU during cleanup.
Failure context names the clip length, update and sigma. It does not claim an unfinished update completed.
The four-GPU test occupies all four devices; stop independent jobs in this pool first.

## Tests

`test_placement.py` checks structured transport, checkpoint gradients and cleanup.
Its native four-device tiny model requires bit-exact outputs, input gradients and parameter gradients against one GPU.
The complete model's parity and memory controls remain separate native gates.
