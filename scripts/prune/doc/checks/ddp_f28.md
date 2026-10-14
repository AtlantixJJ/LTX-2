# checks/ddp_f28

## Objective

Test whether a complete F18 or F28 clip fits per GPU during two-rank DDP learned-head training.
Each GPU stores the full video-only BF16 dev model. There is no weight sharding or CPU offload.

## Usage

Run this bounded diagnostic directly from LTX-2 in the ltx environment, after checking free GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD" \
  python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/prune/checks/ddp_f28.py.txt \
  --study /abs/source/configs/frozen_study.json --output /abs/fresh/diagnostic --updates 3
```

Add `--latent-frames 18` for complete F18 source clips: 137 RGB frames instead of F28's 217.
Both clip lengths use 1024×1024 recorded crops. The default remains F28.

The `.txt` suffix keeps this diagnostic outside existing frozen learner Python inventories.
It is an executable Python file. The explicit PYTHONPATH supplies the package root.
See [launch rules](../usage.md#launch-and-output-rules).

Outputs are `provenance.json`, per-rank stage/memory receipts and saved native input manifests under the fresh output root.

## Data flow

Verify the source study content hash and checkpoint hashes. Record current code independently from the original frozen study.
Select two distinct learning clips with the requested latent length. Keep the original capture, crop, matte and empty text context.
The native input owner creates complete clips with exact clean frame zero and saved noise.
F18 contains 18,432 tokens; F28 contains 28,672. No source clip is truncated.
The native forward predicts velocity. FP32 generated-token MSE and the raw-logit sparsity penalty supply the loss.

## Organization logic

Register the native model and learned gates in one DDP module. Attach the native gate hooks before DDP initialization.
Use native non-reentrant gradient checkpointing. Keep BF16 frozen weights and FP32 adapter/gate parameters.
Each rank computes one undivided clip loss. DDP averages gradients once, giving the two-clip mean and one sparsity penalty.
Three updates exercise forward, backward, Adam state allocation and subsequent DDP bucket handling.
Measure allocation, reservation and device-free memory at each stage. Save a failure receipt before propagating an exception.

## Invariants

No shortened clip or altered spatial geometry. Rank and loss settings come from the source config.
Each GPU holds a full model replica and one clip. No FSDP wrapper or block placement is used.
Adapters and logits must have finite nonzero gradient norms. Frozen parameter versions must remain unchanged.
Both ranks must have identical adapter and gate state hashes after each successful update.
Only the fresh diagnostic output root receives artifacts; source training files stay intact.

## Gotchas

DDP also stores gradient buckets. Forward success alone does not prove training fits.
The first optimizer step allocates Adam moments. A later update can rebuild DDP buckets.
An out-of-memory exception means this exact configuration failed; it does not test quantization or offload alternatives.
Torchrun terminates the peer when a rank fails. Inspect both rank receipts and the launcher log.

## Tests

The native diagnostic itself checks exact first-frame preservation, gradients, frozen versions and rank state equality.
Compile and import the file in the ltx environment before launching. Run at most five updates.
