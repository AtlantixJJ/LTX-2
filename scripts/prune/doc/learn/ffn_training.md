# learn/ffn_training

## Objective

Compute joint head/FFN updates and save exact, versioned learning state.

## Data flow

`JointStep` registers the native model, head logits and FFN scores inside one module.
DDP sees all trainable parameters through that module's forward.
Live providers multiply native activations before their output projections.
The official velocity loss excludes the supplied clean frame.

## Organization logic

The joint arm uses existing sigmoid head gates, legal binary refinement and rank-128 V/output adapters.
The FFN-only arm uses all-one head gates and no attention adapters.
Both arms use rank-16 FFN up/down adapters and individual binary channel decisions.
FFN masks ramp through aligned counts, then keep their final width while identities remain learnable.
Only soft head learning adds the original head sparsity loss.
There is no FFN sparsity loss because the forward already enforces its count.

At selection completion, select and freeze the final original IDs once.
Recovery keeps registered score parameters with zero-gradient paths. Only the adapter optimizer steps.
For example, a 16-channel FFN retaining eight channels has eight exact ones in every forward at its final width.
The sigmoid supplies gradients for all 16 scores during selection, including masked channels.

`head_ffn_learning_state_v1` stores both score families, masks, adapters, Adam states, rank RNG and input chain.
Immutable checkpoint sidecars pin file and state hashes. Resume validates geometry, binary counts and rank count.
The ordinary loader restores RNG. Inference explicitly disables RNG restoration when device numbering changes.

## Invariants

Freeze base weights. Keep score and adapter masters FP32.
Select once per optimizer update and keep the decision through checkpoint recomputation.
Average DDP gradients once. Clip head scores, FFN scores and adapter gradients separately.
Keep fixed masks and gate optimizer moments unchanged during recovery.

## Gotchas

The straight-through gradient is approximate. It is not the derivative of the discrete selector.
The final post-update selection can change IDs relative to that update's forward.
Fusion and compact BF16 execution need independent numerical checks.

## Tests

`test_ffn_learning.py` checks independent mask algebra, live gradients and native GELU boundaries.
It checks frozen weights, exact resumed updates across selection/recovery, original-ID Adam transfer and four-rank DDP averaging.
