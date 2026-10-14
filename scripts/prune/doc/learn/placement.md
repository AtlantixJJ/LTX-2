# learn/placement

## Objective

Place consecutive native video blocks on four GPUs without changing model arithmetic.

## Data flow

The Session loads a native model on GPU zero. The context moves twelve blocks to each GPU.
Input hooks copy complete TransformerArgs to each block owner. Copies retain dtype and autograd connections.
The last output returns to GPU zero for the native final projection and loss.

## Organization logic

`PlacedSession` wraps the public Session and delegates its other properties.
It applies placement to every model lifetime, including the compact recovery model.
Diagnostic execution can specify saved single-GPU reference predictions.
Cleanup removes hooks. The enclosing Session releases the model on its actual devices.

## Invariants

Use exactly four visible devices, numbered zero through three. The first owner is the Session primary.
Preserve all native fields and checkpoint recomputation hooks. Reject audio and causal caches.
Keep model lifetime and scientific schedules with their existing owners.

## Gotchas

Moving the complete model back to GPU zero during cleanup can exhaust memory.
Single-GPU inference phases do not use this wrapper.

## Tests

`test_placement.py` compares checkpointed forward, input gradients and parameter gradients across four devices.
It also verifies exact replay of all four CUDA RNG streams after restore.
