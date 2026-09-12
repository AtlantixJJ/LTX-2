# Accelerate configs for `train.py`

Copies of `packages/ltx-trainer/configs/accelerate/fsdp.yaml` at 2 and 3 processes, with two
deliberate differences. The trainer's own 4-GPU file is left untouched -- the LTX-2.3 I2V run
in `expr/` depends on it.

| Setting | Trainer's fsdp.yaml | Here | Why |
|---|---|---|---|
| `num_processes` | 4 | 2 / 3 | A preliminary run on whatever is free (plan §8.1: four free cards rarely exist on this box). Drop `--lora-rank`, never `--chain-length` -- `K` is the thing the loop exists to exercise. |
| `fsdp_cpu_ram_efficient_loading` | `true` | `false` | `train.py` loads the 42 GB bf16 checkpoint **straight onto each GPU** (`--init-device cuda`) rather than staging it in host RAM. Three ranks staging on the host would want ~126 GB of a machine with ~139 GB free, and host-RAM contention here has hung jobs for hours before. FSDP shards in place, so the 42 GB is transient and fits a 49 GB card. |
| `fsdp_state_dict_type` | `SHARDED_STATE_DICT` | `FULL_STATE_DICT` | `save_lora` gathers the adapter on the main process and writes ONE ComfyUI-compatible `.safetensors`, the same layout `DiffusionStage.with_loras` fuses at load. The adapter is a few tens of MB, so there is nothing to shard. |

Memory at 2 GPUs: ~21 GB of sharded weights per rank plus the all-gather buffer, the
gradient-checkpointed activations of **one** window (4096 tokens), and the LoRA grads/Adam
state. Detaching between chain windows (§4.4) keeps that at one window regardless of `K`.
