# `core/preflight.py` — checkpoint and GPU checks

## Objective

Check registered components, the distilled sigma grid and GPU headroom before loading a model.

## Usage

A human or agent can run this check directly from `LTX-2` in the `ltx` environment.
It reads metadata and queries CUDA. It does not load transformer weights.
There is no `--dry-run` option.

```bash
conda run -n ltx python -m scripts.prune.core.preflight \
  --model 2.5 --gpu-id 0 --min-free-gb 44 --dump-caps
```

| Argument | Meaning |
|---|---|
| `--model` | Registered key; default 2.5 |
| `--gpu-id` | Check and select this visible device |
| `--min-free-gb` | Required free decimal GB; default 4 |
| `--dump-caps` | Save capabilities and provenance |

Output is printed metadata and, when requested, `../expr/refiner_prune/2.5/caps.json`.
See the [run guide](../usage.md#preflight-and-prompt-cache).

## Data flow

`check` resolves model paths and verifies a nine-value sigma grid starting at 1 and ending at 0.
It requires CUDA and sufficient free memory.
`dump_caps` writes the resolved capabilities, grid and provenance.

## Organization logic

With `gpu_id`, check that device's memory and make it the current CUDA device.
Without it, require at least one device meeting the memory threshold.
`free_gpus` returns indices and memory values in decimal GB.

Worked example: a requested device with 3 GB free fails the default 4 GB threshold even if another device is free.

## Invariants

- Check the requested device, not just any free device.
- Select the current CUDA device for backend launches.
- Resolve files and metadata before loading weights.

## Gotchas

The default 4 GB threshold does not establish capacity for a 22B whole-clip forward.
Scoring, ablation and benchmarking use a separate 44 GiB check.
This command's decimal GB differ from GiB.
Check `nvidia-smi` before GPU work. Hidden devices use visible CUDA indices.

## Tests

The CPU suite checks imports and registry interfaces.
Native checks must verify device selection and model-loading capacity.
