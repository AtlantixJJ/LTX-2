# Configuration — Accelerate YAMLs and the D0/D1 run recipes

Two different things live under this heading, and conflating them is the mistake this file
exists to prevent:

* **`fsdp_{2,3,4}gpu.yaml`** configure the *process and sharding topology* — how many ranks, how
  FSDP shards and saves. They say nothing about the experiment.
* **The recipes in §2** configure the *experiment* — arm, objective, forcing, geometry, σ, LoRA,
  schedule. `train.py` is a **CLI-driven trainer**: there is no experiment YAML and no config
  loader. There is deliberately no `d0.yaml`/`d1.yaml`; a file like that could not be loaded.

> Every recipe uses `clean_c0_v1`: the supplied real first frame is clean at timestep zero in
> block 0 and is retained as the pinned cache sink.

Definitions of the arms and axes: [`../doc/experiments.md`](../doc/experiments.md).
Mechanics: [`../doc/core_algorithm.md`](../doc/core_algorithm.md).

---

## 1. Accelerate configs

Copies of `packages/ltx-trainer/configs/accelerate/fsdp.yaml` at 2, 3 and 4 processes, with two
deliberate differences. The trainer's own file is left untouched — the LTX-2.3 I2V run in `expr/`
depends on it — so `fsdp_4gpu.yaml` here is a copy at the same process count, not a reference.

| Setting | Trainer's fsdp.yaml | Here | Why |
|---|---|---|---|
| `num_processes` | 4 | 2 / 3 / 4 | Preliminary runs take whatever cards are free. Drop `--lora-rank`, never `K` — `K` is what the loop exists to exercise. |
| `fsdp_cpu_ram_efficient_loading` | `true` | `false` | `train.py` loads the 42 GB bf16 checkpoint **straight onto each GPU** (`--init-device cuda`) rather than staging it in host RAM. Three ranks staging on the host would want ~126 GB of a machine with ~139 GB free, and host-RAM contention here has hung jobs for hours. FSDP shards in place, so the 42 GB is transient and fits a 49 GB card. |
| `fsdp_state_dict_type` | `SHARDED_STATE_DICT` | `FULL_STATE_DICT` | `save_lora` gathers the adapter on the main process and writes ONE ComfyUI-compatible `.safetensors`, the layout `DiffusionStage.with_loras` fuses at load. The adapter is tens of MB; there is nothing to shard. |

Memory at 2 GPUs: ~21 GB of sharded weights per rank plus the all-gather buffer, the
gradient-checkpointed activations of **one block**, and the LoRA grads/Adam state. Detaching
between blocks keeps that at one block regardless of `K`.

**Pick the YAML that matches the cards you actually have free** (`nvidia-smi` first), and give
each concurrent launch its own `--main_process_port`.

---

## 2. The four named recipes

All commands run **from the LTX-2 repo root** in the `ltx` conda env.

### Path substitutions

| Placeholder | Meaning |
|---|---|
| `<SUBSET>` | a frozen subset JSON under `../expr/onestep_avatar/windows/` |
| `<OUT>` | a **fresh** run directory under `../expr/onestep_avatar/runs/` — a used `--output` is refused unless `--overwrite` archives it |
| `<GPUS>` | the free device ids, e.g. `2,3` |
| `<PORT>` | a free `--main_process_port`, e.g. `29517` |

`--corpus-root` is **not** passed: it defaults to the subset's own recorded `corpus_root`, which
is where `precompute.py` wrote the masters. Pass it only to read a relocated copy.

### Freezing the subset first — where `K` comes from

`K` (blocks per training sample) is a **property of the frozen subset**, chosen at freeze time by
`windows.py --chain-length`. `train.py` has no `--chain-length` flag; it reads `K` from the
subset's chains.

```bash
# bg (needs guide renders; --require-guide checks the render MP4, not the guide latent -- G6/F4)
conda run -n ltx python -m scripts.onestep_avatar.windows \
  --name t2r2 --objective bg --require-guide \
  --max-actors 8 --chain-length 3 --min-holdout-actors 2

# white D0 (capture-only: no --require-guide, since D0 reads no guide artifact at all)
conda run -n ltx python -m scripts.onestep_avatar.windows \
  --name white-d0-t2r2 --objective white \
  --max-actors 8 --chain-length 3 --min-holdout-actors 2
```

`--min-holdout-actors` defaults to 12; at 8 actors use 2, or the training split is left with one
actor. The subset records the objective it was frozen against, and `train.py` refuses a
`--objective` that disagrees.

### R1 — `d0_teacher_forced`

The capacity diagnostic with a clean ground-truth history. Not deployable.

```bash
CUDA_VISIBLE_DEVICES=<GPUS> accelerate launch \
  --config_file scripts/onestep_avatar/configs/fsdp_2gpu.yaml --main_process_port <PORT> \
  -m scripts.onestep_avatar.train \
  --subset <SUBSET> --output <OUT> \
  --model 2.5 --objective white --guide-mode d0 --teacher-forcing \
  --sigma0 0.725 --block-latent-frames 2 --context-latent-frames 15 \
  --lora-rank 32 --lora-alpha 32 --lora-target attn \
  --lr 1e-4 --warmup-steps 20 --steps 2000 --seed 42 \
  --save-every 100 --anchor-weight 0.0
```

### R2 — `d0_self_forced`

The same diagnostic under the regime deployment actually uses. Drop one flag:

```bash
#   ... --guide-mode d0          (no --teacher-forcing)
```

Run R1 and R2 as a pair when the question is exposure-bias drift: one variable, everything else
held.

### R3 — `d1_teacher_forced`

The deployable arm with a ground-truth history — a training ablation, not a deployment mode.

```bash
CUDA_VISIBLE_DEVICES=<GPUS> accelerate launch \
  --config_file scripts/onestep_avatar/configs/fsdp_2gpu.yaml --main_process_port <PORT> \
  -m scripts.onestep_avatar.train \
  --subset <SUBSET> --output <OUT> \
  --model 2.5 --objective bg --guide-mode d1 --teacher-forcing \
  --sigma0 0.725 --block-latent-frames 2 --context-latent-frames 15 \
  --lora-rank 16 --lora-alpha 16 --lora-target attn \
  --lr 1e-4 --warmup-steps 20 --steps 2000 --seed 42 \
  --save-every 100 --anchor-weight 0.0
```

Training refreshes the cache from the **target** `z_y` here, which is correct. The generic
`causal_core.rollout(teacher_forcing=True)` does not — it refreshes from the guide, which equals
the target for D0 only ([G2](../doc/known_gaps.md#g2--generic-teacher-forced-rollout-refreshes-from-the-guide-not-the-target)).
Do not probe a D1 teacher-forced checkpoint through that path expecting the training regime.

### R4 — `d1_self_forced`

**The baseline every other arm has to beat** — the deployable arm under the deployable regime.

```bash
#   ... --guide-mode d1          (no --teacher-forcing)
```

### The `bg` / `white` substitution

Change **two** things together and nothing else:

1. `--objective bg` → `--objective white`;
2. `<SUBSET>` → a subset frozen with `windows.py --objective white`.

Subset requirements per combination:

| Arm × objective | Subset needs | Ready today |
|---|---|---|
| `d0` + `white` | capture masters only | **yes** — 3,360 white captures exist |
| `d0` + `bg` | capture masters only | yes |
| `d1` + `bg` | capture **and** guide masters, current under `dataset.GUIDE_COMPOSITING_VERSION` | **no** — 18 of 19 guide pairs are stale under v2 ([G6](../doc/known_gaps.md#g6--guide-artifacts-on-disk-predate-the-compositing-fix)) |
| `d1` + `white` | same | **no** — zero white guide renders exist |

---

## 3. Which values are defaults, and which are choices

`train.py --help` is the authority; this table is the reading of it at the time of writing.

| Flag | Default in `parse_args` | In the recipes |
|---|---|---|
| `--model` | `2.5` | **checked default**, stated explicitly |
| `--sigma0` | `0.725` | **checked default** — the deployed operating point |
| `--block-latent-frames` | `2` (`causal_core.BLOCK_LATENT_FRAMES`) | **checked default** — the deployed 16-pixel-frame stride |
| `--context-latent-frames` | `15` (`causal_core.CONTEXT_LATENT_FRAMES`, max 16) | **checked default**; a retained history of 16 latent frames including the pinned sink — the compute/quality knob |
| `--guide-mode` | `d1` | experiment choice |
| `--objective` | `bg` | experiment choice; must match the subset |
| `--teacher-forcing` | off | experiment choice |
| `--lora-rank` | `8` | **example choice** (16/32 above) |
| `--lora-alpha` | equal to `--lora-rank` | **keep it equal**: non-unit `alpha/rank` is stamped but not applied at fusion ([G3](../doc/known_gaps.md#g3--checkpoint-and-artifact-conditions-are-recorded-but-not-enforced)) |
| `--lora-target` | `attn` | **checked default**; `attn_ffn` adds the FFN projections |
| `--lr` | `1e-4` | **checked default** |
| `--warmup-steps` | `20` | **checked default** |
| `--steps` | `200` | **example choice** (2000 above) |
| `--seed` | `42` | **checked default** |
| `--save-every` | `100` | **checked default** |
| `--anchor-weight` | `0.0` | **the only accepted value** — the anchor is disabled; any nonzero value is rejected before the subset is read |
| `--max-grad-norm` | `1.0` | default |
| `--init-device` | `cuda` | default; see the FSDP table above |

`K` is **not** a `train.py` flag. `--overwrite`, `--dry-run`, `--skip-subset-check`,
`--save-initial`, `--timing` and the W&B flags are operational, not experimental.

### Fixed σ versus `--sigma-levels`

`--sigma0` trains **one** operating point. `--sigma-levels a b c` instead trains one adapter
across several, assigning `sigmas[(rank + step) % len(sigmas)]` — so levels are mixed *within* a
step and every rank walks through every level over the run. It overrides `--sigma0` and stamps
`onestep_avatar_sigma0 = "mixed"` plus the level list, so a fixed-σ loader cannot mistake it for
calibrated. `σ = 0.0` is refused (it noises nothing: loss and gradient are identically zero), and
repeated levels are refused.

The candidate grid is the distilled checkpoint's own: `{0.421875, 0.725, 0.909375}`. Note that
`train.py` accepts any value in `(0, 1]` while `onestep_core` accepts only on-grid points —
[G5](../doc/known_gaps.md#g5--training-and-deployment-disagree-about-valid-sigma).

### What a run records

`<OUT>/config.json` and the W&B run config record the resolved configuration; each checkpoint's
safetensors metadata carries σ₀ (or `"mixed"`) and the level list, `K`, the schedule, the
attention kind, block and cache geometry, the subset hash, objective, guide mode, anchor weight,
teacher forcing, LoRA rank/alpha/target, and `loss=full_frame_x0_mse`. **That metadata is the
record of a particular run** — but nothing reads it back on load, so it does not currently protect
a probe or a deployment from off-condition use (G3). A run directory name is not provenance.

---

## 4. Sanity and probe commands

```bash
# Zero-init check: step 0's LoRA B must export as exactly zero; step 1 is the first update.
#   ... --guide-mode d0 --save-initial --save-every 1 --steps 1

# Decode a D0 checkpoint: `capture | frozen base | LoRA` per probe sigma.
conda run -n ltx python -m scripts.onestep_avatar.visualize_d0 \
  --subset <SUBSET> --run <OUT> --steps 0 1 --output <OUT>/probes/init --gpu-id <ID>
```

The probe is **D0-only** ([G4](../doc/known_gaps.md#g4--no-d1-probe)) and does not validate the
adapter's recorded conditions (G3): it always uses the default deployed geometry and all three
`PROBE_SIGMAS`. Do not present its output as a D1 result, and label a probe at a σ the adapter was
not trained at as off-condition.
