# `LTX-2/scripts/onestep_avatar/` — design docs

Per-file design docs for the one-step LTX-2.5 avatar renderer. Corpus tooling and model
training live in this one package; see [`../CLAUDE.md`](../CLAUDE.md) for the env split
(only `build_guidance.py` needs `argavatar`) and the documentation contract.

**Plan:** `plans/2026-09-15-ltx25-one-step-argavatar-lora-core.md`. Section references
(SS1.2, SS1.5, SS1.6, …) are that plan's.

## The two rules that shape everything

**`causal_core.py` is the single implementation of "roll a causal block forward."** Training,
deployment, the probe, the benchmark and the subset freezer all call it. A train/deploy
mismatch would have to be an edit to that one file, not a divergence between two.

**One producer per artifact.** Every past bug here has been the same shape: two producers of
something that must have one. Before adding code, ask which artifact it produces or consumes.

| Artifact | Producer | Consumers |
|---|---|---|
| `capture_latent_manifest.json` — the crop box of record | `precompute.py --capture-only` | `build_guidance.py`, `windows.py` |
| `argavatar_render[_white].mp4` + `argavatar_alpha.mp4` | `build_guidance.py` | `precompute.py` (paired) |
| `ltx_vae_latent[_white].pt` — `z_y` | `precompute.py --capture-only` | `train.py`, `stats.py` |
| `argavatar_ltx_vae_latent[_white].pt` — `z_g` | `precompute.py` (paired) | `train.py`, `stats.py` |
| `capture_mask_crop.mp4` | `precompute.py` (paired) | `train.py`, `stats.py`; sampled QA copy for first five subjects/part, view 0 |
| the frozen subset JSON | `windows.py` | `train.py`, `visualize_d0.py` |

## Data flow

```
corpus (rgb.mp4, mask.mp4, bbox.npy, pose3d.npy, meta.json)
   │
   ├─ precompute.py --capture-only ─▶ crop box of record + z_y master (per objective)
   │                                   [geometry.py owns the square-crop rule]
   ▼
build_guidance.py  (argavatar env)  motion.py: pose3d → sam3db
   render into THAT box ─▶ composite over the objective's background
                        ─▶ guide video + argavatar_alpha.mp4   [qa.py scores IoU]
   ▼
precompute.py (paired) ─▶ z_g master · capture_mask_crop.mp4
   mask MP4 pair ─▶ train.py / stats.py pool transient latent grids on read
   ▼
windows.py ─▶ causal block chains + actor-disjoint split + sha256 pin ─▶ subset JSON
   ▼
train.py ── per block: denoise → backward → refresh → evict   [all four in causal_core]
   ▼  LoRA safetensors + metadata
   ├─▶ onestep_core.py   (deployment)
   ├─▶ visualize_d0.py   (decoded probe)
   └─▶ bench_forward.py  (cost vs k2)
```

## Files

### Corpus side

| Doc | Module | One line |
|---|---|---|
| [dataset.md](dataset.md) | `dataset.py` | corpus layout, and the **objective → filename** map every module obeys |
| [geometry.md](geometry.md) | `geometry.py` | the one square-crop rule (SS1.7), pure and GPU-free |
| [hashing.md](hashing.md) | `hashing.py` | the one `sha256(path)`, pure and GPU-free |
| [motion.md](motion.md) | `motion.py` | `pose3d.npy` → ARGAvatar `sam3db`, with the three conversions |
| [qa.md](qa.md) | `qa.py` | mask IoU at the dataset's own threshold |
| [mask_video.md](mask_video.md) | `mask_video.py` | mask storage — lossless gray MP4, 42× smaller than raw, bit-exact |
| [build_guidance.md](build_guidance.md) | `build_guidance.py` | **`argavatar` env.** Render the guide, harvest alpha, composite |
| [windows.md](windows.md) | `windows.py` | freeze the training subset: chains, split, content pin |

### Model side

| Doc | Module | One line |
|---|---|---|
| [causal_core.md](causal_core.md) | `causal_core.py` | **the** rollout: block plan, clip grid, causal mask, K/V cache, the three calls |
| [precompute.md](precompute.md) | `precompute.py` | one continuous VAE encode per view, per objective, resumable |
| [train.md](train.md) | `train.py` | the AR LoRA loop, the loss rule, the checkpoint contract |
| [onestep_core.md](onestep_core.md) | `onestep_core.py` | deployment rollout — the same three calls, one schedule |
| [stats.md](stats.md) | `stats.py` | measurement only: the excursion `a`, the gap `r`, latent moments |
| [bench_forward.md](bench_forward.md) | `bench_forward.py` | wall clock per finalized chunk, causal vs `k2` |
| [plot_training.md](plot_training.md) | `plot_training.py` | per-rank JSONL → training figures + summary |
| [visualize_d0.md](visualize_d0.md) | `visualize_d0.py` | decoded `capture │ base │ LoRA` probe per sigma |
| [report_d0.md](report_d0.md) | `report_d0.py` | artifact-checked handoff record for the D0 arm |

`__init__.py` carries no design. `configs/fsdp_{2,3,4}gpu.yaml` are accelerate configs;
`run_a1.sh` / `run_b2a.sh` / `run_b2b.sh` are launchers documented in [`../README.md`](../README.md).

## Keeping these docs true

A doc here is part of the change, not a write-up after it. When a module's **objective, data
flow, invariants, or contract with another module** changes, update its doc in the same commit.
Inline docstrings answer "why this line"; these docs answer "how this file fits the others" —
which is what a reader cannot reconstruct from one file, and what has actually gone wrong here.
