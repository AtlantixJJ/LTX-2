# Plan — Training-based pruning and recovery for the LTX refiner

Companion to [`2026-08-26-refiner-head-ffn-pruning.md`](2026-08-26-refiner-head-ffn-pruning.md)
(the training-free plan). Everything here updates parameters by gradient descent.
Date: 2026-08-27 · Env: `ltx` conda env · Hardware: 8× RTX A6000 (49 GB)

**Do not start this plan until the training-free plan's gates are measured.** Its Phase 2/3
results are this plan's baseline, its harness is this plan's infrastructure, and its achieved
sparsity tells you whether training is needed at all. Starting here first would mean training
against a selection that a free method could have found.

---

## 1. Why this plan exists

The training-free plan is expected to reach **25–40%** of video-branch parameters at the
quality gates. Two specific things it cannot do:

1. **Joint selection.** Ranking heads independently — even iteratively — is greedy. Two
   heads that duplicate each other both score high; the iterative schedule corrects this one
   removal at a time, but never *optimizes* the set. Learned gates minimize the deployment
   objective over the whole mask at once, which is the difference between a good greedy
   solution and a good solution.
2. **Weight adaptation.** Least-squares re-solve only re-mixes the surviving `to_out` /
   `ff.net.2` columns. It cannot move `to_q`/`to_k`/`to_v` or `ff.net.0.proj` to compensate
   for what was removed. Recovery fine-tuning can.

Together these are worth roughly the gap between **~30%** and **~50%** parameter reduction
at the same quality gates. That is the whole justification for the extra ~4 days.

## 2. Inherited from the training-free plan (do not rebuild)

| Component | Use here |
|---|---|
| `scripts/prune/refine_task.py` | the deployment conditions — identical |
| `scripts/prune/model_registry.py` | `--model {2.3,2.5}`, `ModelPaths`, probed `ModelCaps` |
| `scripts/prune/chunk_states.py` | AR-geometry states (frozen ctx + 1–3 fresh latent frames) |
| `scripts/prune/teacher.py` | `x0*` targets from the deeper `k8` schedule; on-policy + renoised state families |
| `scripts/prune/losses.py` | masked x0-space loss and `rel_l2` |
| `scripts/prune/metrics.py` | T0 / T1 / T2 / T3 tiers, unchanged |
| `scripts/prune/hooks.py` | `attn.to_out[0]` and `ff.net[2]` pre-hooks — the same mask carriers |
| `scripts/prune/gates.py` | the same JSON verdict, so both plans' results are directly comparable |

**Nothing in this plan re-specifies the task, the geometry, the target, or the metrics.**
If a change is needed to any of them, change it in the training-free plan's module so both
plans move together — otherwise the two sets of numbers stop being comparable, which defeats
the purpose of running them in sequence.

**One loss-construction note carries over and matters here too:** the naive self-distillation
loss `‖f(ξ) − f(1)‖²` has zero gradient at ξ = 1. That degeneracy is fatal to the *Taylor*
estimator but harmless during *training*, since training moves ξ off 1 on the first step.
Keep the `x0*` target anyway — it costs nothing and keeps the objective bit-identical across
both plans, which is what makes the comparison honest.

---

## 3. Phase T1 — learned head gates with a hard budget  (~2 days)

Replaces §7.2's greedy ranking with joint optimization over the mask.

Train the ~3072 `ξ` scalars — the only trainable tensor, ~12 KB, negligible optimizer state —
against `L = ‖D_θ − x0*‖² + λ·Ω(ξ)`, with `Ω` an expected-L0 penalty (hard-concrete gates,
Louizos et al. / CoFi-style) or a straight-through top-k with an annealed budget.

```python
xis = attach_head_masks(model, kind="hard_concrete")   # log_alpha params, not raw scalars
for p in model.parameters():
    p.requires_grad_(False)                            # weights stay frozen in this phase
opt = torch.optim.Adam(xis.parameters(), lr=1e-2)

for step, batch in enumerate(loader):
    res, _ = denoiser(wrapped, batch.state, None, sigmas, batch.step_idx)
    loss = x0_loss(res.denoised, batch.x0_star, batch.state) + lam * expected_l0(xis)
    loss.backward(); opt.step(); opt.zero_grad()
    anneal_temperature(xis, step)                      # -> hard 0/1 by the end
# harden: keep top-k by expected gate, write a 0/1 mask, then re-solve to_out[0] (lstsq.py)
```

Design points:

- **Let the L0 penalty allocate globally** across `attn1` / `attn2` rather than imposing
  per-type budgets — then *report* what it chose. That allocation is itself a finding, and
  it is the clearest evidence for or against the "text cross-attention is nearly inert under
  a constant prompt" hypothesis.
- **Compute per-σ masks as well as their union.** The refiner runs only 2–3 forwards, so a
  different head set per step costs nothing at inference when weights are shared; structural
  removal needs the union. The gap between the two quantifies how much σ-specialization is
  available, and it is free to measure once the gates are trained.
- **Follow hardening with the least-squares `to_out[0]` re-solve** from the training-free
  plan (`lstsq.py`). Gates choose the set; the re-solve fixes the mixing. Skipping it leaves
  quality on the table for zero saved effort.
- **Warm-start from the training-free scores.** Initialize `log_alpha` from the §7.2b
  ranking rather than uniformly — it converges faster and makes the comparison a strict
  improvement-over-baseline rather than a different random draw.

Cost: forward+backward ≈ 3–4× a forward ≈ 4–5 s/step at chunk = 4; 500 steps ≈ 45 min per λ
on one A6000. Sweep 6 λ values across 6 GPUs in parallel.

**Gate:** at the training-free plan's achieved sparsity, learned gates must beat the greedy
mask on T0 **on held-out clips**; then push λ until a gate fails, to find the new ceiling.
If gates do *not* beat greedy at matched sparsity, stop — the extra machinery is not earning
its keep, and that is a legitimate result to report.

---

## 4. Phase T2 — LoRA recovery fine-tune  (~2 days)

Self-distillation of the pruned student against the unpruned teacher, restricted to the
deployment regime.

- Teacher: the unpruned **video-only** model (`load_transformer(video_only=True)`).
- Student: the pruned checkpoint from the training-free plan's §9 export (or Phase T1's).
- Loss: `x0_loss` on fresh-chunk tokens at the **refine sigmas only**.
- LoRA targets: `["to_k","to_q","to_v","to_out.0"]` (the config default) plus
  `ff.net.0.proj` / `ff.net.2` — the FFN targets matter here precisely because the
  least-squares re-solve could not touch `ff.net.0.proj`.
- Data: AR chunks from the 40 calibration clips at the {1,2,3} chunk mixture.
- **Fuse the LoRA into the pruned weights** for the shipped artifact — a LoRA at inference
  is an extra GEMM per target module, which directly cancels part of the speedup this whole
  effort is buying.

### Coding guidance

The trainer's timestep sampling (`packages/ltx-trainer/src/ltx_trainer/timestep_samplers.py`)
is distribution-based. Check whether a fixed-set sampler already exists; if not, add a small
`FixedSigmaSampler` drawing uniformly from the k2 schedule. Do not reuse a continuous
sampler with a narrow distribution — the refiner sees exactly two sigmas, and training on a
smear around them wastes capacity on states deployment never visits.

```yaml
# expr/refiner_prune/<model-key>/recovery/config.yaml  (sketch)
model:
  model_source: <pruned transformer .safetensors>
  video_only: true
lora:
  rank: 64
  target_modules: ["to_k","to_q","to_v","to_out.0","ff.net.0.proj","ff.net.2"]
timesteps:
  sampler: fixed_sigmas
  sigmas: [0.725, 0.421875]
```

Launch with the workspace's documented FSDP command:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 29503 \
  --config_file LTX-2/packages/ltx-trainer/configs/accelerate/fsdp.yaml \
  LTX-2/packages/ltx-trainer/scripts/train.py \
  "$PWD/expr/refiner_prune/<model-key>/recovery/config.yaml" --disable-progress-bars
```

The trainer's DDP validation-unwrap fix, per-rank quantization-device fix and
filesystem-based validation-path synchronization must all be present (they are — see the
workspace `CLAUDE.md`; do not revert them, and do not restore GPU `gather_object()` for
validation results).

Budget: 4 GPUs × ~8 h.

**Gate:** T2 rollout drift back to (or below) the unpruned model's intrinsic floor; T1 PSNR
recovered above the gate; **measured speedup unchanged after LoRA fusion** — verify the fused
checkpoint benchmarks the same as the pre-LoRA pruned one, since an unfused LoRA silently
gives back the win.

---

## 5. Phase T3 — full fine-tune of the pruned model  (~2 days, only if needed)

Escalate only if T2 leaves a visible T1/T2 gap at the target sparsity. Same data, same loss,
same sigmas; all weights trainable, low LR, short schedule, early stop on held-out T0.

Two cautions specific to this task:

- **Do not let the fine-tune drift off the refine regime.** Full fine-tuning at two sigmas
  on 40 clips will happily specialize the model onto those clips. Hold out 12 clips and stop
  on their T0, not on training loss.
- **Re-run the full gate suite after**, including T3 human review. A full fine-tune can trade
  a metric win for a visible texture change that PSNR does not penalize.

---

## 6. Targets

| Metric | Gate |
|---|---|
| T0 / T1 / T2 / T3 | **identical to the training-free plan** — same `gates.py`, same thresholds |
| Parameter reduction | **50% of executed video-branch params** (stretch), vs ~30% training-free |
| Speed | ≥1.8× at 50% params off, measured at chunk ∈ {1,2,3} |
| LoRA fusion | fused checkpoint benchmarks within noise of the unfused pruned model |

Report the training-free and training-based results **side by side in one table**. The
decision this plan exists to inform is "is the extra sparsity worth the extra pipeline
complexity", and that is only answerable if both columns use the same gates on the same
held-out clips.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Learned gates do not beat the greedy mask at matched sparsity | Explicit Phase T1 gate — stop and report; the training-free result ships |
| Gate training collapses to a degenerate mask (all one layer) | Per-layer floor on kept heads; monitor the allocation across layers each 50 steps |
| Recovery fine-tune overfits 40 calibration clips | Held-out 12-clip T0 as the stopping signal; never train on eval clips |
| LoRA left unfused at deploy | Benchmark gate on the fused artifact; the fusion step is part of export, not an afterthought |
| Training specializes the model further onto one corpus | Already a task-specialized refiner (§1 of the free plan); document the corpus in provenance and never reuse for general generation |
| Divergence between the two plans' harness modules | Shared modules live in the training-free plan's package; this plan imports, never forks |

## 8. Schedule

| Phase | Days |
|---|---|
| T1 learned head gates | 2 |
| T2 LoRA recovery | 2 |
| T3 full fine-tune (conditional) | 2 |
| **total** | **4 days (6 with T3)** |

Runs after the training-free plan's ~9 days, not in parallel — T1 needs its scores as a
warm start and its masks as a baseline.
