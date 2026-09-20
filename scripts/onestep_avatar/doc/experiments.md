# Experiments — the D0/D1, objective and forcing axes

The canonical definition of every experiment this package can run. The runnable commands are in
[`../configs/README.md`](../configs/README.md); the mechanics are in
[`core_algorithm.md`](core_algorithm.md).

**Three independent axes.** A run picks one value on each. They are orthogonal: the objective
does not change the code path, and the forcing policy does not change the arm.

| Axis | Values | Flag |
|---|---|---|
| **Arm** | `d0` (diagnostic) · `d1` = D1a (deployable) | `--guide-mode` |
| **Objective** | `bg` (the product) · `white` (subject-isolating) | `--objective` |
| **History** | self forcing (default) · teacher forcing (ablation) | `--teacher-forcing` |

All six combinations keep the clean supplied first frame `c0`, independent of arm and forcing
policy.

---

## 1. Arm

Both arms compute the same loss against the same target; they differ in **which master latent is
noised into the block input**.

| | **D0** — capacity diagnostic | **D1 / D1a** — guide adaptation |
|---|---|---|
| Block input | `(1−σ)·z_y + σ·ε` | `(1−σ)·z_g + σ·ε` |
| Loss target | `z_y` | `z_y` |
| Reduces to | ordinary flow matching on real video (`z_g = z_y` ⇒ `v* = ε − z_y`) | the real render→capture task |
| First-frame source (required) | `c0` from capture master frame 0 | the same `c0` — **not** the guide's frame 0, which is a render composite |
| Artifacts required | capture master only (`ltx_vae_latent[_white].pt`) | capture **and** guide masters; the guide must be current under `GUIDE_COMPOSITING_VERSION` |
| Loss | unweighted full-frame latent MSE, block-averaged | identical |
| CLI | `--guide-mode d0` | `--guide-mode d1` (the default) |
| Probe | `visualize_d0.py` (D0-only) | **none** — [G4](known_gaps.md#g4--no-d1-probe) |
| Deployable | **no** — `onestep_core.guide_conditionings` refuses `d0`, because there is no `z_y` at inference | yes, and the only deployable arm |
| Interpretation | the architecture's capacity ceiling at σ₀ when the correspondence gap is zero | whether a LoRA closes the measured render→capture gap |

**D0 is not a competing arm and its loss is not a D1 quality number.** D0 starts from the target
itself (`r = 0`), so its loss measures how well the frozen backbone plus a LoRA can reproduce real
video in one step. D1's loss additionally carries the render→capture correspondence gap
(subject-interior `r ≈ 0.89–0.93` measured 2026-09-12 by `stats.py`). Comparing the two numbers
compares two different problems. The legitimate reading is directional: if **D0** fails, the
bottleneck is capacity and no D1 tuning fixes it; if D0 is easy, the bottleneck is the
correspondence.

**D1b / D1c are deferred proposals, not runnable configurations.** D1b (latent blend of the
guide with the first frame before noising) and D1c (an additive guide embedding after
`patchify_proj`) were sketched in the September 15 plan. `--guide-mode` accepts only `d0` and
`d1`; there is no code for either. D1b additionally contradicts the package's own rule that
composites are built in **pixel** space, never as a latent blend — an unresolved design
contradiction, not queued work. The old plan's "nothing is rejected in advance" was a statement
about the design space; it is not an implementation status, and must not be quoted as one. The
former `d2` extra-token arm was dropped in 2026-09-13 (it cost 1.05× `k2`) and is not
expressible under block-causal attention at all — appended reference tokens are future context.

---

## 2. Objective

Both objectives share **every** code path, differing only in which pixels were encoded and which
filename holds them. An objective is never a second pipeline.

| | **`bg`** (default, the product) | **`white`** |
|---|---|---|
| Guide `z_g` | render composited over the clip's real first frame, in **pixel** space: `R_white + (1−α)·(B_frame0 − white)` | render on white — the compositing identity case |
| Target `z_y` | the real, unmatted capture | the capture with its background matted to white |
| Bundle names | unsuffixed: `ltx_vae_latent.pt`, `argavatar_ltx_vae_latent.pt`, `argavatar_render.mp4` | `_white` suffix: `ltx_vae_latent_white.pt`, `argavatar_ltx_vae_latent_white.pt`, `argavatar_render_white.mp4` (`dataset._suffix`) |
| Ghost band | present by design — `mask_0 \ mask_t`, a stale person-shaped patch from frame 0 — and part of the full-frame loss like everything else | does not exist: nothing to go stale |
| Required `c0` | the capture master's frame 0 for **this** objective | the **white** objective's first frame — never the unmatted `bg` latent, never a guide frame |
| Buys | the whole product | isolates the subject-texture gap from the background question |
| Corpus state (2026-09-18 inventory) | 3,360 capture masters; 19 guide pairs, of which **18 are stale under v2** ([G6](known_gaps.md#g6--guide-artifacts-on-disk-predate-the-compositing-fix)) | 3,360 capture masters; **zero** guide renders |

Consequences for what can run **today**: `white` + `d0` needs only capture bundles and is fully
ready. `bg` + `d1` needs the 19 guides rebuilt under v2 first. `white` + `d1` has no data at all.

A subset records the objective it was frozen against; `train.py` refuses a mismatch.

---

## 3. History — teacher vs self forcing

The two regimes differ in **exactly one tensor**: what the cache refresh is handed after a block
is denoised.

| | **Self forcing** (default, and what deploys) | **Teacher forcing** (`--teacher-forcing`) |
|---|---|---|
| Refresh input | `ẑ₀.detach()` — the block's own prediction at timestep zero | the clean ground-truth target for that block |
| Later blocks see | the model's own accumulated errors | a clean history the model did not produce |
| Role | the production setting; deployment has no ground truth | an **ablation**: isolates exposure-bias drift from everything else the AR loop changes |
| Probing | probe a self-forced checkpoint self-forced | a teacher-forced checkpoint never saw its own errors in the cache, so probe it teacher-forced or the input distribution is one training never produced |

**Teacher forcing is not access to future ground truth at deployment.** It is a training and
evaluation ablation only; nothing in a deployed rollout can supply it.

Two things to keep straight:

* "Clean" cached content means **timestep-zero**, not ground truth. A self-forced refresh writes
  generated content that is clean in exactly that sense.
* The generic `causal_core.rollout(teacher_forcing=True)` currently refreshes from the **guide**,
  which equals the target for D0 only —
  [G2](known_gaps.md#g2--generic-teacher-forced-rollout-refreshes-from-the-guide-not-the-target).
  Training does it correctly.

**Cache priming is a separate teacher-forced seam.** A chain that starts mid-clip fills the cache
from clean ground truth with one no-grad forward, in both forcing regimes. That is a
training/deployment difference in its own right and must be disclosed as one; the escalation is
to train whole clips (`windows.py --chain-length` covering the clip), which needs no priming.

---

## 4. What is implemented, deferred and historical

| | Status |
|---|---|
| D0, D1a; `bg`, `white`; teacher and self forcing | **implemented** and selectable from the CLI |
| Clean supplied first frame `c0` in every block | **implemented** as `clean_c0_v1` |
| D1 probe | **owed** ([G4](known_gaps.md#g4--no-d1-probe)) |
| D1b, D1c | **deferred proposals** — no code, unresolved design contradiction |
| `d2` extra reference tokens | **dropped** 2026-09-13; not expressible under causal attention |
| Anchor loss (`--anchor-weight`) | **disabled** — only `0.0` is accepted; no `base_denoised.pt` producer exists and one frozen per-view tensor cannot represent the anchor across chains/σ/history |
| Masked, subject-weighted or disagreement-weighted loss | **superseded** — the binding decision is unweighted full-frame latent MSE |
| Sliding windows with a frozen carryover | **historical** — replaced by block-causal attention + K/V cache (2026-09-14). `refine_core` keeps it for the `k2` baseline only |
| Pre-causal D0/D1 results, `runs/prelim/` | **historical evidence only.** `runs/prelim/` is an incomplete D1/`bg` causal run with merged launches and duplicate steps — not an experimental control. No `.safetensors` checkpoints exist under the current runs tree |

Old measurements made under masked loss or the pre-causal scheme are historical. They must not be
relabelled as current results, and they must not become configuration defaults.

---

## Related

* [`core_algorithm.md`](core_algorithm.md) — symbols, block algorithm, conditioning contract.
* [`known_gaps.md`](known_gaps.md) — the open defects, with acceptance criteria.
* [`../configs/README.md`](../configs/README.md) — the named, runnable recipes.
* [`train.md`](train.md) · [`windows.md`](windows.md) — the per-module detail.
