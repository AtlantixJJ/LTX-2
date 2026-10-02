# Known gaps — where the code does not meet the contract

Each entry states the **required behavior**, the **current implementation** with its source
symbols, the **impact**, the **acceptance criteria** for a future fix, and a **status**. An entry
stays here, prominently, until a fix is implemented *and* verified. Nothing below is fixed by
this documentation; do not read an acceptance criterion as a passing test.

Status vocabulary: **open** (no fix), **in progress** (a fix is partially landed),
**verified** (fixed and checked — then the entry is removed).

| | Gap | Severity | Status |
|---|---|---|---|
| [G1](#g1--the-supplied-first-frame-is-not-a-model-condition) | The supplied first frame is not a model condition | blocks the product contract | **verified** |
| [G2](#g2--generic-teacher-forced-rollout-refreshes-from-the-guide-not-the-target) | Generic teacher-forced rollout refreshes from the guide, not the target | wrong outside D0 | **verified** |
| [G3](#g3--checkpoint-and-artifact-conditions-are-recorded-but-not-enforced) | Checkpoint/artifact conditions are recorded but not enforced | silent off-condition evaluation | **in progress** |
| [G4](#g4--no-d1-probe) | No D1 probe | D1 checkpoints cannot be looked at | **in progress** |
| [G5](#g5--training-and-deployment-disagree-about-valid-sigma) | Training and deployment disagree about valid sigma (σ) | a trained adapter its own API refuses | **open** |
| [G6](#g6--guide-artifacts-on-disk-predate-the-compositing-fix) | 18 of 19 guide renders predate the v2 compositing fix | D1 data readiness | **in progress** |
| [G7](#g7--cached-history-can-disagree-with-a-causal-prefix) | Cached history can disagree with an explicit causal prefix | continuation quality is unmeasured | **in progress** |
| [G8](#g8--bf16-lora-fusion-weakens-the-trained-adapter) | bf16 LoRA fusion weakens the trained adapter | probe/deploy run a smaller correction than training learned | **open** |

---

## G8 — bf16 LoRA fusion weakens the trained adapter

**Required.** The adapter that probe and deployment run is the function training optimised,
within bf16 rounding of the *delta*, not just of the weights.

**Current.** Training runs the LoRA unmerged (PEFT, fp32 adapter weights against the bf16
base). `Session.transformer(loras=...)` fuses `W + BA` into bf16 weights at load. Measured
2026-10-02 on a real dev D1 adapter (60 updates, block 0, σ .421875, clip 0013_07): with no
adapter the two paths are bit-identical, but the adapter's effect on the block is relative
L2 0.136 unmerged and 0.119 fused, the two effects differ by 16%, and the raw outputs by 2.2%
(`expr/onestep_avatar/dev_training_20261001/analysis/setup/fusion_parity_q0d1_step60.json`).
A small `BA` added to a large bf16 `W` loses low-order bits, so fusion shrinks and perturbs the
learned correction.

**Impact.** Every fused evaluation slightly understates (and perturbs) what the adapter learned;
the effect is largest for small, early adapters.

**Acceptance.** Fuse in fp32 and round once, or run the probe with the adapter unmerged; then
re-measure the effect gap (target well under 5%).

**Status.** Open.

---

## G7 — cached history can disagree with a causal prefix

**Required.** Before treating cached refresh as an optimization of explicit causal
continuation, compare them under matched text conditioning, clean per-token history
timesteps, positions and retention policy on the real checkpoint. If zero-sigma refresh is
instead the intended streaming model, its quality must be established under generated history.

**Current.** `refresh_block` computes history K/V with global sigma zero.
`transformer_args.py` uses global sigma for prompt AdaLN, independently of token timesteps.
The LTX-2.5 checkpoint enables this branch. A small two-layer CPU model differs at block 1
before eviction when this branch is enabled. On the real checkpoint, clean block-0 K/V are
equal at layer 0 but differ from layer 1 onward when only global sigma changes from 0 to
0.909375 or 1. A two-block real rollout differs at block 1 by relative latent L2 0.12084
(D0) / 0.11428 (D1) at 0.909375, and 0.23675 in both arms at sigma 1. Block 0 and a
repeated cached run agree exactly.
`rollout(history_mode="recompute")` now supplies an
explicit block-causal reference for paired inference; its retained history stays at token
timestep zero while global sigma tracks each denoise step. After eviction, recomputation also
changes the old states' available context, so comparisons must split before and after eviction.

**Impact.** The computational difference on real weights is measured on one view; its
contribution to appearance jumps remains unmeasured. Teacher-forced cache history also differs
from the displayed output by definition;
that separate source mismatch is tested by dropping `--teacher-forcing`.

**Acceptance.** Block-1 output and layerwise K/V checks are complete at 0.909375 and sigma 1.
Compare boundary quality under both history policies and both forcing policies on held-out
views. Select the continuation
computation before training a new adapter or changing deployment semantics.

**Status:** in progress. CPU tests, real-weight computation checks and the explicit diagnostic
are present. A same-GPU eight-block raw-only control measured 77.12 s for recomputation versus
26.23 s for caching, with only a small change in latent boundary residual on one view.
The separate joint-window reference improved GT-history latent metrics on three actors, but
generated-history boundary residual was mixed and its one measured full rollout cost 2.66×
cached inference. Visual appearance, motion, additional seeds and broader latency comparisons
are still owed. The
one-view exploratory run is in the workspace
[`expr/onestep_avatar/d1_diagnostic/REPORT.md`](../../../../expr/onestep_avatar/d1_diagnostic/REPORT.md).

---

## G1 — the supplied first frame is not a model condition

The workspace tracks this as **F2 / Stage C** of
the September 18 audit.

### Required

Every generated block, **including block 0**, has the supplied first-frame clean latent `c0` as
initial conditioning — independent of D0/D1 and of teacher/self forcing. `c0` is
objective-consistent, enters block 0 as clean tokens at per-token timestep zero, is preserved in
that block's output and refresh, and remains reachable by every later block through the pinned
cache history. Teacher forcing may add clean completed targets and self forcing adds detached
predictions; neither replaces `c0`. A mid-clip chain start uses the same `c0` semantics. Full
detail: [`core_algorithm.md` §3](core_algorithm.md#3-the-conditioning-contract).

### Current

`train_chain` derives `c0` from the objective-consistent capture master and makes block 0's
leading latent-frame tokens clean at timestep zero. `causal_core.rollout` requires that same
patchified condition, preserves it in the output, and writes it through the pinned cache sink.
`onestep_core.rollout` requires `first_frame_latent`, so deployment cannot silently use the
guide's composited frame 0.

CPU reproduction recorded by the audit, through the real primitives with an identity denoiser: a
first-frame input value of `1.0` comes out as `1.6720136404`, at timestep `0.725`.

### Impact

* The product's defining input — "one real first frame" — never reaches the model.
* The background every later frame is supposed to propagate is itself generated, so identity and
  background drift have no anchor.
* Deployment has no parity with training even in principle, because the interface is missing.
* Under teacher forcing the decoded video (predictions) can disagree with the identity that
  conditioned the next block (ground truth), visible as a transition at the block-0/1 boundary.

### What does **not** count as a fix

Setting a zero timestep while the content stays noised; clamping frame 0 in the output after
denoising; relying on `keyframes_mask`; pinning a generated frame 0 in the cache. "The sink is
pinned" is not evidence — pinning is a *retention* policy over whatever was written.

### Acceptance criteria for the future runtime fix

These are **owed**, not present. They are distinct from the tests already in
`tests/test_causal_core.py` / `tests/test_train.py`; in particular the existing cached-vs-full
attention parity test passes today *with* this defect, so it establishes nothing about it.

1. Inspect block 0's assembled input directly: frame 0's tokens equal `c0` bit-for-bit and its
   per-token timestep is exactly zero, while generation tokens carry `σ`.
2. Frame 0 is preserved in block 0's output and in the tokens written to the cache.
3. The retained cache content still holds `c0` after eviction has discarded every other frame,
   including at `--context-latent-frames 0`.
4. Coverage across D0 × D1 and teacher × self forcing — the invariant is independent of both.
5. Objective consistency: the `white` arm conditions on the white-objective first frame, never an
   unmatted `bg` latent or a guide frame.
6. Mid-clip chain starts use the same `c0` semantics; any additional GT history priming is
   separately asserted and disclosed.
7. D1 teacher forcing refreshes from the **target**, distinguishable from the guide (see G2).
8. Train/rollout parity on a small real transformer, and equal transformer-forward counts on
   every rank (FSDP lockstep) for clip-start and mid-clip chains alike.
9. The conditioning change is **versioned** in checkpoint metadata; results from before and after
   are not pooled.

### Status

**Verified 2026-09-19.** Focused CPU conditioning tests passed, followed by a two-GPU D0
teacher-forced step-0/step-1 debug train and matching D0 probe.

---

## G2 — generic teacher-forced rollout refreshes from the guide, not the target

**Required.** Teacher forcing means the cache refresh is fed the **ground-truth target** for the
completed block.

**Current.** `train.train_chain` does that — `clean = target_tokens[lo:hi]` — and
`causal_core.rollout` now does too: it takes `teacher_tokens` and refreshes from it.

**What was wrong (until 2026-09-21).** `rollout(teacher_forcing=True)` refreshed from
`guide_tokens`, the tensor the block was noised from. Those are the same tensor **for D0 only**,
where the source *is* `z_y`. For D1 the generic rollout teacher-forced on the render guide, which
is not the target.

**Impact.** `visualize_d0.py --teacher-forcing` and any other caller of the generic rollout
evaluated a regime training never ran, without raising. It was silent, and it looked like the
training ablation. No D1 result was ever produced through that path — no D1 run has been
trained — so nothing on disk needs relabelling.

**Acceptance.** Pass an explicit teacher target through the rollout, or restrict
`teacher_forcing=True` to D0 and refuse it otherwise; assert that a D1 teacher-forced refresh
receives `z_y` and not `z_g`.

**Status.** **Verified 2026-09-21.** `rollout` takes `teacher_tokens` and raises when
`teacher_forcing=True` is passed without it — deliberately *required* rather than defaulted, so
the next caller cannot reintroduce the bug by omission. `visualize_d0.py` passes the capture
master. `tests/test_causal_core.py::test_teacher_forcing_refreshes_from_the_target_not_the_guide`
pins all three behaviours on a case where `z_g != z_y`, including that the result no longer
matches a refresh from the guide.

---

## G3 — checkpoint and artifact conditions are recorded but not enforced

**Required.** A fixed-σ, fixed-geometry, fixed-arm adapter must not load off-condition without an
explicit, recorded override.

**Current.** `train.checkpoint_metadata` stamps σ₀/σ levels, `K`, block and cache geometry,
objective, guide mode, anchor weight, teacher forcing, LoRA rank/alpha/target, subset hash and
`loss=full_frame_x0_mse`. Nothing reads it back: `visualize_d0.py` now accepts explicit geometry
and sigma overrides and records them, but still takes those values and `--teacher-forcing` from
the command line rather than the adapter. It checks probe sigmas against the selected base
model's schedule, not the adapter metadata. `onestep_core.rollout` checks only that σ₀ is on the
model grid and the schedule is one step. `sampling.assert_one_step_conditions` has no
production call site.
Related enforcement gaps: LoRA `alpha/rank` scaling is stamped but not folded into the exported
factors nor applied at fusion (safe only at the default `alpha == rank`); `dataset.load_master`
checks schema, not encode contract/objective/crop provenance; the stamped subset hash covers
`subset['sources']` only, so different chains or splits can share one "identity".

**Impact.** A context-16 checkpoint is probed at context 2; a D1 adapter is accepted by the D0
probe; a fixed-σ adapter is probed at other σ without being labelled off-condition. Every one of
those produces a plausible video and a wrong conclusion.

**Acceptance.** One package-owned checkpoint-condition reader used by probe and deployment before
expensive loading; geometry and allowed σ derived from metadata; model/objective/arm validated;
off-condition research allowed only through an explicit recorded override.

**Status.** **In progress (2026-10-02).** `sampling.check_adapter_conditions` is the one
package-owned reader: base variant and weights fingerprint (`backbone.identity`), model key,
objective, arm, `clean_c0_v1`, loss, attention, history computation, geometry, forcing policy,
`alpha == rank`, calibrated σ and schedule. `visualize_d1.py` calls it before loading and refuses
off-condition use unless `--off-condition-override` is passed, which the manifest records. The
trainer now stamps the base identity, the history computation and the **full** subset hash
(`windows.subset_sha256`: sources, chains, splits, span); `windows.py` pins capture/guide latent
and sidecar hashes. Still open: `visualize_d0.py` and `onestep_core.rollout` do not call the
reader; LoRA `alpha/rank` is refused rather than applied at fusion; `dataset.load_master` still
checks schema, not encode provenance.

---

## G4 — no D1 probe

**Required.** A checkpoint from the deployable arm can be decoded and looked at.

**Current.** `visualize_d0.py` is the only probe and is D0-only: it renders
`capture │ frozen base │ LoRA` from the capture source. It does not refuse a D1 adapter either
(that is part of G3).

**Impact.** The arm that actually deploys cannot be inspected; the arm that cannot deploy can.

**Acceptance.** A D1-capable probe sharing the evaluation code with an explicit source choice —
not a second rollout implementation — plus a refusal path in the D0 tool until it exists.

**Status.** **In progress (2026-09-21; exercised 2026-10-02).** Real dev D1 adapters from the
dev-training study (`expr/onestep_avatar/dev_training_20261001/`) were exported, loaded fused
into the dev base by `visualize_d1.py --arms d1 --checkpoint`, and rolled out in-condition; the
step-0 adapter reproduces the bare base bit for bit. `visualize_d0.py --guide-mode d1` exists and shares
one `causal_core.rollout` with D0, changing only the noising source (`_source_master`); the
refusal path is moot for the arm itself. Two things keep this open rather than verified: the D1
path has **not yet been exercised against a real D1 adapter on a GPU** (none exists — no D1 run
has been trained), and the tool still reads arm/σ/geometry from the command line rather than
from adapter metadata, which is [G3](#g3--checkpoint-and-artifact-conditions-are-recorded-but-not-enforced)'s
half of the same problem: nothing stops probing a D0 adapter with `--guide-mode d1`.

---

## G5 — training and deployment disagree about valid sigma

**Required.** One nonzero, on-grid validator for the operating points both sides support.

**Current.** `train.training_sigmas` accepts any value in `(0, 1]` and refuses `0.0`.
`onestep_core.one_step_sigma` accepts only points on the distilled model's 9-point grid — and
would accept `0.0` if it were on that grid. `--sigma0 0.5` therefore trains an adapter that its
own deployment API refuses.

**Impact.** A completed run that cannot be deployed, discovered at deployment.

**Acceptance.** A shared validator against the selected model's grid, with adapter calibration
checked separately, and an explicit research override for deliberate off-grid work.

**Status.** Open (audit F12). Note since 2026-10-02: the dev backbone has no grid at all (any
start in `(0, 1]`), so "on-grid" is a property of the distilled base only; the condition reader
checks an adapter's *calibrated* σ separately from what the base supports, but
`onestep_core.one_step_sigma` still applies the distilled grid to every base.

---

## G6 — guide artifacts on disk predate the compositing fix

**Required.** Every guide latent used for D1 was composited under the current contract,
`dataset.GUIDE_COMPOSITING_VERSION` (2).

**Current.** The v2 background-replacement fix (`R_white + (1−α)·(B−white)`) has landed in
`build_guidance.composite_guide_frame` and is checked by `_render_is_complete`, with no legacy
value grandfathered. One real `bg`/`white` pair (`Part_1/0012_09` view01) has been rebuilt and
reviewed; **the other 18 `bg` pairs on disk are known-stale**, and there are **zero `white` guide
renders**, so white D1 has no data at all.

**Impact.** A D1 run over the current `t2r2` subset would train on v1 guides. D0 is unaffected —
it reads no guide artifact.

**Acceptance.** All guides required by a subset rebuilt under v2 before that subset is used for a
D1 run; the subset's readiness checked against guide *latents*, not just render MP4s.

**Status.** In progress (audit F1/F5, Stage B). **White objective, 2026-10-02:** 51 white guide
renders carry `compositing_version` 2 and all 51 now have guide latents (21 newly encoded; 21
existing ones were re-encoded by a fresh-manifest `precompute --process_syn_latent` pass, not
bit-identically, list in `expr/onestep_avatar/dev_training_20261001/provenance/`).
`windows.py --require-guide-latent` gates a subset on the latent and the sidecar's version. The
`bg` guides are unchanged: 18 of 19 stale.

---

## Related

* [`core_algorithm.md`](core_algorithm.md) — the contract these gaps are measured against.
* [`experiments.md`](experiments.md) — which arms each gap affects.
* [`../configs/README.md`](../configs/README.md) — the recipes, each labelled with G1.
