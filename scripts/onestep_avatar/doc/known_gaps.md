# Known gaps — where the code does not meet the contract

Each entry states the **required behavior**, the **current implementation** with its source
symbols, the **impact**, the **acceptance criteria** for a future fix, and a **status**. An entry
stays here, prominently, until a fix is implemented *and* verified. Nothing below is fixed by
this documentation; do not read an acceptance criterion as a passing test.

Status vocabulary: **open** (no fix), **in progress** (a fix is partially landed),
**verified** (fixed and checked; only a brief historical invariant is retained).

| | Gap | Severity | Status |
|---|---|---|---|
| [G1](#g1--the-supplied-first-frame-is-not-a-model-condition) | Historical clean first-frame defect | preserve conditioning invariant | **verified** |
| [G2](#g2--generic-teacher-forced-rollout-refreshes-from-the-guide-not-the-target) | Historical teacher-target defect | preserve explicit target invariant | **verified** |
| [G3](#g3--checkpoint-and-artifact-conditions-are-recorded-but-not-enforced) | Condition checks are missing from some callers | silent off-condition evaluation | **in progress** |
| [G4](#g4--no-d1-probe) | D1 probe coverage differs between paths | direct-path real-adapter verification remains incomplete | **in progress** |
| [G5](#g5--training-and-deployment-disagree-about-valid-sigma) | Training and deployment disagree about valid sigma (σ) | a trained adapter its own API refuses | **open** |
| [G6](#g6--guide-artifacts-on-disk-predate-the-compositing-fix) | Selected guides need current provenance checks | D1 data readiness | **in progress** |
| [G7](#g7--cached-history-can-disagree-with-a-causal-prefix) | Cached history can disagree with an explicit causal prefix | continuation quality is unmeasured | **in progress** |
| [G9](#g9--a-random-window-c0-is-not-a-keyframe-encode) | a random-window `c0` is not a keyframe encode | training conditions on 8-frame latents that deployment never supplies | **open (accepted 2026-10-05)** |
| [G8](#g8--bf16-lora-fusion-weakens-the-trained-adapter) | bf16 LoRA fusion weakens the trained adapter | shared unmerged path needs native effect/cost verification | **in progress** |
| [G11](#g11--euler-rounding-differs-from-the-stock-step) | Euler rounding differs from the stock step | native sampling parity remains unmeasured | **in progress** |

| [G10](#g10--saved-comparisons-have-unreadable-titles-at-narrow-widths) | Saved render text shrinks below the narrow-width requirement | preserve matched readable formats | **verified** |

---

## G11 — Euler rounding differs from the stock step

**Required.** Positive next levels use the stock step's operation order and
dtype conversions. A direct `[sigma,0]` step returns the prediction exactly.
E1 compares native raw encodings and pixels, including this terminal difference.

**Historical defect.** The custom step used bf16 interpolation at every level.
A fixed CPU sample of 524,288 elements differed from the stock step at 257,310
elements for `[1,0.725]`, with RMS 0.00334952 and maximum 0.03125.
This is arithmetic evidence, not model-output or perceptual evidence.

**Current.** `model.sampling.euler_to` calls the actual native
`EulerDiffusionStep.step` at positive next levels and returns the prediction
directly at zero. The stock bf16 step rounds velocity before reconstructing its
endpoint, so the terminal outputs can differ. Keep the direct endpoint contract;
do not hide this difference with a parity claim or an unmeasured tolerance.

**Acceptance.** CPU multi-interval float32/bf16 tests must match the native
step exactly at positive levels and preserve the exact direct endpoint. E1 must
still measure full-model raw and decoded differences with matched inputs.
Historical outputs keep their original source hashes.

**Status.** In progress. The native E1 comparison remains open.

## G9 — a random-window `c0` is not a keyframe encode

**Required.** Training's `c0` has the distribution of what deployment supplies: a single real
image encoded by the causal VAE as latent frame 0 (one pixel frame).

**Current.** `train.py --random-window-latent-frames W` slices the stored master at a random
latent frame `s` and uses frame `s` as `c0` (user decision, 2026-10-05: slice, do not re-encode).
For `s = 0` this is the keyframe encode; for `s > 0` it is a latent that encodes 8 pixel frames
(pixel frames 8s − 7 … 8s), with different statistics. `keyframes_mask` still marks window
frame 0, and evaluation keeps `s = 0`.

**Consequence.** Part of the training signal conditions on `c0` latents deployment never
produces. The share is `1 − 1/(F − W + 1)` of samples (½ for 18-frame clips, 11/12 for
28-frame clips at `W = 17`).

**Closing it.** Re-encode each window from pixel frame `8s` so its frame 0 is a true
single-frame encode (precompute per offset, capture and guide), or train only on `s = 0`.

---

## G8 — bf16 LoRA fusion weakens the trained adapter

**Required.** The adapter that probe and deployment run is the function training optimised,
within bf16 rounding of the *delta*, not just of the weights.

**Current.** Typed training, ordinary evaluation and product now share
`model.adapters` configuration and saved tensor loading: unmerged PEFT, fp32
adapter weights against frozen bf16 base weights. Inference wraps that velocity
function once in stock x0. CPU controls with real small LTX/PEFT models match the
loaded training reference bit-for-bit in both modes, including zero adapters.
Native E2 effect, views/checkpoint steps, decoded appearance and cost remain open.
`--adapter-application fused_bf16` is an explicit evaluation diagnostic requiring
a recorded research override. Product exposes no fusion choice.

**Historical evidence.** `Session.transformer(loras=...)` fuses `W + BA` into
bf16 weights at load. Measured
2026-10-02 on a real dev D1 adapter (60 updates, block 0, σ .421875, clip 0013_07): with no
adapter the two paths are bit-identical, but the adapter's effect on the block is relative
L2 0.136 unmerged and 0.119 fused, the two effects differ by 16%, and the raw outputs by 2.2%
(`expr/onestep_avatar/dev_training_20261001/analysis/setup/fusion_parity_q0d1_step60.json`).
A small `BA` added to a large bf16 `W` loses low-order bits, so fusion shrinks and perturbs the
learned correction.

**Impact.** Every fused evaluation slightly understates (and perturbs) what the adapter learned;
the effect is largest for small, early adapters.

**Acceptance.** Run ordinary evaluation/product with the unmerged adapter and
measure its effect against the loaded training reference (target under 5%, with
an explicit near-zero-effect rule). Compare fusion separately. An fp32 sum
rounded into bf16 base weights is not the accepted normal function. Run the
bounded native E2 views/checkpoint-step, zero-effect, appearance and cost controls.

**Status.** In progress: shared application implemented and CPU checked; native E2 open.

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

**Verified historical defect, fixed 2026-09-19.** Current `train_chain` and rollout assemble
clean capture/supplied-image `c0` before the model at token timestep zero, preserve it in
prediction/refresh and retain it as the sink. Dedicated conditioning tests and the recorded
two-GPU debug run verified this boundary. Pre-fix results do not share `clean_c0_v1`.
The original missing-condition reproduction is historical, not current behavior.

## G2 — generic teacher-forced rollout refreshes from the guide, not the target

**Verified historical defect, fixed 2026-09-21.** `causal_core.rollout` requires explicit
`teacher_tokens` when teacher forcing is requested and refreshes from that capture target.
`test_teacher_forcing_refreshes_from_the_target_not_the_guide` verifies the distinction when
guide and capture differ. Preserve this invariant during extraction; no implicit guide fallback.

---

## G3 — checkpoint and artifact conditions are recorded but not enforced

**Required.** A fixed-σ, fixed-geometry, fixed-arm adapter must not load off-condition without an
explicit, recorded override.

**Current.** Training stamps exact sigma/levels, geometry, objective/arm/forcing, loss,
base fingerprint and full subset identity. `sampling.check_adapter_conditions` is used by
`visualize_d1.py` before loading; explicit overrides are recorded. `visualize_d0.py` and
the removed old product rollout do not use that full reader. Alpha/rank is refused when unsupported
rather than applied at fusion; master schema checking still does not enforce all encode
provenance. The lower status record describes the existing partial enforcement.

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
and sidecar hashes. Still open: `visualize_d0.py` and the removed old product rollout do not call the
reader; LoRA `alpha/rank` is refused rather than applied at fusion; `dataset.load_master` still
checks schema, not encode provenance.

---

## G4 — no D1 probe

**Required.** A checkpoint from the deployable arm can be decoded and inspected.

**Current.** `visualize_d0 --guide-mode d1` exists and shares its rollout with D0.
`visualize_d1 --arms d1 --checkpoint` provides paired evaluation. The existing 2026-10-02
record reports real dev D1 adapters exercised via `visualize_d1` in the dev-training study.
Claims that no D1 probe or D1 adapter exists are obsolete.

**Status.** In progress: the recorded real-adapter path is `visualize_d1`; direct
`visualize_d0` end-to-end D1 verification is not established here, and its complete adapter
condition enforcement remains G3. This documentation phase supplies no new GPU evidence.

**Acceptance.** Preserve the existing verified D1 behavior in unified evaluation, exercise
its real-adapter path and use the common condition reader for every consumer.

---

## G5 — training and deployment disagree about valid sigma

**Required.** One nonzero, on-grid validator for the operating points both sides support.

**Current.** `train.training_sigmas` accepts any value in `(0, 1]` and refuses `0.0`.
the removed old distilled-only validator accepts only points on the distilled model's 9-point grid — and
would accept `0.0` if it were on that grid. `--sigma0 0.5` therefore trains an adapter that its
own deployment API refuses.

**Impact.** A completed run that cannot be deployed, discovered at deployment.

**Acceptance.** A shared validator against the selected model's grid, with adapter calibration
checked separately, and an explicit research override for deliberate off-grid work.

**Status.** Open (audit F12). Note since 2026-10-02: the dev backbone has no grid at all (any
start in `(0, 1]`), so "on-grid" is a property of the distilled base only; the condition reader
checks an adapter's *calibrated* σ separately from what the base supports, but
the removed old distilled-only validator still applies the distilled grid to every base.

---

## G6 — guide artifacts on disk predate the compositing fix

**Required.** Every guide latent used for D1 was composited under the current contract,
`dataset.GUIDE_COMPOSITING_VERSION` (2).

**Current.** The v2 replacement formula and producer guard are implemented. Some earlier
inventories recorded stale bg guides; the 2026-10-02 status below records white guides built
under v2. These dated inventories are not a live readiness check. Every selected D1 source
must have current render/latent/crop/VAE provenance; documentation design does not regenerate
or validate the full corpus and does not claim that white guide count is zero.

**Impact.** A D1 run that selects a stale guide latent would train on the old compositing
contract. Readiness must be checked for that run's selected sources. D0 reads no guide artifact.

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
* [`../configs/README.md`](../configs/README.md) — current runnable recipes and recorded limitations.

### Product enforcement update — two-mode implementation

The product `infer` CLI now checks version-two records and actual LoRA matrices
before weight/text sessions. It checks the full base hash, D1/background, mode,
shape and exact schedule, and permits no research override. Supplied-image
encoding provenance and actual VAE identity are checked. Distilled levels must
be on the selected base grid; dev levels are continuous in (0,1]. Thus the old
product-side G3/G5 paths described above are removed. Old research visualization
callers still await conversion, so broader G3 enforcement remains incomplete.

### Causal diagnostic condition repair — 2026-10-07

The typed checker now requires causal request fields `history_mode` and
`kv_source`. It maps recorded `cached_refresh_global_sigma0` to cache/refresh.
Recompute/joint history or denoise-produced K/V requires the existing research
override; recorded differences name the field, trained/requested values and
the changed continuation calculation. Missing/unsupported fields or invalid
denoise-history pairs fail even with an override. Typed evaluation binds these
fields before model/text sessions and records them in `conditions`. Product
preflight explicitly uses cache/refresh and exposes no diagnostic override.
Focused CPU checks cover these paths with actual adapter contracts/matrices and
checked input records. This fixes F5's omitted history choices. Legacy
visualization consumers still leave broader G3 incomplete. The shared adapter
repair also binds `application_method` before loading: unmerged is the ordinary method;
fusion requires a research override and is refused by product inference.

The selected G7 default is clean global-sigma-zero cache refresh for training and
product. Recomputed and joint history remain diagnostics. The existing numerical
disagreement is not a default-selection blocker; E3 quality and cost measurement
before/after eviction is still required. G8 requires unmerged fp32 PEFT adapters
for ordinary evaluation/product and E2 effect verification. An fp32 sum rounded
back into bf16 base weights is not accepted as the normal application method.

## G10 — saved comparisons have unreadable titles at narrow widths

**Required.** Keep text at least 16 pixels in a delivered compact video or poster
shown at 480 pixels wide. Preserve exact labels, panel order, source frames and
image aspect ratios. Fail before decoding when no supported layout fits.

**Historical defect.** Saved comparisons only produced a full video with a
1280-pixel viewing width. Its 16-pixel titles on a 1232-pixel canvas shrank to
about 6.23 pixels at 480. A successful receipt did not prove narrow readability.
The initial artifact and its evidence remain preserved.

**Current.** Shared metadata-only geometry selects a measured compact layout.
Schema-three saved renders contain both full and compact media, using the same
decoded RGB panels without a second decode. Completion checks bind layout,
font, dimensions, timing, labels, media hashes and equal panel pixel hashes.
The saved-only report reader exposes both formats and refuses missing or changed
compact output. It never renders missing media.

**Verified.** The native first original corpus comparison produced a 2 × 2
compact canvas at 824 × 1028, with 28-pixel text: 16.31 pixels at width 480.
Both videos contain 129 frames at 30 fps. Inspected full and narrow native frames
and a Chromium player screenshot at 480 pixels. The full compact movie played
to its 4.3-second end at normal speed with no browser media error. Wide panel
records and numeric metrics exactly match the prior native render. Original
specification and four input hashes remain unchanged. The guarded queue finished
with a verified receipt and released its GPU-4 claim.

Workspace evidence: `expr/onestep_avatar/two_mode_restructure_20261005/compact_presentation_acceptance.json`.
The 789-test suite includes compact-selection, pre-decode refusal, altered format
and saved-only report-reader cases. This closes the presentation defect for the
implemented saved-render path. It does not establish transformer/FSDP, adapter
quality, stock sampling or whole-study scientific acceptance.
