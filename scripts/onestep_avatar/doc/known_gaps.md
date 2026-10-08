# Known gaps — where the code does not meet the contract

Each entry separates required behavior, current implementation and remaining
acceptance. A passing check has the scope stated in its original evidence.
This page summarizes current gaps; the
[active handoff](../../../../plans/2026-10-07-onestep-avatar-development-experiment-handoff.md#current-progress-and-revised-work-order--2026-10-08)
owns exact receipts and live progress. Historical failures retain their original
bytes and attribution outside this current summary.

Status vocabulary: **open** (no fix), **in progress** (a fix is partially landed),
**verified** (fixed and checked; only a brief historical invariant is retained).

## Current acceptance and next step

Shared model calculations, typed training, unmerged fp32 adapters, strict input
and adapter checks, fixed preparation, previews and product review are implemented.
Both modes pass the native four-rank/serial numerical update comparison (E4
numerical scope), including all named matrices and exact export/reload. This
does not close preview, product, appearance or whole-plan acceptance.

The causal physical-coverage defect is repaired in evaluation and preparation.
The full CPU suite passes, and fresh public preparation preserves the original
capture/guide/image/text/noise tensors while keeping the original causal
training span null. A fresh causal public preview completes raw generation,
rendering, allocator checks and bounded owned-worker supervision. Full/narrow
media inspection remains required. Fresh current-source first-update E2 in both
modes passes the public saved verifier on one camera at steps zero/one.
Short causal controls and both corrected public preview/product executions
pass their scoped scientific and bounded-supervision checks. Full-clip media
inspection remains separate. Older results retain original attribution
and are not receipts for changed producers.

The immediate gate is to inspect complete matched videos from the current-source
short workflows, including readable playback at 480 pixels wide.
Then run the unchanged four-rank seven-frame E5 pilots through steps 20/60.
Learning acceptance remains open. Full E2 still needs two camera views and both trained steps.
E3 still needs original 17-frame history/K/V and before/after-eviction coverage,
held-out people/seeds and measured cost. Stage D still must move experiment
orchestration, retire the duplicate runtime/parser and remove obsolete executors.
No user decision or access change is required for these steps.

Current evidence is under workspace
`expr/onestep_avatar/handoff_implementation_20261007/`: numerical receipts in
`native_training_deterministic/`, full CPU validation in
`causal_output_coverage_20261008/`,
`current_fixed_preparation_readback_20261008.json`, and causal preview execution
in `native_short_preview_causal_current_20261008.supervision/` and
`native_short_preview_causal_current_20261008.resources/`.

| | Gap | Severity | Status |
|---|---|---|---|
| [G1](#g1--the-supplied-first-frame-is-not-a-model-condition) | Historical clean first-frame defect | preserve conditioning invariant | **verified** |
| [G2](#g2--generic-teacher-forced-rollout-refreshes-from-the-guide-not-the-target) | Historical teacher-target defect | preserve explicit target invariant | **verified** |
| [G3](#g3--checkpoint-and-artifact-conditions-are-recorded-but-not-enforced) | Condition checks are missing from some callers | silent off-condition evaluation | **in progress** |
| [G4](#g4--no-d1-probe) | D1 probe coverage differs between paths | direct-path real-adapter verification remains incomplete | **in progress** |
| [G5](#g5--training-and-deployment-disagree-about-valid-sigma) | Historical sigma consumers remain | typed validation checked; legacy retirement is gated | **in progress** |
| [G6](#g6--guide-artifacts-on-disk-predate-the-compositing-fix) | Selected guides need current provenance checks | D1 data readiness | **in progress** |
| [G7](#g7--cached-history-can-disagree-with-a-causal-prefix) | Cached history can disagree with an explicit causal prefix | continuation quality is unmeasured | **in progress** |
| [G9](#g9--a-random-window-c0-is-not-a-keyframe-encode) | a random-window `c0` is not a keyframe encode | training conditions on 8-frame latents that deployment never supplies | **open (accepted 2026-10-05)** |
| [G8](#g8--bf16-lora-fusion-weakens-the-trained-adapter) | bf16 LoRA fusion weakens the trained adapter | shared unmerged path needs native effect/cost verification | **in progress** |
| [G11](#g11--euler-rounding-differs-from-the-stock-step) | Euler rounding differs from the stock step | measured terminal difference; full E1 scope remains open | **in progress** |
| [G12](#g12--ordinary-global-sigma-loses-stock-precision) | Precision/runtime acceptance is incomplete | shared precision and numerical updates pass; complete workflows remain required | **in progress** |
| [G10](#g10--saved-comparisons-have-unreadable-titles-at-narrow-widths) | Saved render text shrinks below the narrow-width requirement | preserve matched readable formats | **verified** |

---

## G12 — ordinary global sigma loses stock precision

**Required.** Shared ordinary training, evaluation and product conditioning keep
global sigma in float32, as the stock pipeline does. Token timesteps stay
float32 too. Record this precision in new adapter and execution conditions;
do not silently relabel an old calibration whose precision is unknown. Historical
records stay readable. Unknown or invalid execution precision must fail before
weights, including with a research override. Explicit historical bf16 conditions
remain a separate diagnostic comparison.

**Current.** The shared modality builder now defaults to float32 global sigma
and token timesteps in both modes. New adapter contracts and ordinary execution
conditions declare that precision. Historical unknown precision is readable but
cannot execute; research overrides only acknowledge explicitly known differences.
Legacy conversion requires matching original config and metadata precision.
Small-model checks cover both modes. Native ordinary-default and explicit-
float32 bidirectional video-component sampling agree exactly. Typed FSDP
disables root-input casting, preserves fp32 adapter leaves and applies the
measured deterministic numerical policy. Both four-rank numerical update
comparisons and serial references pass. Complete native workflows remain
separate acceptance.

**Evidence.** A native 17-frame D0 comparison reused the verified stock controls
and exactly the same image, text, saved noise, schedule, geometry and fps. It
generated only the missing ordinary arm. The initial tokens, token timesteps,
positions and first-image marks match; only global sigma precision changes.
The first prediction differs at RMS 0.00654027. Final latent RMS is 0.00420113,
maximum 0.05078125, across 1,543,927 different elements. RGB RMS is 0.00111309,
maximum 0.03125. Both custom arms use the exact terminal prediction rule, so
terminal Euler reconstruction does not explain this comparison.

The native preprocessor multiplies global sigma by its timestep scale before
embedding it. bf16 changes this calculation before the model uses prompt
conditioning. Do not fit a larger stock-parity tolerance to hide the difference.
The paired video is numerical sampling evidence, not a perceptual-quality claim.

**Acceptance.** Change the shared default and bind it through new checkpoint
and execution records in the same implementation. Preserve typed rejection of
unknown historical precision. Test first-image/timestep/cache invariants in both
modes and train/evaluation/product parity. Then run a fresh ordinary native
stock check with the new source identity. The old bf16 result remains historical.

**Status.** Shared precision, the scoped stock video-component comparison and
both E4 numerical update comparisons are verified. Full E1 and complete current-
source evaluation/product workflow acceptance remain open.

**Fresh native evidence.** Ordinary-default and explicit-float32 custom outputs
are bit-identical. Their four calls have identical inputs and predictions,
including float32 global sigma. Stock repeats are exact in raw output and decoded
pixels. The independently recomputed raw control isolates the same terminal
rounding difference: 90 values, RMS 0.00000338906, maximum 0.0009765625.
Each encoding has 17 frames; comparison media has 129 RGB frames at 30 fps.
Source/runtime, original inputs and saved output hashes were rechecked. This is
pure-noise base sampling with audio absent. It does not verify D1, adapters,
outer RGB/text preparation, joint audio-video or video quality. The matched
stock and custom videos both become blurred by displayed frame 64; numerical
agreement must not be read as appearance acceptance. Evidence:
`expr/onestep_avatar/handoff_implementation_20261007/native_stock_float32_default_acceptance.json`
in the workspace.
Historical precision diagnosis:
`expr/onestep_avatar/handoff_implementation_20261007/native_stock_default_precision_acceptance.json`.

**Numerical runtime evidence.** `training.numerics` applies the measured
deterministic policy before model work. Original launches and actual per-rank
records bind the policy; serial replay requires those observations. Both modes'
four-rank/serial comparisons pass all 768 named clipped gradients, moments and
exports, with exact exported/reloaded output and original allocator bounds.
The measured norms are below the clipping threshold, so these native cases do
not demonstrate active clipping. Independent readback binds original inputs,
software and output bytes. Earlier failed comparisons remain failed provenance;
they do not describe current numerical acceptance. The active handoff links diagnosis
and current receipts. Complete current-source workflows remain separate.

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

**Status.** In progress. Positive-step arithmetic is verified. Native raw and
decoded terminal differences are measured for the declared base video-component
case. Full E1 scope remains open.

**Native evidence.** A 17-frame base video-component check with float32 global
sigma has exact repeated stock encodings and decoded pixels. All four custom
inputs and predictions equal stock. The final encoding differs at 90 values,
with RMS 0.00000338906 and maximum 0.0009765625; custom output equals the final
stock prediction exactly. Decoded RMS is 0.000657139. This confirms the identified
terminal rounding path for this case. The later ordinary-default precision
control agrees with explicit float32 conditioning; see G12. This does not certify
joint audio-video, product generation, D1 guide mixing or perceptual quality. Evidence:
`expr/onestep_avatar/handoff_implementation_20261007/native_stock_parity_acceptance.json`
in the workspace. Full E1 scope remains open.

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
Fresh first-update comparisons pass the public saved verifier in both modes on
one camera view, with successful bounded supervision. Full E2 still needs
two views, steps 20/60,
decoded inspection and measured cost; see the current acceptance summary.
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

**Status.** In progress: shared application and scoped first-update effect
comparison pass; full E2 acceptance remains open.

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

**Acceptance.** Preserve the selected sigma-zero cached refresh calculation.
Compare native block output and layerwise K/V under matched inputs before and
after eviction at 0.909375 and sigma 1. Compare boundary quality under both
history calculations and both forcing policies on held-out views. Measure
generated-history quality and actual cost; do not claim cache/recalculation
equivalence from earlier block-one observations.

**Status.** In progress. The selected default is established; its numerical
difference from diagnostics remains. CPU controls and scoped earlier native
observations do not replace full E3. Current original-source short seven-frame
controls and fresh current-source successors show repeat/c0/prefix invariance
before eviction. The original 17-frame
K/V and capture/generated-history controls, before/after eviction, plus held-out
appearance, motion, seeds and cost remain required. The active handoff links the exact
protocol and scoped saved observations.

---

## G1 — the supplied first frame is not a model condition

**Verified historical defect, fixed 2026-09-19.** The legacy `training.engine.train_chain`
and shared `model.causal.rollout` assemble clean capture/supplied-image `c0` before
the model at token timestep zero. Both typed modes use `model.common.with_clean_prefix`
and `block_modality`. They preserve the first image in predictions; causal refresh
retains it as the sink. Dedicated conditioning tests and the recorded
two-GPU debug run verified this boundary. Pre-fix results do not share `clean_c0_v1`.
The original missing-condition reproduction is historical, not current behavior.

## G2 — generic teacher-forced rollout refreshes from the guide, not the target

**Verified historical defect, fixed 2026-09-21.** `model.causal.rollout` requires explicit
`teacher_tokens` when teacher forcing is requested and refreshes from that capture target.
`test_teacher_forcing_refreshes_from_the_target_not_the_guide` verifies the distinction when
guide and capture differ. `causal_core` is only a local alias in remaining legacy
callers; the old module path is removed. Typed causal training uses the capture
target through `model.causal.train_sample`. Preserve the explicit target invariant.

---

## G3 — checkpoint and artifact conditions are recorded but not enforced

**Required.** A fixed-σ, fixed-geometry, fixed-arm adapter must not load off-condition without an
explicit, recorded override.

**Current.** Typed training writes version-two contracts with base and data hashes,
mode, exact sigma/schedule, loss, geometry, forcing, adapter application and
global-sigma precision. Ordinary evaluation and product use
`training.checkpoints.check_contract` and matrix validation before model/text
loading. Checked data readers enforce encode/VAE provenance for selected inputs.
Evaluation research changes use the existing recorded override; product permits
none. Unknown historical precision fails execution even with an override.
Legacy `visualize_d1.py` still uses `check_adapter_conditions` through its
historical interface. `visualize_d0.py` does not provide the full typed gate.
Their replacement acceptance and retirement remain open.

**Impact.** A context-16 checkpoint is probed at context 2; a D1 adapter is accepted by the D0
probe; a fixed-σ adapter is probed at other σ without being labelled off-condition. Every one of
those produces a plausible video and a wrong conclusion.

**Acceptance.** One package-owned checkpoint-condition reader used by probe and deployment before
expensive loading; geometry and allowed σ derived from metadata; model/objective/arm validated;
off-condition research allowed only through an explicit recorded override.

**Status.** In progress. Typed enforcement is implemented and CPU checked,
including causal history/KV choices and adapter precision/application. Legacy
consumers remain separate from the typed path until Stage D migration. Native
adapter acceptance remains E2/E4; metadata checks alone do not prove it.

**Causal preview physical coverage.** The original E4
adapter correctly records a null training span and frame counts `[6,7]` from
continuous masters. Ordinary evaluation previously had only `--span-latent-frames` for
physical prefix selection. The first seven-frame fixed preview requested
span 7 and failed the strict checker before loading a transformer. E2 and product
already select physical seven-frame input under the original null setting.
This is an ordinary CLI selection defect, not permission to weaken the checker
or change E4 inputs. Implemented repair: causal-only `--output-latent-frames` selects
physical coverage independently, with positive/exact-block/paired-length checks,
unchanged settings and strict shape/noise/adapter preflight. The
[evaluation design](evaluate.md#causal-physical-output-coverage)
states decisions and worked outcomes. The full CPU suite, fresh public input
preparation and fresh causal public preview execution now pass. Full/narrow
media inspection and complete workflow acceptance remain required. The public fixed-preview argument producer
preserves the causal training span and emits physical coverage separately;
fresh preparation records bind the new producer bytes. Preserve failed runs and
source-bound receipts; the pilot's explicit span-seven lineage stays separate.

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

**Historical defect.** The old trainer accepted continuous values while the old
product applied the distilled grid to every base. This could create an adapter
that its product entrypoint refused. That product entrypoint is removed.

**Current.** Shared `model.sampling` validates schedules and base-supported levels.
Typed training, ordinary evaluation and product use these rules. Dev permits
continuous levels in `(0,1]`; distilled requires its actual nonzero grid.
The checkpoint owner separately checks the adapter's calibrated levels and
schedule. Deliberate supported research differences need the recorded evaluation
override. Product exposes no override. The old trainer/parser and visualizers
remain transitional consumers, pending Stage D retirement.

**Impact.** A completed run that cannot be deployed, discovered at deployment.

**Acceptance.** A shared validator against the selected model's grid, with adapter calibration
checked separately, and an explicit research override for deliberate off-grid work.

**Status.** Typed validation is implemented and CPU checked. Reconcile and remove
remaining historical consumers after native replacement acceptance. Broader
package closure is still open; the removed product validator is historical evidence.

---

## G6 — guide artifacts on disk predate the compositing fix

**Required.** Every guide latent used for D1 was composited under the current contract,
`dataset.GUIDE_COMPOSITING_VERSION` (2).

**Current.** The v2 replacement formula and producer guard are implemented.
Every selected D1 source must have current render/latent/crop/VAE provenance.
Current typed runs check version-two membership and selected producers through
dataset and training/evaluation preflight. Corpus-wide historical inventory
counts cannot establish readiness for a selected run.

**Impact.** A D1 run that selects a stale guide latent would train on the old compositing
contract. Readiness must be checked for that run's selected sources. D0 reads no guide artifact.

**Acceptance.** All guides required by a subset rebuilt under v2 before that subset is used for a
D1 run; the subset's readiness checked against guide *latents*, not just render MP4s.

**Status.** Selected-source readiness is checked; complete input-production and
caller migration acceptance remain open. Preserve old render/encoding records
with their original attribution. Reprepare invalid selected inputs through their
one producer; never weaken provenance checks or repair bytes in a reader.

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
CPU controls include compact-selection, pre-decode refusal, altered format
and saved-only report-reader cases. This closes the presentation defect for the
implemented saved-render path. It does not establish transformer/FSDP, adapter
quality, stock sampling or whole-study scientific acceptance.
