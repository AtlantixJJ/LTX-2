# Smaller LTX avatar model: recovery, motion conditioning and block-causal inference

Date: 2026-10-04. Status: CPU-only execution started at the user's request.
The user authorized sequential execution and said GPUs 0–3 should be available
in about one hour (earliest check: 2026-10-04 15:58 UTC). Do not assume they are
free at that time; inspect availability before any launch. Experiment outputs belong under workspace
`expr/`; this plan belongs in repository-local `LTX-2/plans/`.

## Recommendation

Pursue **a physically smaller student with recovery training, a useful motion
condition, then adaptation to cached block-causal inference**. Run compression
feasibility and motion-conditioning development as independent streams after a
short evidence/data audit. The current CPU execution follows the user's requested
sequential order; the streams may run in parallel in a later funded stage.
Combine them only after each has a useful baseline.

Answers to the five proposed directions:

1. **Report: yes, first, but bound the work.** The October pruning reports already
   contain most required experiments and figures. Refresh the stale evidence
   index and produce a decision-focused synthesis following the repo guide.
2. **New methods: prioritize recovered depth pruning.** Test actual removal of
   intact blocks plus teacher distillation; keep aligned FFN reconstruction and
   recovery as the second candidate. Further local head rankings have lower
   priority because current head exports have little speed benefit and causal
   integration problems.
3. **Clothing: stratify and audit before excluding.** Start the conditioning pilot
   on good pose/camera fits with fitted or moderate clothing. Keep loose clothing
   as a separate evaluation stratum. Exclude corrupted/misaligned examples by
   measured quality, rather than equating every loose garment with bad data.
4. **Pose/MHR rendering: yes, make it an explicit condition.** The current D1
   render is a noisy initialization, not a separate motion condition; its weight
   becomes zero at sigma 1. Begin with calibrated 2D pose, then add MHR geometry
   or ARGAvatar RGB only if the corresponding held-out ablation helps.
5. **Other bases: yes, as bounded quality/cost references.** Reuse the native
   MimicMotion and UniAnimate-DiT implementations with freshly matched inputs,
   extend to Wan2.2-Animate, and include
   Wan-Animate-2 for direct rendered-RGB driving. Wan2.2-S2V belongs in a separate
   pose-plus-audio track when audio is relevant. Do not finetune every base now.

## 1. Evidence that determines the next step

**Terms.** D0 noises the capture target; D1 noises the ARGAvatar render guide.
`c0` is the supplied real first frame's clean latent. A functional mask deletes
contributions in the full-width model. A compact student physically removes
parameters and computation. Recovery updates the remaining model after removal.
A teacher supplies reference predictions; a task teacher must first generate
useful motion-conditioned avatars. Block-causal generation sees completed
generated history and the current block, within a declared lookahead budget.

- The [method screen](../../expr/refiner_prune/2.5/method_screen_20261001/REPORT.md)
  uses two calibration actors, two validation actors and three blind actors with
  three seeds and three sigmas: 27 blind cases per finalist. No finalist passes
  all gates. Worst blind direction deviations are 5.26% for allocated heads,
  15.50% for FFN and 13.39% for depth. Direction agreement measures preservation
  of the teacher, not correct pose or identity.
- Four-block bypass achieves about 1.090x forward speedup in both timing orders
  but fails quality and retains the original checkpoint tensors. A physically
  compressed depth checkpoint had not been demonstrated at that snapshot. The
  October 4 CPU control now proves a physical 44-block export; native BF16 parity,
  speed and quality remain unmeasured. FFN removal of 3,200
  channels reaches 1.056–1.057x but fails functional/export parity. The exact
  one-head export is slightly slower. These suggest recovery and executable
  architecture work, not another broad training-free ranking sweep.
- FFN removal of 3,200 channels is 19.53% of each video FFN's hidden width, but
  only about 6% of stored checkpoint elements. Report full-file, resident-video
  and executed-model denominators separately. The [combined-mask study](../../expr/refiner_prune/2.5/whole_clip_20261001/REPORT.md)
  likewise establishes storage savings, not a useful speed candidate.
- Calibration's flattened stride samples only columns 0 and 16 of a 32x32 grid.
  A balanced spatial sampler remains a worthwhile bounded control. Allocation
  already improved head validation damage from 11.34% to 4.77%; new uniform
  local rankings alone are not the most promising next investment.
- The [dev avatar training study](../../expr/onestep_avatar/dev_training_20261001/REPORT.md)
  already tested generated-history cached D1 training. Capture MSE improves while
  detail and motion deteriorate; at update 100 detail is about 0.77 of capture.
  No adapter is promoted. Guide/capture mismatch plus a mean-seeking objective is
  a plausible explanation supported by D0 and noise controls, not an isolated
  proof that clothing alone causes the failure.
- Its Q8 whole-clip capped-sigma D0 adapters improve one training clip without
  blur but do not meaningfully generalize to an unseen actor. Full-corpus Q8d
  and rank Q8e were **not run at the initial report snapshot**. The owner's
  refreshed report records rank-16 training with rank-32/64 and validation
  queued. Read-only checks during this CPU execution confirmed the existing
  `corpus_cap097_r16` job on GPUs 0–3; its held-out outcome is not available yet. Leave
  that separately launched job and its frozen inputs untouched.
  The prepared corpus contains 3,073 chains from
  117 training actors. This is a denoising-control opportunity, not evidence of
  deployable pose generation or causal success.

**Task boundary:** below sigma 1, D0 contains future capture information in
`x_sigma = (1-sigma)*capture + sigma*epsilon`. At sigma 1, current D1 loses its
render initialization too. Neither a successful low-sigma D0 adapter nor a
smaller D0 model establishes the intended avatar product.

Use the [pruning methods review](../scripts/prune/doc/METHODS.md), the prior
[workspace method plan](../../plans/2026-10-01-prune-method-comparison.md), the
[dev training plan](../../plans/2026-10-01-ltx25-dev-onestep-avatar-training.md)
and [capped-sigma D0 plan](../../plans/2026-10-03-ltx25-dev-d0-capped-sigma-wholeclip.md)
as supporting context. October evidence supersedes older unrun-status statements
in [D1 next actions](d1-next-actions.md); retain its useful cache/decoder controls.

## 2. First deliverable: a current decision report

Budget: approximately one working day of analysis/report work, with no new broad
GPU sweep. Proposed output root: `expr/refiner_prune/2.5/next_stage_20261004/`.

1. Refresh `expr/refiner_prune/2.5/FINDINGS.md`, which still calls September 30
   current. Link the October whole-clip and method-screen reports and identify
   the strongest evidence and its task boundary. Preserve historical reports.
2. Reuse their saved metrics, plots and synchronized media. Correct captions
   that name the wrong actor or arm before importing them into the synthesis.
   List examined videos separately from unexamined cases.
3. Answer four questions: which removals preserve output; which save real work;
   what recovery remains untested; what must change for pose-driven streaming.
   State explicitly that there is no promoted pruning or D1 adapter today.
4. Follow [the workspace report guide](../../CLAUDE.md#structuring-an-experiment-report):
   problem, Terms, shared setup, numbered findings; plain question headings of
   at most 15 words; one varied factor per sub-question; Short answer first;
   baseline/change, variants, output, expectation, coverage and prose results.
   Do not use experiment-summary tables. Scientific figures need PNG, PDF and
   plotted-data JSON; inline posters link synchronized titled MP4s with matched
   capture references and equal-height, aspect-preserving panels. Give each
   experiment a small rendered Mermaid path using the avatar legend.
5. Keep prose in `configs/narrative.json`, layout in `code/report.py`, and an
   independent validator for hashes, case coverage, selection, links and media.
   Rebuild and inspect desktop/mobile rendering. Mark proposed questions
   "not run yet"; a failed scientific gate is a valid negative result.

**Exit:** the report identifies one primary compact architecture and one backup,
with no claimed speed from faithful padding, file size or offline windowing.

## 3. Shared data and evaluation foundation

Before any new recovery or conditioning selection, freeze one actor registry
across pruning, avatar training and external baselines. Previously inspected
blind actors are now development evidence; reserve new actor identities before
further tuning. Split by bare actor ID across all clips/views, including future
corpus training. Hash the manifests and record latent/source identities.

Execution caveat: the existing full-corpus D0 job predates this registry. New
reservations constrain subsequent studies; do not rewrite its running subset.
Any corpus-adapted teacher must be checked against its actual training actors,
and rejected for a claimed unseen-actor comparison when those overlap the new
test identities. A fresh dev task teacher can follow the new registry.

Audit a small paired set for camera/crop/matte correctness, frame synchronization,
pose validity, silhouette mismatch, occlusion, motion amount and garment category.
Define garment labels as fitted, moderate and loose, with an unknown label.
Review disagreements rather than using silhouette error as a clothing detector:
a body mesh is not a clothed-body target. Preserve rejected samples and reasons
in a manifest; filtering must be reversible.

Use four training actors and two validation actors for the initial task pilot.
Reserve at least six additional test actors spanning garment categories, three
evaluation seeds, and at least two views where available. If the available corpus
cannot supply that split, publish the actual coverage and reduce the claim.
Keep capture RGB, capture VAE and guide comparisons separately labeled.

The first clothing question is whether **fit-quality filtering**, at matched
training size, helps held-out pose/detail. Next compare fitted-only versus a
quality-matched mixed-clothing set; evaluate both on the same fitted/moderate/loose
holdout. A fitted-only curriculum is justified if it isolates a learnable task.
Permanent loose-clothing exclusion requires an explicit restricted product scope;
it would not solve general clothing dynamics or misaligned supervision.

Add missing evaluation before scaling: confidence-gated keypoint trajectory error
normalized by subject size; identity consistency from tracked face/body regions;
detail and motion relative to both capture and the same-input full model; visible
garment/hand artifacts; seam resets and long-rollout drift. Feature scores need
valid tracking/visibility. Gradient energy and low pixel change are secondary
diagnostics because noise can look sharp and static outputs can look smooth.
Include capture/teacher ceilings and deliberately mismatched motion/reference
controls. Aggregate by actor; frames and seeds are not independent subjects.

**Exit:** pinned split and quality labels, verified model-coordinate overlays,
and an evaluation that rejects blur, frozen motion and identity substitution.

## 4. Compression stream: recovered depth first, aligned FFN second

### 4.1. Bound the architecture and numerical checks

Keep the existing D0 protocol for continuity: BF16 weights/latents, float32 sigma,
saved noise, clean `c0`, exact direct schedules and stock-stage control. Compare
balanced 2D sampling with the old stride at the same sampled-token count; use
calibration actors only. Broaden direct block screening beyond the eight old
shortlisted positions, with interaction checks for jointly removed sets.

Implement a **static physically shortened checkpoint** with an explicit mapping
from retained blocks to original block IDs. Preserve residual/conditioning
semantics; remap any block-indexed guidance settings; verify serialization,
reload, forward call count and that removed tensors are absent. The current
runtime bypass is a feasibility control, not a small checkpoint.

Begin with four and eight removed blocks, two calibration-selected deletion
patterns per budget. Cheap architecture/numerical checks select one pattern per
budget for the initial recovery pilot; preserve the screened-out patterns as
untested recovery alternatives. Measure actual resident-video parameters and
whole-forward time. Give the two finalists equal recovery, then choose the budget
on validation. This does not establish the globally best recoverable mask.
Expand the budget or fund alternative patterns only after recovery demonstrates
useful generalization. This is a bounded recoverability-inspired study, not a full
implementation of [TinyFusion](https://arxiv.org/abs/2412.01199).

For the backup FFN student, test retained widths aligned to 128, fixed masks and
source-order channel groups. Isolate BF16 reduction-width effects using local
FP32 diagnostics; include FP32 cost if it becomes an execution requirement.
Fit retained output projections by calibration-only ridge reconstruction, then
test full outputs. Local fit alone is insufficient. This adapts reconstruction
ideas already recorded in METHODS; it does not reproduce SparseGPT exactly.

### 4.2. Compare fixed-budget recovery

For a fixed smaller architecture compare: no recovery; reconstruction where
applicable; and teacher-output distillation with small LoRA on retained modules.
Measure backward-memory feasibility first. Pilot limit: 100–200 updates, fixed
effective batch, evaluations at 0/25/50/100 and 200 if reached, and a predeclared
GPU-hour ceiling from measured update cost. Start with at most two recovered
depth candidates and one FFN candidate, using the same data/noise stream.

Use the full-size **distilled D0 model** as the compression teacher for this
screen. Match noisy inputs, prompts, sigma and clean-frame condition; distill
outputs/direction rather than a misaligned guide-to-capture endpoint. Selected
hidden-state or temporal-feature terms can be added as separate ablations after
the simple recovery baseline. Keep dev and distilled fingerprints distinct.
If LoRA underfits and validation supports a capacity limit, test remaining-weight
finetuning on one candidate, with its larger memory/compute budget recorded.

Report deletion damage before recovery, after recovery, and generalization at
each sigma. D0 recovery qualifies compression of this teacher; it cannot qualify
the avatar task. A full-corpus D0 run may be reused as a teacher/control only
after held-out results exist. It is not a prerequisite for the motion pilot,
and this roadmap does not cancel the separately requested Q8 program.

**Fidelity rules:** unchanged or faithful mask export still requires max absolute
latent difference <=0.02. Keep that gate unchanged. A trained compact student is
a changed model: compare in-memory trained execution against its saved/reloaded
checkpoint, then separately judge teacher/task quality. Record a failed
pre-recovery FFN parity test as failed; recovery does not retroactively pass it.
An intentionally changed compact architecture needs its own reference and label.

**Selection:** prefer the recovered candidate with useful quality and measured
cost. Practical research targets are at least 20% fewer resident-video parameters
and at least 1.20x transformer speedup eventually; these are proposed targets,
not existing results. Smaller pilot gains may justify the next budget, but must
exceed measured drift. Report file bytes, resident and trainable parameters,
peak VRAM and actual GEMM/block execution separately. Quantization, token sparsity
and step-count changes remain independent follow-ons.

## 5. Motion stream: establish a task teacher before one-step compression

The current D1 equation cannot retain motion at sigma 1. Repeating longer
unweighted capture-MSE training, higher LoRA rank or sigma sweeps would repeat a
failure already observed. Explicit conditions need a new, documented contract.

1. **Geometry audit first.** Reuse `motion.py` and verified MHR joint mapping to
   project available body/hand landmarks through the calibrated camera and crop.
   Add face cues only with a defined, validated landmark mapping and renderer.
   Verify left/right topology, occlusion, scale and timestamp overlays on capture
   and ARG render. Never copy raw MHR joint ordering into an OpenPose renderer.
2. **Smallest condition first.** Train a pose-conditioned dev teacher with
   persistent `c0`; add depth/normals/silhouette from MHR as a second ablation.
   Treat body geometry as motion/visibility information, not clothing texture.
   ARGAvatar RGB is a third condition for comparing appearance-rich guidance.
3. **Reuse native reference conditioning for the offline pilot.** The trainer's
   [IC-LoRA reference path](../packages/ltx-trainer/docs/training-modes.md)
   concatenates clean reference tokens with noisy target tokens. Use reference
   plus first-frame conditioning with probability 1 for this task. A scaled
   reference reduces token cost; choose its resolution on validation and report
   the condition encoder/tokens/latency. This existing bidirectional path needs
   explicit block visibility and cache work before streaming integration.
4. **Use a denoising teacher objective first.** Noise the clean target during
   training, `x_sigma=(1-sigma)*z_y+sigma*epsilon`, with the guide supplied
   independently and unnoised. Train the native flow/velocity target
   `epsilon-z_y` on generated tokens, with documented weighting and timestep
   sampling; preserve/exclude `c0`. In inference start generated tokens at noise
   (sigma 1), without future capture. Target noising here is standard supervised
   training, not a deployable input. Compare the same schedule between teachers.
5. **Require a useful multistep output.** Keep pose, identity and detail on held-out
   actors before attempting one/few-step distillation. Multistep inference alone
   does not qualify a teacher. If correspondence remains poor, run a bounded
   guide-refitting/correspondence comparison; train-time extra views must be
   labeled privileged. Perceptual/adversarial losses are separate later pilots,
   with concrete frozen features, decoder gradients and visibility masks.

The separate RGB condition and causal training ideas in
[Wan-Animate-2](https://arxiv.org/html/2608.06009v1) motivate this stream; its
architecture and speed are not assumed to transfer to LTX. The paper is recorded
in METHODS with its release and model-size limits.

**Exit:** explicit condition affects motion even at sigma 1, a mismatched-condition
control degrades motion, and the task teacher improves pose/capture fidelity while
preserving identity/detail. If this gate fails, continue fixing conditioning and
supervision before scaling student training.

## 6. Combine the streams and adapt to block-causal inference

Transfer the selected retained-block/FFN **topology** onto fresh dev weights,
attach the explicit condition, then recover against the dev task teacher at its
sampling schedule. This is the default bridge from the distilled compression
screen: it transfers the architecture hypothesis, not its weights or LoRA.
Revalidate the selected pattern on dev; its distilled ranking may not transfer.
Pin the student initialization, source fingerprint, retained-block mapping and
condition branch, and train an adapter against that exact compact base. Any
distilled-weight student trained against the dev teacher is a separate
cross-backbone distillation experiment, not a compatible adapter transplant.
Compare full-size task model, compact
unrecovered model and compact recovered model with identical conditions. Select
size independently of step reduction; then test four/two/one steps as a separate
distillation question. A usable two/four-step student is an acceptable intermediate
result when direct one-step quality fails.

Before training the causal student, close relevant integration gaps:

- `backbone.py` accepts explicit pruned weights, but current sparse active-head
  attention [rejects K/V caches](../packages/ltx-core/src/ltx_core/model/transformer/attention.py).
  Physical compact head widths also conflict with
  [the shared-width cache allocator](../scripts/onestep_avatar/causal_core.py).
  Prefer depth or FFN-only students for the first bridge. Supporting heads needs
  per-layer K/V shapes, head identities and native cache parity; it is new work.
- Use one checkpoint-condition reader in trainer, probes and deployment: base
  fingerprint, retained block mapping/widths, condition type, sigma/schedule,
  crop/objective, and history computation. Close relevant G3/G5 gaps rather than
  silently loading whole-clip adapters into a causal path.
- Fix G8's fused/unmerged adapter discrepancy. FP32 addition followed by BF16
  rounding may still lose a small delta: test trained effects, not only raw weight
  error. Keep an unmerged evaluation reference; choose the deployment path based
  on measured effect parity and include its runtime cost.

Keep one rollout owner in `causal_core`. Start with the existing block-two,
context-eight, sink-one geometry, clean real `c0`, global positions, zero-sigma
history refresh and the student's generated outputs. Document this as the chosen
streaming computation; it is not automatically equivalent to causal-prefix
recomputation, given global-sigma prompt modulation.

Port the condition so **all token paths** obey availability: current output sees
only permitted current/past guide tokens and generated history. Reference tokens
must not relay future output information. Separate condition and history caches
where their lifetimes differ; rebuild caches after optimizer updates. Train from
clip start or from a student-generated detached prefix, never GT history priming
as the deployable default. Run beyond eviction using 17-latent/129-RGB-frame
chains and 13-block evaluations within the existing RoPE time limit.

Use teacher predictions on the same student history with the teacher's own cache,
or declare a bounded-lookahead teacher. A future-informed offline teacher is a
quality reference, not an identical causal target. A few-step generator should
not be treated as a calibrated score model. If endpoint distillation still blurs,
consider a separately budgeted [Self Forcing](https://arxiv.org/abs/2506.08009)
distribution objective with a verified score teacher and auxiliary score network.

**Causality controls:** alter future noise, guide/pose and future source frames
independently at fixed length and absolute randomness. Already emitted latents
must remain identical within the declared backend tolerance. Audit crop/pose
preprocessing too: full-clip crop unions, multiview/full-sequence refinements and
leading-gap future fill can use unavailable future data. Existing ARG reconstruction
can use three frame-zero views; test a single-image arm for a single-image claim.
Declare offline-prepared driving streams separately from live-input streams.

**Decoder controls:** equal-length offline decoding suffices for current quality
comparisons. Before online emission, align decoder noise by absolute coordinates,
measure future-latent support, set an overlap/delay policy and preserve emitted RGB.
Transformer-only causality cannot establish end-to-end streaming causality.

**Exit:** compact save/load correctness, preserved `c0`, future-intervention
controls, useful generated-history motion/detail after eviction, and measured
first-output/per-block/total latency including refresh, conditioning and decode.
At 30 fps a two-latent-frame block covers 16 RGB frames (~0.533 s); call the
system real time only if measured steady-state service time meets that budget,
including declared buffering. Size reduction alone is still a useful result.

## 7. External baselines: reuse, then extend one shared benchmark

The existing [MimicMotion](../../expr/mimicmotion_argavatar_20260927/README.md)
and [UniAnimate-DiT](../../expr/unianimate_argavatar_20260928/README.md) native
DWPose runs cover one actor, 33 frames at 512 square, one seed and 20 steps.
They are smoke evidence, not a model ranking. Their old MHR generations used
wrong topology and must not enter the benchmark; corrected guide files alone
do not repair the generated outputs. UniAnimate's local face-enabled condition
also needs a native-default arm.
The local preprocessing scripts extracted those DWPose conditions from ARG RGB,
not capture RGB. Existing Wan-Animate capture-face videos use future capture
frames and belong to a privileged track. Recorded MP4 fps also differs from
MimicMotion's motion microconditioning; relabeling playback fps cannot repair
the native temporal contract. Fresh matched runs are required.

Use native inputs and distinguish three tracks:

- **Pose:** MimicMotion, UniAnimate-DiT, Wan2.2-Animate and conditioned LTX.
  Reuse common permitted landmarks with each model's native renderer. Report
  missing hands/face and confidence; DWPose from capture is an oracle compared
  with MHR/render-derived driving. Wan's native face input must obey the same
  source policy or be labeled privileged.
- **Rendered RGB:** LTX render-conditioned model and Wan-Animate-2. The
  [existing Wan-Animate-2 run](../../expr/wan_animate2_rgb_drive_20260927/README.md)
  joins overlapping offline clips using the previous last frame. It is not a
  verified block-causal/KV-cache implementation. Score raw output separately
  from capture-mask/background postprocessing.
- **Pose plus audio, optional:** Wan2.2-S2V supports image/audio and optional
  pose video. Match permitted audio and pose; do not rank an audio-only run by
  pose-following fidelity against differently conditioned models.

Primary implementations: [MimicMotion](https://github.com/Tencent/MimicMotion),
[UniAnimate-DiT](https://github.com/ali-vilab/UniAnimate-DiT),
[Wan2.2-Animate](https://github.com/Wan-Video/Wan2.2#run-wan-animate),
[Wan2.2-S2V](https://github.com/Wan-Video/Wan2.2#run-speech-to-video-generation)
and [Wan-Animate-2](https://github.com/Wan-Video/Wan-Animate-2).
Specify UniAnimate-DiT, the local Wan2.1-14B-based implementation, rather than
silently conflating it with original UniAnimate. These 14B-class references
are quality comparisons, not evidence of a smaller deployment model.

Start with four matched cases, then evaluate surviving references on the shared
reserved actor set. Fix reference image, physical timestamps, crop, coverage and
permitted condition sources. Compare shared valid output frames: the old Wan
demos expose 29 frames versus the other models' 33. Keep a native recommended
resolution/schedule track and a controlled-resolution track; same seed numbers
do not imply identical noise between architectures. Record steps, guidance,
dtype, offload and sampling choices rather than forcing one model's schedule
on all models. Include reference/condition preprocessing, bytes/parameters,
VRAM and complete generation time; compute pose/identity/detail/drift using the
same evaluator and raw outputs.

Keep external models zero shot first. Consider adapting UniAnimate-DiT only if
its official training path and held-out evidence justify an alternative to LTX;
match training data and budget in that comparison. Do not launch five finetuning
projects. Verify checkpoint/component terms when packaging a chosen deployment.

## 8. Order, budgets and decision points

1. **Next 1–2 working days:** evidence report/index, shared split and clothing/fit
   audit, baseline provenance triage, conditioning overlays and evaluation spec.
2. **Next bounded pilots:** compression/export/recovery stream and full-size
   explicit-conditioning stream can proceed independently. Each begins with
   memory/time measurements and a fixed update/GPU-hour budget. Coordinate
   GPU reservations so the two streams do not compete for the same four cards.
3. **After both pass:** task-aware compact recovery at fixed steps; then causal
   conditioning/cache integration and generated-history training. Only then
   spend a larger step-distillation budget.
4. **Confirmation:** freeze candidates and thresholds, run reserved actors once,
   publish the quality/size/latency frontier and guide-compliant report. If a gate
   fails, record the negative result and return to the specific failing stream.

Use the current `ltx` environment; `build_guidance` alone uses `argavatar`.
Inspect `nvidia-smi` before GPU work and honor existing reservations. Earlier
training plans used GPUs 0–3 and excluded 6–7; do not infer availability from
those historical assignments. Fresh runs pin revision, model/VAE/prompt/source
hashes, actual epsilon, geometry, schedules, dtype/backend, condition and history
policy. Changed production modules update their matching design docs and focused
CPU tests; native export/cache changes additionally require real-weight controls.

Provisional recovery screen: retain the existing 5% direction/5% capture-MSE
damage gates for D0 teacher preservation, plus detail/motion/video checks. Task
selection must improve pose/identity fidelity without new serious artifacts;
predeclare metric thresholds from frozen-reference variability before training.
A practical pilot guardrail is no more than 5% degradation in detail/motion
relative to the matched full task model, supported by video review. Require a
speed gain in both ABA/BAB orders beyond twice bracket drift; repeat unstable
timings. Do not waive a failed gate because a value is close.

### Implementation checklist

- [x] Current pruning decision report and evidence index, rebuilt and validated.
- [ ] Shared actor split, reversible fit-quality/clothing labels and evaluator.
- [x] Physically shortened checkpoint with retained-block mapping and CPU reload checks.
  Native full-size BF16 parity remains pending GPU.
- [ ] Bounded depth recovery; one aligned-FFN reconstruction/recovery backup.
- [ ] Explicit pose condition and held-out multistep task teacher at sigma 1.
- [ ] Compact task recovery, separating model size from sampling steps.
- [ ] Checkpoint/fusion enforcement and compact-model causal/cache compatibility.
- [ ] Block-causal condition visibility, generated-history training and eviction tests.
- [ ] Raw-output native external baseline comparison with declared input tracks.
- [ ] End-to-end causal/decoder checks and final quality/size/latency report.

Progress from this planning task: local pruning, dev D1 and Q8 evidence reviewed;
external implementations checked against primary sources; new Wan-Animate-2 and
Self Forcing references recorded in METHODS. No future checklist item is claimed
complete by writing this plan.

### CPU execution record

- **Stage 1 complete, 2026-10-04:** [decision report](../../expr/refiner_prune/2.5/next_stage_20261004/REPORT.md)
  and [evidence index](../../expr/refiner_prune/2.5/FINDINGS.md) rebuilt. The
  [independent validator](../../expr/refiner_prune/2.5/next_stage_20261004/validation_checks.json)
  passed selection/input identity checks for 81 blind cases, gate/timing claims,
  eight selected MP4s, 95 artifact pins, corrected captions, reproducibility and
  desktop/mobile layout. This is reused experimental evidence, not a new GPU run.
- **Stage 2 CPU foundation complete:** [registry and reversible labels](../../expr/refiner_prune/2.5/data_audit_20261004/manifests/selected_cases.json)
  index 360 sources from 12 actors; 26 cases were reviewed, with 22 eligible for
  further checks and four preserved exclusions. Actor 7 has visible-hand mask
  omissions; actor 171's original clip has severe ladder occlusion, not a proven
  mask defect. Two views of an intact alternate clip preserve that actor's test
  reservation. The first pose pilot uses two moderate-clothing training actors
  (11/16), two validation actors (13/17), and six reserved actors with two views
  each and seeds 42/43/44. This falls short of the four-training-actor target and
  has no fitted-clothing coverage; the reserved set has two loose and four
  moderate actors. All six overlap the separate existing D0 corpus job, so its
  adapter cannot support a local unseen-actor claim on this test set.
  [Validation](../../expr/refiner_prune/2.5/data_audit_20261004/validation.json)
  checks pins, topology/crop conversion, actor separation and exposure rejection.
  CPU evaluator controls reject frozen motion, substituted synthetic identity and
  blurred real capture. Learned output extraction, real scientific thresholds and
  seam/drift qualification remain pending; the full evaluation checklist stays open.
- **Stage 3 depth CPU proof complete:** [physical checkpoint control](../../expr/refiner_prune/2.5/depth_export_20261004/README.md)
  removes original blocks 3/7/8/15, yielding 44 blocks, 8.19% fewer resident video
  parameters and 7.36% less stored tensor payload. Independent streaming validation
  compared all 4,013 retained tensors (38,923,327,488 bytes) bitwise and passed
  metadata, mapping, native provenance/accounting and corruption controls. The
  full-size native CPU loader also loaded all 1,250 video tensors in BF16, with
  44 blocks and no meta/uninitialized parameters; no full-size forward was run. Focused
  production CPU suites passed 57 tests, including compact-cache eviction and
  retained-model backward gradients. This set retains its previously failed quality
  status; it is an architecture diagnostic. The opt-in balanced 2D sampler and
  bounded FP64 dual-ridge FFN helper pass independent-solve, coverage and cache/input
  mutation controls. The native held-out guard now enforces bare actor IDs across
  clips, Parts and filesystem aliases. Combined focused regression passed 121 tests; the complete CPU pruning suite
  passed 138 tests (one GPU-only lifetime test deselected). Ruff and whitespace
  checks passed.
  No new activations have been scored and no fitted FFN model was produced.
  At 15:58 UTC GPUs 0–3 remained occupied by the pre-existing corpus training job;
  [native GPU parity](../../expr/refiner_prune/2.5/depth_export_20261004/gpu_queue.json),
  recovery, speed and quality are queued, and no existing job was changed. The
  separately refreshed dev report also queues rank-32/64 training and evaluations
  on GPUs 0–3. Honor that reservation across inter-job gaps; memory appearing free
  briefly is not permission to race the existing queue.

- **Stage 4 CPU preparation complete:** [native pose-reference inputs and launch bundle](../../expr/refiner_prune/2.5/pose_condition_20261004/README.md)
  bind all 20 selected cases to exact capture-master prefixes, capture first frames,
  129-frame pose videos, physical timestamps and genuine native pre-connector
  empty-text features. The native flow-training configurations preserve clean
  reference and first-frame tokens; 23 CPU controls verify their sigma-1 behavior,
  loss masks and reference scaling. One reserved actor-188 view has an invalid MHR
  prediction at source frame 84; raw validity is preserved and its condition frame
  is blank. Training/validation prefixes are finite and valid throughout all
  129 frames. [Independent validation](../../expr/refiner_prune/2.5/pose_condition_20261004/manifests/validation_checks.json)
  preserves full sequential decode evidence for 2,580 pose and 516 capture frames,
  rechecks immutable input/media hashes, and verifies exact target slicing and RGB
  first frames. [Launch controls](../../expr/refiner_prune/2.5/pose_condition_20261004/manifests/launch_guard_controls.json)
  pass 57 checks for stale inputs/producer pins, invalid sigma, four-rank fresh
  output reservations and exact reviewed case/video hashes. The real VAE producer
  passes CPU dry-run; all 20 genuine pose-reference latent files remain missing,
  so launch readiness is false. No fixtures enter the training dataset.
  GPU VAE encoding, the 50-update memory/validation sanity run, the separately
  gated 200-update teacher pilot and pose-generation quality remain unrun.
