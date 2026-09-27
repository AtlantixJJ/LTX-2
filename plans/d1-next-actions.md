# D1 next actions

Updated: 2026-09-27. This is the active plan; it supersedes the six [archived plans](archive/d1-2026-09-26/README.md). The work below is proposed unless explicitly marked complete. This consolidation does not launch experiments or change model behavior.

**User constraint: cached history only.** Training and deployment use generated-history clean K/V refresh and the same retention policy. Joint/recomputed history results remain diagnostic evidence; further selection or expansion of those deployment options is closed. Training must address the previously observed detail loss, not merely lower GT MSE.

## Objective

Produce render-guided avatar continuation with stable identity and clothing, without abrupt block-boundary changes or loss of motion. Measure three outcomes separately: boundary resets, gradual appearance drift, and guide following.

All inference comparisons use the **same frozen LTX-2.5 distilled base without a LoRA**. One-step and multistep describe sampling schedules. Historical LoRA outputs are excluded from these comparisons.

## Established evidence

| Question | Current result | Consequence |
|---|---|---|
| Does extending the tested causal rollout overwrite its past? | Earlier output latents are bit-exact in independent shorter reruns | Causal prefix reproducibility is established for the tested configuration |
| Is bidirectional joint generation prefix-invariant? | No. Independent 2/4/6-frame runs differ from the eight-frame run despite shared prefix noise | Adding future tokens can change earlier output; shared noise does not make causal and joint generation equivalent |
| Does varying future **noise values** at fixed length affect earlier predictions? | **Yes (Action 1 complete).** Replacing only generated frames 3–8 changes joint frames 1–2 by 23–31% relative L2 across three replacement seeds; repeat run and causal first block are bit-exact | A joint teacher uses future noise a causal student cannot see; this does not by itself explain appearance jumps |
| Is the clean-history cache equivalent to an explicit active-sigma prefix? | No; real-checkpoint K/V and outputs differ before eviction | A computation discrepancy exists, but recomputation has not demonstrated a clear visual fix |
| Does generated history help? | It reduces the sharpest GT-history resets in reviewed videos, with appearance and pose drift | Use generated history for deployment-relevant evaluation; track drift separately |
| Has self-rollout been swept over initialization sigma for both GT and ARG-Avatar sources? | Only partial coverage: generated-history controls at 0.909375 and 1; the broad historical sweep used GT history | Complete the source-by-sigma self-rollout sweep before selecting a training sigma |
| Does longer-block continuation solve the problem? | A matched 129-frame pair is complete. Joint history lowers its single-boundary residual by about 3.7%, with mixed visual results | Confirm this candidate on a small set before selecting it |
| Are separately decoded prefixes pixel-identical? | No, even with the same decoder seed | Compare matched decode lengths; do not mistake decoder differences for changed transformer latents |

Evidence: [continuation dossier](../../expr/onestep_avatar/d1_diagnostic/HUMAN_REVIEW.md), [causal-prefix validation](../../expr/onestep_avatar/base_distill_noise_prefixes_20260926/REPORT.md), [AR rerun](../../expr/onestep_avatar/base_distill_noise_prefixes_20260926/ar_incremental_8/REPORT.md), and [bidirectional-prefix study](../../expr/onestep_avatar/base_distill_noise_prefixes_20260926/bidirectional_prefix_8/REPORT.md).

The prefix studies use `bg`, `0012_09/view01_cam52`, sigma 1 and saved block noise. The D1 continuation comparisons use `white`, render guidance at sigma 0.909375 and globally indexed noise. These are separate controls; do not pool their quality scores. At sigma 1 the current render guide disappears, so smooth but static output does not establish avatar quality.

## Action 1 — Finish the fixed-length future-noise test (complete)

Result: [report](../../expr/onestep_avatar/joint_future_noise_influence_20260926/REPORT.md).

**Purpose:** determine whether future noise values influence earlier predictions, independently of sequence length. This is a dependency test, not a proposed seam fix.

Use the prefix-study source and objective, frozen base, prompt and spatial geometry. Keep nine total latent frames: clean `c0` at index 0 and eight generated frames at indices 1–8. Use the full sigma-1 schedule:

`[1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0]`.

Construct A from the saved eight-frame noise. Construct B by cloning A and replacing **only generated frames 3–8** with a new saved Gaussian realization. Keep generated frames 1–2 exactly equal. Clamp only `c0` throughout denoising; let every generated frame evolve.

| Run | Attention/grouping | Initial noise |
|---|---|---|
| J-A | One bidirectional block `[0,9)`, no history cache | A |
| J-A-repeat | Same computation | A |
| J-B | Same length, positions, schedule and attention | B |
| C-A | Causal blocks `[0,3), [3,5), [5,7), [7,9)`; generated-history refresh | A sliced by global index |
| C-B | Same causal computation | B sliced by global index |

The joint run must denoise all eight new frames together. `history_mode=joint` with small sequential blocks is a different experiment: it only includes clean past and the current noisy block.

Compare final raw latent indices `[1,3)` before decoding. Record maximum/mean absolute difference, relative L2, changed-element count and exact equality. J-A versus J-A-repeat supplies the numerical floor. Verify actual input slices, `c0` preservation and reset state between runs. Use the same device/backend; do not generate fresh geometry-dependent noise.

**Decision:** a joint effect clearly above repeatability with an unchanged causal first block establishes future-noise dependence. If the causal control changes, debug the controls first. If the joint effect is absent, verify the mask/intervention and try a small number of replacement-noise seeds; a negative sample does not prove independence. Stop once the question is resolved. Optional intermediate predictions can localize the first affected denoising call.

Deliverable: five raw outputs, A/B noise, manifest, metrics and a short conclusion under a fresh `../expr/onestep_avatar/joint_future_noise_influence_20260926/`. No VAE run is necessary.

## Action 2 — Sweep initialization sigma under self-rollout (complete)

Complete 2026-09-27 ([report](../../expr/onestep_avatar/d1_actions_report_20260927/REPORT.md)). Four actor/seed cases, D0/D1 × σ {0.421875, 0.725, 0.909375, 1}; D0=D1 at σ=1. No systematic boundary resets with generated history; σ=0.725 is the balanced D1 operating point; 0.909375 drifts in outfit/pose and ghosts on a fast-motion boundary; σ=1 is static.

**Purpose:** determine whether boundary discontinuity persists when history always comes from generated outputs, and whether the current-block initialization source or strength changes it. This directly tests the proposed configuration; it can proceed independently of Action 1.

The [historical sigma sweep](../../expr/onestep_avatar/d1_comparison/high_noise/REPORT.md) used teacher-forced capture history. Its one-step arm also used a LoRA, so it does not answer this question with a frozen base. Current generated-history diagnostics cover sigma 0.909375 and sigma 1, not the complete sweep below.

Keep the frozen base without a LoRA, clean capture `c0`, generated history only (`teacher_forcing=False`), cached history, white objective, block size two, context eight plus `c0`, and 129 pixel frames. For each new block initialize `x_sigma = (1-sigma) * z_source + sigma * epsilon`, where D0 uses encoded GT capture and D1 uses encoded ARG-Avatar RGB. Noising happens in latent space. GT as the D0 initialization source does **not** mean GT history refresh.

- Start with `0008_01/view00_cam51`, seed 42, and start sigmas `0.421875, 0.725, 0.909375, 1.0`, paired D0/D1. Hold the starting sigma fixed across all blocks of each rollout; changing sigma between windows is a separate, untested intervention.
- Use each sigma's official remaining distilled schedule (one, two, three and eight calls, respectively). This tests practical starting-strength settings; call count also changes, so do not attribute differences solely to sigma. If comparing one-step against multistep, add direct-to-zero runs with the **same frozen base**.
- Use one saved globally indexed epsilon field shared across all sources and sigmas, and reset history for every rollout. Reuse the existing 0.909375 cells only after matching source, prompt, geometry, noise and model provenance. Existing sigma-1 grouping controls cover 65 frames; complete the matched 129-frame control if missing. D0/D1 must agree at sigma 1 within repeatability.
- The current `--trajectory-only` path skips the final one-call schedule. Run sigma 0.421875 through the normal one-step path with no `--checkpoint` adapter; do not silently omit that cell.
- Decode complete equal-length sequences with matched decoder settings/randomness. Score actual boundaries at frames 17, 33, 49, 65, 81, 97 and 113 against matched interior transitions, separating appearance jumps, identity drift, guide motion and detail. Mark boundaries before versus after history eviction.

**Interpretation:** self-rollout removes displayed-versus-GT-history mismatch but does not guarantee continuity. A mismatch can still arise between the generated past and the new GT/render-based noisy block. Lower sigma retains more source content; whether this improves continuity or preserves render artifacts must be measured. Sigma 1 removes current-source guidance and is a dependency control, not a motion-quality candidate. D0 is a diagnostic oracle, not an available deployment input.

If both arms jump at the same sigma, ARG-Avatar input errors are not necessary for that failure. If D0 is stable and D1 jumps, investigate guide-domain/conditioning sensitivity, without claiming this isolates a unique cause. Confirm a promising D1 setting on `0012_09` and `0025_11`, then an additional seed, before selecting it. Save raw latents, provenance, boundary crops and a source-by-sigma results table under a fresh experiment directory. Expand the high-sigma grid only if the initial results require it.

## Action 3 — Cache/joint comparison (closed as a deployment decision)

The September 27 review verified all five pairs and found modest latent-metric gains with mixed pixel results. The user has selected cached history only. The experiment specification below is retained as historical context, not an instruction to launch more joint-history work or select joint deployment. See [implementation review](assets/2026-09-27-d1-implementation-review.md).

The original 129-frame experiment is **complete**; do not repeat its setup as new work. It starts cache and joint clean-history continuation from identical nine-latent-frame outputs and crosses its only generation boundary at pixel frame 65.

Keep its conditions: frozen base, `white` objective, sigma 0.909375 with `[0.909375, 0.725, 0.421875, 0]`, generated history, block size eight, context eight plus `c0`, 17 latent frames / 129 pixel frames. Compare `history_mode=cache` with `history_mode=joint` using the same global epsilon and inputs.

Add four paired cases:

- `0012_09/view00_cam51`, seed 42.
- `0025_11/view00_cam51`, seed 42.
- `0008_01/view00_cam51`, seeds 43 and 44.

Together with the completed `0008_01` seed-42 case, this gives five actor/seed pairs. Assert identical first-block raw latents within every pair. Decode equal-length complete sequences with the same resolved decoder and matched decoder randomness. Inspect frames 57–73 and the remainder of block 2.

For each pair report:

- Face/clothing change across the boundary relative to the preceding generated frames.
- Identity/clothing drift relative to `c0` and pose-matched capture observations.
- Pose trajectory and motion preservation relative to guide/capture, with tracking and occlusion confidence.
- First-output and per-call latency, peak memory, and full-rollout time.

Use tracked crops, actual boundary annotations, lossless frames and common comparison regions. Retain latent residuals/RGB-change plots as secondary diagnostics; lower change can mean frozen motion. Review individual pairs instead of treating frames as independent samples.

**Decision:** select joint history as a quality reference only if its continuation benefit repeats without losing identity or motion. If results remain mixed, retain cached generation as the simpler baseline and close the inference-only cache/recompute sweep. A 3.7% improvement in one latent boundary metric is not a selection criterion by itself.

## Action 4 — Resolve decoder stability only where it affects the decision (complete)

Steps 1 and 3 complete 2026-09-27 ([report](../../expr/onestep_avatar/d1_actions_report_20260927/REPORT.md)): repeat decode is bit-exact; substituting future latents changes earlier pixels only within ~2 latent frames of the endpoint; a 9- vs 17-latent decode with the same seed shifts every frame 5–50× more (noise-shape misalignment). Step 2 (canonical absolute-indexed decoder noise) not run.

Current offline comparisons should use complete equal-length decodes. The original seam videos were decoded once, so the short/full decode discrepancy alone does not explain their internal seams. Do not make a decoder redesign a prerequisite for the raw-latent test in Action 1.

Before online incremental decoding, or if pixel changes are being attributed to future latent influence, run one decoder-only diagnostic using saved latents:

1. Repeat a full decode with identical explicit noise/settings to establish repeatability.
2. Decode short/full inputs using corresponding slices of a canonical absolute-frame-indexed decoder noise field, including an explicit padding convention.
3. Keep total length, noise and latent prefix fixed; substitute future latent content and measure earlier decoded pixels.

Report per-frame and foreground/face/clothing differences versus distance from the endpoint. Same seed with a different noise tensor shape is not a guarantee of coordinate-wise identical randomness. These controls distinguish noise alignment from future-context/padding effects.

If noise alignment is responsible, stabilize decoder noise by absolute coordinates. If the effect is localized near the endpoint, determine an overlap/context margin and delay emission of unresolved frames. Preserve emitted RGB. A broad temporal effect requires a separate streaming-decoder design or a documented approximation. Optional visuals for Action 1 must also use equal-length decodes.

## Action 5 — Cached-history training with an explicit detail-preservation test

The prior recommendation to continue full-frame GT MSE training is insufficient given the observed blur. Start with a bounded objective comparison under the exact cached deployment computation. New losses below are **proposed**, not implemented or already approved training contracts. No training is launched by this plan.

### Evidence informing the change

The [September 21 study](../../expr/onestep_avatar/runs/causal-one-step-study-20260921/report_payload.json) recorded a historical D0 detail/correspondence tradeoff: at step 800, sigma 0.909375 had Laplacian/high-frequency ratios about 0.79/0.68 versus base, while 0.421875 was about 0.99/0.99. This does not establish that lower-sigma D1 is adequate. High sigma also dominated the measured initial D1 gradient norm. The later D1 learning-rate runs trained with GT history; their self-rollout evaluation is not evidence that generated-history training was tested. At 50 steps the low-rate adapter barely changed, so its preserved detail does not demonstrate a learned solution. Detail energy can also measure artifacts; compare against capture and inspect semantic detail.

**Working hypothesis:** one-step regression toward a single capture target can suppress uncertain or misaligned texture and disturb the distilled sampler. GT-history training adds a separate deployment mismatch. These mechanisms are plausible, not isolated causes of the historical blur. Merely changing history, learning rate, or loss from L2 to L1 is not an established cure.

### Training computation

1. Select a useful D1 start sigma from the self-rollout sweep, considering guide fidelity and detail as well as boundaries. Use one operating point initially. One-step versus multistep deployment remains to be specified; the current trainer implements only a one-call output objective. A multistep deployment needs a corresponding rollout/training design, not reuse of a one-step adapter without validation.
2. Start from the frozen distilled base with a small zero-initialized attention LoRA. Keep the backbone, VAE and text encoder frozen. Fix rank across objective comparisons. Small updates limit intervention; they do not guarantee preservation.
3. Generate from clip start with clean capture `c0`, noised ARG-Avatar initialization, and **only the student's generated outputs** for cache refresh. Use the exact deployment global positions, block geometry, sigma schedule, zero-sigma clean refresh and eviction. Do not prime from GT history.
4. Train on trajectories that cross eviction: for block size two/context eight, use the 17-latent/129-pixel-frame chain. If full-chain cost is excessive, generate the prefix with the current student without gradients and train sampled continuation blocks, including blocks after eviction. Detach past outputs/cache; gradients update the current continuation. Rebuild/reset cache after parameter updates, and keep distributed forward counts consistent.

### Bounded comparison before a long run

- **B0:** frozen base, same inference schedule and inputs.
- **B1:** generated-history training with current full-frame GT latent MSE, as the control for changing the history distribution. Do not repeat a long MSE-only run.
- **B2 (proposed):** B1 plus a frozen-base output-preservation term on exactly the same current noisy input, sigma, conditioning, and generated latent history. Compute the base reference with adapters disabled and a separate base K/V cache refreshed from those same student history latents; student K/V cannot be reused as base features. Target is stop-gradient. This limits departure from the existing generator; it can also preserve its mistakes and is not a guarantee against blur.

The existing `--anchor-weight` is disabled and is not a working B2 implementation. A single `base_denoised.pt` per view is invalid because predictions depend on noise, sigma and history. Implement the reference/cache ownership and update the module contracts before using a preservation objective. Compare B1/B2 at matched meaningful learning progress as well as compute budget; an unchanged adapter is not a successful blur fix. Use fixed evaluation seeds, held-out actors and checkpoints early enough to detect degradation.

If B2 has a useful learning/detail tradeoff but residual boundary errors remain, separately ablate motion-aligned appearance-feature supervision across boundaries, with reliable correspondence and visibility gating. Previous output can be detached while the current output receives gradients. This requires a concrete feature/decoder/gradient implementation; current training never decodes RGB. Do not use adjacent-frame equality or maximize high-frequency energy as a substitute for correct moving detail.

If useful learning still necessarily loses detail, stop increasing MSE training time. The next larger project is cached self-rollout with a video-level perceptual/distribution objective, retaining explicit avatar identity/pose supervision. [Self Forcing](https://arxiv.org/abs/2506.08009) supports the general combination of KV-cached self-generated histories and video-level distribution matching; it does not establish a ready-made solution for this LTX/ARG-Avatar setting. A DMD-style method needs an appropriate diffusion score teacher and auxiliary learned score model, not the distilled few-step generator assumed to be a calibrated score model. Alternatively adapt a non-distilled teacher and re-distill into the cached student. These require new infrastructure, teacher/data verification and a separate bounded plan.

| Remaining failure | Pilot priority |
|---|---|
| Abrupt reset despite generated history | Adapt continuation under the selected causal/history computation |
| Smooth continuation with outfit/identity drift | Strengthen persistent appearance conditioning and train on generated-history rollouts |
| Lost or incorrect guide motion | Address guide conditioning/strength; sigma 1 removes the current guide entirely |

Use paired render/capture data, clean `c0`, generated history, a small fixed subset and held-out actors. Specify the update budget, evaluation cadence, loss weights and detail/motion rejection criteria in the pilot configuration before launch. Full-frame MSE remains the implemented control objective; additional losses require explicit implementation and documentation changes.

The existing trainer supports a one-step objective. Compare a pilot against the frozen base using the same one-step schedule; keep multistep references separate. Multistep distillation, a new guide/appearance branch or a temporal loss requires an explicit implementation and documentation change. Do not turn an inconsistent multistep output into a teacher merely because it uses more steps.

Joint teacher outputs can depend on future inputs unavailable to a causal student. Action 1 quantifies this mechanism for noise; teacher construction must also match guide/context availability. Demonstrating that dependency does not itself establish that it causes the observed appearance jumps.

**Pilot acceptance:** fewer resets/drift at preserved guide motion and detail on held-out generated-history videos. Reject apparent gains caused by blur or static motion. Training-loss improvement alone is insufficient.

## Execution and handoff

- Check current artifacts before launching each action; extend completed work rather than overwriting it.
- Use the `ltx` environment, inspect GPU availability, and record resolved model/VAE identities, source hashes, prompt, epsilon, geometry, schedule, masks, history policy, dtype/backend and code revision/snapshot.
- Keep one rollout implementation in `causal_core`; follow package documentation and focused-test requirements for behavioral changes.
- Use fresh experiment directories. Keep metrics and generated media under `expr/`; keep this file as the active action list.

The completed prefix, sigma-removal, cache-parity, recompute, decode-review and first-continuation experiments remain evidence. Repeat them only if a code change or a failed control requires it. The broad bidirectional prefix-invariance question is answered; Action 1 addresses only the narrower fixed-length intervention.
