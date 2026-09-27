# D1 next actions

Updated: 2026-09-26. This is the active plan; it supersedes the six [archived plans](archive/d1-2026-09-26/README.md). The work below is proposed unless explicitly marked complete. This consolidation does not launch experiments or change model behavior.

## Objective

Produce render-guided avatar continuation with stable identity and clothing, without abrupt block-boundary changes or loss of motion. Measure three outcomes separately: boundary resets, gradual appearance drift, and guide following.

All inference comparisons use the **same frozen LTX-2.5 distilled base without a LoRA**. One-step and multistep describe sampling schedules. Historical LoRA outputs are excluded from these comparisons.

## Established evidence

| Question | Current result | Consequence |
|---|---|---|
| Does extending the tested causal rollout overwrite its past? | Earlier output latents are bit-exact in independent shorter reruns | Causal prefix reproducibility is established for the tested configuration |
| Is bidirectional joint generation prefix-invariant? | No. Independent 2/4/6-frame runs differ from the eight-frame run despite shared prefix noise | Adding future tokens can change earlier output; shared noise does not make causal and joint generation equivalent |
| Does varying future **noise values** at fixed length affect earlier predictions? | Not yet isolated; the bidirectional-prefix test also changes length | Run the bounded intervention below to answer this narrower question |
| Is the clean-history cache equivalent to an explicit active-sigma prefix? | No; real-checkpoint K/V and outputs differ before eviction | A computation discrepancy exists, but recomputation has not demonstrated a clear visual fix |
| Does generated history help? | It reduces the sharpest GT-history resets in reviewed videos, with appearance and pose drift | Use generated history for deployment-relevant evaluation; track drift separately |
| Does longer-block continuation solve the problem? | A matched 129-frame pair is complete. Joint history lowers its single-boundary residual by about 3.7%, with mixed visual results | Confirm this candidate on a small set before selecting it |
| Are separately decoded prefixes pixel-identical? | No, even with the same decoder seed | Compare matched decode lengths; do not mistake decoder differences for changed transformer latents |

Evidence: [continuation dossier](../../expr/onestep_avatar/d1_diagnostic/HUMAN_REVIEW.md), [causal-prefix validation](../../expr/onestep_avatar/base_distill_noise_prefixes_20260926/REPORT.md), [AR rerun](../../expr/onestep_avatar/base_distill_noise_prefixes_20260926/ar_incremental_8/REPORT.md), and [bidirectional-prefix study](../../expr/onestep_avatar/base_distill_noise_prefixes_20260926/bidirectional_prefix_8/REPORT.md).

The prefix studies use `bg`, `0012_09/view01_cam52`, sigma 1 and saved block noise. The D1 continuation comparisons use `white`, render guidance at sigma 0.909375 and globally indexed noise. These are separate controls; do not pool their quality scores. At sigma 1 the current render guide disappears, so smooth but static output does not establish avatar quality.

## Action 1 — Finish the fixed-length future-noise test

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

## Action 2 — Confirm the existing continuation candidate

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

## Action 3 — Resolve decoder stability only where it affects the decision

Current offline comparisons should use complete equal-length decodes. The original seam videos were decoded once, so the short/full decode discrepancy alone does not explain their internal seams. Do not make a decoder redesign a prerequisite for the raw-latent test in Action 1.

Before online incremental decoding, or if pixel changes are being attributed to future latent influence, run one decoder-only diagnostic using saved latents:

1. Repeat a full decode with identical explicit noise/settings to establish repeatability.
2. Decode short/full inputs using corresponding slices of a canonical absolute-frame-indexed decoder noise field, including an explicit padding convention.
3. Keep total length, noise and latent prefix fixed; substitute future latent content and measure earlier decoded pixels.

Report per-frame and foreground/face/clothing differences versus distance from the endpoint. Same seed with a different noise tensor shape is not a guarantee of coordinate-wise identical randomness. These controls distinguish noise alignment from future-context/padding effects.

If noise alignment is responsible, stabilize decoder noise by absolute coordinates. If the effect is localized near the endpoint, determine an overlap/context margin and delay emission of unresolved frames. Preserve emitted RGB. A broad temporal effect requires a separate streaming-decoder design or a documented approximation. Optional visuals for Action 1 must also use equal-length decodes.

## Action 4 — Choose one bounded adaptation pilot

After the dependency test and continuation comparison, write a concrete pilot configuration around the remaining failure. This is the next implementation/training phase, not a claim that the present reference already solves continuity.

| Remaining failure | Pilot priority |
|---|---|
| Abrupt reset despite generated history | Adapt continuation under the selected causal/history computation |
| Smooth continuation with outfit/identity drift | Strengthen persistent appearance conditioning and train on generated-history rollouts |
| Lost or incorrect guide motion | Address guide conditioning/strength; sigma 1 removes the current guide entirely |

Start with paired render/capture data, clean `c0`, generated history and the approved full-frame latent MSE objective. Prefer clip-start/whole-clip chains where feasible; disclose remaining GT priming. Choose a single training sigma and deployment geometry, a small fixed subset, held-out actors, an update budget and evaluation cadence before launch.

The existing trainer supports a one-step objective. Compare a pilot against the frozen base using the same one-step schedule; keep multistep references separate. Multistep distillation, a new guide/appearance branch or a temporal loss requires an explicit implementation and documentation change. Do not turn an inconsistent multistep output into a teacher merely because it uses more steps.

Joint teacher outputs can depend on future inputs unavailable to a causal student. Action 1 quantifies this mechanism for noise; teacher construction must also match guide/context availability. Demonstrating that dependency does not itself establish that it causes the observed appearance jumps.

**Pilot acceptance:** fewer resets/drift at preserved guide motion and detail on held-out generated-history videos. Reject apparent gains caused by blur or static motion. Training-loss improvement alone is insufficient.

## Execution and handoff

- Check current artifacts before launching each action; extend completed work rather than overwriting it.
- Use the `ltx` environment, inspect GPU availability, and record resolved model/VAE identities, source hashes, prompt, epsilon, geometry, schedule, masks, history policy, dtype/backend and code revision/snapshot.
- Keep one rollout implementation in `causal_core`; follow package documentation and focused-test requirements for behavioral changes.
- Use fresh experiment directories. Keep metrics and generated media under `expr/`; keep this file as the active action list.

The completed prefix, sigma-removal, cache-parity, recompute, decode-review and first-continuation experiments remain evidence. Repeat them only if a code change or a failed control requires it. The broad bidirectional prefix-invariance question is answered; Action 1 addresses only the narrower fixed-length intervention.
