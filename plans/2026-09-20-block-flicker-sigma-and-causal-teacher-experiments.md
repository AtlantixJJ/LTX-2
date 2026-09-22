# Block-boundary flicker: noise level, cache behavior, and teacher supervision

Date: 2026-09-20. Status: proposed experiment plan; no new GPU probe or training launched.

## Objective and first experiment

Determine why the unadapted distilled checkpoint is temporally coherent within a generated block but changes appearance across autoregressive block boundaries. Start with the user's proposed comparison: **sigma 0.909375 versus 1.0**, holding the original probe's other conditions fixed. Then distinguish teacher-forced history effects from cache implementation and causal adaptation issues. If a larger jointly generated span is coherent, evaluate using it to supervise a student that emits that span in two autoregressive calls.

This plan belongs to `LTX-2/plans/`, as requested. It is an experiment proposal; implementation contracts remain in [the package docs](../scripts/onestep_avatar/doc/README.md).

## Evidence already inspected

Reference video: [step 0, sigma 0.909375](../../expr/onestep_avatar/runs/white-d0-tf-c0-debug/probes/init/step_00000_sigma_0.909375.mp4).

Its [manifest](../../expr/onestep_avatar/runs/white-d0-tf-c0-debug/probes/init/manifest.json) records:

| Item | Reference setting |
|---|---|
| Source | `Part_1/0152_01/views/view00_cam51` |
| Objective / input | `white`, D0: capture latent is the noising source |
| Model | Run config records model key `2.5`; resolve and record actual checkpoint files before reproduction |
| Panels | Capture / frozen base / step-0 adapter |
| History | **Teacher forcing**: completed capture tokens enter the cache |
| Block geometry | 2 generated latent frames per call; block 0 additionally contains `c0` |
| Retained history | 15 context latent frames plus the pinned 1-frame sink |
| Clip | 27 latent frames, 13 autoregressive calls, 209 decoded frames at 30 fps |
| Seed | 42 |
| First-frame condition | Clean capture frame-0 latent, timestep zero |

Boundary-adjacent frames were extracted and visually inspected. They support investigating changes in generated pose and appearance, but no quantitative flicker result has yet been computed.

The current `test_causal_core.py` suite was run on CPU: **18 passed**. It includes cached-versus-explicit-block-causal equivalence on a small real transformer. This supports the implementation of the tested semantics; it does not establish historical checkpoint/GPU parity or equivalence to full bidirectional attention.

The current rollout appends nonoverlapping generated latent blocks and decodes the assembled latent video. It does not independently decode and stitch overlapping pixel windows. The suspected seams should therefore be described as autoregressive block boundaries, while checking decoder effects separately.

An existing unrelated local modification adds progress logging to `visualize_d0.py`. Preserve it during any later implementation.

## Hypotheses

| Hypothesis | Prediction / discriminator |
|---|---|
| H1: residual current-block capture content at 0.909375 conflicts with generated appearance | Removing that content at sigma 1 may reduce boundary changes. The paired sigma experiment tests this operational hypothesis. |
| H2: teacher forcing creates disagreement between displayed output and subsequent history | The next block is conditioned on the real capture, even when the previous displayed prediction has different clothing, pose, or face details. Compare teacher forcing with generated-history refresh. |
| H3: cache, timestep, position, or assembly defect | Cached output differs from an explicit forward implementing the **same causal mask**, history contents, and timesteps. Diagnose in latent space before attributing a difference to decoding. |
| H4: the bidirectional base is not adapted to causal history | Cache parity passes, yet jointly denoised spans are more coherent than cached autoregressive continuations. Access to history is available but insufficiently used under the changed attention/noise configuration. |
| H5: sigma 1 is a poor operating point for a single denoising call | Sigma 1 produces poor generation despite being on the schedule. A normal multi-step sampling control can distinguish this from an unsupported numerical timestep. |

These explanations can coexist. A K/V cache provides information; it does not enforce equal appearance between separately sampled blocks. In particular, an independently refreshed clean-history cache is not the same computation as bidirectional joint denoising, in which past hidden states can respond to current/future tokens.

For generated tokens, the existing input is

\[
x_\sigma=(1-\sigma)z_y+\sigma\epsilon.
\]

At sigma 1 it becomes pure noise. The supplied `c0` remains clean, and teacher-forced history remains real capture history. The sigma comparison also changes the model timestep and noise scale, so improvement supports a useful operating-point change but does not uniquely prove that the residual signal caused flicker. In D1, sigma 1 also removes the current render guide entirely; a successful D0 diagnostic would not establish a deployable guide-controlled solution.

## P0 — prepare a matched, reproducible probe

1. Resolve the reference capture, objective-consistent `c0`, text context, model/VAE files, dtype, backend, and temporal positions. Save exact paths and identifiers. Use the recorded model family to reproduce this run; do not substitute a different checkpoint while changing sigma.
2. Set block size 2 and context size **15 explicitly**. Current code defaults to context 8, unlike the historical manifest. If resource limits require another geometry, run both sigma values under that geometry and label it as a new experiment rather than the historical reproduction.
3. Use the frozen base without a LoRA. A duplicate zero-initialized adapter panel adds no information to the primary comparison. The original run's step-0 panel can remain a visual reference.
4. Generate and save identical per-block epsilon tensors for both sigma arms, following the original seed-42/block-index convention. Reset the cache between arms. Assert that generation noise differs only through the sigma mixture and that `c0` is unchanged.
5. Save the assembled output latent before decoding. Decode every arm with the same VAE path over the complete latent sequence. Keep measurements free of captions and MP4 compression.

### Small implementation work required before execution

The current `visualize_d0.py` hardcodes three sigma values, uses default geometry, and requires a checkpoint argument even though it computes a base panel. It cannot express this matched base-only experiment through its current CLI.

Extend the existing probe to accept explicit probe sigmas, explicit block/context geometry, and a base-only mode. These are **proposed capabilities**, not currently runnable flags. Continue to use `causal_core` for all assembly, denoising, refresh, and eviction; do not create a second rollout implementation or monkeypatch global constants. Validate nonzero sigmas against the selected model's schedule. The current schedule includes 1.0; membership does not imply that a direct `[1.0, 0.0]` jump has good sample quality.

Record resolved geometry, conditioning, history policy, actual sigma list, model identity, noise provenance, and output paths in the manifest. Update the probe doc and configuration examples alongside these changes. Run focused probe tests and the required package checks before GPU execution; additional `scripts/prune` tensor changes require its applicable parity gate.

## P1 — four matched frozen-base rollouts

Start with the reference clip and seed 42:

| Arm | Sigma | History refresh | Purpose |
|---|---:|---|---|
| A | 0.909375 | Real capture | Reproduce the original base condition |
| B | 1.0 | Real capture | User's proposed sigma test |
| C | 0.909375 | Generated output | Test whether displayed-history mismatch contributes |
| D | 1.0 | Generated output | Test the sigma change with generated history |

Execute A/B first and inspect them, then C/D. All four use one denoising call per autoregressive block, identical geometry, the same per-block noise, and the same clean initial frame. At 13 blocks, each rollout currently requires 13 denoise plus 13 refresh forwards; all four total 104 transformer forwards, excluding loading and decoding. Measure wall time and peak memory rather than borrowing a training estimate.

Suggested output root, relative to the LTX-2 repository:

`../expr/onestep_avatar/runs/base-block-flicker-sigma-20260920/`

Save a side-by-side `capture | A | B` comparison and a corresponding `capture | C | D` comparison, individual uncaptioned results, raw latents, logs, and the manifest. Never overwrite the historical probe.

If the first comparison is valid, repeat A/B with two additional fixed seeds and two held-out actor clips at seed 42. This produces five clip/seed pairs including the original. Repeat C/D over those pairs when the initial generated-history result is promising or needed to explain the outcome. Keep per-pair results rather than pooling frames as independent observations.

## Measurements and interpretation

Nominal block transitions in the reference video occur at zero-based decoded frame indices **17, 33, 49, ..., 193**. Derive these from the actual block plan and VAE temporal scale. Decoder temporal context can spread a latent discontinuity across several frames, so inspect short windows around each transition as well as the exact boundary frame. Analyze early boundaries before any cache eviction separately from later ones.

For each arm:

- Inspect face identity, clothing color/texture, body shape, pose transitions, and hand motion around boundaries. Show normal-speed playback and boundary contact sheets.
- Measure temporal change in matched subject regions using appearance features and, where reliable, motion-compensated frame residuals with occlusion masking. Compare boundary windows with motion-comparable within-block windows; report both absolute changes and boundary/interior differences. Simple adjacent-pixel differences alone confuse motion with flicker.
- Record subject reconstruction/perceptual error against the capture and the capture's VAE reconstruction, but do not use reconstruction as the sole ranking at sigma 1: that arm has deliberately lost current-block capture information.
- Measure motion magnitude and inspect motion plausibility, identity, and sharpness alongside temporal stability. A frozen pose, blur, or collapsed subject must not count as a consistency improvement.
- For teacher-forced arms, record previous prediction-versus-capture discrepancy near each boundary. The cache contains the latter, which can explain why the next prediction appears to reset.

Use the exact same measurement pipeline and frame-selection rules for all arms. Mark optical-flow failures and subject-tracking failures instead of interpreting their numbers as appearance changes. The first seed is exploratory; a claim of improvement requires the direction to persist over the confirmation pairs without gross motion/identity degradation. Do not claim that the issue is “gone” from one smooth video or one scalar score.

| Outcome | Interpretation / next action |
|---|---|
| B improves over A, and D improves over C | Sigma change is promising for this diagnostic. Check motion and guide dependence before considering deployment. |
| B improves but D does not | Improvement relies on clean capture history; it is not evidence of stable deployed generation. |
| C improves over A | Displayed prediction versus GT-history mismatch contributes. Quantify whether drift increases later. |
| All arms flicker at boundaries | Investigate causal conditioning and cache parity; residual capture signal is not a sufficient explanation. |
| Sigma 1 loses coherence or motion entirely | The one-step sigma-1 test is inconclusive about residual-signal conflict. Use a supported multi-step control if resolving that uncertainty matters. |
| Cached versus explicit causal outputs disagree | Treat as an implementation defect and fix before further model-training conclusions. |

## P2 — targeted cache and conditioning checks

Run these only to resolve uncertainty left by P1; avoid a large uncontrolled sweep.

**Cache parity:** on a short, memory-feasible span with the real checkpoint, compare cached denoising against explicit history tokens under the identical block-causal mask, global RoPE positions, and per-token timesteps. History must contain the same clean tokens in both cases, generated or GT as appropriate. Begin before eviction to avoid changing available history. Establish numerical repeatability for the actual dtype/backend and compare relative latent errors against that floor. The existing small-transformer CPU test is supporting evidence, not a substitute for this checkpoint-level check.

**History availability:** assert cache length, retained token spans, and preserved `c0`; inspect denoise read-only behavior and refresh writes. Compare full intended history with a `c0`-only control to determine whether history affects the output. This control establishes sensitivity, not quality by itself.

**In-window versus cached history:** compare a mid-clip continuation with the immediately preceding clean latent frame as an explicit timestep-zero prefix against the cache-only path. Preserve global positions and avoid duplicate prefix tokens in the cache. Distinguish an explicit computation with the same causal mask (parity check) from bidirectional in-window conditioning (a changed architecture). Improved quality in the latter would support a conditioning limitation, not automatically identify a cache bug.

If sigma 1 failed badly, a further control can run the checkpoint's normal multi-step schedule on the same short span. First establish the bidirectional reference, then compare a causal implementation of the same schedule if needed. Do not reinterpret a distilled output as a calibrated continuous velocity and integrate arbitrary timesteps. This control requires explicit sampler support and separate provenance from the one-step experiment.

## P3 — test the larger-span teacher proposal

Interpret the user's “two-step rollout” as **two autoregressive generation calls**, each emitting two of the current temporal blocks. Keep the number of diffusion denoising steps per call fixed and report it separately.

For a clip-start example:

| Configuration | Generated latent frames per call | Calls | Total latent frames including `c0` |
|---|---|---:|---:|
| Current baseline | 2, 2, 2, 2 | 4 | 9 |
| Proposed student | 4, 4 | 2 | 9 |
| Joint teacher reference | 8 | 1 | 9 |

All cover the same 65 pixel frames at temporal scale 8. The first call contains `c0` in addition to its generated frames. Use one saved noise tensor over the entire span, assembled from the original baseline block noises and sliced for each configuration; changing block grouping must not silently change noise. Decode the whole span identically.

First verify the premise: the joint teacher must actually produce coherent, sharp, moving content across all four original blocks. Then test the unadapted base with the student's 4+4 grouping. A larger group may already reduce the number of visible seams, with increased latency and memory. It does not demonstrate that a remaining seam has been solved by learning.

### Candidate supervision, if the teacher is good

Generate one joint teacher target `Y_T[1:4]`. Train the student to emit blocks 1–2, refresh its **own** cache, and then emit blocks 3–4. Begin with teacher-target history to isolate continuation fitting, and evaluate with student-generated history; eventually train under generated history if that mismatch matters. Targets are generated teacher outputs, not the original capture, and `c0` remains the supplied real frame. No DMD is required to run this regression experiment.

There is a remaining information mismatch: the joint teacher's first half can depend on second-half noise/content that the student's first call cannot see. Shared noise does not remove that dependence. Measure it by fixing `c0` and first-half inputs, resampling only second-half noise, and checking how much the teacher's first half changes. This is the architectural issue analyzed in [Causal Forcing](https://arxiv.org/abs/2602.02214); a coherent joint teacher alone does not guarantee a learnable causal sample-by-sample mapping.

If dependence is large, test a teacher restricted to the information available at each student call: generate the first half from its own inputs, then generate the second half with the first half supplied as clean context. Recomputing a clean prefix in-window may provide a useful teacher for the student's cached continuation, provided teacher quality is demonstrated. This is a different teacher construction and must be labeled accordingly. It does not require pretending that the original teacher's joint trajectory was causal.

P3 training is a follow-up decision after P1/P2 and teacher validation. Its dataset, regression target, checkpoint metadata, and changed block geometry need a separate concrete implementation recipe. Success means reduced boundary flicker in generated-history rollouts with preserved detail/motion on held-out clips, not just lower teacher-forced MSE.

## Execution order and deliverables

1. Implement the minimal explicit probe configuration and validate it.
2. Run matched A/B, then C/D; retain complete artifacts and timing.
3. Confirm a promising or ambiguous result across the specified clip/seed pairs.
4. Perform only the P2 checks needed to discriminate remaining hypotheses.
5. Preview joint teacher and 4+4 student grouping before deciding on any training.

Before GPU execution, inspect current device usage and select free resources. Run in the documented `ltx` environment. The original 15-frame context has substantial K/V memory requirements; do not silently reduce it to make one arm fit. The initial experiment is inference-only and does not require a long-running training job.

The result report should state: exact conditions reproduced; videos and latent artifacts; boundary and motion measurements; evidence for/against each hypothesis; limitations of sigma 1; and whether to fix implementation, adapt causal conditioning, or proceed to the larger-span teacher experiment. Record what was actually run separately from planned follow-ups.
