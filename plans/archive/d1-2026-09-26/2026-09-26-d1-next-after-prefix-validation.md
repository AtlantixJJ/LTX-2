> Archived and superseded. Historical findings and proposals are retained as written; use [the current next-actions plan](../../d1-next-actions.md) for active work.

# Next steps after latent-prefix and decoder validation

Date: 2026-09-26. Evidence: [noise-prefix report](../../../../expr/onestep_avatar/base_distill_noise_prefixes_20260926/REPORT.md), its verification scripts/JSON, and the [latest continuation dossier](../../../../expr/onestep_avatar/d1_diagnostic/HUMAN_REVIEW.md). No new GPU run or training was launched for this recommendation.

## Decision

Stop expanding the investigation of whether future generation overwrites earlier latents: that invariant now passes on the real checkpoint. Do one decoder-only isolation experiment, then a small confirmation of the motion-bearing continuation result. Use those results to choose the conditioning contract for a bounded adaptation pilot. Avoid another broad noise-level or cache-recomputation sweep.

The original deployment problem has three distinct outcomes to measure: abrupt boundary resets, gradual identity/clothing drift, and loss of guide motion. No current frozen-base configuration has demonstrated a reliable solution to all three.

## What the new evidence changes

The prefix experiment independently reruns four shorter causal rollouts with saved epsilon and obtains bit-exact earlier latents. This is useful evidence against future-token leakage or accidental rewriting of past outputs in the tested path. It does **not** test whether the newly generated frames continue the same appearance: an immutable earlier block can still be followed by a different-looking block.

The same latent prefix decoded at different total lengths produces a mean RGB difference of approximately 0.0028–0.0035 and local maxima of 0.44–0.67. Neither statistic alone establishes an identity change. The global mean can conceal localized effects; the maximum can come from a few edge pixels. Per-frame, foreground and face/clothing localization is required.

The report changes decoder input length and redraws noise from a fresh same-seed generator simultaneously. That does not isolate random-noise alignment, padding, temporal context or tile setup. `DiffusionVideoDecoder._decode_pixels` draws a `[B,C,T,H,W]` pixel-noise tensor using the total target shape. The same seed is not a guarantee of identical noise at each absolute `(channel, frame, y, x)` coordinate across shapes.

The report's eight short videos are cuts of one complete decode. They prove prefix presentation is consistent when sourced that way, not that repeated online decoding is stable. Also, the earlier D1 comparisons decoded each complete generated latent sequence once. Differences between short and long decode calls therefore cannot, by themselves, explain their within-video generation-boundary jumps.

Do not pool the new run's quality with the white-background D1 runs: it uses the `bg` objective, `0012_09/view01_cam52`, sigma 1 and block-indexed noise. Its strongest result is the prefix invariant, not a matched comparison of avatar quality or block geometry. The named first and continuation segments each contain four two-latent-frame calls; this is not a new two-call eight-frame-block experiment.

The actual long-block continuation experiment is already complete: two calls covering 129 pixel frames at sigma 0.909375. Cache and joint clean-history modes share exactly the same first nine latent frames, which I rechecked on the saved tensors. Joint lowers the single-boundary latent residual from 0.2377209 to 0.2289050 (3.71%) with mixed appearance results. This supersedes the previous recommendation to run that experiment; the remaining question is its repeatability and visual value.

## 1. Isolate decoder effects once, using saved latents

No transformer generation is needed. Use one existing sigma-1 latent sequence and one motion-bearing sigma-0.909375 sequence. Add a diagnostic mechanism to supply decoder noise explicitly; avoid global RNG monkeypatches in production code.

Compare:

1. **Full length, repeated twice**, identical explicit noise and settings: establish the numerical repeatability floor.
2. **Short versus full**, using slices of one canonical absolute-frame-indexed decoder noise field: isolate length/context/padding effects with noise correspondence controlled. Define padding noise consistently as well; log the actual shapes and coordinates.
3. **Fixed total length and noise, identical latent prefix but altered future latents**: measure how far future context changes earlier decoded pixels. This tests decoder sensitivity, not whether the generated prefix changed. Use a plausible alternate future as well as a simple diagnostic replacement if necessary.

Save uncompressed pixels, per-frame foreground/face/clothing errors and temporal distance from the truncated endpoint. Inspect whether differences are confined to the end of the decoded prefix, spread throughout it, or align with the actual generation boundaries. Compare their magnitude and appearance with the reported seam itself. Do not infer causality merely because both effects exist.

If stable noise eliminates the meaningful difference, use absolute-frame-indexed decoder randomness for streaming. If finite temporal context explains an endpoint region, use overlapping decoding with a measured context margin and delayed emission of unresolved tail frames; retain already emitted RGB frames. Establish the needed overlap experimentally rather than assuming one latent frame is sufficient. If the decoder has material long-range dependence, independent online chunks need a distinct decoder design or a documented approximation.

For current offline evaluation, continue decoding equal-length complete sequences with matched noise; take all prefix views from those decodes. This removes avoidable evaluation ambiguity while keeping the generator comparison intact.

## 2. Confirm continuation quality on a small fixed set

Use the existing 129-frame cache-versus-joint protocol at sigma 0.909375, generated history, identical frozen base, block size eight, context eight plus `c0`, and common global epsilon. Add only four actor/seed pairs: the other two actors at seed 42 and two extra seeds for `0008_01`. Together with the completed pair, this yields five comparisons.

For each pair, assert common first-block raw latents before inspecting the frame-65 continuation. Use the decoder protocol above. Score three outcomes separately:

- Immediate appearance change relative to the last generated frames, using tracked face/clothing regions with occlusion checks.
- Drift relative to the supplied image and corresponding capture appearance, allowing for pose and visibility changes.
- Pose trajectory and motion amplitude relative to the driving guide/capture, including hands when reliable.

Report individual pairs, actual boundary neighborhoods and comparable interior motion. Treat latent MSE and raw RGB-change curves as secondary diagnostics. Report first-output latency and steady-state cost separately from full-video time.

If joint history is consistently better at preserved motion and identity, keep it as the more expensive reference for adaptation. If the effect is mixed or negligible, retain the simpler cached baseline and stop spending on inference-only recomputation variants. Neither outcome licenses a claim that the prompt-sigma cache discrepancy was nonexistent; it determines its practical value for this task.

## 3. Move to a bounded adaptation pilot, not another open-ended diagnostic sweep

Once the decoder confound is controlled and the small continuation comparison is complete, test whether training under the deployed history distribution addresses the residual appearance drift. The existing package supports a practical initial pilot:

- Use the paired render/capture data and the approved full-frame latent MSE objective.
- Keep clean capture `c0`, but feed generated history for subsequent blocks. Use clip-start/whole-clip chains where feasible so GT priming is not silently reintroduced.
- Train at one explicit sigma and the intended deployment geometry. Sigma 0.909375 is a reasonable first hypothesis because it preserves some guide signal and has matched diagnostics; it is not an established optimum.
- Start with the existing one-step training contract. Compare against the frozen base with the **same one-step schedule**. Keep the frozen three-step result as a separate reference. The current training loop is not a drop-in multistep distillation trainer.
- Use a small fixed subset with held-out actors and a bounded update budget specified before launch. Evaluate actual generated-history videos periodically; stop if lower loss is accompanied by reduced motion, blur or identity drift.

This is a capacity/conditioning feasibility test, not evidence that a production architecture or a distillation teacher has already been selected. Do not distill the current inconsistent continuation outputs merely because they use more denoising steps. If a joint reference is used later as a teacher, account for the future noise/guide information unavailable to a causal student.

If guide motion or identity remains poorly constrained despite adaptation, propose an explicit persistent guide/appearance conditioning path. At sigma 1 the current guide disappears entirely, so that operating point cannot by itself solve the pose-controlled avatar task. Adding persistent conditioning is an architectural proposal requiring corresponding training and documentation, not an existing inference flag.

## Deliverable

The next deliverable should combine the decoder attribution test, five matched continuation comparisons, and a decision to retain cache or use joint history as a reference. Then specify the small generated-history adaptation pilot around the actual remaining failure. Prefix immutability is now a passed invariant; continuing to re-prove it will not resolve new-block identity or motion quality.
