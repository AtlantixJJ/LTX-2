> Archived and superseded. Historical findings and proposals are retained as written; use [the current next-actions plan](../../d1-next-actions.md) for active work.

# D1 continuation: review after decoding and grouping experiments

Date: 2026-09-26. This supersedes the pending-work recommendations in the earlier [agent review](2026-09-26-d1-agent-review-next-step.md). Scope: current implementation, saved manifests/tensors, metric scripts, and selected visual contact sheets. No new GPU experiments were launched.

## Recommendation

**Correct the evaluation tooling, then test the first continuation after a 65-frame generated block at sigma 0.909375.** The agent completed the earlier missing controls. Repeating cache/recompute comparisons or adding more sigma-1 runs is now lower value than checking whether the motion-bearing joint result survives its first real continuation boundary.

The cache-conditioning discrepancy remains real, but neither GT-history nor generated-history recomputation has demonstrated a useful visual improvement. Keep it documented as a semantic difference from the explicit causal reference; there is insufficient evidence to promote recomputation as the quality fix. Training should follow a successful continuation reference, or a deliberately scoped feasibility experiment, rather than an assumption that the cache discrepancy explains all jumps.

## What is now complete

- Full saved recompute/joint latents have been decoded, including the other actors' available cache/joint comparisons.
- The generated-history recompute condition exists: boundary residual 0.20331, versus 0.20729 for cache and 0.20786 for the joint clean-history window on `0008_01`. The roughly 1.9% reduction from cache is small, and the reviewed clothing sheet shows distinct outfit drift in all three modes rather than a clear recompute advantage.
- Genuine 65-frame generation in four, two and one call exists at sigma 1 and 0.909375. At 0.909375, the one-call result has lower latent MSE (0.16064 versus 0.16837 for four calls), but this is one actor/seed and contains no continuation after the long block.
- Joint-mode attention metadata is corrected, and raw-latent analysis now reports an interior metric excluding the special first latent transition.

I checked model/prompt/source/epsilon provenance across each grouping, verified the saved D0/D1 latent hashes for all six grouping runs, confirmed nine latent frames in each output, and compared the actual epsilon tensors: they are bit-identical within each grouping experiment. All six use the frozen base without a LoRA and generated history. The five reviewed decode manifests contain distinct output filenames for their current entries. The prior review's 41 passing tests remain evidence for the core code; I did not rerun unchanged core tests during this review.

Selected sheets inspected directly:

- [Sigma 0.909375 clothing at frame 33](../../../../expr/onestep_avatar/d1_diagnostic/decoded_sigma0909_65f/sigma0909_boundary_33_clothing.jpg).
- [Generated-history clothing at frame 65](../../../../expr/onestep_avatar/d1_diagnostic/decoded_review/0008_generated_boundary_65_clothing.jpg).
- [Sigma 1 faces at frame 49](../../../../expr/onestep_avatar/d1_diagnostic/decoded_sigma1_65f/sigma1_boundary_49_face.jpg).

These show different appearance trajectories between configurations; differences between rows are not themselves evidence of temporal jumps within a row. This review inspected still neighborhoods, not every full video at normal speed. The agent's broader playback observations are recorded in the [experiment report](../../../../expr/onestep_avatar/d1_diagnostic/REPORT.md).

## Implementation findings

### 1. Boundary annotations are incorrect for the new grouping comparison

`plot_decoded.py:66` draws boundaries every 16 frames, and line 71 calls them new-block starts. `decode_saved.py:96` likewise selects a fixed list of boundary neighborhoods. The saved block plans actually give:

| 65-frame grouping | Actual zero-based pixel boundaries |
|---|---|
| Four calls | 17, 33, 49 |
| Two calls | 33 |
| One call | None |

Frame 49 is an interior frame in the two-call run and all frames after zero are within one generation block in the one-call run. The frame-49 face sheet can compare appearance at a shared time, but cannot establish a two-call boundary jump there.

**Fix:** propagate each run's block plan into the decode manifest. Derive pixel starts as `(latent_start - 1) * temporal_scale + 1`, excluding block 0. Mark boundaries per series or use separate panels. Label shared-time crops as checkpoints, and separately label actual boundaries. Include every relevant boundary in longer runs; the current fixed list omits 97 and 113.

This affects interpretation of existing plots, not the generated latents or the raw-latent boundary calculation, which already uses each run's plan.

### 2. The decode utility can silently overwrite results in a broader sweep

At `decode_saved.py:61–67`, each `video` entry loops over all latent records attached to its view, without filtering `latent['sigma'] == video['sigma']`. The probe attaches the view's complete latent list to every sigma entry. A three-sigma view can therefore enqueue each D1 latent three times.

The output name at line 92 is only `{actor}_{run_name}`. It omits view and sigma. Multiple sigmas or views within one run overwrite the same MP4 and PNGs. The capture cache at line 74 is only `{actor}_capture`; it can reuse another view/objective/length's target simply because the file exists.

The current reviewed datasets avoid these collisions because each relevant directory has one sigma and one view per actor. That is a limitation of the utility, not evidence that these existing videos are wrong.

**Fix before expanding the sweep:** uniquely identify outputs by run, source/view, sigma, objective, covered span and arm; filter latent entries by sigma; key capture reuse by source hash, objective, covered length and decoder identity/seed. Reject existing conflicting outputs rather than overwriting them. Add a small enumeration test for multiple sigmas, views and lengths; it needs no VAE or GPU.

### 3. Decode provenance checks do not bind the decoder actually loaded

At `decode_saved.py:57`, the script verifies the VAE at the original recorded path, then `open_session(args)` independently resolves the currently configured VAE. It never compares that resolved path/fingerprint with the expected decoder. The source capture is also reloaded without checking its recorded capture hash. The decode manifest lacks the resolved VAE identity, decode seed and source-manifest hash.

**Fix:** validate the resolved decoder against the generation manifest after resolution and before decoding; verify the capture artifact hash; record exact decode provenance. This matters when expanding runs or moving model roots. There is no observed VAE mismatch in the reviewed runs.

### 4. The pixel metric remains a diagnostic, not a selection objective

`plot_decoded.py:37–40` uses a fixed central rectangle and a prediction-dependent foreground mask. It may exclude extended hands/sleeves, and each method's mask can change which pixels contribute. Adjacent RGB change conflates pose, motion, texture, blur and alignment. A low value can reward a nearly static result.

**Fix:** retain the current plots as an explicitly approximate visualization; use common source-derived regions for comparisons and add tracked face/clothing crops plus pose trajectory error. For continuation, measure two separate quantities: appearance change relative to the last generated block, and identity/clothing drift relative to the supplied first frame. Report tracking/occlusion confidence. Use lossless frames for scored measurements; the current plot reads compressed MP4s. Avoid claiming that motion preservation follows from the average RGB-change magnitude alone.

## Interpretation of the latest results

The sigma-1 control successfully removes the current guide, but it also removes the main current-block motion signal. Near-static output under a generic quality prompt is therefore not by itself another bug. Its locally stable identity is not a suitable avatar teacher. More sigma-1 sampling cannot establish guide-following quality in this architecture.

At 0.909375, all groupings move and produce different clothing/face trajectories. Joint generation's modest MSE improvement suggests it is worth testing as a longer block, but it does not demonstrate stable continuation: the existing joint result ends before the first boundary it would need to cross. Likewise, the similar 65-frame runtimes do not measure first-output latency or steady-state streaming throughput. The long-block path emits a different amount per call and consumes more peak memory.

## The next concrete experiment

### A. First continuation after the long block

Use `0008_01/view00_cam51`, seed 42, frozen base, sigma 0.909375 and its three-step tail, generated history, context eight plus `c0`, and the existing globally indexed epsilon. Generate **129 pixel frames / 17 latent frames** using block size eight: `[0,9)` followed by `[9,17)`. The first real transition is **pixel frame 65**.

Compare two conditions:

1. **Cached continuation**, retaining the first block's clean history features.
2. **Joint clean-history continuation**, recomputing those same clean latent values with bidirectional attention to the next noisy block (`history_mode=joint`).

Both modes should produce the same first nine latent frames: block 0 has no history and the same inputs. Verify this equality against each other and the existing one-call 65-frame raw latent, at the measured backend repeatability floor. This makes the continuation comparison start from the same generated appearance, rather than two already-divergent histories. At the first new block there has been no history eviction: all nine prefix frames fit the retained policy.

Decode each full 129-frame output once. Compare the *raw prefix* for reproducibility; a full-length DiffVAE decode need not reproduce the independently decoded 65-frame prefix pixel-for-pixel because temporal context can differ. Review frames 57–73 as a transition neighborhood, then the rest of the second block for drift and pose following.

The joint condition processes up to 17 latent frames together and can exceed the memory of the prior nine-frame experiment. Check GPU capacity before launch. If it does not fit, reduce the common spatial geometry for **both** conditions and rerun the common prefix; do not silently give only one arm fewer history tokens. Report the change as a separate diagnostic. If only cached continuation is feasible at native size, its video still answers whether the apparent long-block quality survives one transition.

Reuse the existing full block-size-two generated-history video as a reference for continuity, identity and motion; it does not have the same generated prefix, so it is an end-to-end comparison rather than the controlled prefix test.

### B. Confirm only an informative result

If either continuation mode improves the frame-65 transition while preserving identity and pose, repeat the matched pair on the other two actors at seed 42 and on two additional seeds for the original actor. This is five actor/seed pairs in total. Expand further only if the effect survives. Record per-call latency, first-output latency, memory, and quality separately.

If both modes jump from the same prefix, the issue remains even with explicit clean-history access. At that point prioritize a small continuation-adaptation feasibility study with the proven input/conditioning contract, not another cache-only sweep. If both are smooth but drift in outfit/identity, prioritize persistent appearance conditioning and supervision of generated-history rollouts. If motion fails, first address guide strength/conditioning; increasing sigma toward one would remove more of the signal needed for the task.

## Deliverable and stopping rule

The immediate deliverable is corrected plot/decode bookkeeping and a two-condition, 129-frame continuation comparison from an identical long-block prefix. A larger training run becomes justified only after choosing which failure it is intended to repair: abrupt continuation reset, gradual identity drift, or lost guide motion. Treat those as separate measured outcomes.
