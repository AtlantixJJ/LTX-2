# Review of the D1 continuation implementation and experiments

Date: 2026-09-26. Reviewed the current uncommitted implementation, saved manifests and raw artifacts under `../expr/onestep_avatar/d1_diagnostic`. This review launched CPU checks only; it did not change inference code or start new GPU experiments.

## Decision

The agent has established a real cache-conditioning discrepancy, but has **not established a visual fix**. Keep the new modes as diagnostics. The next task is to decode the saved alternatives and complete the generated-history comparison before selecting an architecture or starting training.

The [experiment report](../../expr/onestep_avatar/d1_diagnostic/REPORT.md) is appropriately cautious in its final decision. Its strongest result is computational: sigma-dependent prompt conditioning changes later-layer history K/V, including at sigma 1. Its quality evidence is weaker because it uses uncompensated latent differences, largely on one seed, and the full joint/recompute alternatives have not been decoded.

## Implementation review

The core implementation is consistent with the intended diagnostic modes:

- `cache` retains the existing zero-sigma history refresh.
- `recompute` rebuilds the retained clean prefix at the active global sigma, keeps history token timesteps zero, and applies a block-causal mask.
- `joint` uses the same clean prefix and current noisy block without that mask. Only the new block is emitted; the history latent values stay fixed.
- Global-frame epsilon permits geometry comparisons without changing the noise. The main saved comparisons use no adapter and have matching model/VAE fingerprints, prompt hash, capture/guide hashes and epsilon hash.

I reran the causal-core and shared-probe tests: **41 passed**. I independently recomputed the full-prefix GT and `0008` joint-generated raw-latent measurements; they exactly reproduce the saved JSON values. These checks support the diagnostic machinery, not the quality of any continuation mode.

### Findings to address

1. **The quality evaluation does not yet measure the reported symptom directly.** `analyze_raw_latents.py:34–39` averages differences across latent channels and the entire spatial grid. It mixes appearance, motion, alignment and background effects; its interior set also includes the special frame-0-to-frame-1 latent transition. Lower boundary residual can reflect drift or altered motion rather than stable appearance. Preserve this metric as a secondary diagnostic; add decoded foreground/appearance checks, and report interior statistics excluding that special transition.
2. **The generated-history causal-recompute arm is missing.** Saved runs contain cache/generated and joint/generated, but all recompute runs use GT history. Thus the small GT-history benefit does not establish the recompute mode's value under deployment conditions. One full generated-history recompute run is needed to complete that comparison.
3. **`joint` is not a jointly denoised long-span reference.** In `causal_core.py::denoise_with_clean_history`, past latent values remain clean and fixed. Their hidden representations respond to the current block, but future new blocks are absent. Its mixed result does not rule out the hypothesis that jointly denoising several new blocks is coherent while sequential generation is not.
4. **Joint-run attention metadata is wrong.** `visualize_d1.py:139` writes `attention: "block_causal"` for every mode, including `joint`, which passes `attention_mask=None`. `history_mode` records the truth, so the computation remains identifiable, but consumers of the attention field can misclassify it. Derive that field from the selected mode in the next probe maintenance change; distinguish bidirectional attention within the current window from access to future blocks.
5. **Full-rollout recomputation changes more than prompt sigma after eviction.** Recomputing retained tokens omits evicted ancestors that influenced their original cached features. The block-1/pre-eviction parity evidence isolates the sigma issue; later quality differences combine sigma and history recomputation effects. The new tests recognize this; preserve that qualification in experiment conclusions.
6. **Some report text is stale.** The diagnostic report says no full recomputed rollout was measured at line 39, then reports that completed run later. The analysis plan also retains prospective descriptions above its implementation follow-up. Update those statements when the next results are added so they cannot be mistaken for the current status.

No defect in the reviewed modes has been shown to explain the remaining visible jumps by itself. In particular, correctly preserving clean history does not guarantee the frozen base will use it to continue identity and texture.

## What the experiments now establish

| Finding | Supported interpretation |
|---|---|
| Real-checkpoint D0/D1 outputs are identical at sigma 1 | Current source removal works in the tested runs; there is no evidence of hidden render dependence there |
| Block-1 cached/recomputed relative L2 is about 0.114 for D1 at 0.909375 and 0.237 at sigma 1 | The cache differs substantially from the explicit active-sigma causal reference before eviction |
| Full GT-history recompute boundary residual changes 0.23775 → 0.23581, runtime 26.23 → 77.12 seconds | About 0.8% improvement in this proxy at 2.94× cost; insufficient to select a production fix, and not evidence that visual effects are necessarily small |
| Generated cache lowers boundary residual but raises reconstruction/interior errors across three actors | Changing history source trades resets against drift; lower boundary residual alone cannot select it |
| Joint/generated boundary error improves for only one of three actors | No consistent proxy improvement; saved videos are needed to assess appearance |

## Next experiment, in order

### 1. Decode existing results first

Create a decode-only comparison from saved latents; do not rerun the transformer for artifacts already available. Use the same VAE, seed and complete sequence per decode. For `0008_01`, show GT, cache/GT, recompute/GT, joint/GT, cache/generated and joint/generated. For the other two actors, show the available cache/joint GT/generated pairs.

Deliver normal-speed videos and enlarged face/clothing crops around frames 17, 33, 49, 65 and 81, plus per-block appearance/motion plots. Compare both boundary and within-block motion. Record whether each alternative has abrupt resets, gradual drift, sleeve/texture changes, blur or reduced motion. If optical-flow residuals are added, mask occlusions and unreliable correspondence; do not call raw frame differences motion-compensated.

This is the cheapest missing evidence and should precede another broad inference sweep.

### 2. Complete the missing generated-history arm

Run the same frozen base on `0008_01/view00_cam51`, seed 42, sigma 0.909375 three-step tail, block size two and context eight, with `history_mode=recompute` and generated history. Keep model, prompt, input and global epsilon identical to the existing cache/generated and joint/generated artifacts. Save and decode the full eight-block output.

This supplies the missing causal-recompute cell. First compare boundaries before eviction to isolate continuation behavior without the retained-prefix confound. Extend to other actors/seeds only if appearance improves or the result is needed to resolve an ambiguity.

### 3. Establish a genuine jointly denoised reference

If all continuation modes still jump, compare the same first nine latent frames (65 pixel frames) using:

| Grouping | New latent frames per call | Calls | Purpose |
|---|---:|---:|---|
| Current continuation | 2 | 4 | Three boundaries |
| Larger continuation | 4 | 2 | One boundary |
| Joint generation | 8 | 1 | All eight new frames denoised together with clean `c0` |

Use generated history, the same frozen weights, the same global frame-indexed epsilon, identical source/c0/prompt, and matched decoded coverage. Begin at sigma 1 with the full eight-step schedule to remove residual guide dependence. A nine-latent-frame joint span is a memory-dependent target; use five latent frames for a shorter comparison if necessary and disclose the change. Repeat at 0.909375 only after the sigma-1 comparison is informative.

The existing four-frame experiment covers only one block of five latent frames. It provides a short joint control but has no remaining boundary, so it cannot establish that larger-block continuation solves seams. The proposed two-call four-frame condition tests that explicitly.

If joint generation is coherent and continuation jumps, prioritize causal/continuation adaptation with an information-matched teacher. If joint generation also has abrupt changes, examine the frozen-base conditioning, prompt, source/decoder behavior and sampler reference before investing in causal distillation. If jumps move to the new block boundaries, increasing block size reduces their frequency rather than fixing their cause.

## Gate before training

Select a continuation mode only after decoded generated-history videos show improved identity/clothing continuity with preserved motion, detail and guide control where a guide is present. Verify the promising configuration on the three actors and at least two additional seeds, with latency/memory reported. At sigma 1 the current guide is absent, so judge it as a conditioning diagnostic rather than a pose-controlled avatar result.

The immediate deliverable is a visual comparison of the saved results plus one missing generated-history recompute run. A broad LoRA or sigma sweep would be premature.

## Follow-up completed 2026-09-26

The saved full recompute/joint outputs were decoded from raw latents, as were the available
cache/joint pairs for `0012_09` and `0025_11`. The missing generated-history recompute cell
was run with matched source, model, seed, epsilon and three-step tail. Its `0008_01` D1
boundary residual is 0.20331 versus 0.20729 for cache and 0.20786 for the joint clean-history
window. Visual review finds GT-history face/clothing resets, while generated history reduces
the sharpest resets but drifts in costume and pose. Recompute does not yield a clear visible
improvement. The manifest attention field is corrected for joint runs, including saved
manifests.

The genuine first-nine-latent-frame grouping was run in four, two and one call at both sigma 1
and 0.909375, with matching global epsilon and frozen weights. The sigma-1 outputs are nearly
static; the guide-bearing 0.909375 joint result has slightly lower latent MSE on this actor
but changes the face and outfit trajectory and lacks full-clip continuation evidence. No
training or deployment change was selected. Full measurements, normal-speed videos, boundary
crops and decoded-pixel diagnostics are in
[`expr/onestep_avatar/d1_diagnostic/REPORT.md`](../../expr/onestep_avatar/d1_diagnostic/REPORT.md).
