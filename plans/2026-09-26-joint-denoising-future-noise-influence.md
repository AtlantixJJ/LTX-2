# Does future noise affect earlier outputs during joint denoising?

Date: 2026-09-26. Status: proposed experiment; no implementation or GPU execution performed.

## Question

When eight new latent frames are denoised together with bidirectional attention, does changing only the noise in generated frames 3–8 change the output in generated frames 1–2?

The previous [autoregressive prefix experiment](../../expr/onestep_avatar/base_distill_noise_prefixes_20260926/ar_incremental_8/REPORT.md) compares two executions of the same sequential block-causal algorithm. Its bit-exact prefixes establish reproducibility under that algorithm. It does not test influence between frames inside one jointly denoised block.

## Fixed setup

- Frozen LTX-2.5 distilled base, no LoRA. Resolve and record the same model used by the prefix study.
- Use that study's source, `Part_1/0012_09/views/view01_cam52`, and `bg` objective. Take `c0` from the corresponding capture master.
- Keep prompt/context, spatial geometry, fps, dtype, device and attention backend fixed.
- Total sequence: **nine latent frames** = clean `c0` plus eight generated frames, covering 65 pixel frames.
- Initial sigma: **1.0**. Full schedule: `[1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0]`.
- Clamp only `c0` at every step, with its token timestep zero. All eight generated latents evolve freely. Do not clamp the first two generated latents after initialization.
- No current guide information survives the sigma-1 mixture. Keep any required source tensor identical between runs regardless.

Indexing is important: latent index 0 is `c0`; generated frames 1–2 are latent indices `[1,3)`; generated frames 3–8 are indices `[3,9)`.

## Construct the intervention

Build and save two explicit epsilon tensors in the same token layout:

1. **A:** concatenate the first four saved block epsilons from the parent prefix study, covering nine latent frames including the unused/noise-overridden `c0` portion. Validate shape and dtype against the new grid.
2. **B:** clone A. Replace only latent indices `[3,9)` with a separately sampled Gaussian realization from a recorded seed, for example 43. Preserve indices `[0,3)` exactly.

Before running the model, assert:

- A and B have identical shape, dtype and finite values.
- Their `[0,3)` token spans are bit-exact.
- Their `[3,9)` spans differ.
- The actual assembled inputs contain the same clean `c0`, independent of its unused epsilon entries.

Draw/save this intervention once. For every run, slice these tensors by **global latent-frame index**. Do not regenerate noise according to the selected block geometry.

## Runs

| Run | Attention and grouping | Initial epsilon | Purpose |
|---|---|---|---|
| J-A | One bidirectional block: `[0,9)` | A | Joint reference |
| J-A-repeat | Identical to J-A | A | Numerical repeatability |
| J-B | Identical to J-A | B | Test influence of later noisy frames |
| C-A | Four causal blocks: `[0,3)`, `[3,5)`, `[5,7)`, `[7,9)` | A, sliced by span | Causal reference |
| C-B | Identical to C-A | B, sliced by span | Verify intervention is blocked across causal blocks |

For the joint runs, all nine frames must participate in every denoising call with unrestricted within-block attention and no history cache. A single block of eight new latent frames naturally provides this behavior. Do not substitute `history_mode=joint` with two-frame generation blocks: that mode sees clean past and the current block, not all eight noisy frames together.

For the causal runs, use generated-history refresh, the existing pinned `c0` policy, and context eight. Reset cache state between runs. Each block receives the same complete denoising schedule.

Use the existing rollout/conditioning primitives where possible. The experiment requires explicit injected epsilon and optional intermediate-output capture; the current CLI alone should not be assumed to expose both. Follow the package documentation and test requirements for any implementation change.

## Measurements

The primary result is computed **before VAE decoding** on the final clean predictions for latent indices `[1,3)`:

- Maximum absolute difference.
- Mean absolute difference.
- Relative L2 difference, with a small protected denominator.
- Number/fraction of changed elements and `torch.equal` result.

Measure these for:

1. **J-A versus J-A-repeat:** empirical repeatability floor.
2. **J-A versus J-B:** effect of the future-noise intervention under joint attention.
3. **C-A versus C-B:** corresponding causal negative control.

Also verify that `c0` is identical to the supplied condition in every output. Confirm that the changed future inputs reached J-B/C-B rather than being ignored or overwritten by the harness.

If practical, save the predicted clean latents for `[1,3)` after each denoising call in J-A/J-B. A difference on the first call shows immediate attention-mediated influence; later calls show how it propagates. Keep instrumentation identical across paired runs.

Do not use differences between J-A and C-A to answer the primary question: that comparison changes the computation and history representation. The controlled interventions are A versus B **within** each attention regime.

## Interpretation and stopping rules

| Result | Interpretation / next action |
|---|---|
| J-A/J-B differs well beyond repeatability; C-A/C-B agrees at its repeatability floor | Later noisy frames influence earlier outputs during joint denoising; the causal partition blocks that dependency |
| Both comparisons agree | No measurable influence for this sample/configuration. Verify masks, intervention delivery and measurement slices; if valid, repeat a few future-noise realizations before making a broader claim |
| C-A/C-B differs materially | Investigate unequal first-block inputs, cache/RNG leakage, conditioning differences, masks or backend repeatability before interpreting joint results |
| J-A/J-A-repeat is unstable | Establish a reliable numerical floor first; use a consistent supported backend if necessary |

When the repeatability floor is nonzero, report its absolute magnitude and the intervention effect together. An order-of-magnitude separation is a useful investigation threshold, not a claim of statistical significance. Add a C-A repeat if the causal control has a nonzero unexplained discrepancy. Do not equate any floating-point inequality with a meaningful effect.

One clear controlled counterexample is sufficient to reject the claim that future noisy frames *cannot* affect earlier outputs under joint denoising. Several negative trials cannot prove universal independence.

This experiment establishes computational dependence, not whether the dependence causes the appearance jumps or how large its visual effect is. Those are subsequent questions.

## Artifacts and execution

Suggested fresh output directory: `../expr/onestep_avatar/joint_future_noise_influence_20260926/`.

Save:

- A/B epsilon tensors, exact changed token/latent spans and generation seeds.
- All five raw output latents, optional intermediate predictions, and a metrics JSON.
- Manifest with input hashes, model/VAE paths and fingerprints, text-context hash, `c0` hash, schedule, block plans, actual masks, forcing/cache policy, dtype/backend and source-code revision or snapshot.
- A short report stating the observed dependency, numerical floor and limits.

No decoding is required for the primary gate. Optional visualizations must decode equal-length nine-frame latent tensors with matched decoder noise/settings; comparing separate short/long decode calls would reintroduce the known decoder-length confound.

Run in the `ltx` environment, inspect GPU availability before loading weights, and execute paired cases sequentially on the same device. This is a bounded inference diagnostic, not a training job.
