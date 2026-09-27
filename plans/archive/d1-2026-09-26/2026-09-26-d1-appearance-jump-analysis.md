> Archived and superseded. Historical findings and proposals are retained as written; use [the current next-actions plan](../../d1-next-actions.md) for active work.

# D1 appearance jumps across autoregressive blocks

Date: 2026-09-26. Scope: analysis of `../expr/onestep_avatar/d1_comparison`, current code inspection, saved-video measurements, and CPU diagnostics. No new GPU inference or training was launched; the fixes below are proposals.

**Correction following the user's clarification:** the intended one-step and multistep comparison uses the **same frozen base**, without a LoRA in either arm. “One step” describes the sampler, not a different set of weights. Adapter calibration/training is not a proposed cause of that comparison's jumps. Some existing files record an older, different setup; their provenance is retained below rather than treating them as measurements of the intended comparison.

## Assessment

**The leading explanation is inconsistent continuation conditioning across separately generated blocks. More denoising steps within each block do not repair that relationship.** The saved multistep experiment uses the frozen base under custom causal attention, refreshes history from ground truth rather than the displayed prediction, and never jointly refines adjacent output blocks. At high noise, the current guide supplies less information, so this continuation mechanism matters more.

There is also a concrete implementation concern worth testing before further training: **the clean-history cache is not generally equivalent to an explicit causal forward for this checkpoint's sigma-dependent text conditioning.** An expanded small-model CPU probe reproduces that difference. Its contribution to the real videos is unmeasured; it is not yet a demonstrated sole cause of flicker.

My recommended order is: verify/fix the intended cache-conditioning semantics, compare generated-history and GT-history rollouts with identical weights, test an explicit clean-prefix continuation, and only then adapt/distill a model under the successful continuation configuration. Increasing sigma or training longer on the existing teacher-forced MSE objective is not a sufficient fix by itself.

**The reported jump at sigma 1 strengthens the case for a continuation problem.** It eliminates residual current-guide content as a necessary cause, and the completed frozen-base eight-step run removes the one-step shortcut as a sufficient explanation. It does not alone distinguish a cache bug from a valid computation whose conditioning/model is unsuitable for continuation. Section 4 gives the exact invariants that can make that distinction.

## 1. What the saved experiment actually runs

Evidence: the [comparison report](../../../../expr/onestep_avatar/d1_comparison/REPORT.md), [multistep launcher](../../../../expr/onestep_avatar/d1_comparison/run_official.sh), [example manifest](../../../../expr/onestep_avatar/d1_comparison/videos/official/gpu0/manifest.json), and [adapter config](../../../../expr/onestep_avatar/runs/causal-one-step-study-20260921/E2/lr_1e-4/config.json).

| Property | Saved setting / implication |
|---|---|
| Views | Nine views from three clips: `0008_01`, `0012_09`, `0025_11` |
| Model | Logs identify LTX-2.5; current registry resolves the 22B distilled transformer |
| Objective / panels | White background; decoded capture, capture-sourced D0, render-sourced D1 |
| High-noise multistep schedule | `0.909375 → 0.725 → 0.421875 → 0`: **three denoising calls per block**, plus one cache refresh |
| Multistep weights | Frozen base, `checkpoint: null` |
| Intended one-step weights | Same frozen base as the multistep arm; no adapter |
| Older saved one-step artifacts | The inspected launchers/manifests still record a step-50 D1 LoRA; excluded from same-base sampler conclusions |
| History | Both runs set `teacher_forcing: true`: refresh from clean capture `z_y` |
| Geometry | Manifest blocks `[0,3), [3,5), …, [15,17)`; current/launcher-default context is eight latent frames plus the pinned first frame |
| Noise | Seed 42; matched per-block epsilon between D0 and D1 |
| Video | 129 decoded frames; the inspected source bundle records 30 fps |

“Official” in the filenames means the distilled **schedule tail**, not the stock full-video inference computation. `visualize_d1.main` calls `visualize_d0._run_chain`, which calls `causal_core.rollout` for both schedules. The frozen base is still run as a custom autoregressive model.

The **intended** comparison changes only the schedule. However, `run.sh`, `high_noise/run.sh`, and the inspected one-step manifests still specify `lora_weights_step_00050.safetensors`. The corresponding multistep manifests specify `checkpoint: null`. Those particular saved one-step files cannot establish the same-base schedule comparison. Future matched runs must omit `--checkpoint` in both arms, record identical resolved base-model identities, and use fresh output directories. No existing files should be relabeled as base-only without rerunning or stronger provenance evidence.

The September 26 `high_noise/` sweep has now completed and has a [report](../../../../expr/onestep_avatar/d1_comparison/high_noise/REPORT.md). The [sigma-1 multistep manifest](../../../../expr/onestep_avatar/d1_comparison/high_noise/videos/gpu6/official/manifest.json) records the frozen base, GT refresh, three views, and the full eight-step schedule `1 → 0.99375 → 0.9875 → 0.98125 → 0.975 → 0.909375 → 0.725 → 0.421875 → 0`. This supersedes the earlier observation that jobs were incomplete. Its reported D0 and D1 reconstruction means agree to the displayed precision: 19.564 dB PSNR and 0.9017 SSIM. Similar aggregate scores are compatible with source invariance but do not prove per-frame or latent equality.

### These are latent block boundaries

The current rollout writes nonoverlapping latent blocks into one output tensor (`causal_core.py`, `rollout`, approximately lines 829–860). `visualize_d1.py` decodes the assembled latent once per arm; `scripts/prune/evaluate/decode.py::decode_latent` passes it to `decode_video(..., None, generator)`.

Thus there is no per-generation-window RGB decode and stitch in this path. The nominal new-block starts are zero-based pixel frames **17, 33, 49, 65, 81, 97, 113**, one transition every 16 frames after the first 17-frame block. Decoder temporal context can spread a latent discontinuity around these indices.

Global RoPE positions, the pinned clean `c0`, and the explicit GT refresh target are present in current code. Older documentation contains stale statements about absent `c0` conditioning or D1 refreshing from the render; those statements do not describe this implementation.

## 2. Evidence from the saved videos

I inspected a [contact sheet](../../assets/2026-09-26-d1-boundaries/official_sigma0909375_contact.png) of `0008_01/view00_cam51` at frames 14/16/17/19, 30/32/33/35, and 46/48/49/51. Rows are GT, D0, and D1. Sleeve shape, clothing coverage, and texture change around the generated-block transitions; the GT row provides a smoother motion reference. The sheet supports the reported symptom, but is not an identity-recognition measurement.

I also measured all nine views at sigma 0.909375 for both saved schedules. Each panel was downsampled to 256². A foreground mask was derived from the union of adjacent decoded-GT frames (`min RGB < 0.93`) and dilated by seven pixels. The boundary region comprises transitions within ±2 frames of each nominal boundary; the interior excludes them and transitions before frame 5. Each video contributes 35 boundary and 89 interior transitions; table entries average the per-view means.

Define temporal residual error as

`E(t) = mean_foreground |(prediction[t] − prediction[t−1]) − (GT[t] − GT[t−1])|`.

All RGB values are normalized to `[0,1]`.

| Saved run | Arm | Boundary E | Interior E | Boundary / interior |
|---|---|---:|---:|---:|
| Three-step frozen base | D0 | 0.07102 | 0.06278 | 1.131 |
| Three-step frozen base | D1 | 0.07687 | 0.06752 | 1.138 |

This supports investigating boundary-localized error in **both** multistep arms. The raw CSV also preserves measurements of the older LoRA one-step files; those rows are excluded here because they do not measure the requested same-base comparison. These measurements do not prove appearance flicker independently of pose: there is no optical-flow compensation, and GT itself has more motion near these boundaries (raw temporal MAE 0.05700 versus 0.05088). MP4 compression, the foreground threshold, and repeated views of only three clips further limit the result. A blurred or nearly static output can score well on raw frame differences.

The existing full-frame reconstruction report gives multistep D1 PSNR/SSIM of **22.451 dB / 0.9381**, versus **24.375 dB / 0.9455** for D0 at 0.909375. These are reconstruction metrics, not seam metrics; the white background also dilutes subject errors.

Artifacts: [per-video measurements](../../assets/2026-09-26-d1-boundaries/boundary_metrics.csv), [aggregates](../../assets/2026-09-26-d1-boundaries/summary.json), and [reproduction script](../../assets/2026-09-26-d1-boundaries/analyze_saved_videos.py).

## 3. Ranked causes and discriminating tests

### A. GT cache history disagrees with displayed history — directly established mechanism

For each block, the code does:

```text
displayed block = denoise(current noised guide, cached history)
cached next history = features(clean GT block, timestep 0)
```

If the generated block changes a sleeve, face detail, or clothing pattern, the next block is conditioned on the **GT version** of that feature. The displayed prediction is not fed forward. This can cause apparent resets even though each block is reasonably plausible given its conditioning.

Teacher forcing controls error accumulation against clean history; it does not establish continuity of the displayed generated sequence. Self forcing will align the history source with the displayed output, but can introduce accumulated drift. This issue exists with the frozen base and does not require an adapter. Treat it as a paired diagnostic first, then train under generated history if it helps.

**Test:** same frozen base, sigma tail, epsilon, `c0`, context depth and views; change only `--teacher-forcing`. Run this at sigma 1 as well as 0.909375. A reduction in jumps implicates this source mismatch; no improvement points toward conditioning capacity or implementation as well.

### B. Frozen bidirectional model under short causal blocks — leading modeling hypothesis

Within a block, tokens attend bidirectionally. Across blocks, past hidden states were computed earlier and cannot respond to the current noisy block. The current block is only two latent frames (16 pixel frames); earlier generated content cannot be jointly revised during its denoising.

Adding the 0.725 and 0.421875 calls improves the current block's estimate under the **same fixed cache**. It does not change its available information or make the model learn continuity across that boundary. An on-grid schedule is necessary for this distilled model but does not establish calibration for the new attention/history regime.

**Test:** a memory-feasible short span with full bidirectional denoising, then the same span with causal blocks and matched inputs. Also compare two versus four generated latent frames. If larger blocks only move the jumps to the new boundaries, they mitigate frequency without solving continuation.

The general architectural gap is discussed in [Causal Forcing](https://arxiv.org/abs/2602.02214). Its analysis motivates checking teacher/student information access; it does not diagnose this LTX run by itself.

### C. Sigma-dependent text conditioning breaks a claimed cache equivalence — reproduced on a small model

This is an additional finding beyond the earlier [September 20 plan](2026-09-20-block-flicker-sigma-and-causal-teacher-experiments.md):

1. `causal_core.refresh_block` assembles history with `sigma=0.0`, which also sets clean per-token timesteps.
2. `transformer_args.py::prepare` separately feeds **global `modality.sigma`** to `prompt_adaln`, when enabled (approximately lines 279–287).
3. `transformer.py::apply_cross_attention_adaln` uses that result to modulate text-context K/V. Text attention changes video hidden states, and therefore later layers' **video self-attention K/V**.
4. Cached history was computed with prompt sigma zero. An explicit full-prefix causal forward at current sigma computes the same clean history with a different prompt modulation. Equal history-token timesteps alone do not make these computations equivalent.

The current resolved LTX-2.5 checkpoint has `cross_attention_adaln=True`; `use_prompt_adaln_single` is absent from its config and resolves to **True**, as confirmed by `resolve('2.5').caps`. The original tiny parity fixture leaves cross-attention AdaLN disabled.

The existing causal-core suite passes: **25 tests**. I then reused its explicit-causal parity check with a two-layer model, cross-attention AdaLN enabled, and deterministic random weights scaled by 0.2 so the effect was measurable:

| Prompt AdaLN | Current sigma | Block-1 maximum absolute output difference | Relative L2 | Existing `2e-4` tolerance |
|---|---:|---:|---:|---|
| Disabled | 0.725 | 1.19e-7 | 3.05e-8 | Pass |
| Enabled | 0.725 | 0.004527 | 0.001113 | Fail |
| Disabled | 1.0 | 2.38e-7 | 3.73e-8 | Pass |
| Enabled | 1.0 | 0.018322 | 0.003920 | Fail |

Block 0 agrees exactly in both configurations. With the fixture's original smaller weight scale 0.05, both configurations passed the loose tolerance; that is why the diagnostic reports its changed initialization explicitly. These are synthetic-model values, **not an estimate of the 22B model's error**. See [CPU reproduction](../../assets/2026-09-26-d1-boundaries/probe_prompt_cache.py).

This demonstrates a limitation of the claimed cached-versus-explicit equivalence, not necessarily a violation of an intentionally defined streaming model: a model can be trained to use zero-sigma history features. The current frozen-base comparison has not established that this choice preserves quality.

**Test/fix direction:** compare the real checkpoint before eviction against an explicit causal prefix at every active sigma. For a reference that must match the full-prefix computation, rebuild prefix features under that step's global prompt sigma **while retaining history-token timesteps at zero**. Simply setting history tokens to the active noisy timestep would change the condition incorrectly. Recomputing only the newest block is insufficient if older cached states retain a different prompt modulation. A faster design needs explicitly defined, trained cache-compatible conditioning semantics; do not blindly disable a trained AdaLN branch.

### D. High sigma weakens the sole current-guide input — directly established, contribution unmeasured

For D1, `x_sigma = (1−sigma) z_g + sigma epsilon`. At 0.909375, the render coefficient is **0.090625**, versus **0.275** at sigma 0.725. These are mixing coefficients, not percentages of retained information. Since jumps also occur at sigma 1, disagreement with residual current-guide content cannot be the sole explanation.

No separate clean guide is supplied at each subsequent denoising call: the evolving state is all that carries this current-block guide. At sigma 1 the guide disappears completely. With teacher forcing, matched epsilon, and shared capture `c0`, D0 and D1 then have identical rollout inputs; they should agree up to backend nondeterminism. That is a useful implementation control but no longer tests render-driven pose control.

Independent per-block noise is not intrinsically a bug—ordinary joint video generation also starts from random noise. Here it gives each weakly coupled block freedom to choose different details. Reusing an identical noise pattern in every block can produce repeated textures or motion artifacts and is not a principled continuity fix.

**Fix direction:** resolve the continuation behavior of the frozen base first. If high sigma remains necessary for the eventual guide-driven product, propose a persistent pose/render conditioning path and train it. Such a branch is new architecture; it is not an existing `guide_mode` option.

### E. Adaptation is a possible later remedy, not the present cause

For the intended same-frozen-base experiment, adapter training length, sigma calibration and loss are not explanatory variables. The earlier discussion of the step-50 LoRA concerned legacy artifacts and should not be used to diagnose this comparison. Fine-tuning is a later remedy only after the cache/conditioning implementation has been checked.

**Fix direction:** establish a coherent causal multistep reference, then train the adapter with generated history at the intended sigma/schedule and geometry. Include clip-start sequences and longer histories so training sees deployment conditions. If short chains are needed for memory, prime with a generated prefix when feasible and disclose any remaining GT priming. [Self Forcing](https://arxiv.org/abs/2506.08009) provides the relevant rationale for training on generated context.

### Lower-priority explanations

- **Cache eviction:** with context eight plus the sink, boundaries 17/33/49/65 precede eviction's first effect on a later denoise call, at frame 81. Visible early changes cannot all come from eviction. Examine later boundaries separately.
- **Per-window position reset / missing first frame:** current `ClipGrid` uses global positions; `rollout` clamps `c0` and sets its token timestep to zero. No evidence of these historical failures in this path.
- **Source or VAE seams:** the inspected capture/guide bundles agree on shape `[128,18,32,32]`, crop, fps, schema 2 and encode-contract 1. The output uses a continuous decode with matched decoder RNG. This reduces concern about obvious assembly mismatches, but does not prove source continuity or rule out DiffVAE amplification. Inspect raw predicted latents and decoded guide controls before excluding it completely. Logs record a Triton neighborhood-attention fallback, not a demonstrated decode defect.

## 4. Why sigma 1 can still jump, and what would establish a bug

### What this experiment rules out

At sigma 1 the initial generated tokens satisfy `x_1 = epsilon`. The current-block capture/render source is absent. However, clean `c0`, text, GT or generated cached history, positions, and independent noise for each block remain. During a multistep rollout the state evolves from epsilon; the code does not reinsert the render as sigma decreases.

Therefore, persistent jumps with the same frozen base and a full eight-step trajectory mean:

- A residual render-versus-capture conflict is **not required** to create the jumps.
- Adapter misuse is **not required** to create them.
- Taking additional denoising steps does not remove the cause under the current continuation computation.

They do **not** imply that all conditioning has been removed or that continuity should emerge automatically. For GT refresh, block `i+1` is conditioned on `GT_i` rather than the displayed `prediction_i`. For generated refresh, the right pixels/latents are available, but a model still has to use their cached representation well. At sigma 1 the current guide cannot compensate for a weak history mechanism. Every block has a new random realization, not a new identity constraint.

If the reported jump is from a direct `[1,0]` call, an excessively coarse sampler remains a plausible contributor. The saved eight-step frozen-base run is the stronger diagnostic because it removes that shortcut while preserving the same causal setup. Its mismatch with capture pose at sigma 1 is not itself a bug: current guide-driven motion control has been removed, although GT history still supplies past motion.

### A strict source-invariance test

With identical weights, schedule, geometry, text, epsilon, `c0`, and history policy, replacing `z_g` with any other **finite** tensor of the same shape at sigma 1 must not change the output beyond numerical repeatability. For teacher forcing the target history must also stay identical; for self forcing the histories should stay identical inductively from block 0 onward.

I ran this through the existing `causal_core.rollout` using the small CPU model, two substantially different source tensors, and both one-step and eight-step schedules. **All four combinations of schedule and forcing policy produced exactly identical D0/D1 output latents (maximum difference 0), and preserved `c0` exactly.** [Script](../../assets/2026-09-26-d1-boundaries/probe_sigma1_invariance.py) · [results](../../assets/2026-09-26-d1-boundaries/sigma1_invariance.json).

This checks source removal and shared rollout plumbing on the small model; it does not test the 22B model's continuity. A material D0/D1 difference in a matched real sigma-1 run would reveal hidden source dependence, unequal noise/history/model settings, or a cache reset/leakage problem. Compare raw latents before decoding: separately compressed MP4 panels are not an exact equality test.

### Cache-equivalence is the stronger current bug lead

The expanded parity probe in section 3C also fails at sigma 1 when prompt AdaLN is enabled, while its disabled control agrees to float32 roundoff. Thus setting sigma to 1 does **not** remove this cache-conditioning discrepancy. There are two conditioning values here: clean history-token timestep 0 and the global prompt-conditioning sigma. The cache was computed with both at 0; the explicit current-step reference uses history-token timestep 0 but global sigma 1.

If the intended cache is an optimization of that explicit causal reference, the non-equivalence is a correctness defect relative to that contract. If zero-sigma history features are deliberately the definition of a different streaming model, the question becomes whether the frozen checkpoint is suitable for it. We have reproduced the discrepancy on a synthetic model, not proved its size or visual effect on the production checkpoint. Do not infer from these random weights that real error must grow monotonically with sigma.

### Order the bug checks to locate the failure

| Check | Expected result | Meaning of a failure |
|---|---|---|
| Replace current guide at sigma 1, keep all other inputs fixed | Same raw output latents | Hidden guide dependence or mismatched experiment state |
| Run the same arm twice from a reset cache and fixed RNG | Agreement at measured numerical floor | RNG/state leakage or backend nondeterminism requiring isolation |
| Cached versus explicit **same causal** reference, before eviction | Agreement under matched global and token conditioning | Cache/conditioning implementation discrepancy |
| Keep current noise fixed; substitute or ablate history | Detectable, sensible continuation dependence | If unchanged, investigate ignored cache, masking or ineffective history use; sensitivity alone does not prove quality |
| Joint short-span reference versus correctly computed causal continuation | Measure remaining quality gap | If only causal continuation jumps, architectural/model adaptation is implicated |

Before training, prioritize the real-checkpoint parity test at block 1 and sigma 1, with clean `c0` handled identically in both paths. Compare outputs and layerwise K/V; determine whether divergence begins after text cross-attention, as the current code predicts. Follow with GT-versus-generated history and a recomputed clean prefix. These tests distinguish an implementation defect from a model that executes the requested computation correctly but is poor at continuation.

## 5. Concrete fix and validation sequence

### Phase 1: make the comparison diagnostic

Extend the existing probe, without duplicating `causal_core.rollout`, to save raw predicted latents and full resolved provenance: base/VAE paths and fingerprints, block/context/sink geometry, history source, actual schedule, text-context identity, noise tensors or reproducible provenance, dtype/backend, and prompt-AdaLN capability flags. Require `checkpoint: null` for both arms of this comparison. Current D1 manifests omit several of these details. The present CLI already supports frozen-base one-step inference by omitting both `--checkpoint` and `--trajectory-only`; adding `--trajectory-only` selects the multistep arm with the same base.

Start on `0008_01/view00_cam51`, seed 42; then validate on additional actors and seeds. Run the following paired controls before an expensive training sweep:

| Control | Hold fixed / change | Decision it supports |
|---|---|---|
| Base one step versus three steps | Same base, sigma 0.909375, history and inputs | Isolate schedule benefit |
| Base one step versus eight steps | Same base, sigma 1, history and inputs | Separate one-step approximation from continuation failure |
| GT versus generated refresh | Same base and three-step tail | Quantify displayed-history mismatch |
| Cached versus explicit causal prefix | Same clean prefix, positions, per-token and global sigma semantics | Measure implementation/conditioning divergence |
| Causal versus joint short span | Same weights, current inputs, schedule, `c0` | Measure cost of causal restriction |
| Two versus four latent frames/block | Same base and guide/noise indexed by global frame | Distinguish fewer boundaries from better boundaries |

A geometry sweep must not regenerate epsilon using `seed + block_index` with differently shaped blocks: that changes the noise too. Save a global frame-indexed epsilon tensor and slice it consistently for that particular comparison.

### Phase 2: fix continuation before distillation

1. **Resolve cache semantics first.** Expand permanent parity tests to the actual checkpoint capabilities, including prompt AdaLN, clean `c0`, multiple sigma levels, and retention policy. Validate a short real-checkpoint span against the correct explicit reference. If matching a full-prefix model is required, implement the per-step prefix recomputation reference described above and measure its cost/quality.
2. **Evaluate generated history.** Omit `--teacher-forcing` in the matched probe. If it helps, adapt training to generated-history continuation rather than silently deploying a teacher-forced adapter under a different distribution.
3. **Test a recomputed clean prefix.** Supply recent generated latent frames as explicit timestep-zero tokens alongside the current noisy block, preserve global positions, clamp their values at every step, and emit only new frames. Remove duplicate copies from the cache. First use the same causal mask to isolate representation/parity effects; then separately test allowing bidirectional attention within this conditioning window. The latter changes the architecture and must be trained/evaluated as such. It is proposed work, not a current CLI flag.
4. **Use larger blocks only if latency permits.** They can improve within-block coherence but must be evaluated at the remaining boundaries. Pixel crossfading may hide a color jump while creating ghosted anatomy; it does not correct latent identity changes.

### Phase 3: adapt the successful architecture

Train at the deployment history policy, geometry, and noise schedule. A fixed-sigma one-step LoRA is not automatically a calibrated multistep denoiser. For eventual one-step distillation, demonstrate that the causal multistep teacher already produces coherent boundaries; otherwise the student inherits its defect.

A joint long-span teacher can be a quality reference, but paired endpoint targets may depend on future guide/noise unavailable to an online student. Before using such targets for regression, fix the student's available inputs, vary the teacher-only future, and measure changes in the teacher's early outputs. Prefer a teacher with matching information access when that dependence is substantial.

If continuity remains weak after these changes, propose temporal supervision across generated boundaries using motion/occlusion-aware feature consistency and appearance/pose evaluation. This is a **new training-objective proposal**: the current contract specifies unweighted full-frame latent MSE. Do not silently substitute a masked or temporal loss. Avoid plain adjacent-frame equality losses, which reward freezing or blur.

## 6. Acceptance criteria and limits

Report boundary and interior behavior separately, split boundaries before/after eviction, and preserve per-actor/per-seed results. Use raw latents and lossless decoded frames for final measurements. Add motion-compensated foreground residuals, face/clothing appearance features where reliable, and pose/motion checks; review normal-speed playback and boundary contact sheets together.

A successful fix must reduce boundary appearance changes **without** achieving that through blur, frozen motion, loss of guide control, or steadily drifting identity. It must work with generated history, beyond the three-block training horizon, and on held-out actors. Latency and memory increases from prefix recomputation or larger blocks must be reported alongside quality.

The current evidence establishes the rollout conditions, visible early-boundary changes, a coarse boundary-error concentration, and a synthetic cache-equivalence counterexample. It does **not** establish the relative causal contributions on the real checkpoint or verify a production fix.

## Reproduction and source pointers

Commands run from the LTX-2 repository in the `ltx` environment:

```bash
conda run -n ltx python -m pytest scripts/onestep_avatar/tests/test_causal_core.py -q
conda run -n ltx python plans/assets/2026-09-26-d1-boundaries/analyze_saved_videos.py
PYTHONPATH=. conda run -n ltx python plans/assets/2026-09-26-d1-boundaries/probe_prompt_cache.py
PYTHONPATH=. conda run -n ltx python plans/assets/2026-09-26-d1-boundaries/probe_sigma1_invariance.py
```

The prompt-cache script is an investigative reproduction: it catches and prints the expected enabled-prompt parity failure; a zero shell exit code does not mean all configurations passed. Its monkeypatches are confined to that diagnostic process. The sigma-1 source-invariance script asserts equality and fails if that invariant is broken.

Current checkout HEAD: `fdf80f59ecaa642226ca99655f9316828cedbc45`, with pre-existing uncommitted D1 probe files. Historical manifests do not pin a source revision; conclusions about implementation are from the current files and available logs, not a reconstructed historical checkout.

- [D1 probe](../../../scripts/onestep_avatar/visualize_d1.py): schedule selection, weight loading, manifests, decoding.
- [Shared probe](../../../scripts/onestep_avatar/visualize_d0.py): `_run_chain`, `_decode`, paired epsilon construction.
- [Causal core](../../../scripts/onestep_avatar/causal_core.py): `ClipGrid`, `mix_block_noise`, `block_modality`, `refresh_block`, `rollout`.
- [Transformer argument preparation](../../../packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py): global sigma versus token timesteps.
- [Transformer blocks](../../../packages/ltx-core/src/ltx_core/model/transformer/transformer.py): `apply_cross_attention_adaln`.
- [Model configurator](../../../packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py): prompt-AdaLN defaults.
- [Training](../../../scripts/onestep_avatar/train.py): per-block MSE and refresh source.

## Implementation follow-up (2026-09-26)

The diagnostic probe now saves raw D0/D1 latents, global frame-indexed epsilon, resolved
weight/text fingerprints, schedule, geometry, history policy, attention backend, and rollout
timing/memory. It refuses a nonempty output directory. `--max-blocks` and `--raw-only` make
short real-checkpoint checks practical. `causal_core.rollout` offers inference-only
`history_mode=recompute` (clean retained prefix, block-causal mask, current global sigma)
and `history_mode=joint` (the same tokens with bidirectional window attention); training and
deployment keep the existing cached mode. Permanent CPU tests cover prompt AdaLN, clean
`c0`, eviction, geometry-invariant noise and joint-window information access.

The [diagnostic report](../../../../expr/onestep_avatar/d1_diagnostic/REPORT.md) records one view
each from three actors, matched GT/generated-history runs, sigma-1 source invariance, real
checkpoint layerwise K/V comparisons, and raw latent boundary/interior metrics. Cached and
explicit block-1 outputs differ substantially before eviction, but recomputed causal prefix
has only a small boundary-metric effect on the full tested clip at roughly 3× rollout time.
Joint-window attention improves GT-history metrics; its generated-history boundary result is
mixed. No training or deployment change was made because the successful continuation
configuration required by Phase 3 has not been established. Visual, motion-compensated and
pose/appearance review, additional seeds, and model adaptation remain open.
