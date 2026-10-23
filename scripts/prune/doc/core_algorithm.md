# Pruning calculations

Pruning changes selected transformer units while keeping the saved denoising task fixed.
This page defines the shared input, scoring and learned-selection calculations.
The [glossary](glossary.md) defines terms; the [overview](overview.md) lists the branches each method can remove.

## Terms and shapes

Capture latents have shape `[1,C,F,H,W]`: batch, channels, latent frames, latent height and latent width.
With patch size one, tokens have shape `[1,F*H*W,C]`.
Token `(f,h,w)` has index `(f*H+h)*W+w`; one latent frame occupies `H*W` consecutive tokens.

Use registered VAE factors from `core.geometry`.
The aligned pixel **frame count** is `temporal_factor*(F-1)+1`; the supported temporal factor is eight.
Spatial height and width use their own registered factors.
Preserve the recorded fps and source coordinates.

## Input reconstruction

`data.whole_clip._reconstruct_input` owns the calculation for both public input-loading paths.
The [input contract](data/whole_clip.md) explains validation and candidate-family dispatch.

1. Validate the saved D0 manifest and requested view/sigma row.
2. Verify capture, VAE, saved epsilon and actual text-context bytes.
3. Rebuild the grid from capture shape, recorded fps and registered VAE factors.
4. Mix in FP32: `x_sigma=(1-sigma)*capture+sigma*epsilon`, then cast to BF16.
5. Replace the first `H*W` tokens with the clean capture's first latent frame, c0.
6. Set those tokens' timesteps to zero; all remaining token timesteps equal sigma.
7. Keep the global sigma and token timesteps in FP32.
8. Use `attention_mask=None`, one full bidirectional block and no cache.
9. Predict clean latents with the session's X0 wrapper.
10. Restore exact c0 in the predicted output.

The first latent stays clean because it is the image condition, not a generated frame.
Using a dense all-ones mask can allocate quadratic attention bias; `None` expresses full visibility directly.
BF16 weights and latents do not require rounding the schedule.

Worked input: let `F=3`, `H=W=2` and `C=128`. There are 12 tokens; tokens `[0,4)` contain c0.
A later scalar with capture 2, epsilon 6 and sigma 0.5 becomes 4.
A first-frame scalar with the same values stays 2 and has timestep zero. Global sigma stays 0.5.

## Training-free scores and masks

`score.whole_clip_d0_scores.score` observes activations immediately before head and FFN output projections.
For head h, let `a_h` be its activation vector and `W_h` its columns in the output projection.
The default score is:

`score_h = sqrt(mean_cases(mean_sampled_tokens_and_head_channels(a_h**2))) * norm_F(W_h)`.

Each calibration case has equal weight, even when sample counts differ.
For an FFN channel, use its scalar post-activation RMS and the norm of its down-projection column.
The scorer checks a fresh unmasked forward against the saved baseline before accepting each case.

`score.token_sampling.sample_indices` excludes the clean frame.
`stride` samples flat spatial offsets; `balanced_grid` spreads a comparable sample count over rows and columns.
For a 32×32 latent frame and stride 16, flat offsets visit columns 0 and 16, giving 64 sampled tokens per generated frame.
The balanced sampler uses an 8×8 midpoint grid instead.

`score.estimators.fractional_masks` removes `round(unit_count*fraction)` lowest scores per branch, retaining at least one unit.
Stable ties remove lower original IDs first.
For scores `[1,1,3,4]` and fraction 0.5, the mask is `[0,0,1,1]`.
Head masks zero complete head vectors before `to_out[0]`; FFN masks zero post-activation channels before `ff.net[2]`.
The exact branch coverage is defined in the [overview](overview.md#parts-that-can-change).

The RMS proxy ignores cancellation between channels, interactions between heads and downstream behavior.
It observes selected query tokens rather than the full spatial grid.
`score.estimators.projected_energy` instead returns `mean_tokens(norm(W_h @ a_h)**2)` for each head.
It uses FP32 head-local Gram matrices and FP64 accumulation; it returns mean squared energy, without a square root.
`learn.heads.energy_phase` uses this helper for the fixed-energy comparison arm.
Both measures still require held-out interventions to assess model quality.

## Learned velocity and binary selection

Learning uses `core.ltx_adapter.velocity_forward` to obtain native velocity and its stock timestep-based clean conversion.
The target is `epsilon.float()-capture.float()`.
Compute FP32 mean squared velocity error over generated tokens and channels; exclude c0 from the loss.
Do not reconstruct velocity from an already rounded clean prediction, because that changes the training target calculation.

Head learning converts FP32 logits to sigmoid amplitudes during soft selection.
The configured sparsity penalty and coefficient ramp encourage removal.
Binary refinement uses `decision + (sigmoid(alpha)-stopgrad(sigmoid(alpha)))`.
The forward is exactly binary while backward uses the sigmoid derivative.

`learn.gates.select_legal` solves an exact block/count dynamic program for the configured total retained count and allowed block widths.
Stable original IDs break ties. One decision serves every draw and checkpoint recomputation within an update.
The [head phase contract](learn/heads.md) describes phase inputs and outputs; [preset values](usage.md#preset-values) have one home in the run guide.

The FFN-mask variant ranks individual channel scores per block and keeps an aligned configured count.
Its count ramps from dense to the requested width; its binary surrogate has a fresh graph for every forward.
The fixed count supplies the FFN budget, so this variant adds no FFN sparsity penalty.
After selection, retained original IDs stay fixed during adapter recovery.

## Joint clipped selection with return

V3 uses one FP32 score per video self-attention head and per FFN channel.
`learn.readmission` owns selection state; `learn.joint_training` owns gradients and optimizers.
The default geometries are 48×32 heads and 48×16,384 channels.

For each non-permanent unit, forward amplitude is `clamp(alpha,0,1)`.
The straight-through expression differentiates `clamp(alpha,-3,3)` instead, including PyTorch's endpoint derivatives.
This lets a zero-amplitude removed unit receive gradients and return.
Permanently removed units multiply by zero and have zero score gradient.

The sparsity term is:

`lambda(t)/head_count * (sum_heads(relu(alpha+3)) + rho*sum_channels(relu(alpha+3)))`.

Only enabled, non-permanent units contribute.
`learn.joint_training.JointStep.forward` derives lambda from `lambda_max` with its 100-update ramp;
`rho` balances channel and head costs. The frozen study sets `rho=1/704`.
The head-only arm `heads_clip_reset_v3` disables FFN selection and FFN adapters.

### Event defaults

These are `learn.readmission.JointPolicy` defaults and the standard joint preparer's policy.
The policy's schedule is frozen under `config.policy`; family thresholds and caps are code constants.

| Quantity | Source key or owner | Default |
|---|---|---:|
| Event interval | `policy.interval` | 50 updates |
| Late-window length | `policy.late` | 10 updates |
| Earliest quiet stop | `policy.earliest` | 200 |
| Hard selection limit | `policy.limit` | 1,500 |
| Fixed-mask recovery | `policy.recovery` | 200 updates |
| FFN alignment | `policy.alignment` | 128 channels |
| Strong drop/rise, heads | `Family.config.strong` | 0.25 |
| Persistent drop, heads | `Family.config.persistent` | 0.15 |
| Strong drop/rise, FFN | `Family.config.strong` | 0.35 |
| Persistent drop, FFN | `Family.config.persistent` | 0.25 |
| Maximum give-back | `Family.candidates` | 0.1 |
| Event removal cap and separate return cap | `Family.event` | `floor(0.04*pre_event_kept)` |
| Per-block minimum | `Family.config.minimum` | 1 head; alignment channels |
| Per-block FFN removal cap | `Family.config.block_cap` | `floor(0.25*pre_event_block_kept)` |

### One selection event

Each unit keeps only its window minimum, maximum, final score and score at the late-window start.
This bounds memory by the number of units instead of storing every score trajectory.
All decisions use the pre-event state so a unit cannot leave and return in the same event.

1. For a kept unit compute `drop=1-final`, `down_giveback=(final-min)/(1-min)` and `late_change=final-late_start`.
   Both give-back denominators use an FP64 floor of `1e-30`, preventing division by zero when there was no movement.
2. A removal candidate must have give-back at most the limit and negative late change.
   Its drop must reach the strong threshold, or the persistent threshold in two consecutive events without an earlier return.
3. For a removed, non-permanent unit, compute `rise=final` and `up_giveback=(max-final)/max`.
   It can return if rise reaches the strong threshold, give-back is small enough and late change is positive.
4. Rank removals by decreasing drop, then increasing give-back, then original ID.
   Apply per-block limits before the family-wide cap. Rank returns by the mirrored rule with their own cap.
5. Remove selected units and restore selected returns. A unit removed after any earlier return becomes permanent.
6. Reset scores to one for kept units and zero for removed units; clear **gate** Adam state and score gradients.
   Adapter parameters and their Adam state continue across events.

Resetting scores makes the next window measure movement from the new binary decision.
Clearing gate moments prevents momentum from the previous window from driving that new measurement.
A removed but non-permanent score still learns between events; its continuous forward amplitude can rise before the binary return.

Worked event: a kept head ends at 0.70, with minimum 0.68 and late-start score 0.74.
Its drop is 0.30, give-back is `0.02/0.32=0.0625` and late change is −0.04, so it is a strong removal candidate.
With 100 pre-event kept heads, at most four heads are removed and at most four returned, subject to block limits.
The candidate can still lose its place to a higher-ranked head.
If it later ends a removed window at 0.30, maximum 0.32 and late-start 0.26, it qualifies to return.
If removed again after that return, it becomes permanent for the remaining selection stage.

### Selection end and recovery

Two consecutive quiet events can end selection at or after `policy.earliest`, while removal events remain enabled.
Quiet means no score-eligible removal or return candidates, rather than merely no applied changes after caps.
At the hard limit selection always stops. The final two scheduled events allow returns only.
With default timing those events occur at updates 1,450 and 1,500.

At selection end, align each block's FFN count upward to `policy.alignment` by restoring removed channels.
Restoration prefers non-permanent units, larger last rise, smaller removal drop and lower original ID, in that order.
This final alignment can restore a permanent unit; permanence governs selection events, not the alignment pass.
Keep heads unchanged, reset gate state again, and train adapters with the resulting fixed masks for `policy.recovery` updates.
The final checkpoint is `selection_end + recovery`. The [state contract](learn/readmission.md) covers exact resume and review pauses.

### Matched-budget control

`budget_joint_v3` uses binary projection instead of removal/return events.
`learn.readmission.budget_projection` reserves one highest-score head and the minimum channels in every block, charging those units to the budget.
It then takes a globally descending score prefix; it does not divide each score by cost.

One head costs `cost=704` FFN-channel units. Selection stops at the first unaffordable unit, leaving any slack unused.
The configured budget ramps to a pinned joint reference budget and stops at that reference's selection endpoint.
It then uses terminal alignment and fixed-mask adapter recovery like the other joint arms.

## Export modes and runtime

| Mode | Storage | Execution |
|---|---|---|
| `masked_full` | Full source tensors | All heads/channels computed; inactive activations zeroed before output projections |
| `sparse` | Full source tensors | Full projections; attention selects active heads and scatters back; FFN remains full |
| `compact` | Retained V/output, gate and FFN slices; full Q/K | Retained execution widths; native head-gate projection keeps original output geometry |
| `compact_faithful` | Compact slices | Reconstruct original GEMM and attention widths from zero-padded weights |

For retained heads 0 and 2 at head dimension 4, keep channels 0–3 and 8–11 in V and output weights.
Full Q/K normalization still sees all original heads; original head positions select rotary frequencies.
The [export/runtime contract](score/export_pruned.md) owns each metadata key and its runtime reader.
[Depth export](score/export_depth.md) removes whole blocks in original order.

## Metrics and evidence

Generated-frame direction is `d=(x_sigma-predicted_x0)/sigma`.
Direction relative L2 is `norm(d_candidate-d_baseline)/norm(d_baseline)`; cosine compares their angle.
For baseline direction 4 and candidate direction 2, relative L2 is 0.5 and cosine is 1.
Capture MSE measures squared latent error to the recorded capture. Clean c0 is excluded.

Export parity measures the maximum absolute difference from the declared functional reference.
Decoded evidence assesses identity, motion and detail on matched videos.
Timing measures warmed, synchronized forwards; speedup is `baseline_ms/candidate_ms`.
ABA/BAB brackets repeat the outer arm to estimate drift.
The [verification guide](VALIDATION.md#evidence-gates) owns tolerances, required coverage and the limits of each claim.

## Diagram legend

Use small left-to-right Mermaid flows, with green rounded tensor inputs, blue rectangular operations and purple saved artifacts.
The [overview diagram](overview.md#shared-path) shows the package flow.
Render and inspect changed diagrams, including at narrow reading widths.
