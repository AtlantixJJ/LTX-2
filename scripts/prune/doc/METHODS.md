# Structured pruning: objectives, mathematics and candidate methods

This is the conceptual reference for head and FFN-channel pruning in this package.
It distinguishes methods used to produce masks, estimators merely implemented,
and proposed extensions. File-level implementation notes remain in the existing
module docs until the [simplification proposal](../../../../plans/2026-09-29-prune-simplification.md)
is implemented.

## 1. What we are optimizing

Let \(f_\theta(x_\sigma,c)\) be the video transformer's prediction for a noised
capture, first-frame condition and text context. A binary structural mask \(m\)
removes complete attention heads or FFN hidden channels. The desired result is a
model with lower **measured forward latency**, acceptable held-out prediction and
decoded-video differences, and lower memory where possible.

A useful formulation is

\[
\min_{m,\tilde\theta}\;\mathbb E_{x,\sigma}\,
\ell(f_{\tilde\theta,m}(x_\sigma,c),f_\theta(x_\sigma,c))
\quad\text{subject to}\quad T(m,\tilde\theta)\le T_{\rm budget}.
\]

This is a proposed selection objective, not an optimization the current screen
solves. Preservation of the baseline and fidelity to the capture are separate
measurements. An apparent improvement against capture at one noise level does
not establish that the pruned model preserves the baseline or improves generally.

There are two distinct input distributions in our experiments:

| Task | Inputs and prediction region | Status |
|---|---|---|
| Whole-clip D0 | One bidirectional forward at a specified sigma; clean first latent frame; generated frames scored | Current pruning evaluation target |
| k2 refiner | Sliding windows, frozen history, fresh chunk, deployed two-step schedule | Historical pruning study; rollout code still serves other scripts |

Scores calibrated on one task must not silently select a mask for the other.
Capture, noise, prompt, geometry, fps and condition span must match between
baseline and candidate. Hold out actors and retain per-sigma results.

The user-confirmed active requirement is **native bidirectional pruning**. Selection,
ablation, functional-mask/export parity, quality and timing all use that method.
AR rollout cannot supply its importance scores or acceptance criteria. A historical
AR-derived checkpoint can be an explicitly labeled transfer control; it is not a
native-calibrated result. Reusing an estimator's mathematics is valid only after
collecting activations/gradients under native inputs and defining the native objective.

## 2. Structural units

At the attention output projection, write a branch as

\[
y=\sum_h W_h a_h+b,
\]

where \(a_h\) is the post-attention, post-head-gate activation for head \(h\),
and \(W_h\) is its block of output-projection columns. An FFN has

\[
y=W_2\phi(W_1x+b_1)+b_2=\sum_j w_j a_j+b_2.
\]

Here \(j\) is a hidden channel, not a residual-stream coordinate. Removing it
couples a row of \(W_1\), its bias and a column of \(W_2\).

Masking activations is useful for ablation, but leaves full-width GEMMs.
Compact export physically changes supported dimensions. In this LTX implementation
Q/K normalization couples original heads: the exporter retains full Q/K projections
and normalization before selecting heads. It can slice V, output projection and
head gates. Consequently, head-count reduction does not imply the same fraction
of attention-projection savings. See [export implementation](../score/export_pruned.py).

## 3. Methods employed and implemented

### Activation RMS × output-weight norm: the latest D0 mask

For FFN channel \(j\), our score is

\[
I_j=\sqrt{\mathbb E[a_j^2]}\,\|w_j\|_2.
\]

For a head of width \(d_h\), the sampled proxy is

\[
I_h=\sqrt{\mathbb E[\|a_h\|_2^2]/d_h}\,\|W_h\|_F.
\]

The D0 implementation accumulates mean squared activations over sampled generated
tokens and gives each calibration view/sigma forward equal weight before taking
the square root. It is not an average of already square-rooted per-forward scores.
It samples space across all generated latent frames, excluding the clean first frame.

For a scalar channel, this equals the RMS norm of that channel's local projected
contribution. For a multi-dimensional head, the factored expression is a proxy:
it discards covariance within the head. Neither accounts for cancellation between
units, residual gates after the projection, or amplification by downstream layers.

This is **Wanda-inspired structured scoring**, not a reproduction of Wanda.
Wanda scores individual weights using activation magnitude and allocates sparsity
per output row; we aggregate output columns into channels/head blocks and allocate
per branch/layer. [Sun et al., *A Simple and Effective Pruning Approach for Large
Language Models*, ICLR 2024](https://arxiv.org/abs/2306.11695).

Our recorded D0 mask used two actors at sigma 0.725, 0.909375 and 1.0, stride 16,
with a third actor held out. Every calibration forward exactly matched its saved
baseline. Per-branch 10% rounding removed 3/32 heads (9.375%); per-layer FFN
rounding removed 1,638/16,384 channels. Across 48 layers: 288 heads and 78,624
FFN channels. This was a one-shot screen, without iterative rescoring or recovery.
It changed the output substantially and demonstrated no forward speedup. See
[score artifact](../../../../expr/refiner_prune/2.5/whole_clip_d0/scores_d0_h10_f10.json)
and [measured findings](../../../../expr/refiner_prune/2.5/FINDINGS.md).

### Exact local head contribution: implemented for k2

The older contribution estimator and the native-token primitive in
`score/estimators.py` compute

\[
C_h=\sqrt{\mathbb E_{t\in\mathcal T}\|W_h a_{t,h}\|_2^2}
\]

on task tokens \(\mathcal T\). The older k2 CLI additionally L2-normalizes
scores within each branch; the native primitive returns raw per-head energy
for a caller-defined generated-token index set. This retains within-head
covariance and is more faithful to local output energy than factored RMS × norm.
It is still not the change in final model output or task loss after deletion.
The D0 screen did not use it. See [native primitive](../score/estimators.py) and
[historical k2 implementation](../score/head_scores.py).

### Gate-gradient / first-order Taylor: the historical p05 mask

Introduce a continuous gate \(g_h\) multiplying each head, initially one. Deleting
it changes \(g_h\) by −1. First-order Taylor expansion gives

\[
\Delta L\approx-\frac{\partial L}{\partial g_h},\qquad
I_h=\mathbb E\left|\frac{\partial L}{\partial g_h}\right|.
\]

The k2 scorer uses clean-latent reconstruction loss on fresh chunk tokens,
accumulates absolute mask gradients, normalizes within each branch, and selects
the lowest scores globally. The model weights remain frozen; only mask gates
require gradients. Taking the absolute value before accumulation prevents signed
gradient cancellation across examples. [Michel, Levy and Neubig, *Are Sixteen
Heads Really Better than One?*, NeurIPS 2019](https://arxiv.org/abs/1905.10650).

The schedule supports pruning a small batch, recomputing scores under the current
mask and repeating. The recorded p05 smoke run used two records and one round;
it does not demonstrate the benefits of a multi-round schedule. It also does not
provide D0-calibrated gradient scores.

### Random-projection Jacobian sensitivity: implemented, not used for the latest masks

Let \(J_h=\partial f/\partial g_h\) on task outputs and let \(r\) have independent
Rademacher entries. Then

\[
\mathbb E_r[(r^T J_h)^2]=\|J_h\|_2^2.
\]

Our implementation normalizes \(r\) by \(\sqrt M\), where \(M\) is the number of
task-output scalar elements, and estimates
\(\sqrt{\mathbb E\|J_h\|^2/M}\) using repeated VJPs. It then normalizes scores
within each branch. The code calls this `gauss_newton`; more precisely it is a
random-projection estimate of diagonal output-Jacobian energy. It corresponds to
a Gauss–Newton curvature diagonal for a squared preservation loss, up to the loss
normalization, but is not a full Hessian, Fisher estimator or exact deletion cost.
The stochastic identity is the same underlying principle as Hutchinson-style
trace estimation; our code does not implement Hutch++.
[Meyer et al., *Hutch++: Optimal Stochastic Trace Estimation*, 2021](https://arxiv.org/abs/2010.09649).

### Ridge reconstruction: available infrastructure, not employed by the D0 screen

After selecting retained features \(X_K\), fit an output projection to reproduce
the original local branch output \(Y\):

\[
\min_{\widetilde W}\|Y-\widetilde W X_K\|_F^2+
\lambda_{\rm eff}\|\widetilde W\|_F^2,
\quad
\widetilde W=YX_K^T(X_KX_K^T+\lambda_{\rm eff}I)^{-1}.
\]

The code accumulates the Gram and cross matrices in FP32, uses a linear solve,
and sets \(\lambda_{\rm eff}\) to the user ridge coefficient times the mean Gram
diagonal. It fits the bias-free projection contribution and preserves the output
bias separately. This is closed-form calibration, not gradient training.
The Gram matrix is quadratic in retained width: streaming examples does not remove
that memory cost. Local reconstruction needs held-out validation and may fail to
preserve downstream behavior. [Implementation](../score/lstsq.py).

## 4. Candidate methods and priority

These are adaptations to investigate, not claims of established LTX performance.

| Priority | Candidate and paper basis | What changes in our experiment | Main limitation |
|---|---|---|---|
| 1 | Runtime-aware structural selection: [ZipLM, Kurtic et al., NeurIPS 2023](https://arxiv.org/abs/2302.04089) | Measure actual supported head/channel group sizes; select among groups using held-out damage versus measured time saved | LLM results do not establish diffusion quality; some groups save no time |
| 2 | Covariance-aware selection and compensation: [SparseGPT, Frantar & Alistarh, ICML 2023](https://proceedings.mlr.press/v202/frantar23a.html) | Use local activation covariance to model redundant features; shortlist structural groups and compensate surviving output weights | SparseGPT is principally weight sparsity; head/channel groups require an explicit adaptation and export support |
| 3 | Diffusion-aware Taylor ranking: [Diff-Pruning, Fang et al., NeurIPS 2023](https://arxiv.org/abs/2305.10924) | Calibrate gate gradients on D0 inputs at relevant sigmas; test per-sigma robustness and importance aggregation | Published method includes recovery training; it is not evidence for training-free LTX pruning |
| 4 | Dependency-aware pruning plus recovery: [LLM-Pruner, Ma et al., NeurIPS 2023](https://arxiv.org/abs/2305.11627) | If recovery training is in scope, compare structural masks followed by a small LoRA/distillation recovery run | Separate training budget and held-out checks required; language-model evidence does not transfer automatically |

Structural dependencies also need an explicit specification. [DepGraph, Fang et
al., CVPR 2023](https://arxiv.org/abs/2301.12900) motivates grouping coupled
parameters; it does not justify blind application of a generic pruner to LTX's
normalization, RoPE and head-gate semantics.

My recommended combination is **a cheap activation screen, direct grouped ablation,
runtime-aware allocation, and optional local reconstruction**. Exact local energy
can improve the screen; covariance and task-gradient methods cost more and should
be added only when they improve held-out selection.

For a proposed group \(G\), a simple *local adaptation* of runtime-aware selection is

\[
R_G=\frac{\Delta L_G}{\Delta T_G},\quad \Delta T_G>0.
\]

Select low-damage groups with measured positive savings, rescore after changes,
and remeasure the complete model because latency is not additive. This ratio is
our proposed heuristic, not a statement of ZipLM's exact algorithm. Use prediction
preservation for \(\Delta L_G\), track capture fidelity separately, and reject
groups that only reduce parameter counts without a timing gain. Also keep memory
and latency as separate objectives.

## 5. What would make a result trustworthy

1. Fix calibration and held-out subjects, prompt, condition, fps, geometry, epsilon
   and exact sigmas before selection. Inspect sampling convergence and per-sigma ranks.
2. Compare activation ranking with direct ablation on a manageable shortlist. Test
   head-only and FFN-only candidates before a combined candidate.
3. Export to a supported dense structure. Compare **functional mask versus export**
   separately from **baseline versus pruned model**. BF16 shape changes can alter
   numerical results; report a stated tolerance and decoded effects.
4. Measure warm forward time on the same device with synchronized, bracketing
   baseline arms. Record drift, memory, supported matrix sizes and timed boundaries.
5. Inspect held-out VAE decodes and synchronized capture/baseline/candidate videos.
   A small activation score, a FLOP count or a calibration loss alone is insufficient.

The current evidence supports further method investigation, not deployment of the
combined 10% head/FFN mask.
