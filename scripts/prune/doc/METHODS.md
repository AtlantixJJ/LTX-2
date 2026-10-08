# Pruning LLMs and video generators: literature, evidence and LTX-2 test plan

Literature checked 1 October 2026, with avatar/causal-training references added
4 October 2026. This review covers representative weight,
channel, head and block pruning methods, including video-diffusion work through
2026. Primary papers and author project pages support the claims. Results belong
to their stated models and protocols; they do not form a common leaderboard or
establish that their gains transfer to LTX-2.5.

Our question is: **which structures can we remove from LTX-2.5 to obtain a
smaller, faster executable network while retaining useful whole-clip predictions?**
The immediate scope is one full-clip bidirectional D0 forward at each exact
$[\sigma,0]$ schedule, with the capture's first latent frame clean. There is no
sliding window, generated history or K/V cache. Methods that change connectivity,
recurrence, resolution or sampling steps are related work, rather than equivalent
implementations of this task.

## 1. What pruning removes, and what performance means

A pruning method specifies the **removal unit**, **importance criterion**,
**budget allocation**, and **compensation or recovery**. Comparing importance
scores alone misses three of those decisions.

| Removal unit | Executable change | Requirement for acceleration |
|---|---|---|
| Individual weights | Irregular zeros in fixed-size matrices | Sparse storage and kernels that skip zeros; dense GEMMs still execute the original dimensions. |
| $N:M$ weights, such as 2:4 | Regular sparsity within fixed-size matrices | Supported hardware, dtype, shapes and sparse kernels. |
| FFN channels / attention heads | Coupled rows and columns disappear | Smaller dense operations, correct dependency handling, and no full-width reconstruction during each forward. |
| Residual blocks / layers | Fewer blocks execute | Omit computation; gating a block after computing it saves no work. |
| Tokens / attention edges | Less attention work | Sparse/reduced-token execution; a dense visibility mask alone need not accelerate attention. This changes our full-attention contract. |

**Quality** may mean LLM perplexity, downstream accuracy, image FID, video FVD,
VBench dimensions, or paired teacher agreement. These are not interchangeable.
Low latent error does not prove identity or motion preservation; high motion
smoothness can coexist with nearly static video. Content and motion need separate
assessment.

**Efficiency** may mean checkpoint bytes, resident parameters, peak memory,
operation count, kernel latency, complete transformer latency, or end-to-end
latency. Faster VAE decoding, lower resolution and fewer sampling steps can improve
a system without establishing a faster pruned transformer.

For our measurements, define speedup as

$$
S=\frac{T_{\mathrm{baseline}}}{T_{\mathrm{candidate}}}.
$$

$S>1$ is faster. Removal percentages must identify their denominator: all
checkpoint tensors, the executed video branch, one FFN, or one attention branch.
A 10% head budget is not 10% total-model compression.

## 2. LLM methods and their transferable ideas

### 2.1. Magnitude and activation-aware weight pruning: Wanda

Magnitude pruning ranks weights by $|w_{ij}|$. Wanda incorporates input activity:

$$
s_{ij}=|w_{ij}|\,\|X_j\|_2.
$$

It removes low-scoring weights within each output row without retraining or
updating retained weights. Its original unit is an individual weight, not a head
or channel. Grouping scores requires separate evaluation.
[Wanda, ICLR 2024, §§3–4](https://arxiv.org/html/2306.11695v2).

### 2.2. Reconstruction and curvature: SparseGPT and LLM Surgeon

SparseGPT approximately preserves each linear layer's outputs using
activation-derived second-order information and compensation of surviving
weights. It supports unstructured and $N:M$ sparsity. The authors report pruning
OPT-175B and BLOOM-176B in less than 4.5 hours, with small accuracy loss at high
sparsity. Compression time is not inference speed, and no gradient retraining
does not mean unchanged weights. [SparseGPT, ICML 2023](https://arxiv.org/abs/2301.00774).

LLM Surgeon uses Kronecker-factored curvature, global allocation and repeated
pruning/update steps, including structured removal. It reports 20–30% row/column
pruning on OPT models and Llama-2-7B with small performance loss. It offers a
stronger alternative to independent local scores, at the cost of curvature
collection and model updates. [LLM Surgeon, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/38a1671ab0747b6ffe4d1c6ef117a3a9-Abstract-Conference.html).

**LTX hypothesis:** output reconstruction may compensate for removed structures
better than freezing every retained weight. A full 22B curvature calculation is
not our first implementation target.

### 2.3. Dependency-aware groups and recovery: LLM-Pruner

LLM-Pruner identifies coupled structures, scores them with gradients, and
recovers performance through LoRA. It evaluates LLaMA, Vicuna and ChatGLM; the
authors report recovery using 50K examples in approximately three hours.
Recovered performance must be separated from immediate deletion damage.
[LLM-Pruner, NeurIPS 2023](https://arxiv.org/abs/2305.11627).

**LTX adaptation:** gate executable groups rather than assuming the LLM dependency
graph transfers unchanged. Channel deletion couples producing and consuming
projections; attention also imposes normalization and positional constraints.
The loss must match our noise levels and prediction target.

### 2.4. Variance, allocation and mean compensation: FLAP

FLAP scores activation fluctuations, standardizes scores for adaptive allocation,
and compensates missing outputs through bias terms without recovery training.
Its matched quality and throughput tables are
reviewed in §3. [FLAP, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/download/28960/29826).

**LTX hypothesis:** centered contribution may predict dispensability better than
uncentered RMS when compensation is allowed. Video activations vary with sigma,
text and time, so we must compare compensation on/off for the same mask and
check that calibration means generalize.

### 2.5. Smaller dense dimensions: SliceGPT

SliceGPT rotates representations using computational invariances, then slices
low-information dimensions to produce smaller dense matrices. It is not simply
an original-coordinate activation mask. The authors report up to 25% parameter
removal with 99%, 99% and 90% of dense zero-shot task performance on Llama-2-70B,
OPT-66B and Phi-2. This does not mean numerical equivalence or equal retention
on every task. Its perplexity comparison appears in §3.
[SliceGPT, ICLR 2024](https://arxiv.org/html/2401.15024v2).

**LTX implication:** export should create genuinely smaller dense operations.
Transferring the rotations would require proving compatibility with modulation,
normalization, residual connections and positional operations; it is not a
drop-in exporter change.

### 2.6. Depth pruning: ShortGPT

ShortGPT ranks blocks by cosine change between input and output, then removes
low-influence blocks. Its original preprint reports 27.1% parameter removal on
Llama-2-7B, aggregate score 44.52→42.60 (95.69% retention), but XSum
19.40→0.67. Aggregate retention can hide severe generation failure. Competing
results in that table are imported from LaCo, rather than all reimplemented
under one controlled pipeline. [ShortGPT, original preprint, Table 1](https://arxiv.org/html/2403.03853v1).

**LTX hypothesis:** executing fewer intact blocks may save more useful work than
small width reductions. Block influence is a screening statistic; actual deletion
damage and decoded motion determine acceptability.

## 3. LLM performance comparisons

### 3.1. Matched individual-weight sparsity

These results share **Wanda Table 3's LLaMA-7B protocol**. WikiText perplexity is
lower-is-better. “Update” means compensation of retained weights, not necessarily
gradient training. [Source: Wanda, Table 3](https://arxiv.org/html/2306.11695v2).

| Method | Retained-weight update | 50% unstructured PPL | 2:4 PPL |
|---|---|---:|---:|
| Dense | — | 5.68 | 5.68 |
| Magnitude | No | 17.29 | 42.13 |
| SparseGPT | Yes | 7.22 | 11.00 |
| Wanda | No | 7.26 | 11.53 |

Activation awareness closes much of the reconstruction gap here. The 2:4
constraint imposes greater quality loss. Wanda separately reports about 1.6×
linear-layer acceleration and 1.24× end-to-end acceleration on LLaMA-7B
(312→251 ms) with 2:4 support on A6000. These are different timing scopes.
[Source: Wanda, §4.3](https://arxiv.org/html/2306.11695v2).

### 3.2. Matched structured pruning

**FLAP Tables 1, 2 and 4** supply this LLaMA-7B comparison. PPL uses WikiText2
**validation**, rather than the setup above; accuracy averages seven tasks.
Values follow the tables where neighboring prose is inconsistent.
[Source: FLAP, Tables 1, 2 and 4](https://ojs.aaai.org/index.php/AAAI/article/download/28960/29826).

| Method | Nominal pruning | Recovery | Validation PPL ↓ | Task average ↑ | Tokens/s ↑ |
|---|---:|---|---:|---:|---:|
| Dense | 0% | — | 12.62 | 63.25 | 25.84 |
| Wanda-sp | 20% | None | 22.12 | 63.35 | Not reported here |
| LLM-Pruner | 20% | None | 19.77 | 56.82 | 32.57 |
| LLM-Pruner | 20% | LoRA | 17.37 | 60.07 | Not separately reported |
| FLAP | 20% | Bias compensation | 14.62 | 62.08 | 33.90 |

FLAP's configuration contains 5.07B parameters versus 6.74B dense and gives
approximately 1.31× throughput. Nominal group pruning and actual parameter
removal have different denominators. Compensation and allocation deserve
separate tests rather than attributing the entire gain to a score.

### 3.3. Smaller dense dimensions

**SliceGPT Table 1** uses 1,024 calibration sequences of length 2,048. At 25%
slicing, Llama-2-7B has PPL 7.24 versus 5.47 dense; 70B has 4.60 versus 3.32.
SparseGPT 2:4 gives 8.69 and 4.98 in that table. These are unequal budgets:
25% slicing versus 50% sparse weights. They compare operating points, not
equal-cost algorithms, and must not be combined with Wanda's different dense
PPL. [Source: SliceGPT, Table 1](https://arxiv.org/html/2401.15024v2).

**Our comparison rule:** compare quality at matched executable budgets and latency
at matched quality. Nominal sparsity alone does not control calibration,
compensation, parameter count or kernel support.

## 4. Diffusion pruning: the bridge from LLMs to video

Importance can vary with noise level and sampling objective. Multi-step sampling
can accumulate errors; our one-step task removes that recurrence but still needs
calibration at its actual sigmas. Causal next-token loss is not a substitute for
whole-clip bidirectional video prediction.

### 4.1. Diff-Pruning: timestep-aware Taylor importance

Diff-Pruning aggregates informative diffusion-loss gradients, excludes unhelpful
timestep contributions, and structurally prunes with recovery. It reports about
50% FLOP reduction at 10–20% of original training cost. This supports
timestep-aware selection and budgeted recovery, not a training-free guarantee or
an LTX latency claim. [Diff-Pruning, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/35c1d69d23bb5dd6b9abcd68be005d5c-Abstract-Conference.html).

### 4.2. EcoDiff: learn masks against final generation

EcoDiff learns differentiable structural masks against the end-to-end generation
result, with timestep gradient checkpointing to manage memory. Its current title
is **Learnable Sparsity for Vision Generative Models**; the earlier title was
**Effortless Efficiency**. The authors report 20% parameter pruning on SDXL and
FLUX using 100 samples and about 10 A100 GPU-hours. No retraining after pruning
still involves mask optimization. This is image evidence, not video quality or
speed. [EcoDiff, ICLR 2026, author project](https://yangzhang-v5.github.io/EcoDiff/).

**LTX hypothesis:** one-step final-latent mask optimization avoids unrolling a
long sampler, but its discrete export still needs independent validation.

### 4.3. TinyFusion: optimize recoverability of shallow DiTs

TinyFusion learns depth masks with updates that estimate future recoverability,
then fine-tunes the shallow model. It reports a 14-layer student from DiT-XL/2,
2× speedup and ImageNet FID 2.86, at less than 7% of pretraining cost. The best
immediate-error mask need not be the best recoverable mask. This is an image-DiT
result, not full-clip LTX video evidence. [TinyFusion, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Fang_TinyFusion_Diffusion_Transformers_Learned_Shallow_CVPR_2025_paper.pdf).

## 5. Video-model methods and performance

### 5.1. Temporal-attention pruning: F³-Pruning

F³-Pruning ranks aggregated temporal attention and removes redundant temporal
attention work without training. It evaluates original CogVideo and
Tune-A-Video, not CogVideoX or today's joint space-time DiTs. Tune-A-Video changes
45.08→43.83 s; CogVideo achieves approximately 1.35× acceleration. These are
attention-work results, not evidence of a comparably smaller LTX parameter
checkpoint. [F³-Pruning, AAAI 2024, Tables 1–2](https://arxiv.org/html/2312.03459v1).

### 5.2. Blocks and content/motion recovery: ICMD and V.I.P.

ICMD constructs VDMini through block pruning and recovery targeting per-frame
content and multi-frame dynamics, including adversarial training. Its revised
paper reports 2.5× speedup on SF-V, 1.4× on T2V-Turbo-v2 and 1.25× on
HunyuanVideo. These are distinct starting architectures and sampling protocols.
Its shallow/deep observations should not become a universal LTX deletion rule.
[ICMD, ACM MM 2025 / revised paper](https://arxiv.org/html/2411.18375v3).

V.I.P. uses iterative preference distillation, combining supervised learning
with ReDPO to recover compressed video generators. It reports 36.2% parameter
reduction on VideoCrafter2 and 67.5% on AnimateDiff with maintained or improved
quality. This includes trained recovery and preference-data curation; parameter
reduction alone does not establish latency. [V.I.P., ICCV 2025](https://arxiv.org/abs/2508.03254v1).

**LTX implication:** evaluate frame content and dynamics separately, and separate
selection quality from what recovery training fixes.

### 5.3. Mobile structural compression: MobileVD and Neodragon

MobileVD combines SVD temporal-block pruning and channel funnels with lower
resolution, temporal multiscale processing and single-step adversarial training.
Its ablations attribute 13% additional phone latency reduction to temporal-block
pruning and 9% to channel funnels. The 523× full efficiency claim includes other
interventions; 1.7 s measures latent generation. SVD temporal blocks have no
direct LTX counterpart. [Mobile Video Diffusion, ICCV 2025, §4.2](https://openaccess.thecvf.com/content/ICCV2025/papers/Yahia_Mobile_Video_Diffusion_ICCV_2025_paper.pdf).

Neodragon prunes Pyramidal-Flow MMDiT blocks using importance and visual impact,
then data fine-tuning and teacher distillation. Table 4 isolates 24→18 blocks:
2.028→1.518B parameters, NPU latency 1.15→0.74 s, and VBench 80.31→80.21
with recovery. Latency sums one step across three pyramid stages. Its 6.7 s
full-system result also includes text-encoder, decoder and step distillation.
[Neodragon, ICLR 2026 / preprint Table 4](https://arxiv.org/html/2511.06055v1).

### 5.4. Modern DiT size compression: FastLightGen and PARE

FastLightGen scores blocks by deletion impact, trains stochastic block-skipping
configurations, then co-distills size and sampling steps. It favors 30%
parameter pruning with four-step sampling in its tested video generators. The
approximately 35.71× figure combines 70% retained size, 50→4 steps and CFG
removal: a theoretical compound ratio, not measured single-forward pruning
speedup. Its Hunyuan sweep reports average quality 0.849 dense, 0.829 at 30%
and 0.785 at 50% pruning. [FastLightGen, CVPR 2026, §1 and Table 7](https://arxiv.org/html/2603.01685v1).

PARE is a May 2026 preprint combining spatial/temporal-aware width pruning,
input/timestep-dependent routing and optional step distillation. Its Wan2.1-14B
width-only arm has 9.8B active parameters and a six-dimension VBench average
77.46 versus 77.70 teacher after width distillation. Training uses 30K video-text
pairs and FFN widths aligned to multiples of 128. Its approximately 50× total
figure is projected and includes routing, step reduction and CFG removal.
[PARE, preprint, Table 1 and §4.1](https://arxiv.org/html/2605.27336v1).

**LTX implication:** directly test deletion impact and feasible dense widths.
Router and step-distillation gains cannot be attributed to static pruning.

### 5.5. Projected head scores versus learned gates: MobileWan

MobileWan compares projected head-contribution scoring plus recovery with
learned gates and joint fine-tuning. At approximately 33% head pruning, its
noise-bias ablation gives VBench totals 80.19 for low-noise, 80.91 for standard
and 82.19 for high-noise-biased training. Its 20 s full-system result additionally
uses recurrent/causal attention, step distillation and decoder optimization.
We can test the score and gate-learning ideas while retaining full-clip
bidirectional attention; the recurrent renderer is outside scope.
[MobileWan, July 2026 preprint, §3.1 and Table 1](https://arxiv.org/html/2607.06173v1).

### 5.6. Related work: sparse attention and quantization

Efficient-vDiT combines attention-tile sparsification with few-step consistency
distillation. It reports 7.4–7.8× acceleration on Open-Sora-Plan-1.2 for
29/93-frame 720p generation using 0.1% of pretraining data. This is a joint
attention/sampling result, not head/FFN parameter pruning.
[Efficient-vDiT, 2025](https://arxiv.org/abs/2502.06155v2).

QuantSparse combines quantization, attention sparsification and distillation.
Its HunyuanVideo-13B results report 3.68× storage reduction, 1.88× end-to-end
acceleration and PSNR 20.88 versus 16.85 for Q-VDiT. PSNR is paired fidelity
evidence rather than a general perceptual ceiling.
[QuantSparse, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94359ca6e248af69b8b6854668ae9782-Abstract-Conference.html).

Both demonstrate useful interactions between compression techniques. Sparse
attention changes our connectivity contract; quantization changes precision.
Neither belongs in the first fixed-precision structural comparison.

### 5.7. Scope of video efficiency claims

This table compares **published claims**, not our experiments. Primary sources
are linked above. Hardware, resolutions, quality metrics and recovery budgets
differ; rows cannot be sorted into a common performance ranking.

| Method | Removal / adaptation | Recovery | Reported gain | Scope |
|---|---|---|---|---|
| F³-Pruning | Temporal attention | None | Tune-A-Video 45.08→43.83 s; CogVideo ≈1.35× | Model-specific attention pruning |
| ICMD | Blocks | Content/motion recovery | SF-V 2.5×; T2V-Turbo-v2 1.4×; HunyuanVideo 1.25× | Separate compressed-model comparisons |
| V.I.P. | Compressed generators | SFT + preferences | 36.2% / 67.5% fewer parameters | Size; no latency inferred here |
| MobileVD | Temporal blocks + channel funnels | Fine-tuning | 13% / 9% less phone latency | Incremental ablations inside an optimized system |
| Neodragon | 24→18 MMDiT blocks | Two-stage recovery | 1.15→0.74 s, ≈1.55× calculated | One step summed across pyramid stages |
| FastLightGen | Blocks + few-step student | Co-distillation | ≈35.71× theoretical ratio | Size, steps and CFG changes |
| PARE | Width + routed depth | Distillation | Width-only 14→9.8B active parameters | Size arm; compound speed projection also changes steps/CFG |
| MobileWan | Heads + recurrent reformulation | Gate/weight training | 20 s full-system generation | Multiple optimizations; no head-only speed ratio inferred |
| Efficient-vDiT | Attention edges + fewer steps | Distillation | 7.4–7.8× | Joint attention/sampling acceleration |
| QuantSparse | Quantization + sparse attention | Distillation | 1.88× end-to-end; 3.68× storage | Multiple compression mechanisms |

**Synthesis.** Learned/recovered structural models can be useful; temporal/content
sensitivity matters; execution determines realized speed. None of these papers
establishes a universally best pruning percentage or score for LTX-2.5.

### 5.8. Related work: explicit driving-video conditioning and Wan-Animate-2

Wan-Animate-2 consumes driving RGB through a separate motion branch. Its Lite
design uses causal teacher-forcing pretraining, an error buffer, then Self-Forcing
distillation with chunk-wise gradient accumulation. This is relevant to our
render-guided avatar and streaming stages, rather than evidence that static
pruning alone works. The paper describes 14B-scale models; do not interpret
"Lite" as a measured parameter-size reduction.
[Wan-Animate-2, August 2026 preprint, §§3–4](https://arxiv.org/html/2608.06009v1).

**LTX hypothesis:** separate the render/pose condition from the noisy generation
state, then adapt to generated history. This could preserve guidance at high
sigma; it requires new conditioning and training contracts. The official release
lists Base and Distilled weights, which does not establish availability of a
separate causal Lite checkpoint.
[Official code and release notes](https://github.com/Wan-Video/Wan-Animate-2).

### 5.9. Related work: generated-history distribution matching with Self Forcing

Self Forcing trains autoregressive video diffusion on its own generated histories,
using K/V-cached rollouts and a video-level distribution objective. It addresses
the gap between teacher-forced training and autoregressive deployment; it is not
a structural parameter-pruning method.
[Self Forcing, 2025, paper](https://arxiv.org/abs/2506.08009).

**LTX implication:** generated-history training is necessary to test deployment,
but history matching alone does not guarantee sharp, correct avatars. Our
[October dev study](../../../../expr/onestep_avatar/dev_training_20261001/REPORT.md)
already trained cached D1 on generated history and still learned blur under
capture-latent MSE. A distribution-matching follow-on needs an appropriate score
teacher and an auxiliary learned score model; the few-step distilled generator
cannot simply be assumed to supply calibrated diffusion scores. Keep causal,
step-count and structural-compression gains separately measured.


## 6. Current implementation and measured limits

### 6.1. Activation screening

Our current FFN channel score uses post-activation RMS and output-column norm:

$$
I_j=\sqrt{\mathbb{E}[a_j^2]}\,\|w_j\|_2.
$$

A head of width $d_h$ uses the factored proxy

$$
I_h=\sqrt{\frac{\mathbb{E}[\|a_h\|_2^2]}{d_h}}\,\|W_h\|_F.
$$

Sampled tokens cover every generated latent frame; the clean first frame is
excluded. Squared statistics are averaged with equal weight per calibration
clip/sigma before RMS. Allocation removes a fixed fraction per attention branch
and FFN layer, with stable ties and at least one retained unit. This is
activation-aware, but **not Wanda's individual-weight algorithm**. The head proxy
ignores covariance, cross-unit cancellation and downstream sensitivity.

The existing `exact_local_head_energy` primitive computes

$$
E_h=\sqrt{\mathbb{E}[\|W_h a_h\|_2^2]}.
$$

It retains within-head covariance and resembles MobileWan's projected-contribution
screen. It still ignores cross-head cancellation and downstream deletion damage.
The default calibration CLI does not select masks using this helper.
See [estimators](score/estimators.md), [hooks](score/hooks.md) and
[native calibration](score/whole_clip_d0_scores.md).

### 6.2. Export and current evidence

FFN deletion couples an input-projection row/bias with an output column. LTX
attention Q/K normalization spans original heads, so the compact exporter retains
full Q/K projections and normalization before head selection, while slicing V,
gate and output dimensions and preserving RoPE identities.

`masked_full` is a full-width reference. `sparse` selects retained heads with
full-width projections. `compact_faithful` stores sliced tensors but reconstructs
full-width weights during every `ShapeFaithfulLinear` forward and executes the
original GEMM dimensions: it is a **storage/fidelity control**. `compact` reduces
supported dimensions and requires independent numerical validation.
See [export](score/export_pruned.md) and [parity](checks/export_parity.md).

The [fresh whole-clip study](../../../../expr/refiner_prune/2.5/whole_clip_20261001/REPORT.md)
found 4.510% fewer stored elements and 1.229 GiB lower peak memory for faithful
compact, but 3.797–3.917% longer forwards. Reduced-width compact also runs slower
and fails functional-mask parity; combined pruning changes held-out content.
This rejects the present mask/export as a speed optimization, not structural
pruning generally. Coverage is one held-out actor/view, one seed, three sigmas.

The subsequent [method screen](../../../../expr/refiner_prune/2.5/method_screen_20261001/REPORT.md)
adds separated rankings, allocation, compensation and block-bypass experiments,
then 27 blind actor/seed/sigma cases per finalist. None passes all gates.
Four-block bypass reaches about 1.090× forward speedup but fails quality and
retains the source checkpoint tensors. An aligned FFN reduction reaches
1.056–1.057× but fails functional/export parity; the exact one-head export is
slower. The subsequent [CPU depth export proof](../../../../expr/refiner_prune/2.5/depth_export_20261004/README.md)
physically removes original blocks `[3,7,8,15]`, leaving 44 blocks. All 4,013
retained tensor payloads match the source bytes. Resident-video elements fall
from 13,123,337,344 to 12,048,169,856 (8.19%); checkpoint payload falls 7.36%,
and full-file bytes fall from 42,018,190,584 to 38,923,958,248. This deletion set
failed its earlier functional quality gate and remains unqualified. Native BF16
export parity, decoded quality, measured timing and recovery training remain
pending; CPU payload fidelity does not establish them.

The [October 4 roadmap](../../../plans/2026-10-04-prune-finetune-causal-avatar.md)
prioritizes recoverable compact depth, an aligned-FFN backup and explicit motion
conditioning before cached avatar integration. It separates preserving D0 from
learning the deployment task, whose future capture frames are unavailable.

## 7. Methods we plan to test

The production workflow supports RMS scoring, uniform allocation, functional
masks, export controls and matched evaluation; exact projected head energy also
exists as a helper. The broader agenda below separates production capabilities
from study-specific implementations. Gradient, reconstruction/rotation and recovery
training methods remain unmeasured; results from a training-free screen do not
establish their performance.

The [archived 1 October experiment plan](../../../../plans/history/2026-10-08-superseded/2026-10-01-prune-method-comparison.md)
executes a bounded training-free comparison of rankings, compensation,
allocation and static block removal using isolated calibration, validation and
blind-test actors. Its study helpers live under ignored `expr/`; they do not
change the production workflow. Gradient learning and recovery remain follow-on
studies, conditional on those results. The [method-comparison report](../../../../expr/refiner_prune/2.5/method_screen_20261001/REPORT.md)
records the fresh measured outcomes, frozen blind split, export parity and
same-GPU timing checks. These experiment helpers do not add production CLI modes.

### 7.1. First establish executable dense savings

Hold structures fixed; compare full-width masking with genuinely smaller dense
execution. Start FFN-only, then heads-only, then combined. Profile GEMMs,
attention, head selection, copies, allocations and complete forward time.
Loading/noising/VAE remain outside forward timing; end-to-end cost is separate.

Test aligned width groups. The present 16,384-channel FFN retains 14,746 channels
at the current 10% request. Retaining 14,848 instead removes 1,536 (9.375%) and
preserves a multiple of 128. Alignment is a **hardware hypothesis**, not a proven
optimum: scan nearby widths and head counts on the actual GPU. Every ranking
method must use the same feasible groups and realized budgets.

Preserve full Q/K normalization when it defines the intended mask. Changed
normalization defines a changed-model arm. Do not loosen parity tolerance to
conceal a failure. Reconstructing original weights per forward cannot support a
claim of dense compute reduction.

### 7.2. Training-free rankings at fixed budgets

**Random and magnitude controls.** Compare seeded random group selection and
group weight norms with several random masks and identical executable budgets.
This establishes whether activation calibration improves ranking.

**Current RMS proxy.** Retain it as the cheap baseline, testing heads, FFN and
combined families separately.

**Implemented sampling control, not yet evaluated on GPU.**
`score.token_sampling` and the D0 scorer now offer opt-in
`balanced_2d_midpoint_v1`, preserving default stride and its token budget.
Actual latent H/W and deterministic integer-index hashes are recorded. At
32×32/stride 16 the control uses an 8×8 midpoint grid, expanding sampled columns
from two to eight; it does not cover all 32 rows or columns. CPU tests verify
native patchifier coordinates and budget/c0 invariants. Production native BF16
baseline parity, ranking comparison and held-out deletion results remain pending.

**Exact projected contribution.** Rank heads by $E_h$ on the same sampled tokens.
Single-channel FFN contribution already factorizes into RMS times column norm
at its local output projection. Test whether removing the head covariance
approximation improves whole-transformer deletion behavior.

**FLAP-inspired variance and compensation.** Let $\mu_j=\mathbb{E}[a_j]$ and

$$
V_j=\sqrt{\mathbb{E}[(a_j-\mu_j)^2]}\,\|w_j\|_2.
$$

For removed channels $R$, compare zero deletion with an added output bias

$$
\Delta b=\sum_{j\in R}w_j\mu_j.
$$

For heads, use centered projected energy and mean projected compensation.
Compare compensation on/off for the same mask before changing the ranking.
Begin with one pooled static bias; per-sigma compensation would define explicitly
sigma-dependent behavior and must be evaluated as such.

**Direct grouped deletion.** Measure final-output damage for executable head/FFN
groups and candidate blocks. Shortlist with cheaper scores, then compare ranking
with actual damage. Test combinations because isolated errors do not add.
A proposed calibration metric for removal set $R$ is

$$
D(R)=\mathbb{E}_{c,\sigma}\left[
\frac{\|P(v_{\theta\setminus R}(x_{c,\sigma})-v_\theta(x_{c,\sigma}))\|_2^2}
{\|P v_\theta(x_{c,\sigma})\|_2^2+\epsilon}
\right].
$$

Here $c$ is a calibration clip, $P$ selects generated latent frames and
$\epsilon>0$ prevents division by zero. Reconstruct direction consistently as
$v=(x_\sigma-\hat{x}_0)/\sigma$. Teacher agreement selects candidates; it does
not replace perceptual evaluation.

### 7.3. Adaptive allocation and sigma robustness

First keep uniform budgets to isolate score choice. Then compare uniform with
layer/branch-adaptive allocation using calibration deletion damage and measured
marginal time savings. For candidate group $G$, consider

$$
\Delta T(G)=T_{\mathrm{current}}-T_{\mathrm{current}\setminus G},\qquad
\frac{D(G)}{\Delta T(G)}\quad\text{only if }\Delta T(G)>0.
$$

This is a proposed greedy heuristic, not a proven optimizer. Recompute damage
and marginal times after accepted removals. Reject time-increasing groups when
speed is the objective.

Compare equal-sigma pooling with a worst-sigma damage constraint over exact
$[0.725,0]$, $[0.909375,0]$ and $[1.0,0]$ schedules. Protect structures that look
harmless on average but alter high-noise content/motion. Calibration selects
the mask; held-out test actors never choose its hyperparameters.

### 7.4. Static depth pruning

Compare ShortGPT/Neodragon-inspired block-influence screening with actual block
deletion damage. Start with one-, two- and four-block removal sets, including
interaction checks. Whole multimodal-block removal must preserve stream and
conditioning dependencies; it needs exporter/manifest support and is outside
today's head/FFN schema. Retained blocks keep full bidirectional connectivity.
No timestep router or windowed path is introduced.

Test whether fewer intact dense blocks beat narrower blocks at matched quality.
Infer sensitive positions from LTX measurements rather than importing another
model's depth pattern.

### 7.5. Gradients and learned masks

**LLM-Pruner / Diff-Pruning-inspired gate saliency.** Introduce structural gates
$g_G$ and evaluate a first-order deletion screen such as

$$
T_G=\mathbb{E}\left[\left|g_G\frac{\partial\mathcal{L}}{\partial g_G}\right|\right].
$$

Use a documented data/flow-matching loss or finite perturbed-gate objective.
Squared error to an identical frozen teacher has zero gradient at the unpruned
point and cannot supply meaningful first-order importance there. Compare
per-sigma gradients, calibration cost and peak memory. Full LLM Surgeon curvature
is deferred in favor of cheaper local reconstruction.

**EcoDiff / MobileWan-inspired learned gates.** Optimize fixed-budget head/channel
masks against one-step final-latent agreement, harden them, then test the exported
discrete network. Compare equal-sigma with high-noise-biased sampling, while
validating all sigmas. Separate frozen-weight mask learning from joint gate/weight
training. Soft training gates are not an inference speed result.

### 7.6. Reconstruction and budgeted recovery

First fit retained output projections to the original local outputs with
calibration-only ridge regression. Evaluate unseen actors and complete forward
outputs, not just the fitted layer. This is a SparseGPT/LLM-Surgeon-inspired
adaptation rather than an exact reproduction.

**Implemented CPU helper, no LTX fitting result.** `score.ffn_reconstruction`
provides a bounded FP64 dual ridge correction around retained source weights,
with output-channel chunks, sample/memory caps and unchanged bias. A separate
calibration-cache validator pins the native distribution, source/mask,
actual capture/noise geometry/content, retained order/alignment, equal case
quotas and sampler/payload hashes. Synthetic CPU controls verify the algebra
and rejection rules. Activation collection, fitted checkpoint serialization,
BF16 execution and held-out full-model improvement remain unrun.

Next compare no recovery with fixed-budget LoRA/student distillation. Keep masks,
inputs and training budget matched, checking per-frame content and multi-frame
dynamics as motivated by ICMD. Merge LoRA or include its runtime overhead.
Report before/after quality, recovery steps, data and GPU-hours.

If recovery helps but mask selection limits quality, test TinyFusion-inspired
learnable block masks and recoverability. Preference/adversarial recovery from
V.I.P./FastLightGen comes after a simpler distillation baseline because it adds
curation and training variables. Sampling stays one-step in this pruning study;
step-distillation gains are not credited to structural pruning.

## 8. Protocol and acceptance

**Matched inputs.** Fix checkpoint, VAE, source crop/matte/span, fps, clean first
latent frame, saved epsilon, prompt, guidance and exact sigma per comparison.
Use float32 sigma/timesteps and BF16 weights/latents. Include nonempty prompts
before a broader claim. Capture-conditioned D0 does not establish ordinary
text-to-video or autoregressive-avatar performance.

**Splits.** Calibration selects structures and fits compensation; a separate
validation split chooses budgets/thresholds; a blind test split supports final
claims. Actors remain disjoint across all views. Proposed initial expanded
coverage is at least three held-out actors and three fixed seeds, including
low-motion and active-motion cases. The method screen completed three blind
actors and three seeds; future tuning requires newly reserved identities rather
than treating those inspected actors as blind again.

**Controlled questions.** Compare rankings at fixed feasible widths, allocation
with a fixed score, and compensation/recovery with a fixed mask. Finally compare
complete candidates at matched quality and plot quality/latency/memory trade-offs.
Include unpruned, no-prune exported, functional-mask and dense-export controls.

**Correctness.** Calibration must reproduce saved baselines. Export must match
its intended functional intervention with the current maximum absolute latent
tolerance of 0.02. Changed normalization or recovery weights require a new
reference; they cannot silently reuse the original-mask fidelity claim. Verify
source/noise hashes, paired tensors and clean-frame preservation.

**Quality.** Record generated-frame direction relative L2/cosine, capture latent
MSE and synchronized decoded comparisons against both teacher and capture VAE.
Inspect identity, facial detail, pose/gesture, motion amount and temporal
consistency. Add individual VBench dimensions for a sufficiently broad suite.
Small-sample FID/FVD or one aggregate score cannot substitute for matched visual
evidence. Predeclare numeric and visual acceptance thresholds before blind tests.

**Cost.** Report selection forwards/backward passes, calibration wall time and
memory; recovery budget; checkpoint bytes and executed-branch parameter count;
warmed synchronized forward latency and peak allocated memory. Use repeated
samples and ABA/BAB brackets on the same device/software/precision. Profiling
explains overhead but does not replace complete forward timing. Claim speed
only when the gain survives both orders and exceeds measured drift.

**Order of work.** Establish executable dense widths, compare inexpensive
rankings, test adaptive allocation and static block removal, then add compensation
and explicitly budgeted recovery. Accept candidates when correctness, held-out
content/motion and measured cost agree. If only storage improves, label the
result as storage compression.

Implementation contracts remain in the source-matching [design index](README.md)
and [validation guide](VALIDATION.md). Revise this agenda when new methods are
implemented or matched LTX evidence changes the priorities.
