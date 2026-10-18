# Pruning LLMs and video generators: literature, evidence and LTX-2 test plan

DiT pruning literature checked 11 October 2026, including September 2026 work.
FFN gate, grouping and recovery references checked 12 October 2026 (§§2.7–2.9).
The earlier LLM comparisons and avatar/causal-training reviews remain dated
1–4 October 2026.
This review covers weight, channel, head, block and expert pruning, plus related
low-rank and dynamic-computation methods. Primary papers and author project
pages support the claims. Results belong
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

Its dependency group contains parameters that must be removed together.
This does not establish a semantic group of many FFN neurons sharing one gate.

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

### 2.7. Learning individual FFN gates under an architecture budget

An **FFN channel** is one intermediate neuron, including its producing row and
consuming column. One gate per channel can produce an aligned final width.
The alignment constraint controls the number retained; it need not tie their gates.
The following methods give direct precedent for learning these choices.

**Louizos et al.: the gate foundation.** Hard-concrete gates permit exact zeros
while providing gradients for their distribution parameters. Their differentiable
expected $L_0$ penalty encourages removal. The framework supports learning
structures, but an expected sparsity penalty alone does not guarantee a specified
binary width in every forward. §4.7 gives the gate and penalty equations.
[Learning Sparse Neural Networks through $L_0$ Regularization, ICLR 2018](https://arxiv.org/abs/1712.01312).

**CoFi: individual channels with multiple mask scales.** CoFi learns
hard-concrete masks for FFN intermediate dimensions, attention heads, hidden
dimensions and whole attention/FFN modules. Lagrangian sparsity control and
prediction/layer distillation guide optimization. Its FFN intermediate mask has
one variable per neuron. Several mask scales can control a weight, but this
does not require several neurons to share one intermediate gate.
This is direct evidence that per-channel gates are a workable design.
[CoFi, Structured Pruning Learns Compact and Accurate Models, ACL 2022, §3](https://aclanthology.org/2022.acl-long.107.pdf).

**Compresso: masks and LoRA learned together.** Compresso freezes pretrained
weights and jointly optimizes structural masks and LoRA adapters. Its masks
include individual FFN intermediate dimensions, attention heads and shared
residual dimensions. It uses augmented $L_0$ optimization to target model size.
Its collaborative prompting and instruction data are LLM-specific components;
the transferable mechanism is constrained mask learning with bounded recovery.
It is a closer reference for our joint gate/adapter proposal than post-pruning
LoRA alone.
[Compresso, 2023 preprint, §3.3](https://arxiv.org/html/2310.05015v1).

**Sheared LLaMA: choose channels while targeting a model shape.** It learns
hard-concrete masks for layers, heads, hidden dimensions and FFN intermediate
dimensions, with constraints on a specified target architecture. It jointly
updates model weights and masks, then continues pretraining. The relevant lesson
is that individual channel choices and prescribed final widths are compatible.
Its weight training and data budget differ from a frozen-backbone LTX pilot.
[Sheared LLaMA, ICLR 2024, §2.1](https://arxiv.org/html/2310.06694v1).

**GRASPrune: learn the discrete budgeted selection.** GRASPrune learns scalar
scores for FFN channels and attention KV head groups while freezing the backbone.
Each forward uses a binary budgeted mask; a straight-through estimator supplies
gate gradients. After selection, it calibrates retained-unit scales and folds
them into compact weights. This directly supports learning choices under the
deployment budget. Its score-ordered greedy projection is not an exact knapsack
solver. Its GQA head costs and KV-cache claims do not transfer to our D0 call.
[GRASPrune, ACL 2026, §3](https://aclanthology.org/2026.acl-long.491.pdf).

**Diffusion connection.** Among the methods reviewed here, EcoDiff (§4.7) is the closest diffusion
reference for learned head and neuron gates using generation reconstruction and
sparsity. The methods above strengthen its FFN precedent and add explicit shape
constraints and gate/adapter recovery. Their LLM results do not establish video
quality or LTX speed.

### 2.8. How neurons are grouped, ordered or protected

“Group” can mean linked parameters, neurons with similar function, a contiguous
execution tile, or a nested width prefix. These are different design choices.
A functional group can guide calibration without sharing one removal gate.

**FANG: preserve specialized contributions during selection.** FANG clusters
semantic token contexts, constructs a cluster–neuron sensitivity matrix and
assigns neurons to functional groups. It prunes neurons **within** each group
using context-weighted importance. It also protects a shared group that serves
multiple contexts and allocates sparsity by block complexity. It does not delete
each functional group through one binary switch. Its sensitivity construction
uses the pretraining loss; it is not a forward-only grouping method.
[FANG, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40021),
[method details](https://arxiv.org/html/2512.23014v1).

**LTX hypothesis:** region-weighted calibration and loss may protect human
contributions hidden by global averages. Semantic token clusters in an LLM are
not established person regions in a video DiT. Test that adaptation separately;
placing all human-related neurons behind one removable gate could concentrate
damage. A human-focused map alone does not establish necessity or a safe group.

**MoEfication: balanced groups from weights or co-activation.** MoEfication
partitions FFN neurons into equal-sized experts using balanced weight clustering
or a co-activation graph split, then routes inputs among those experts. It gives
concrete grouping algorithms. Its goal is dynamic expert execution, rather than
one permanently narrower FFN. Its co-activation construction uses ReLU-based
models; an LTX GELU adaptation must define activity and similarity explicitly.
[MoEfication, Findings of ACL 2022, §3.2](https://aclanthology.org/2022.findings-acl.71.pdf).

**WIDE: contiguous groups chosen for GPU execution.** WIDE learns token-dependent
selection of attention-head and FFN-channel groups. Its contiguous FFN group
sizes follow matrix-multiply tiling choices. Router training and optional LoRA
recovery work with specialized kernels that reorder masks and skip computation.
It is direct precedent for one gate controlling several channels, with a hardware
rationale. Its dynamic routing and kernel speedups do not establish a benefit
from fixed groups in an ordinary compact dense LTX checkpoint.
[WIDE, 30 July 2026 preprint, §§3.1–3.3](https://arxiv.org/html/2607.28418v1).

**DynaBERT: importance ordering and nested prefixes.** DynaBERT ranks and reorders
heads and FFN neurons, then distills subnetworks that retain prefixes at selected
widths. Important neurons participate in more subnetworks. This provides an
ordering strategy when several nested widths must share one model; it does not
establish semantic groups. A single fixed LTX width does not need prefix nesting.
[DynaBERT, NeurIPS 2020, §3.1](https://arxiv.org/html/2004.04037v1).

**LTX grouping choices.** Compare importance ordering, balanced weight or
co-activation grouping, and contiguous hardware groups at the same retained
widths. Include seeded random groups to test whether grouping information helps.
Keep functional weighting separate from gate sharing. §7.5.2 defines the proposed
FFN comparison and its export constraints.

### 2.9. Additional width selection and recovery references

**Minitron.** It ranks neurons and other structural dimensions using calibration
activations, trims the corresponding weights, and recovers with distillation.
Architecture search and recovery are separate stages. This supports an
activation-selected compact-width comparator; it does not require learned gates
or fixed semantic groups.
[Compact Language Models via Pruning and Knowledge Distillation, July 2024](https://arxiv.org/html/2407.14679v1),
[LLM Pruning and Distillation in Practice: The Minitron Approach, August 2024](https://research.nvidia.com/publication/2024-08_llm-pruning-and-distillation-practice-minitron-approach).

**LoRAPrune.** It estimates importance from LoRA weights and gradients, rather
than pretrained-weight gradients, and iteratively removes channels and heads.
It supports adapter-based selection and recovery with a frozen base. It is a
different selection mechanism from jointly learning a binary gate per channel.
[LoRAPrune, Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.178/).

**LoRAShear.** It builds dependency graphs over LoRA modules, progressively
removes structures, transfers knowledge and performs recovery training.
Its dependency groups identify valid removals, rather than demonstrating
semantic groups of many FFN channels with shared gates.
[LoRAShear, 2023 preprint](https://arxiv.org/abs/2310.18356).

**Bonsai.** Bonsai uses forward-only perturbative evaluation to select structured
removals. It is a useful alternative if complete-clip backward propagation is
too expensive. Forward-only selection still has a selection cost and does not
establish that recovery is unnecessary.
[Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes, 2024 preprint](https://arxiv.org/abs/2402.05406).

**Block Pruning for Faster Transformers.** This method extends movement pruning
to blocks inside weight matrices and can learn removal of larger components.
A matrix block does not generally include every producing and consuming weight
of an FFN hidden channel. Learned weight-block sparsity is therefore a separate
execution design from channel groups exported as smaller dense matrices.
[Block Pruning for Faster Transformers, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.829/).

**Scope of existing baselines.** Wanda and SparseGPT (§§2.1–2.2), and DiT-Pruning
(§4.12), primarily supply weight or patterned-sparsity methods. SliceGPT (§2.5)
removes rotated representation dimensions. They offer useful criteria and
reconstruction controls, but less direct precedent for gates on original FFN
intermediate channels. LLM Surgeon, FLAP and §§4.3–4.6 contribute structured
selection, allocation and compensation. PARE (§4.15) selects individual neurons
with diversity and aligned widths; dynamic width methods (§4.14) have a different
deployment contract. Keep these distinctions when naming an LTX adaptation.

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

## 4. DiT pruning: categories, priorities and mathematical methods

Importance varies with noise level, conditioning and sampling objective.
Multi-step sampling can accumulate errors. Our one-step task removes that
recurrence, but still needs calibration at its actual sigmas.

### 4.1. Categorized and ranked reading list

**Rank means our priority for investigation on LTX-2.5 D0, not published superiority.**
This is an editorial ranking. We first consider compatibility with static dense
head/FFN/block export, then relevance to video and multimodal dependencies,
then selection/recovery cost and the evidence available. No numeric composite
score is claimed. A low rank can mean an incompatible execution contract rather
than a weak paper. Sections 4.2–4.16 explain the algorithms; §5 retains the
video results and their measurement scopes.

Categories are **static width**, **static depth**, **stage-dependent width/depth**,
**weight sparsity**, **low rank**, **expert pruning**, and **token computation**.
“Training-free” permits calibration and algebraic weight updates; mask learning
and recovery training are listed separately. Dates identify the conference or
first preprint, not an inferred acceptance. The list is representative, not exhaustive.

| Rank | Work and date | Category / removal unit | Selection and recovery | Evidence / reason for priority |
|---:|---|---|---|---|
| 1 | [OBS-Diff, ICLR 2026](https://openreview.net/forum?id=eYdPMpW5os) | Static width; also weight and N:M sparsity | Timestep-weighted curvature; OBS compensation; no gradient recovery | SD3/SD3.5/FLUX; explicit heads, neurons and multimodal rank fusion. |
| 2 | [MobileWan, Jul 2026 preprint](https://arxiv.org/abs/2607.06173) | Static width / heads | Projected contribution or learned gates; joint gate/weight recovery | Wan2.2-5B video; closest head-scoring comparison, with recurrence isolated. |
| 3 | [TinyFusion, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_TinyFusion_Diffusion_Transformers_Learned_Shallow_CVPR_2025_paper.html) | Static depth / blocks | Learn local masks with LoRA recoverability estimates; fine-tune | Image DiT/SiT; addresses why low immediate deletion error can select poorly. |
| 4 | [PPCL, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Ma_Pluggable_Pruning_with_Contiguous_Layer_Distillation_for_Diffusion_Transformers_CVPR_2026_paper.pdf) | Static depth plus branch replacement | Linear probes, contiguous-interval distillation; optional full recovery | FLUX/SD3.5/Qwen-Image; preserves interval endpoints and multimodal structure. |
| 5 | [PARE, May 2026 preprint](https://arxiv.org/abs/2605.27336) | Static width plus dynamic depth | Temporal-head protection, diverse FFN selection; progressive distillation | Wan2.1-14B; use its width arm first, assess routing separately. |
| 6 | [SVDtrunc, Sep 2026 / ECCV 2026 author release](https://vislearn.github.io/SVDtrunc/) | Low rank / projection matrices | Compression probes, nonuniform ranks; flow and module distillation | FLUX; important alternative when head/block deletion damages content. |
| 7 | [Neodragon, ICLR 2026](https://arxiv.org/abs/2511.06055) | Static depth / MMDiT blocks | Importance and visual impact; fine-tune then distill | Pyramidal-Flow video; useful isolated depth/recovery ablation in §5.3. |
| 8 | [FastLightGen, CVPR 2026](https://arxiv.org/abs/2603.01685) | Static depth / blocks | Deletion impact, stochastic skips, size/step co-distillation | Modern video DiTs; separate size selection from step and CFG savings. |
| 9 | [EcoDiff, ICLR 2026](https://yangzhang-v5.github.io/EcoDiff/) | Static width / heads and neurons | End-to-end hard-concrete mask learning; optional recovery | FLUX/SDXL; one-step adaptation avoids sampler unrolling. |
| 10 | [HierarchicalPrune, AAAI 2026](https://arxiv.org/abs/2508.04663) | Static depth and subcomponents | Position-aware removal; freeze early weights; sensitivity-guided distillation | Large MMDiTs; a recovery baseline, not a universal late-block deletion rule. |
| 11 | [MosaicDiff, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Guo_MosaicDiff_Training-free_Structural_Pruning_for_Diffusion_Model_Acceleration_Reflecting_Pretraining_ICCV_2025_paper.pdf) | Stage-dependent width / heads and neurons | SNR-specific calibration, grouped OBS; stitch subnetworks | DiT/SDXL; directly relevant curvature method, multi-stage execution differs. |
| 12 | [Diff-ES, Mar 2026 preprint](https://arxiv.org/abs/2603.05105) | Stage-dependent width/depth | Evolutionary sparsity schedules; cached compensation weights | DiT/SDXL; improves allocation/search beyond MosaicDiff's heuristic stages. |
| 13 | [FastFLUX, AAAI 2026](https://arxiv.org/abs/2506.10035) | Residual-branch replacement | Least-squares linear surrogate; neighboring-block LoRA recovery | FLUX; tests replacement rather than pure removal. |
| 14 | [DiT-Pruning, Jul 2026 preprint](https://arxiv.org/abs/2607.00927) | Individual-weight sparsity | Squared-weight activation score; distribution-aware comparison groups | DiT/PixArt/FLUX; recent DiT-specific score, requires sparse execution. |
| 15 | [DyDiT++, Apr 2025 extended preprint](https://arxiv.org/abs/2504.06803) | Dynamic width and token computation | Timestep routers, spatial FFN bypass; flow/PEFT extensions | DiT/SiT/Latte/FLUX; includes video, with a dynamic execution contract. |
| 16 | [DyDiT, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a44a70acd5d0abc1a252ada9719dd06d-Abstract-Conference.html) | Dynamic width and token computation | Head/channel routers and token FFN masks; FLOPs-constrained training | Image DiT; foundational adaptive-capacity baseline for the later extension. |
| 17 | [E-DiT, Feb 2026 preprint](https://arxiv.org/abs/2602.13993) | Dynamic depth/FFN width plus caching | Content/timestep routers; quality and capacity losses | Image/3D DiTs; runtime slicing is useful, caching cannot help a lone D0 call. |
| 18 | [DIET, Sep 2026 preprint](https://arxiv.org/abs/2609.37829) | Expert pruning / whole MoE experts | Cached deletion/re-routing signatures; coverage and budget search; no recovery | LingBot-Video; newest video parameter-pruning evidence, dense LTX has no expert bank. |
| 19 | [TAPE, May 2026 preprint](https://arxiv.org/abs/2605.17837) | Token computation | Temporally smooth importance, reselect tokens, schedule budgets; no training | Video DiTs; useful coherence control, does not remove model parameters. |

The older **Diff-Pruning** is the gradient-based bridge below. F³-Pruning,
ICMD, V.I.P. and MobileVD remain in §5 because their backbones or recovery
questions differ from modern dense DiT pruning. Efficient-vDiT and QuantSparse
are attention/precision comparisons, rather than static parameter deletion.

### 4.2. Shared notation and an exact local reconstruction calculation

Use column-vector tokens. Let $X\in\mathbb{R}^{d\times n}$ contain $n$
calibration inputs, $W\in\mathbb{R}^{p\times d}$ be a projection, and $Y=WX$.
$R$ denotes removed input columns; $K$ denotes retained columns.
$\theta$ is the frozen teacher; $\phi$ is a recovered student.
$m_G\in\{0,1\}$ is a structural keep gate.
For a complete block $B_\ell$, bypass means

$$
h_{\ell+1}=h_\ell+m_\ell\bigl(B_\ell(h_\ell)-h_\ell\bigr).
$$

The FFN form is $W_{\rm down}\,a$, where
$a=\psi(W_{\rm up}h+b_{\rm up})$.
Deleting neuron $j$ couples an up-projection row/bias and a down-projection column.
Deleting a head couples its output columns and the permitted Q/K/V/gate slices.
LTX's cross-head Q/K normalization still imposes the restrictions in §6.2.

Here is a **shared derivation**, not a claim that every paper implements the
same group optimizer. Set $H=XX^\top+\lambda I$ with $\lambda>0$.
For one output row, represented by column vector $w$, use the quadratic objective

$$
\min_{\delta w}\frac12\delta w^\top H\delta w
\quad\text{subject to}\quad \delta w_R=-w_R.
$$

Let $E_R$ select removed coordinates, so $w_R=E_R^\top w$.
The Lagrange equations give
$\delta w=-H^{-1}E_R\nu$ and
$\nu=((H^{-1})_{RR})^{-1}w_R$.
Thus

$$
\delta w^*=-H^{-1}E_R\bigl((H^{-1})_{RR}\bigr)^{-1}w_R,
\qquad
I(R)=\frac12w_R^\top\bigl((H^{-1})_{RR}\bigr)^{-1}w_R.
$$

Across output rows, the compensation and predicted cost are

$$
\Delta W=-W_{:,R}\bigl((H^{-1})_{RR}\bigr)^{-1}(H^{-1})_{R,:},
\qquad
I(R)=\frac12\operatorname{tr}\!\left[
W_{:,R}\bigl((H^{-1})_{RR}\bigr)^{-1}W_{:,R}^\top\right].
$$

**Invert first, then select the block.** $(H^{-1})_{RR}$ generally differs
from $(H_{RR})^{-1}$. The former lets retained columns compensate for deletion.
Scalar removal gives $I(j)=\|W_{:,j}\|_2^2/[2(H^{-1})_{jj}]$.
Joint group costs need not equal sums of scalar costs.

For a worked check, let $H=\left[\begin{smallmatrix}1&0.5\\0.5&1\end{smallmatrix}\right]$
and $w=(2,1)^\top$. Removing coordinate 1 gives
$\delta w=(-2,1)^\top$, a retained weight of 2, and cost 1.5.
Frozen deletion gives $\delta w=(-2,0)^\top$ and cost 2.
This shows what compensation buys under this local objective.
It does not establish full-transformer or decoded-video fidelity.

Paper equations below use this orientation where possible. Factors of two
depend on the reconstruction-loss convention and do not change a local ranking.
Specific noise weights, mask distributions and execution policies remain distinct.

### 4.3. Diff-Pruning: timestep-aware Taylor importance

Diff-Pruning structurally prunes using diffusion-loss gradients, filters
uninformative timesteps, then recovers the model. Its reported roughly 50%
FLOP reduction uses 10–20% of original training cost.
[Diff-Pruning, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/35c1d69d23bb5dd6b9abcd68be005d5c-Abstract-Conference.html).

To explain its gradient principle in common notation, deletion gives
$\Delta\theta_G=-\theta_G$. First-order Taylor expansion yields

$$
\mathcal L(\theta+\Delta\theta)-\mathcal L(\theta)
\approx-\sum_{i\in G}\theta_i\frac{\partial\mathcal L}{\partial\theta_i}.
$$

This is the underlying approximation, not a replacement for the paper's
timestep filter and aggregation. For our gate adaptation, use
$\left|\partial\mathcal L/\partial m_G\right|_{m_G=1}$ with a stated diffusion
or flow target. It measures downstream sensitivity rather than local output energy.
Squared error to an identical frozen teacher has zero first derivative at
the dense solution. It therefore cannot provide this deletion ranking.

### 4.4. OBS-Diff: noise-weighted curvature and coupled multimodal heads

OBS-Diff solves weighted local reconstruction with algebraic compensation.
Its $t=1,\ldots,T$ is the **inference iteration index**: small $t$ means
early denoising, not low noise. For $T>1$, its exact weighting is

$$
\alpha_t=\alpha_{\min}+
\frac{\alpha_{\max}-\alpha_{\min}}{\ln T}\ln(T-t+1),
\qquad
H_\ell=2\sum_{t=1}^{T}\alpha_t\mathbb E[X_{\ell,t}X_{\ell,t}^\top].
$$

It uses scalar OBS for weights/FFN neurons. For heads it aggregates
within-head saliencies using a head Hessian block, rather than solving every
candidate with §4.2's joint-group optimum. For shared multimodal heads it fuses
the two modality rankings:

$$
S_j^{\rm RRF}=\frac1{k+\operatorname{rank}_A(j)}+
\frac1{k+\operatorname{rank}_B(j)}.
$$

Module packages collect several Hessians per calibration trajectory.
Packages are processed sequentially, so later packages see earlier pruning.
No gradient recovery is required, but retained weights change.
[OBS-Diff, §§4.1–4.3](https://arxiv.org/abs/2510.06751).

**LTX adaptation:** collect curvature on saved D0 inputs at the exact sigmas.
For one step, use a positive constant weight; the $\ln T$ formula is undefined
at $T=1$. LTX's separate cross-modal attention branches do not automatically
share the same heads as image MMDiT. Apply rank fusion only to actually coupled
removal groups. Full-width normalization remains an export constraint.

### 4.5. MosaicDiff: different compact widths for different noise stages

MosaicDiff divides the trajectory, calibrates a width-pruned network for each
stage, then stitches those networks during sampling. It combines a closed-form
estimate of latent-MSE change with SNR:

$$
\operatorname{SNR}(t)=\frac{\bar\alpha_t}{1-\bar\alpha_t},\qquad
q(t)=\mathbb E[\operatorname{Grad}(t)]+\lambda\ln\operatorname{SNR}(t),
\qquad \tau=M\max_t q(t).
$$

The threshold defines three stages. Higher average stage scores receive less
pruning; specific sparsities are assigned from the paper's empirical schedule,
rather than solved by a global optimum. Each stage collects activations from
its own noised calibration data, including unconditional inputs when CFG is used.
It removes whole attention-head/MLP column groups and compensates retained
weights with the joint OBS calculation in §4.2. Here
$\bar\alpha_t=\prod_{s=1}^t(1-\beta_s)$ uses the diffusion noise schedule $\beta_s$.
[MosaicDiff, ICCV 2025, §§3.2–3.4](https://openaccess.thecvf.com/content/ICCV2025/papers/Guo_MosaicDiff_Training-free_Structural_Pruning_for_Diffusion_Model_Acceleration_Reflecting_Pretraining_ICCV_2025_paper.pdf).

**LTX adaptation:** stage-specific curvature motivates comparing per-sigma
rankings before pooling. Three saved one-step sigmas are three independent
conditions, not three consecutive sampler stages. A static exported mask must
satisfy all of them; routing among stage models defines a separate arm.

### 4.6. Diff-ES: search the allocation and reuse compensated weights

Diff-ES replaces manually chosen stage sparsities with evolutionary search.
Let $L_i$ be a discrete sparsity level for stage $i$, $f(L)$ its generated-image
fitness, and $G(L)$ its global sparsity. Its formulation is

$$
\hat L=\arg\max_L f(L)\quad\text{subject to}\quad G(L)\ge L_{\rm target}.
$$

Its level-switch mutation uses

$$
L_i\leftarrow L_i+\Delta,\qquad L_j\leftarrow L_j-\Delta,
\qquad \sum_i L_i=B.
$$

Candidates use lightweight TOPIQ, CLIP-IQA or SSIM fitness, depending on
the model. Stage-specific OBS pruning orders and compensated projection weights
are computed once. Search and inference retrieve the needed projections into
one shared backbone. This reduces full-model duplication; the projection
repository still has storage and transfer costs. The framework also accepts
other pruning backends, including depth pruning.
[Diff-ES, Mar 2026 preprint, §§3.1–3.3](https://arxiv.org/abs/2603.05105).

**LTX adaptation:** search allocation across layers/branches under measured
latency and quality constraints. Preserving a sum of sparsity levels preserves
compute only when stage costs and durations match. Use actual operation/time
costs for unequal layers, and keep all candidate selection off blind test actors.

### 4.7. EcoDiff: optimize complete-generation error with hard-concrete gates

EcoDiff's current title is **Learnable Sparsity for Vision Generative Models**.
Its structural gates cover attention heads and FFN neurons; they are direct
diffusion precedent for learning individual channel decisions.
Let $\mathcal F_\theta(z_T,y)$ be the complete sampler. Its objective is

$$
\min_m\mathbb E\left[
\|\mathcal F_\theta(z_T,y)-\mathcal F_{\theta,m}(z_T,y)\|_2\right]
+\beta\|m\|_0.
$$

For logit $\lambda_j$, sample $u_j\sim U(0,1)$ and define the hard-concrete gate

$$
s_j=\operatorname{sigmoid}\!\left(
\frac{\log u_j-\log(1-u_j)+\lambda_j}{a}\right),\qquad
\tilde m_j=\operatorname{clip}_{[0,1]}\bigl(s_j(\zeta-\gamma)+\gamma\bigr),
$$

where $a>0$ is temperature and $\gamma<0<1<\zeta$ are stretch endpoints.
The expected nonzero-gate penalty is

$$
\mathcal L_0=\sum_j\operatorname{sigmoid}
\left(\lambda_j-a\log\frac{-\gamma}{\zeta}\right).
$$

Timestep checkpointing recomputes activations for backward propagation.
Optimization learns masks; thresholding creates the discrete network.
The authors report 20% parameter pruning on SDXL/FLUX in about 10 A100 GPU-hours.
“No retraining after pruning” still includes mask optimization.
[EcoDiff, ICLR 2026, §§5.1–5.3](https://arxiv.org/html/2412.02852v2).

**LTX adaptation:** replace the unrolled sampler with the exact one-step map
and exclude the clean frame from the loss. Frozen-weight gate learning and
joint weight recovery must remain separate comparisons.

### 4.8. TinyFusion: select blocks for their recoverability

TinyFusion targets the post-recovery objective

$$
\min_m\min_{\Delta\Phi}\mathbb E_x\mathcal L(x,\Phi+\Delta\Phi,m).
$$

It divides depth into local groups of $M$ layers and retains $N$ per group.
This is an **N-of-M layer selection**, not hardware N:M weight sparsity.
For each group, enumerate its $\binom MN$ legal masks in matrix $\hat m$.
Candidate probabilities $p_i$ and independent noise $g_i\sim\operatorname{Gumbel}(0,1)$
generate a Gumbel-softmax sample:

$$
\pi_i=\frac{\exp((\log p_i+g_i)/\tau)}
{\sum_j\exp((\log p_j+g_j)/\tau)},\qquad
y=\operatorname{onehot}(\arg\max_i\pi_i),\qquad m=y^\top\hat m.
$$

A straight-through estimator uses the soft sample for backward propagation.
Local distributions factorize, $p(m)=\prod_k p(m_k)$.
LoRA updates $\Delta W=BA$ are co-optimized to estimate recoverability cheaply.
The highest-probability structures are retained, then the shallow model is
fine-tuned. DiT-XL/2 produces a 14-layer student with reported 2× speedup and
ImageNet FID 2.86 at less than 7% of pretraining cost.
[TinyFusion, CVPR 2025, §§3.1–3.2](https://openaccess.thecvf.com/content/CVPR2025/papers/Fang_TinyFusion_Diffusion_Transformers_Learned_Shallow_CVPR_2025_paper.pdf).

**LTX adaptation:** compare fixed-budget recovered masks with fixed-budget
deletion-error masks. Learnability does not validate our present deletion set.
Keep recovery data and iterations matched before attributing improvement to selection.

### 4.9. PPCL: replace a redundant interval with a distilled block

**Pluggable Pruning with Contiguous Layer Distillation** trains residual linear
probes $h+W_i h$ against teacher blocks $T_i$. With teacher inputs $H$, the
least-squares initialization is

$$
W_i^*=(T_i(H)-H)H^\top(HH^\top)^{-1}.
$$

It replaces candidate layers with probes during simulation. Let $C(u,k)$ be
CKA similarity between the teacher at $u$ and the surrogate at $k$.
The interval boundary follows

$$
\Delta(u,k)=-(C(u,k)-C(u,k-1)),\qquad
v=\min\{k-1:\Delta(u,k)>\Delta(u,k-1)\}.
$$

For $[u,v]$, initialize student block $S_u$ from $T_u$ and train it on
**teacher inputs**, with normalized endpoint loss

$$
\mathcal L_{[u,v]}=
\|\operatorname{Norm}(S_u(h^T_{u-1}))-
\operatorname{Norm}(h^T_v)\|_2^2.
$$

Independent interval training avoids feeding one interval's training errors into
the next. Its width stage replaces selected text-stream components and FFNs
with linear projectors; it is not ordinary neuron slicing. Optional global
fine-tuning follows assembly.
[PPCL, CVPR 2026, §§3.2–3.4](https://openaccess.thecvf.com/content/CVPR2026/papers/Ma_Pluggable_Pruning_with_Contiguous_Layer_Distillation_for_Diffusion_Transformers_CVPR_2026_paper.pdf).

**LTX adaptation:** distill a short sequence into one native multimodal block.
Teacher-input success needs an assembled-model check. Normalized features can
match direction while missing amplitude; include unnormalized final predictions.

### 4.10. HierarchicalPrune: position-aware removal and protected recovery

HierarchicalPrune measures quality change $\Delta P(i,c)$ when block $i$ or
component $c$ is removed. Its published position-weighted score is

$$
\operatorname{Score}(i,c)=-|\Delta P(i,c)|\,W_{\rm pos}(i),
\qquad W_{\rm pos}(i)=\exp\!\left(\frac{i-|B|}{|B|}\right).
$$

It preserves early semantic blocks, freezes protected retained parameters, and
distills final velocity plus intermediate features. For aggressive compression,
SGDistill scales **parameter updates**, not only feature-loss weights, inversely
with sensitivity. In common notation, for unfrozen components,

$$
\mathcal L=\mathcal L_{\rm feat}+\mathcal L_{\rm KD},\qquad
\Delta\theta_{i,c}\propto-
\frac{\eta}{|\Delta P(i,c)|}\nabla_{\theta_{i,c}}\mathcal L.
$$

The zero-sensitivity case needs a defined safeguard in any implementation.
Its reported large memory reductions also include W4A16 quantization.
[HierarchicalPrune, AAAI 2026, §2](https://arxiv.org/abs/2508.04663).

**LTX adaptation:** test freezing sensitive retained blocks at a fixed mask and
training budget. Position is a model-specific prior. A later block can still
be essential for motion or identity, and the score's sign/sort convention must
be checked before reproducing selection.

### 4.11. FastFLUX: fit cheap residual branches and recover their neighbors

FastFLUX replaces $h+f_i(h)$ with $h+A_i h+b_i$, preserving the shortcut.
For calibration columns $H$ and residual targets $F_i$, its stated fitting
objective is

$$
\min_{A_i,b_i}\|F_i-A_iH-b_i\mathbf1^\top\|_F^2.
$$

With $\bar H=[H;\mathbf1^\top]$, the ordinary least-squares solution is

$$
[A_i\ b_i]=F_i\bar H^\top(\bar H\bar H^\top)^{\dagger}.
$$

This is our derivation from the stated objective. The paper's displayed Eq. (1)
contains the fitted $g(X)$ inside its target and is not a clean closed-form
solution as written; do not copy it literally. Sandwich training fits the
replacement and LoRA adapters in neighboring retained blocks to match a
teacher three-block segment.
[FastFLUX, AAAI 2026 / preprint, §§3.1–3.2](https://arxiv.org/html/2506.10035v1).

**LTX adaptation:** branch replacement is a changed architecture with learned
weights. It needs its own reference and cannot claim parity with binary deletion.

### 4.12. DiT-Pruning: a DiT-specific sparse-weight score

The July 2026 **Post-Training Pruning for Diffusion Transformers** paper uses
a squared transformation of weights, **STW**:

$$
I_{ij}=|W_{ij}|^2\|X_j\|_2.
$$

Here $X_j$ is input channel $j$ over calibration tokens, using row-token notation
for this equation. This differs from Wanda's $|W_{ij}|\|X_j\|_2$ and from
squared Wanda, $|W_{ij}|^2\|X_j\|_2^2$.
Its clustering-aware granularity uses

$$
\bar I_i=\frac1{d_{\rm in}}\sum_j I_{ij},\qquad
G_{\rm out}=\frac{\operatorname{Var}_i(\bar I_i)}
{(\operatorname{Mean}_i\bar I_i)^2}.
$$

Low heterogeneity uses layer-wide ranking; high heterogeneity uses output-channel
comparison groups. The mask still removes **individual weights**:
$\tilde W=W\odot M$. “Per-channel” describes the ranking scope, not whole-channel
deletion. The authors report only 0.001 CLIP-score loss at 50% FLUX.1-dev
sparsity on MJHQ at 512×512.
[DiT-Pruning, Jul 2026 preprint v3, Eqs. (7)–(10)](https://arxiv.org/html/2607.00927v3).

**LTX assessment:** STW is an empirical score, not an exact Hessian derivation.
Squaring changes weight/activation trade-offs; a uniform scale change alone
cannot change within-layer ranking. A summed STW head/channel score would be
a new grouped adaptation. Dense matrices with zeros do not demonstrate speedup.

### 4.13. SVDtrunc: compress projections while retaining every block

The September 2026 **Importance-Aware Low-Rank Distillation** work probes one
block at a time using matched prompts/seeds and ranks sensitivity by CMMD.
It approximates each selected projection $W\in\mathbb R^{n\times m}$ as

$$
W\approx U_r\Sigma_rV_r^\top=BA,\qquad
B=U_r\sqrt{\Sigma_r},\quad A=\sqrt{\Sigma_r}V_r^\top.
$$

Storage changes from $nm$ to $r(n+m)$. For target reduction $c$, choose

$$
r=\left\lfloor\frac{(1-c)nm}{n+m}\right\rfloor.
$$

All projections in a block share a compression ratio. Less sensitive blocks
receive more compression through a linear schedule
$c_i=c_{\max}-\alpha\operatorname{order}(i)$ under the global parameter budget.
Recovery combines flow-target loss, teacher velocity loss and per-block feature
distillation. The work reports near-full FLUX performance at 68% retained parameters.
[SVDtrunc, Sep 2026, §§3.1–3.3](https://arxiv.org/html/2609.04646v1).

**LTX adaptation:** retain nonlinear blocks but factor their projections.
For a square $4096^2$ matrix, $r=1024$ halves weight elements; it also creates
two GEMMs and an intermediate. Measure actual latency. CMMD tolerates content
changes that our capture-conditioned identity/pose contract may reject; retain
paired fidelity and motion checks alongside distribution scores.

### 4.14. Dynamic capacity: DyDiT, DyDiT++ and E-DiT

DyDiT learns timestep-dependent head/channel masks and spatial FFN token masks.
In common residual notation, token $i$ uses

$$
h'_i=h_i+m^{\rm token}_i\operatorname{FFN}_{m^{\rm width}(t)}(h_i).
$$

Its budget loss controls mean FLOPs rather than static parameter count:

$$
\mathcal L=\mathcal L_{\rm diffusion}+w
\left(\frac1B\sum_b\frac{F_{\rm dynamic}(t_b)}{F_{\rm static}}-\lambda\right)^2.
$$

Routers train with Gumbel-sigmoid/straight-through gradients. Width choices can
be precomputed for a known timestep schedule; spatial choices remain input-dependent.
[DyDiT, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/a44a70acd5d0abc1a252ada9719dd06d-Paper-Conference.pdf).
DyDiT++ extends the framework to flow matching, Latte video and FLUX, and
adds timestep-dependent LoRA. Its token mechanism bypasses FFN computation,
so it should not be described as deleting the attention sequence wholesale.
[DyDiT++, extended paper, §§3–7](https://arxiv.org/abs/2504.06803).

E-DiT conditions each router on content and timestep. With block probability
$p_\ell$ and width distribution $q_\ell$, inference chooses

$$
g_\ell=\mathbf1[p_\ell\ge0.5],\qquad
s_\ell\in\{1/4,1/2,3/4,1\},\qquad
h_{\ell+1}=h_\ell+g_\ell(B_{\ell,s_\ell}(h_\ell)-h_\ell).
$$

It slices retained FFN rows/columns at inference. Gate and width losses are
$(\bar p-\rho_g)^2$ and $(\bar r-\rho_w)^2$, added to flow loss.
Borderline active blocks may instead reuse a cached residual from a prior step.
[E-DiT, Feb 2026 preprint, §§3.2–3.3](https://arxiv.org/html/2602.13993v1).

**LTX assessment:** dynamic computation can keep the full weight bank resident.
A fixed one-step sigma can freeze timestep-only choices; content-dependent
routing remains dynamic. Token bypass and residual caching change the execution
contract and cannot enter our first static full-attention comparison.

### 4.15. Modern video width pruning: PARE and MobileWan

PARE classifies attention heads by their intra-frame attention ratio

$$
r_h=\frac1{|\mathcal Q|}\sum_{q\in\mathcal Q}
\frac{\sum_{k\in\mathcal S(q)}A^{(h)}_{qk}}{\sum_k A^{(h)}_{qk}},
\qquad
\tilde s_h=\begin{cases}\alpha_{\rm temp}s_h,&r_h<0.3,\\s_h,&\text{otherwise},\end{cases}
$$

where $\mathcal S(q)$ contains keys in the query's latent frame.
Its $s_h$ pools $\|W_{O,h}\|_F\|a_h\|_F$ with high-noise timestep weights
$w(t)=\log(1+t/T)$. The temporal multiplier exceeds one.
FFN neuron scores are

$$
s_j=\|W_{{\rm up},j,:}\|_2\|W_{{\rm down},:,j}\|_2.
$$

Greedy selection skips neurons whose concatenated up/down weight signatures
have excessive cosine similarity to retained ones, then relaxes the threshold
until the aligned width budget is filled. Width distillation precedes dynamic
block-router training.
[PARE, May 2026 preprint, §3.2](https://arxiv.org/html/2605.27336v1).

An aligned retained width does not imply fixed channel groups sharing gates.
PARE selects neurons individually using scores and diversity.

MobileWan instead measures projected output energy and normalizes within blocks:

$$
s_{b,h}=\mathbb E\|W^{\rm out}_{b,h}a_{b,h}\|_2^2,\qquad
\tilde s_{b,h}=\frac{s_{b,h}}{H_b^{-1}\sum_j s_{b,j}}.
$$

Its learned alternative jointly optimizes gates and weights:

$$
g_{b,h}=\operatorname{sigmoid}(\alpha_{b,h}/\eta),\qquad
\mathcal L=\mathcal L_{\rm FM}+\lambda\sum_{b,h}g_{b,h}.
$$

It anneals temperature, hardens gates, then recovers the fixed architecture.
During gate learning, $\sigma\sim\operatorname{LogitNormal}(1.5,1)$;
final recovery returns to $\operatorname{LogitNormal}(0,1)$.
[MobileWan, Jul 2026 preprint, §3.1](https://arxiv.org/html/2607.06173v1).

**LTX assessment:** projected energy includes within-head covariance; PARE's
factored norm product does not. For $y=\sum_h c_h$, joint local deletion error is

$$
\mathbb E\left\|\sum_{h\in R}c_h\right\|_2^2
=\sum_{h\in R}\mathbb E\|c_h\|_2^2+
2\sum_{h<j;\,h,j\in R}\mathbb E\langle c_h,c_j\rangle.
$$

Neither scalar ranking captures all cross-head interactions or downstream
damage. Compare these scores at the same mask budget before adding temporal
protection, recovery or a changed noise distribution.

### 4.16. Latest scope boundaries: DIET expert pruning and TAPE token bypass

DIET measures the response of removing each MoE expert **after re-routing**:

$$
\Delta_{\ell,e}(x)=y_{\mathcal E\setminus\{e\}}(x)-y_{\mathcal E}(x),
\qquad
s_{\ell,e}=\operatorname{Concat}_c
\left[\frac1{|\mathcal T_c|}\sum_{i\in\mathcal T_c}\Delta_{\ell,e}(x_{c,i})\right].
$$

It caches all expert outputs/router states once and replays layer-local deletions.
For retained expert set $K_\ell$ of size $k_\ell$, its diversity loss is

$$
K_\ell^*=\arg\min_{|K_\ell|=k_\ell}
\sum_{e\notin K_\ell}\min_{f\in K_\ell}
\left(1-\frac{\langle s_{\ell,e},s_{\ell,f}\rangle}
{\|s_{\ell,e}\|_2\|s_{\ell,f}\|_2}\right).
$$

Local combinatorial search selects experts; regression guides budgets across
layers. On LingBot-Video 30B-A3B, 50% expert removal gives 57→30 GB checkpoints
and reported VBench Total 0.7941→0.8115 on 284 cases without fine-tuning.
All-expert calibration is costly: roughly 16× normal expert computation and
375 GB cached states in its reported setup. Frozen-state replay is not an exact
full-network prediction after multiple deletions.
[DIET, Sep 2026 preprint, §§3–4](https://arxiv.org/html/2609.37829v1).

**LTX assessment:** dense LTX has no MoE expert bank. Its deletion-response
and diversity ideas motivate controls, but routing-aware expert pruning cannot
be relabeled as ordinary dense FFN pruning.

TAPE stabilizes token selection across adjacent frames with

$$
\tilde s_i^{n}=\alpha s_i^{n}+(1-\alpha)s_i^{n-1},\qquad \alpha=0.5.
$$

It reselects important tokens in chosen layers and uses timestep-dependent
budgets, skipping computation for low-ranked tokens while retaining the full
output lattice for reconstruction. No learned weights or parameter deletion
are required.
[TAPE, May 2026 preprint, §§4.1–4.3](https://arxiv.org/html/2605.17837v2).
This is related video-computation work. It changes token execution and does
not supply a smaller static LTX checkpoint.

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

All scores below refer to one fixed checkpoint and one fixed block/branch.
An activation is the value consumed by that branch's output projection, not
an attention probability or the original block input.

Our current FFN channel score uses post-activation RMS and output-column norm:

$$
I_j=\sqrt{\mathbb{E}[a_j^2]}\,\|w_j\|_2.
$$

A head of width $d_h$ uses the factored proxy

$$
I_h=\sqrt{\frac{\mathbb{E}[\|a_h\|_2^2]}{d_h}}\,\|W_h\|_F.
$$

**Symbols in the screening equations**

| Symbol | Type | Definition and shape |
|---|---|---|
| $j$ | Scalar (integer index) | Index of an FFN intermediate channel. Each channel is one scalar per token. |
| $a_j$ | Scalar | Post-activation FFN value in channel $j$, immediately before the FFN output projection. It varies with the input clip, sigma and token. |
| $w_j$ | Vector | Column $j$ of the FFN output-projection matrix. It is a vector with one entry per FFN output channel. Thus $a_jw_j$ is this channel's local output contribution. |
| $I_j$ | Scalar | FFN screening score: activation RMS multiplied by the Euclidean norm of its output column. |
| $h$ | Scalar (integer index) | Index of an attention head within one attention branch. |
| $d_h$ | Scalar (integer count) | Number of channels in head $h$, before the attention output projection. It is not the full model width or the number of tokens. |
| $a_h$ | Vector | Vector of shape $d_h$ consumed by the attention output projection for head $h$ at one token. It already includes the native head gate when enabled. |
| $W_h$ | Matrix | Output-projection column slice belonging to head $h$, with shape $D_{\rm out}\times d_h$. It maps head channels to attention output channels. It is the same slice called $W_{O,h}$ in §6.1.1. |
| $D_{\rm out}$ | Scalar (integer count) | Number of channels in the attention output, after the output projection. Heads in the same branch share this width. |
| $I_h$ | Scalar | Factored head-screening proxy: RMS across head channels and calibration observations, multiplied by the Frobenius norm of $W_h$. |
| $E_h$ | Scalar | Root mean squared vector norm of the projected head output on the explicitly selected tokens; unlike $C_h$, it does not divide energy by output width. |
| $\mathbb E[\cdot]$ | Operator | Mean over the specified calibration observations. The production proxy first averages sampled tokens within each clip/sigma pair, then gives each pair equal weight. |
| $\lVert v\rVert_2$ | Scalar | Euclidean vector norm: $\sqrt{\sum_r v_r^2}$. The index $r$ enumerates vector coordinates. |
| $\lVert W\rVert_F$ | Scalar | Frobenius matrix norm: $\sqrt{\sum_{r,s}W_{rs}^2}$. Here $r,s$ enumerate rows and columns of the matrix. |
| RMS | Operator | Root mean square: square values, average them over the stated axes, then take the square root. It is not the mean absolute value. |

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

$W_ha_h$ is a vector with $D_{\rm out}$ entries. $E_h$ is the root mean of its
squared vector norm over the selected tokens. The helper uses the explicit
token indices passed to it for one forward. It sums output-channel squares
inside the norm; unlike the explorer's $C_h$, it does not divide by
$D_{\rm out}$. With the same observations, $E_h=\sqrt{D_{\rm out}}C_h$.

It retains within-head covariance and resembles MobileWan's projected-contribution
screen. It still ignores cross-head cancellation and downstream deletion damage.
The default calibration CLI does not select masks using this helper.
See [estimators](score/estimators.md), [hooks](score/hooks.md) and
[native calibration](score/whole_clip_d0_scores.md).

#### 6.1.1. Whole-clip head contribution for the attention explorer

The explorer's head-ranking definition is **whole-clip projected output RMS**.
For one fixed clip, block, attention pathway, sigma and conditioning branch,
let $\mathcal Q$ contain all query tokens and $\mathcal K$ all key tokens.
The mask specifies which query/key pairs are permitted. For head $h$, define

$$
A_h(q,k)=\operatorname{softmax}_{k\in\mathcal K}
\left(\frac{Q_h(q)^\top K_h(k)}{\sqrt{d_h}}+M(q,k)\right),
\qquad
z_h(q)=g_h(q)\sum_{k\in\mathcal K}A_h(q,k)V_h(k),
\qquad
c_h(q)=W_{O,h}z_h(q).
$$

**Indices, token sets and dimensions**

| Symbol | Type | Definition |
|---|---|---|
| $\sigma$ | Scalar | Exact noise level for this independent forward: zero is clean and one is pure noise under the saved linear noising rule. The current explorer records 0.4, 0.7 and 0.9 separately, using the same saved noise. |
| $h$, $H$ | Scalars (integers) | Head index and total head count in the selected attention pathway. Indices run from $0$ to $H-1$. |
| $q$ | Scalar (integer index) | Query-token index, from zero through $N_Q-1$. |
| $\mathcal Q$ | Set | All query-token indices in the selected clip/pathway. These queries produce the attention output being measured. |
| $N_Q=\lvert\mathcal Q\rvert$ | Scalar (integer count) | Number of query tokens. |
| $k$ | Scalar (integer index) | Key-token index, from zero through $N_K-1$. |
| $\mathcal K$ | Set | All key-token indices. These tokens supply the information attended to. |
| $N_K=\lvert\mathcal K\rvert$ | Scalar (integer count) | Number of key tokens; it can differ from $N_Q$ in cross-attention. |
| $d_h$ | Scalar (integer count) | Head width. In this LTX capture, each head's query, key and value vectors have this width. |
| $D_{\rm out}$ | Scalar (integer count) | Width of the attention output vector after projection; this is the divisor used to convert projected energy into per-channel RMS. |
| $r$ | Scalar (integer index) | Output-channel index, running from $1$ to $D_{\rm out}$ in the component-wise score formula below. It is separate from the FFN channel index $j$ in §6.1. |

For video-to-video attention in the current saved DNA clip,
$N_Q=N_K=18\times32\times32=18{,}432$, $H=32$, $d_h=128$ and $D_{\rm out}=4096$.
The 18 frames here are latent frames, derived from 137 RGB frames. In
video-to-text attention, $N_Q=18{,}432$ and $N_K=1024$ tokenizer slots,
including padding. These dimensions are read from the selected pathway;
they are not constants for all clips or attention types.

**Vectors, matrices and gates**

| Symbol | Type | Definition and shape |
|---|---|---|
| $Q_h(q)$ | Vector | Query vector of shape $d_h$ for query token $q$, after any executed query normalization and positional encoding. Capital $Q_h$ is a tensor; calligraphic $\mathcal Q$ is a token-index set. |
| $K_h(k)$ | Vector | Key vector of shape $d_h$ for key token $k$, after any executed key normalization and positional encoding. It is distinct from the index set $\mathcal K$. |
| $V_h(k)$ | Vector | Projected value vector of shape $d_h$ at key token $k$. It carries the information that attention weights combine. |
| $M(q,k)$ | Scalar | Additive attention mask: zero for a permitted pair and $-\infty$ for a blocked pair. It is zero everywhere for unmasked attention. |
| $A_h(q,k)$ | Scalar | Scalar attention probability for this query/key pair. Each permitted query row sums to one over keys. The full head map has shape $N_Q\times N_K$. |
| $\operatorname{softmax}_{k\in\mathcal K}$ | Operator | Exponentiate the scaled, masked query/key scores and normalize their sum over keys for each fixed query and head. |
| $\ell_h(q)$ | Scalar | Scalar gate logit produced by the existing model for head $h$ at query token $q$. It depends on the query input. |
| $g_h(q)$ | Scalar | Executed scalar head gate. It equals $2\operatorname{sigmoid}(\ell_h(q))$ for LTX gated attention and one if gating is disabled. It is not a learned pruning mask introduced by this diagnostic. |
| $\operatorname{sigmoid}(u)$ | Scalar | Logistic function $1/(1+\exp(-u))$. The argument $u$ is a real scalar. |
| $z_h(q)$ | Vector | Gated attention result of shape $d_h$, immediately before output projection: a gate times the attention-weighted sum of values. This is $a_h$ from §6.1 evaluated at query $q$; do not apply the gate twice. |
| $W_O$ | Matrix | Full output-projection weight, with shape $D_{\rm out}\times(Hd_h)$ for equal-width heads. It maps concatenated head activations to the attention output. |
| $W_{O,h}$ | Matrix | Slice of $d_h$ columns from $W_O$ for head $h$, with shape $D_{\rm out}\times d_h$. |
| $c_h(q)$ | Vector | Projected contribution vector of shape $D_{\rm out}$ for head $h$ and query $q$. Its entries can be positive or negative. |
| $b_O$ | Vector | Shared output-projection bias vector of shape $D_{\rm out}$. It cannot be uniquely assigned to individual heads. |
| $y(q)$ | Vector | Combined attention output vector of shape $D_{\rm out}$: $b_O+\sum_hc_h(q)$, before later block operations. |
| Superscript $\top$ | Operator | Vector or matrix transpose. Thus $Q_h(q)^\top K_h(k)$ is a scalar dot product. |

The shared output bias is excluded from head scores. This definition stops
before the block's residual modulation and any later model layers.

The whole-clip energy and displayed contribution score are

$$
s_h=\frac{1}{|\mathcal Q|}\sum_{q\in\mathcal Q}\|c_h(q)\|_2^2,
\qquad
C_h=\sqrt{\frac{s_h}{D_{\rm out}}}
=\sqrt{\frac{1}{|\mathcal Q|D_{\rm out}}
\sum_{q\in\mathcal Q}\sum_{r=1}^{D_{\rm out}}c_h(q)_r^2}.
$$

**Energy, contribution and averaging**

| Symbol | Type | Definition |
|---|---|---|
| $c_h(q)_r$ | Scalar | Output-channel $r$ of the projected contribution vector for head $h$ at query $q$. |
| $\lVert c_h(q)\rVert_2^2$ | Scalar | Sum of squared contribution values over all output channels for one query. This is projected energy, not a sum of attention probabilities. |
| $s_h$ | Scalar | Mean projected energy over all $N_Q$ queries for this one clip/block/pathway/sigma/branch. It sums channel squares but averages query tokens. |
| $C_h$ | Scalar | Whole-clip contribution shown by the explorer: RMS over both query tokens and output channels. It has activation units; $s_h$ has squared activation units. |
| $\lvert\mathcal Q\rvert$ | Scalar (integer count) | Number of query tokens. Dividing by it prevents clip length alone from multiplying the energy score. |
| $s_h/D_{\rm out}$ | Scalar | Mean squared projected contribution per output channel and query. Taking its square root gives $C_h$. |

**Worked arithmetic example.** Let one head have two query tokens and two output
channels, with $c_h(0)=(3,4)$ and $c_h(1)=(0,0)$. Then $N_Q=2$,
$D_{\rm out}=2$, $s_h=(25+0)/2=12.5$ and $C_h=\sqrt{12.5/2}=2.5$.
The vector-norm score is $E_h=\sqrt{12.5}\approx3.536$ on those same tokens.
Even if the selected query has zero projected contribution, the whole-clip
score and rank stay the same; its attention probabilities still sum to one.

Rank heads by descending $C_h$,
with head index as the stable tie breaker. The score uses every query token;
changing the displayed query must not change this ranking. Compute separate
scores for each block, pathway, sigma and branch. Do not pool the three sigma
levels or rank heads from different output spaces together.

This is the full-query version of §4.15's MobileWan projected-energy screen,
with the LTX head gate included. For fixed $D_{\rm out}$, $s_h$, $C_h$ and §6.1's
$E_h=\sqrt{s_h}$ give the same ordering on the same tokens. MobileWan's
within-block mean normalization also preserves that order. It does not make
scores a percentage of explained output energy. PARE's factored norm proxy
does not retain the same within-head covariance. Its temporal protection is
a separate pruning choice and is not added to this contribution definition.

The measurement scope differs from MobileWan's pruning procedure. Its heuristic
score averages over a sample set and diffusion timesteps, then normalizes by
the block's mean head energy to rank self-attention heads globally across
blocks. The explorer measures one recorded clip at each exact sigma, ranks
within one block/pathway, and applies the definition to all six pathways.
It does not use that global allocation or perform pruning/recovery training.
For a fixed query count and output width, taking RMS instead of total projected
energy changes units but preserves head order. LTX's existing input-dependent
head gate is part of its native activation; it is distinct from MobileWan's
separately learned scalar pruning gate (§4.15).

Do not sum heatmap brightness to obtain contribution. Each head's attention
weights sum to one for each query; summing over the whole clip gives the same
$|\mathcal Q|$ for every head. The explorer also scales each displayed panel
by its own maximum. Its Q/K-only dumps cannot compute $C_h$: capture the
executed gated head outputs and their output-projection weights, or accumulate
equivalent exact projected statistics during a new forward. The ranking
definition above does not imply that existing Q/K artifacts contain scores.
The explorer's separate `score` command now accumulates these full-query
statistics from the native gated input to each attention output projection.
It writes scalar sidecars with input/noise/context/checkpoint provenance and
keeps Q/K arrays intact. See the
[explorer guide](../../../../scripts/ltx_diag/attention_explorer/README.md#whole-clip-contribution)
for the command and capture checks.

The explorer includes all queries in its recorded clip. Pruning calibration
in §6.1 instead samples tokens and excludes the conditioned clean first frame;
record that different token set when comparing scores. For multiple calibration
clips or sigmas, average $s_h$ with an explicit weighting rule before taking
the square root, rather than averaging per-clip RMS values.

Whole-clip contribution measures local output magnitude. Cross-head inner
products still affect joint deletion error (§4.15), and downstream layers can
amplify or suppress a local change. A large score is not proof that a head
improves the final video. Centering the outputs for FLAP-style mean compensation
(§7) defines another score and must be evaluated separately.

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

The retired October screen and depth-export observations are preserved in the
[historical record](../../../plans/HISTORY.md#october-compression-and-conditioning-record).
The source reports and native receipts are unavailable at their original paths.
The October roadmap later records native depth parity passing, superseding its
earlier pending status. That dated observation does not qualify quality, recovery
or speed; the deletion set retained its failed functional quality gate.

The [dormant research backlog](../../../plans/RESEARCH_BACKLOG.md) retains compact
depth recovery, aligned-FFN reconstruction and explicit motion conditioning as
alternatives. It defines no current execution priority. Preserving D0 remains
separate from learning the deployment task, whose future capture is unavailable.


## 7. Methods we plan to test

The production workflow supports RMS scoring, uniform allocation, functional
masks, export controls and matched evaluation; exact projected head energy also
exists as a helper. The broader agenda below separates production capabilities
from study-specific implementations. Gradient, reconstruction/rotation and recovery
training methods remain unmeasured; results from a training-free screen do not
establish their performance.

For the current human-region question, the recommended next experiment is
**static sparse head gates with bounded LoRA recovery** (§7.5.1).
This is a proposed experiment, not an implemented learner or accepted candidate.
The [2,000-view partial report](../../../../expr/refiner_prune/2.5/mobilewan_head_scores_dev8_20261011/partial/milestone_2000/REPORT.md)
shows stable descriptive energy ranks, but measures no removal quality.
The resident MobileWan scorer covers every generated query at stride one.
More energy observations alone cannot establish that a head can be removed.

For a separate FFN study, start with **individual binary channel gates and
constrained final widths** (§7.5.2). Keep attention unchanged in that study.
The new literature supports this design; it does not provide an LTX result.
The linked head experiment plan remains head-only.

The earlier method screen uses **distilled** weights and 19 latent frames.
The current head study uses **dev** weights and complete 18- or 28-frame clips.
Its earlier quality and timing results motivate controls; they are not measured
outcomes for the current dev task.

The retired 1 October method-comparison plan covered rankings, compensation,
allocation and static block removal with isolated calibration, validation and blind actors.
Its source attribution is in the [workspace history](../../../../plans/history/README.md#dormant-proposals).
The former report is unavailable at its original path; this historical agenda does not establish current measured outcomes.

### 7.1. First establish executable dense savings

Hold structures fixed; compare full-width masking with genuinely smaller dense
execution. Start FFN-only, then heads-only, then combined. Profile GEMMs,
attention, head selection, copies, allocations and complete forward time.
Loading/noising/VAE remain outside forward timing; end-to-end cost is separate.

Test aligned retained widths. The present 16,384-channel FFN retains 14,746 channels
at the current 10% request. Retaining 14,848 instead removes 1,536 (9.375%) and
preserves a multiple of 128. Alignment is a **hardware hypothesis**, not a proven
optimum: scan nearby widths and head counts on the actual GPU. Every ranking
method must use the same feasible widths and realized budgets. Individual channel
gates can meet this constraint without sharing gates in predefined groups.

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

**EcoDiff / MobileWan-inspired learned gates.** Optimize head/channel
gates against one-step final-latent agreement and sparsity, harden them at matched
final budgets, then test the exported
discrete network. Compare equal-sigma with high-noise-biased sampling, while
validating all sigmas. Separate frozen-weight mask learning from joint gate/weight
training. GRASPrune supports a separate hard-forward budgeted variant.
CoFi and Sheared LLaMA support individual channels with shape constraints;
Compresso supports joint masks and adapters (§2.7).
Soft training gates are not an inference speed result.

#### 7.5.1. Recommended next experiment: sparse head gates with bounded recovery

**Status: historical proposal.** Learned-head training owners are implemented;
their current contracts and limits are in the run guide and validation docs.
The [workspace history](../../../../plans/history/README.md#stopped-learned-head-lineages)
records the retired experiment plan and its distinct stopped lineages.
Use a new reviewed study specification for fresh preparation, not this historical agenda's defaults.

**Question.** Can learned head choices preserve people better than energy choices
at the same executable budget and recovery cost?

Local contribution, downstream damage and recoverability are different quantities.
Attention to a human region is a reason to inspect a head, not proof of necessity.
The explorer scales each attention panel by its own maximum (§6.1.1).
Small values, the native gate or output projection can suppress a visible map.
Check the same block, pathway, query set, clip and sigma before comparing scores.
The explorer includes clean-frame queries; this pruning study excludes them.

**Selection variables.** Add one scalar pruning decision per original video
self-attention head: 48 blocks × 32 heads, or 1,536 decisions.
Apply it after the native input-dependent gate and before `to_out[0]`.
Keep the native gate, full Q/K normalization and original RoPE identities.
Keep all text cross-attention heads and FFN channels for this first experiment.
The learned decisions are static across clips, queries and sigmas at deployment.

Following the 12 October gate-policy amendment, initialize every FP32 logit at
$\alpha_{b,h}=+3$ and use continuous sigmoid gates with fixed temperature one:

$$
g_{b,h}=\operatorname{sigmoid}(\alpha_{b,h}),\qquad
g_{b,h}(0)=0.9525741268\ldots,\qquad
L_{\rm sparse}=\frac1{1536}\sum_{b,h}g_{b,h}.
$$

Use ordinary sigmoid gradients for 700 updates, with no gate noise or temperature
annealing. A positive mean-gate penalty encourages suppression in this soft phase.
The soft forward has no fixed binary head count. An exact count would make a
head-count penalty constant and remove its selection pressure.
The initialized gates are near one, rather than an exact dense reproduction;
verify native parity separately with explicit all-one/zero-adapter gates.

After 700 updates, save soft/hardened predictions with identical adapter weights.
For the next 300 updates, select
$\bar m=\operatorname{SelectLegal}(\alpha,K)$ at the declared final budget and use
$m=\operatorname{stopgrad}(\bar m-g)+g$: binary forwards with biased STE gradients.
Disable the soft sparsity penalty in this fixed-count phase. Freeze the final
mask after update 1,000, then perform 500 compact adapter updates.
`SelectLegal` is a proposed legal-count selector, not an existing API.
The plan starts with retained counts $\{16,20,24,28,32\}$ per block and global
budgets near 10%, 23% and 33% removal. Record actual counts.
Retained pruning gates become one; removed heads become zero.
Use stable original-ID ties and the same feasible counts in all comparison arms.

Soft learning can produce uniform attenuation or adapter compensation rather
than useful discrete choices. Plot gate distributions and measure the soft-to-hard
quality change before recovery. Do not fold soft amplitudes into retained weights
or interpret logits as causal importance. Search all eligible heads; energy
supplies a comparator. Compare a 1,000-soft/500-compact control at the same total
updates. The FFN proposal in §7.5.2 remains a separate experiment.

**Bounded recovery.** Freeze the pretrained backbone and jointly train the head
logits with a small, fixed LoRA adapter on declared attention projections.
Keep its rank, targets, updates and data budget fixed across comparison arms.
Match total adapter updates, including selection and final compact recovery.
The energy arm receives the same adapter updates with its fixed-ranking ramp;
its gate trajectory differs from learned soft suppression. Match final budgets
and adapter updates, and report that difference.
This estimates which sets work with limited adaptation, rather than only measuring
their immediate frozen-weight damage. It does not establish optimal recoverability.
MobileWan jointly trains gates and weights (§4.15); TinyFusion motivates selection
with recovery (§4.8). This soft-selection/binary-export/LoRA combination is an LTX adaptation.
Use a short frozen-weight gate control to separate selection from adaptation.
Also freeze the learned and energy masks, discard selection adapters, and give
each fresh identically initialized LoRA recovery for 500 updates. Include energy
identities at the learned block counts and an unpruned fresh-adapter control.
Report their additional cost; they isolate mask quality under common recovery.

**Official training loss.** Use native velocity MSE against the saved noise minus
clean capture, as in the local official trainer's
[`FlexibleStrategy`](../../../packages/ltx-trainer/src/ltx_trainer/training_strategies/flexible.py).
The original dev teacher supplies evaluation references; it is not the target.
For clean capture $y$, noise $\epsilon$, native velocity $v_S$ and generated-token
set $Q$, define

$$
v^*=\epsilon-y,\qquad
L_v=\frac1{C|Q|}\sum_{q\in Q}\|v_{S,q}-v^*_q\|_2^2,\qquad
L=L_v+\lambda_s(t)L_{\rm sparse}(g).
$$

Exclude c0, normalize per clip and reduce errors in FP32. Obtain native velocity
through a new public helper in `core.ltx_adapter`; do not divide rounded BF16
clean outputs by sigma for the training loss. Verify its stock clean conversion
against ordinary inference. The baseline adds no teacher-x0, direction, regional
or temporal loss. The adopted target/reduction does not reproduce every part of
official training: this plan retains its fixed D0 inputs and eight sigmas.

Use $\lambda_s(t)=10^{-3}\min(t/300,1)$ during soft updates only. Binary refinement
and compact recovery use $L_v$ alone. This initial coefficient must be checked
against actual velocity gradients before confirmation; it is not an optimal value.
Compare a zero-penalty control. Exact final counts constrain binary forwards,
rather than soft-gate mass.

**Specific human-protection comparisons.** Keep them separate from the baseline.
On learning GT only, define six motion/size buckets using motion tertiles and
median person area. Compare bucket/actor-balanced draws with the original draw
policy, preserving official loss and paired evaluation inputs.
Then compare $L_v$ with an added person velocity MSE, coefficient one.
Against that person-only arm, test added face and mean-valid-hand velocity MSE,
each coefficient 0.5. Normalize each region per clip using fixed GT coverage;
map verified face/hand annotation boxes through the recorded crop and VAE pooling.
The plan defines 25% box expansion, 32-pixel minimum crops, absent-region rules,
draw cycles, artifacts and one-factor comparisons. These reductions extend the
stock unweighted loss; they are hypotheses motivated by specialized contributions.
No semantic label ties a group of heads to one switch.

A dense model can have nonzero official velocity loss and gradients.
Verify zero loss only for predictions equal to the exact velocity target.
Required decoded person, face and hand checks can reject low-loss candidates.
Official training may improve capture fidelity while changing original-teacher
behavior; keep both measurements and the unpruned adapter control.

**Distribution and splits.** Initially sample the eight exact sigmas from the
current dev study equally. Preserve complete 18- and 28-frame clips, clean c0,
empty context, white objective and full bidirectional attention.
Compare high-noise emphasis only as a later controlled variable; Wan's sampling
distribution is not automatically the LTX D0 distribution.
Cover multiple actors and motions. Use multiple noise draws with saved identities,
and pair every teacher/student call on the same draw.
Use actor-disjoint learning, validation and blind-test sets.
An actor inspected for selection or tuning is no longer blind.
Recompute the energy comparator from the same learning membership, excluding
validation actors. Do not reuse an all-train pooled mask across a new split.

**Code and feasibility gate.** Reuse `data.whole_clip` for saved inputs and
`score.hooks` for the intervention boundary. Add a prune-owned learning entrypoint
and registered logit/adapter state; the current raw mask tensors are not a complete
bounded-gate learner. Their initializer detaches supplied tensors.
`core.session.Session.transformer` surrounds its lifetime with `torch.no_grad()`.
The student path needs explicit gradient enablement and backbone freezing.
Teacher references use `no_grad`, pruning disabled and adapter delta zero.
Generate required original-teacher velocity/clean records once and reuse exact
inputs across arms and optimization replicates in a capped shared store.
Official training needs no teacher forward or cached target for every draw.

Before sustained training, verify all-one/zero-adapter baseline reproduction,
exact +3 initialization, finite velocity and sparsity gradients, unchanged
frozen weights and exact c0.
Check complete-clip backward memory at both lengths with activation checkpointing.
Small optimizer state does not make backward propagation through 48 blocks cheap.
Do not substitute shorter training segments without recording a distribution change.
Also benchmark a few candidate compact widths on the current dev geometry.
Include aligned FFN-only widths before choosing the next speed study.
Full Q/K normalization keeps those projections full-width; head reduction does
not imply proportional parameter or latency reduction.

**First comparison.** Begin near 10% removal, using a feasible realized budget.
Compare the energy-selected and learned masks with the same adapter budget,
input draws and compact execution policy. Include the unchanged model and a seeded
random-mask control. Compare gate-only learning separately from joint learning.
Measure both hard-mask quality before final recovery and quality after matched recovery.
Expand to 23% and 33% only after the first budget passes the declared gates.
Use a few single-head removals as sanity checks; an exhaustive deletion sweep is
not a prerequisite for this learner.

**Compact recovery and acceptance.** Freeze the selected binary mask, create the
actual reduced-width network, and complete the bounded recovery in that network.
Retained pruning decisions become one; removed heads disappear.
Merge LoRA before timing, or include its measured runtime overhead.
`compact_faithful` remains a storage/fidelity control, not the speed candidate.

Report raw functional-mask versus compact-export parity before recovery.
If that fails, preserve the failed result. Training the compact network defines
a changed-model arm, not a repair of the original parity claim.
For that arm, use the trained in-memory compact network as the new reference and
check saved/reloaded execution with the unchanged 0.02 tolerance.
Evaluate its quality separately against both the original teacher and capture.
Keep full Q/K normalization and original head identities in either arm.

Inspect synchronized decoded videos on unseen actors and independent noise at
every required sigma. Report foreground and face/hand damage alongside motion,
whole-frame errors and worst-case failures. Measure complete forward latency,
memory and storage separately on the target GPU, using the existing ABA/BAB gate.
Accept the hard compact candidate only when execution fidelity, human quality and
the declared cost goal all pass. If head widths offer no useful measured speed,
use a promising aligned-FFN profile to prepare the separate channel study;
otherwise test recoverable static depth. Soft loss alone does not justify a
larger head-mask speed study.

#### 7.5.2. FFN recommendation: individual choices, aligned widths and bounded recovery

**Implementation status, 18 October 2026:** individual binary FFN masks,
joint head/FFN training, resumable state and coupled export are implemented.
Use the [FFN plan](../../../../plans/2026-10-18-learned-ffn-gate-mask-pruning.md)
and [entrypoint design](learn/ffn.md) for the executable recipe.
The first formal preset adds FFN masks alongside the established head recipe.
The separate FFN-only comparison below keeps every attention head.
Native acceptance, trained quality and speed require their own saved evidence.

**Question.** Can learned channel choices preserve people better at the same FFN width?

**Gate and weights.** For one token vector $x$ in video block $b$, let
$W_{{\rm up},b}\in\mathbb R^{H_b\times D_{\rm in}}$ and
$W_{{\rm down},b}\in\mathbb R^{D_{\rm out}\times H_b}$. LTX uses GELU:

$$
a_b=\operatorname{GELU}(W_{{\rm up},b}x+b_{{\rm up},b}),\qquad
y_b=W_{{\rm down},b}(m_b\odot a_b)+b_{{\rm down},b}.
$$

There is one FP32 logit $\alpha_{b,j}$ for each original intermediate channel.
The current 48 video FFNs with width 16,384 give 786,432 decisions.
The mask is static across tokens, clips and sigmas at deployment.
It acts after GELU, immediately before `ff.net[2]`.

For the first pilot, fix the same retained width $K_b=14{,}848$ in every block.
This removes 9.375% of eligible FFN channels, rather than 9.375% of the model.
Use binary forwards with the surrogate gradient

$$
p_{b,j}=\operatorname{sigmoid}(\alpha_{b,j}/\tau),\qquad
\bar m_b=\operatorname{TopK}(p_b,K_b),\qquad
m_b=\operatorname{stopgrad}(\bar m_b-p_b)+p_b.
$$

`TopK` returns a binary keep mask with stable, recorded tie handling.
Its gradient estimator is approximate. Start with all channels present, then
ramp removal through legal widths; reuse the same decision during recomputation.
A fixed hard width needs no additional sparsity penalty to enforce that width.
Hard-concrete masks with explicit width constraints remain an alternate estimator,
not the default first pilot. Validate the final binary budget in either case.

The final-width constraint is $\sum_j\bar m_{b,j}=K_b$, with
$K_b\in128\mathbb N$ and a declared positive minimum.
This does **not** impose $m_{b,j}=m_{b,k}$ on neighboring channels.
First hold all $K_b$ fixed to isolate selection. Later compare width allocation
over a predeclared legal set, using the same total parameter budget:

$$
C_{\rm FFN}=\sum_b K_b
\left(D_{{\rm in},b}+D_{{\rm out},b}
+\mathbf 1_{\text{up bias},b}\right).
$$

This counts the retained channel-dependent weights and up biases; down biases
remain unchanged. Count all other parameters separately. Parameter cost is not
latency; benchmark each candidate architecture and complete D0 calls (§8).
Alignment and the initial width remain hardware hypotheses (§7.1).

**Recovery and loss.** Freeze pretrained weights and jointly learn logits with
rank-16 LoRA on `ff.net[0].proj` and `ff.net[2]`, with scale $16/16=1$.
For each targeted weight, use $W'=W+sBA$, where $A\in\mathbb R^{r\times d_{\rm in}}$
and $B\in\mathbb R^{d_{\rm out}\times r}$. Initialize $A$ with seeded random
values and $B=0$; record the seed and initializer. Keep adapter parameters and
optimizer states FP32, with BF16 projection arithmetic as in the head plan.
Use identical adapter targets, rank, data, draws and total updates in every
FFN comparison. Include a frozen-weight gate control. This adapts Compresso's
joint-learning mechanism; it does not reproduce its LLM prompting procedure.

Use the official velocity loss and target in §7.5.1:

$$
\mathcal L=L_v=\operatorname{MSE}_{f>0}(v_S,\epsilon-y),
\qquad \sum_j\bar m_{b,j}=K_b.
$$

This baseline excludes the clean first latent frame and uses no auxiliary term.
Average per clip; preserve actor-disjoint splits,
paired saved noise and all eight exact sigmas from the head plan.
Test the plan's person/small-region weighting and balanced calibration as separate
extensions, stating which question is tested. Faces and hands require decoded checks.
FANG motivates attention to specialized contributions, but neither region weights
nor learned logits prove that a channel has a single semantic function.

**First comparison.** Compare the current RMS-selected mask plus LoRA, learned
individual channels plus the same LoRA budget, and seeded random channels plus
that budget. Measure quality after selection and after matched compact recovery.
Also disable adapters to measure each selected mask's immediate deletion damage.
Official velocity loss can have nonzero gradients at the unchanged dense point.
Its target is saved noise minus capture, rather than the original teacher.

**Grouping comparison.** Run this after the individual-channel pilot passes.
Hold each block's retained width, loss, adapter budget and inputs fixed.

- **Importance ordering:** retain prefixes of an ordering, following DynaBERT's
  idea. At one width this is ranking selection; nested training is a separate
  multi-width question with its extra training cost reported.
- **Balanced grouping:** use balanced weight clustering or a defined GELU
  co-activation similarity on learning data. For a shared-gate arm, create
  equal-sized groups before training and record their original channel indices.
- **Contiguous grouping:** tie one gate across 128 adjacent original channels
  and retain 116 groups per block at the pilot width. This tests the hardware
  grouping hypothesis without adopting WIDE's dynamic routers or kernels.

Use 128 channels per shared-gate group for the first matched comparison, including
seeded random groups of that size. Report grouping time and memory.
Balanced clusters and original contiguous groups must keep the same channel count.
Use a separate loss-weighting comparison for functional protection; do not tie
all person-related neurons to one removable gate.
Once static selected channels are packed into compact dense matrices, their
original grouping need not save extra runtime. Compare quality and measured cost
against free per-channel selection, rather than assuming a grouping speed benefit.

**Export and code boundary.** For retained original indices $S_b$, export
$W'_{{\rm up},b}[S_b,:]$, $b_{{\rm up},b}[S_b]$ and
$W'_{{\rm down},b}[:,S_b]$; keep $b_{{\rm down},b}$ unchanged.
Preserve the same channel order in both projections and in adapter transfer.
Slice up-projection $B$ rows and down-projection $A$ columns for compact recovery;
merge adapters before timing. No runtime gather or full-width reconstruction
is needed for these static FFNs.

`score/hooks.py:attach_ffn_masks` provides the static boundary and detaches supplied tensors.
The live `attach_ffn_provider` preserves the surrogate graph during checkpoint recomputation.
`learn/ffn_gates.py` owns individual hard-forward decisions and aligned counts.
`learn/ffn_training.py` reuses the head gates, official loss and projection adapters.
`data/ffn_study.py` and `learn/ffn.py` provide separate frozen preparation and execution.
`score/export_learned_ffn.py` validates both original-ID families and merges sliced adapters.
The explicit joint preset uses 1000 selection updates and 300 fixed-mask adapter updates.
It uses 50 initially dense FFN updates and a count ramp ending at update 300.
Mirrored designs document current ownership; scientific comparisons below remain distinct from implementation checks.

`score/export_pruned.py:_slice_ffn` already handles arbitrary retained channels.
Use that surgery after adapter fusion. The existing feed-forward implementation
supports real compact widths; shared groups require no new upstream FFN layer.
Preserve raw functional-versus-compact parity results, including failures.
If compact recovery changes weights, use its trained in-memory network as the
reload reference, retaining the 0.02 tolerance and independent quality gate.
Use full and region quality–latency plots, per-sigma/worst-actor errors, and
matched capture/teacher/student videos. All-one reproduction, finite gradients,
unchanged frozen weights and complete-clip backward memory precede sustained work.
No FFN training, export or speed result is claimed here.

### 7.6. Reconstruction and budgeted recovery

For the recommended head experiment, bounded LoRA recovery in §7.5.1 comes first.
Algebraic reconstruction remains a separate alternative, not its prerequisite.
For that reconstruction arm, first fit retained output projections to the original
local outputs with calibration-only ridge regression. Evaluate unseen actors and complete forward
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

**Order of work.** For the current human-region question, first check compact
execution and full-clip gradient feasibility. Then compare energy-selected and
learned binary heads with matched recovery (§7.5.1). Freeze the selected mask,
recover in compact execution and test unseen actors and noise.
Other rankings, reconstruction and static depth remain separate experiments.
Accept candidates when correctness, held-out content/motion and measured cost
agree. If only storage improves, label the result as storage compression.

Implementation contracts remain in the source-matching [design index](README.md)
and [validation guide](VALIDATION.md). Revise this agenda when new methods are
implemented or matched LTX evidence changes the priorities.

### Learned-head implementation status, 12 October 2026

The initial learned-head implementation is attributed in the [workspace history](../../../../plans/history/README.md#stopped-learned-head-lineages).
See the [phase run guide](usage.md#learned-head-phases) and [CPU/native controls](VALIDATION.md#learned-head-controls).
Implemented: +3 sigmoid gates, legal binary STE refinement, official velocity loss, rank-16 V/output adapters,
original-ID moment transfer, shared teacher records, frozen preparation, execution probes, recovered export and paired evaluation.
Implemented code and passing CPU fixtures do not establish measured pruning quality, native feasibility or speed.
Use the experiment's saved stage receipts for measured status.
