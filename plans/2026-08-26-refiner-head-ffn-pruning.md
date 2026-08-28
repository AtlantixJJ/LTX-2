# Plan — Training-free head + FFN pruning of the LTX distilled transformer for the VAE-refine task

Target script: `scripts/vae_refine_sliding_window.py`
Models: **LTX-2.3 and LTX-2.5 distilled, selected by one `--model` flag** (§4)
Deployment geometries: (a) today's sliding-window refiner, (b) the upcoming **autoregressive**
refiner that predicts **1–3 latent frames per chunk** with the preceding latents frozen.
Written 2026-08-26 · revised 2026-08-27 (LTX-2.5 pack landed; training work split out)
Env: `ltx` conda env · Hardware: 8× RTX A6000 (49 GB, Ampere, ~155 bf16 TFLOPS, ~768 GB/s)

**Scope: training-free only.** Nothing here updates a model weight by gradient descent.
Every method is one-shot or iterative-with-recompute: statistics, a mask-gradient
importance score (a backward pass, not an optimizer), and closed-form least-squares
re-solves. Learned gates and recovery fine-tuning live in the companion plan
[`2026-08-27-refiner-pruning-training.md`](2026-08-27-refiner-pruning-training.md), which
starts only after this plan's gates are measured.

Every section carries a **Coding guidance** block: the exact modules, hook points and call
signatures to build against, so the implementation does not re-derive them.

---

## 1. What this is actually optimizing

The refiner is a *very* narrow slice of what the checkpoint can do:

| Full model does | The refine task does |
|---|---|
| all sigmas 1.0 → 0.0 | only the k-step tail (`k2` = σ ∈ {0.725, 0.422, 0.0}, i.e. **2 forwards**) |
| arbitrary text prompts | **one constant generic prompt** |
| video **and** audio, bidirectional A↔V cross-attention | video only; `audio=None` |
| 121-frame bidirectional windows (16 latent frames, 16k tokens) | AR chunks of **1–3 latent frames**, rest of the context frozen |
| generates content from noise | **repairs** an already-correct VAE latent |

Each of those narrows what the weights must do, and head/FFN importance is conditional on
all of them. **All importance statistics must therefore be collected at the deployment
geometry** — short chunk, frozen context, refine sigmas only, the fixed prompt. Calibrating
on 16-latent-frame bidirectional windows would systematically over-value long-range temporal
heads that the AR chunk mode never exercises.

**The output is a task-specialized refiner, not a general LTX model.** Name it accordingly
and never reuse it for T2V.

### Coding guidance

Put the deployment conditions in exactly one module every other script imports, so "the
task" is never re-specified inconsistently:

```python
# scripts/prune/refine_task.py
REFINE_PROMPT = "a high quality, sharp, detailed video with fine texture and natural lighting"
K_STEP          = "k2"          # student schedule: the deployed one
CHUNK_LATENT_FRAMES = (1, 2, 3) # AR chunk sizes to calibrate over
CTX_LATENT_FRAMES   = 4         # frozen context carried into each chunk
CALIB_CLIPS, EVAL_CLIPS = ...   # split by CLIP, never by window
```

Do not import from `vae_refine_sliding_window.py`; it is a run script, not a library. Reuse
the *patterns* in it (listed throughout), write fresh probes.

---

## 2. Facts established from the repo (do not re-derive)

**Architecture** (`packages/ltx-core/src/ltx_core/model/transformer/`)

- `BasicAVTransformerBlock` (`transformer.py`), 48 blocks. Video path per block:
  `attn1` (self-attn) → `attn2` (text cross-attn) → `ff`. Audio path (`audio_attn1/2`,
  `audio_ff`) and A↔V cross-attn are **skipped entirely when `audio is None`**
  (`run_ax` / `run_a2v` / `run_v2a` guards in `forward`).
- `Attention` (`attention.py`) is **plain MHA, no GQA**: `to_q/to_k/to_v` each
  `Linear(dim → heads*dim_head)`, `to_out[0]` is `Linear(heads*dim_head → dim)`.
  `q_norm`/`k_norm` are `RMSNorm(inner_dim)` (element-wise weight → sliceable).
  `self.heads` is a plain attribute read at forward time.
- **Per-head gating is ON in both generations** (`apply_gated_attention=True`, verified in
  the 2.3 and 2.5 distilled configs): `to_gate_logits: Linear(dim → heads)`, output scaled
  by `2*sigmoid(logits)` per head per token (`PytorchGatedAttention`, `ops.py`). A free
  trained per-head signal.
- `FeedForward` (`feed_forward.py`) is a **plain non-gated MLP**:
  `net = Sequential(GELUApprox(dim→4·dim), Identity(), Linear(4·dim→dim))`. `inner_dim` is
  hardcoded `dim * mult` → needs an explicit argument before per-layer widths are
  expressible (§10).
- **The transformer is an `X0Model`**: `transformer(video=Modality, audio=None,
  perturbations=None)` returns the **denoised x0 prediction**, not a velocity.
  `to_velocity(sample, σ, denoised) = (sample − denoised)/σ`; `EulerDiffusionStep.step` is
  `sample + velocity·(σ_next − σ)`. **Write all losses in x0 space** — it is what the model
  emits, and it avoids the 1/σ blowup as σ→0.
- `post_process_latent(denoised, denoise_mask, clean)` restores frozen tokens.
  **`state.denoise_mask` is exactly the "fresh chunk tokens" selector** — mask every loss
  and every statistic with it.
- RoPE (`rope.py`) is applied per head via `unflatten(-1, (h, -1))` where `h` comes from the
  **freqs** tensor. **Verify** the freqs are built from the model's head count before
  trusting any exported model — the one place head surgery can break silently (§10.6).
- `LTXVideoOnlyModelConfigurator` exists; `ltx_trainer/model_loader.py` documents
  `video_only=True` as **lossless** because the audio branch never executes.
- `guidance/perturbations.py` gives a ready-made *block-level* self-attention skip
  (`PerturbationType.SKIP_VIDEO_SELF_ATTN`, per-block list) — free coarse ablation, no
  model code.

**Parameter inventory** — measured from both checkpoints' safetensors headers. The video
branch is **structurally identical across generations**, so every budget, score shape and
surgery routine transfers unchanged:

| Group | LTX-2.3 distilled | LTX-2.5 distilled | Notes |
|---|---|---|---|
| `ff` (video FFN, 48 blocks) | **6.44 B** | **6.44 B** | 2 × 3.22 B — biggest executed group; 2.5 has **no ff biases** (`ff_bias=false`) |
| `attn1` (video self-attn) | 3.22 B | 3.22 B | 4 × 0.805 B |
| `attn2` (video text cross-attn) | 3.22 B | 3.22 B | 4 × 0.805 B |
| **video blocks total (executed)** | **12.88 B** | **12.88 B** | 268 M/block |
| audio branch + A↔V cross-attn | ~5.6 B | ~5.6 B | **never executed** for refine |
| video embeddings connector | ~1.07 B | ~1.07 B | once per prompt |
| file total | 23.07 B (monolith, incl. VAEs) | 21.00 B (transformer only) | 2.5 is a split pack |

Shared config in both: `num_layers=48`, `num_attention_heads=32`, `attention_head_dim=128`
(dim 4096, FFN inner 16384), `cross_attention_dim=4096`, `cross_attention_adaln=True`,
`apply_gated_attention=True` → **3072 video heads** (48 × 32 × {attn1, attn2}).

**Measured baseline** (`expr/sam3dgs_vae_refine/*/k2_*/profile.json`, 1024×1024, A6000, bf16, 2.3):

| Geometry | tokens | fwd/step |
|---|---|---|
| 25 frames (4 latent frames) | 4096 | **1.25 s** (×2 steps = 2.5 s/window) |
| 121 frames (16 latent frames) | 16384 | ~9 s (derived) |
| transformer build | — | **22–28 s** (amortized by `--batch-windows`) |
| peak alloc during refine | — | 39–43 GB |

Roofline at 4096 tokens: GEMMs 2·12.9e9·4096 ≈ 105 TFLOP + attention ≈ 13 TFLOP ≈ 118 TFLOP
/ 1.25 s ≈ **95 TFLOPS ≈ 61% MFU**. The refiner is **compute-bound, not weight-bandwidth-
bound**, at chunk ≥ 1 latent frame (bandwidth floor for 26 GB of video weights is ~34 ms).
**A structured parameter cut therefore converts roughly linearly into wall-clock** — the
justification for structured over unstructured pruning. Re-verify at chunk = 1 (1024
tokens), where MFU drops.

**Corpus already on disk**: `expr/sam3dgs_vae_refine/<52 clips>/source.mp4` plus their
existing `k1`…`k8` refined outputs and `sweep_metrics.json`.

### Coding guidance — the two hook points everything hangs off

**No `ltx-core` edits are needed before the export step.** Every score, mask and activation
statistic attaches at one of two `forward_pre_hook`s, because the module boundaries land
exactly where the math wants them:

```python
# Per-HEAD: attn.to_out[0] receives (B, T, H*D) -- post-attention, post-gate,
# pre-output-projection. Precisely where a head mask xi_h belongs.
def head_hook(mod, args):
    (x,) = args                                    # (B, T, H*D)
    b, t, _ = x.shape
    x = x.view(b, t, heads, dim_head) * xi.view(1, 1, heads, 1)
    return (x.view(b, t, heads * dim_head),)       # returning a tuple replaces the input

handle = attn.to_out[0].register_forward_pre_hook(head_hook)

# Per-FFN-CHANNEL: ff.net[2] receives (B, T, inner_dim) -- the post-GELU activation.
# The same hook serves activation collection (scoring) and channel masking.
handle = block.ff.net[2].register_forward_pre_hook(ffn_hook)
```

Autograd flows through a hook's returned tensor, so the *same* mechanism serves the
zeroth-order contribution norm (§7.2a) and the mask-gradient estimator (§7.2b). It is also
what makes the gating question moot — the hook does not care whether `to_gate_logits` exists.

Write one `scripts/prune/hooks.py` exposing `attach_head_masks(model) -> dict[str, Tensor]`,
`attach_ffn_masks(model)`, `collect_activations(model, which)` and `detach_all()`.

Canonical module iteration (video branch only):

```python
for i, block in enumerate(model.transformer_blocks):
    for kind in ("attn1", "attn2"):
        attn = getattr(block, kind)          # heads=attn.heads, dim_head=attn.dim_head
    ff = block.ff
```

Checkpoint-space key names (verified against **both** checkpoints; used only by the exporter):

```
model.diffusion_model.transformer_blocks.{i}.{attn1|attn2}.to_{q,k,v}.{weight,bias}     [4096,4096] / [4096]
model.diffusion_model.transformer_blocks.{i}.{attn1|attn2}.{q_norm,k_norm}.weight       [4096]
model.diffusion_model.transformer_blocks.{i}.{attn1|attn2}.to_out.0.{weight,bias}       [4096,4096] / [4096]
model.diffusion_model.transformer_blocks.{i}.{attn1|attn2}.to_gate_logits.{weight,bias} [32,4096] / [32]
model.diffusion_model.transformer_blocks.{i}.ff.net.0.proj.weight [+ .bias on 2.3 only]  [16384,4096]
model.diffusion_model.transformer_blocks.{i}.ff.net.2.weight      [+ .bias on 2.3 only]  [4096,16384]
```

---

## 3. Open questions

1. **The AR refiner runtime does not exist in this repo.** No KV cache (`grep kv_cache` →
   only a VAE kernel), no causal mask path in the DiT — though 2.5's config does set
   `causal_temporal_positioning=true`, worth understanding before the AR workstream starts.
   This plan covers *pruning* against the AR geometry; the AR chunk scheduler and context
   caching are separate. §6 emulates AR geometry with the mechanism the refine script
   already uses — `VideoConditionByLatentIndex(strength=1.0)` frozen context + 1–3 fresh
   latent frames — which is geometrically faithful without the cache.
2. **Chunk size to optimize for.** Calibrate on the mixture {1,2,3} latent frames and
   validate at all three rather than picking one. Confirm target resolution (1024×1024 in
   the existing runs → 1024 tokens per latent frame).
3. **2.5's conv VAE variant is not downloaded.** Only `ltx-2.5-video-vae-bf16.safetensors`
   (the diffusion decoder) is on disk. If Phase 0 shows decode time dominating the
   calibration loop, pull `ltx-2.5-video-vae-conv-bf16.safetensors` — one `hf download`
   line — rather than working around it.
4. `vae_refine_sliding_window.py` still hardcodes a monolith 2.3 path and feeds one
   checkpoint string to the transformer, the encoder and the decoder alike. §4 fixes this
   as a Phase 0 deliverable, not an afterthought at export time.

### Coding guidance

Write `scripts/prune/preflight.py` first and run it before anything else:

```bash
conda run -n ltx python -m scripts.prune.preflight --model 2.5
```

It must resolve every component path, print `ModelCaps` (§4) and `detect_model_version`,
assert the distilled sigma list, list free GPUs, and exit non-zero with the exact
`hf download` line if a component is missing. Every later script calls `preflight.check()`
at start-up so a wrong env or half-downloaded pack fails in one second, not 25.

---

## 4. Model selection — one `--model` arg covering both 2.3 and 2.5

Every script takes exactly **one** generation flag — `--model {2.3, 2.5}`, default `2.5`.
A shared registry turns that key into paths, sampler, sigma schedule and probed
capabilities. Per-component flags stay available and always override the registry.

**Both packs are on disk** (`checkpoints/LTX-2.3/`, `checkpoints/LTX-2.5/`), so this is
implementable and testable today.

### What actually differs — probed, not assumed

| | **LTX-2.3 distilled** | **LTX-2.5 distilled** |
|---|---|---|
| layout | **monolith** — transformer + video VAE + audio VAE + vocoder + scheduler in one file | **split pack** — `config` sections are `[transformer, scheduler]` only |
| `ModelPaths` factory | `from_monolith(ckpt, gemma_root)` | `from_split(transformer_path=…, text_encoder_path=…, video_vae_path=…)` |
| transformer | `LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors` (23.07 B incl. VAEs) | `LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` (21.00 B) |
| text encoder | gemma-3-12b **directory** (`--gemma-root`) | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` — 26 GB single file, projection bundled; stock Google Gemma 4 is **rejected** (`gemma_source_checkpoint` must match) |
| video VAE | the same monolith file | `vae/ltx-2.5-video-vae-bf16.safetensors` (diffusion decoder; conv variant not downloaded — §3.3) |
| `model_version` | `2.3.0` | `2.5.0` |
| stage-1 sampler | plain Euler | **ancestral** — `detect_model_version() >= (2,5)` triggers `EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0)` |
| `ff_bias` | `True` | **`False`** — no `ff.net.0.proj.bias` / `ff.net.2.bias` keys at all |
| `apply_gated_attention` | `True` | `True` — per-head gates available on both |
| `frequencies_precision` | float32 | **float64** → `double_precision_rope=True` |
| `causal_temporal_positioning` | absent | **`True`** |
| `use_keyframes_abs_pos_embedding` | absent (False) | **`True`** |
| embeddings connector | present (~1.07 B) | present, config-declared: 8 layers, 128 learnable registers, gated attention |
| `caption_channels` | gemma3 | 3840 (gemma4) |
| video block shapes | dim 4096, 32×128 heads, FFN 16384 | **identical** |

Two corrections to earlier assumptions, both from reading the real 2.5 config:

- **`use_prompt_adaln_single` is absent from the 2.5.0 distilled config**, so it defaults to
  `True` — the cross-attention K/V are *still* timestep-modulated. The "KV-cacheable"
  comment in `model_configurator.py` describes specific checkpoints that set it false, not
  the 2.5 generation. **Cache cross-attn K/V per σ on both generations** (3 tensors for k2),
  not once per run.
- **The distilled sigma schedule is shared.** 2.5's `scheduler` metadata carries no sigma
  list (`RectifiedFlowScheduler` / `LinearQuadratic`, same as 2.3) and
  `DISTILLED_SIGMA_VALUES` has no version dispatch, so both generations use the same 8-step
  schedule. Only the *sampler* differs.

### Decisions this forces

1. **Sampler.** Refine is a 2–3-step *tail* schedule — precisely where `distilled.py`
   deliberately keeps plain Euler for stage 2 ("its 3-step schedule is too short to remove
   freshly injected noise"). Default the refiner to **plain Euler on both generations**,
   expose `--sampler {euler, ancestral, auto}`, and **A/B once on 2.5 in Phase 1** before
   locking it. Re-injecting noise every step is also bad for AR rollout stability — but
   measure.
2. **Latent geometry is probed, not assumed.** Check the window rules (`F % 8 == 1`,
   `(overlap−1) % 8 == 0`) against the *probed* temporal scale factor, not a literal 8.
3. **Prompt context is precomputed** (§5) — with a constant prompt, neither generation needs
   its text encoder resident at all during calibration or deployment.

### Coding guidance — `scripts/prune/model_registry.py`

```python
from dataclasses import dataclass
import json, os
from pathlib import Path
from safetensors import safe_open

from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.constants import detect_model_version, DISTILLED_SIGMA_VALUES

CKPT_ROOT = Path(os.environ.get("LTX_CHECKPOINTS", Path(__file__).resolve().parents[3] / "checkpoints"))

@dataclass(frozen=True)
class ModelCaps:                 # every field PROBED from metadata, never keyed off the version
    num_layers: int; num_heads: int; head_dim: int; cross_attention_dim: int
    ff_inner_dim: int; ff_bias: bool
    apply_gated_attention: bool; cross_attention_adaln: bool
    latent_channels: int; double_precision_rope: bool

def probe_caps(transformer_path: str) -> ModelCaps:
    with safe_open(transformer_path, framework="pt") as f:
        cfg = json.loads((f.metadata() or {}).get("config", "{}")).get("transformer", {})
    dim = cfg.get("num_attention_heads", 32) * cfg.get("attention_head_dim", 128)
    return ModelCaps(
        num_layers=cfg.get("num_layers", 48),
        num_heads=cfg.get("num_attention_heads", 32),
        head_dim=cfg.get("attention_head_dim", 128),
        cross_attention_dim=cfg.get("cross_attention_dim", 4096),
        ff_inner_dim=cfg.get("ff_inner_dim", 4 * dim),
        ff_bias=cfg.get("ff_bias", True),
        apply_gated_attention=cfg.get("apply_gated_attention", False),
        cross_attention_adaln=cfg.get("cross_attention_adaln", False),
        latent_channels=cfg.get("in_channels", 128),
        double_precision_rope=cfg.get("frequencies_precision", False) == "float64",
    )

@dataclass(frozen=True)
class RefinerModel:
    key: str                      # "2.3" | "2.5" -- namespaces EVERY output directory
    version: tuple[int, ...]
    paths: ModelPaths
    sigmas: list[float]
    stepper_kind: str             # "euler" | "ancestral"
    caps: ModelCaps

def resolve(key: str = "2.5", *, sampler: str = "euler",
            transformer_path=None, text_encoder_path=None, video_vae_path=None) -> RefinerModel:
    if key == "2.3":
        ckpt = str(transformer_path or CKPT_ROOT / "LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors")
        paths = ModelPaths.from_monolith(
            ckpt,
            gemma_root=str(text_encoder_path or CKPT_ROOT / "google/gemma-3-12b-it-qat-q4_0-unquantized"),
            video_vae_path=str(video_vae_path) if video_vae_path else None,
        )
    elif key == "2.5":
        root = CKPT_ROOT / "LTX-2.5"
        paths = ModelPaths.from_split(
            transformer_path=str(transformer_path or root / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
            text_encoder_path=str(text_encoder_path or root / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
            video_vae_path=str(video_vae_path or root / "vae/ltx-2.5-video-vae-bf16.safetensors"),
        )
    else:
        raise ValueError(f"unknown --model {key!r}; expected 2.3 or 2.5")
    _require_files(paths)                       # -> the `hf download` line on failure
    version = detect_model_version(paths.transformer())
    kind = ("ancestral" if version >= (2, 5) else "euler") if sampler == "auto" else sampler
    return RefinerModel(key, version, paths, list(DISTILLED_SIGMA_VALUES), kind, probe_caps(paths.transformer()))
```

Downstream, **always** go through the accessors, never a bare checkpoint string:

```python
m = registry.resolve(args.model, sampler=args.sampler)
prompt_encoder = PromptEncoder(m.paths, DTYPE, device)                 # whole ModelPaths
stage = DiffusionStage.from_checkpoint(m.paths.transformer(), DTYPE, device,
                                       model_configurator=LTXVideoOnlyModelConfigurator)
conditioner = ImageConditioner(m.paths.video_vae(), DTYPE, device)
decoder     = VideoDecoder(m.paths.video_vae(), DTYPE, device)
stepper = EulerDiffusionStep() if m.stepper_kind == "euler" else EulerAncestralDiffusionStep()
```

### What is never shared between generations

Scores, masks, reconstructions and pruned checkpoints are **per generation**: head index
spaces are not comparable and FFN activation statistics do not transfer. Namespace every
artifact `expr/refiner_prune/<model-key>/<run-id>/`, stamp the model key **and the
transformer file hash** into provenance, and refuse to load a mask whose key does not match.

**Sequencing this buys:** run 2.5 as the deliverable and 2.3 as a cross-check. Because the
video branch is structurally identical, the *same* score tensors and surgery code apply to
both — so if the head-importance *pattern* (which layers, which attention types) holds
across generations the finding is about the task; if not, about the checkpoint. That is a
free, strong sanity check on every conclusion in this plan.

---

## 5. Phase 0 — registry, free wins, honest baseline  (~1 day)

1. **Land the registry (§4)** and port `vae_refine_sliding_window.py` onto it. Everything
   downstream is written against `RefinerModel`; doing this last would mean rewriting every
   script.
2. **Drop the audio branch** — `model_configurator=LTXVideoOnlyModelConfigurator`.
   Documented lossless; saves ~5.6 B params (~11 GB bf16) and a large slice of the 22–28 s
   build. No FLOP change (it never executed), but it buys the memory headroom §7 needs on
   one 49 GB card.
3. **Precompute the prompt context to disk.** The prompt is one constant string, so encode
   it once per (model, prompt) and cache the tensor. This removes the text encoder — **26 GB
   on 2.5** — and the 8-layer embeddings connector from every calibration run and from
   deployment entirely. Biggest single load-time win available, and it costs nothing.
4. **Cache the text cross-attention K/V per σ.** `attn2`'s K/V depend only on the (constant)
   context and σ, so on a k2 schedule that is 3 cached tensors per layer, reused across
   every window and every chunk.
5. **Baseline measurement** — ms/step and peak memory at chunk ∈ {1,2,3,4,16} latent frames
   × resolution, plus a FLOP count, **per generation**. This table is the denominator for
   every later speedup claim. Measure `torch.compile` + CUDA graphs on/off separately so
   pruning gains are not confounded with compilation gains.
6. **Freeze the source-target corpus split** (§6).

### Coding guidance

```python
# 2. video-only build, and the equivalence check that it is lossless
from ltx_core.model.transformer.model_configurator import LTXVideoOnlyModelConfigurator
stage = DiffusionStage.from_checkpoint(m.paths.transformer(), DTYPE, device,
                                       model_configurator=LTXVideoOnlyModelConfigurator)
assert (x_av - x_vo).abs().max() < 1e-2          # bf16 noise floor, not exact zero

# 3. prompt context cache
ctx_path = cache_dir / f"prompt_ctx_{m.key}_{hashlib.sha1(REFINE_PROMPT.encode()).hexdigest()[:8]}.pt"
if not ctx_path.exists():
    (ctx,) = PromptEncoder(m.paths, DTYPE, device)([REFINE_PROMPT])
    torch.save(ctx.video_encoding.cpu(), ctx_path)       # then never load the TE again
video_context = torch.load(ctx_path).to(device, DTYPE)
```

Cross-attention K/V cache — memoize the projections keyed by σ; validate by asserting the
cached run matches the uncached run bit-for-bit before trusting it:

```python
class CrossKVCache:
    def __init__(self): self.store = {}
    def wrap(self, attn, sigma_key):
        ok, ov = attn.to_k, attn.to_v
        attn.to_k = lambda c, _o=ok: self.store.setdefault(("k", sigma_key, id(c)), _o(c))
        attn.to_v = lambda c, _o=ov: self.store.setdefault(("v", sigma_key, id(c)), _o(c))
```

Benchmark harness:

```python
from torch.utils.flop_counter import FlopCounterMode
with FlopCounterMode(display=False) as fc:
    denoiser(wrapped, video_state, None, sigmas, 0)
flops = fc.get_total_flops()   # time with cuda.Event around a warmed loop -> ms/step, TFLOPS, peak
```
Reuse the `StageTimer` pattern from `vae_refine_sliding_window.py` (reset peak stats,
synchronize, `perf_counter`) — copy it into `scripts/prune/timing.py`, do not import it.

**Gate:** baseline table with a row per generation; a registry-driven 2.3 run matches the
pre-refactor script bit-for-bit; video-only matches AV within bf16 noise on 3 clips; the
prompt-context and K/V caches are bit-exact against the uncached path; `ModelCaps` dumped
to JSON per generation.

---

## 6. Phase 1 — calibration + evaluation harness  (~1 day)

New package `scripts/prune/` (top-level `scripts/`, same `sys.path.append` convention as the
existing probes).

### The target — and why the loss is nonzero

The refiner has no ground-truth "correct" x0. Two constructions matter:

- **Naive self-distillation is degenerate.** `L(ξ) = ‖f(ξ) − f(1)‖²` has gradient
  *identically zero* at ξ = 1, which silently kills the mask-gradient estimator (§7.2b).
- **The construction that works**: use the VAE-encoded **source latent** `x_source` as x0
  target.  For a noised source state `z_σ`, `L = ‖D_θ(z_σ, σ) − x_source‖²` is ordinary
  diffusion denoising supervision and is nonzero at ξ = 1. Its mask gradient is the VJP
  `2 J_ξᵀ(D_θ(z_σ,σ) − x_source)`, so the target supplies a meaningful output-space direction.
- **Renoising generates unlimited calibration states.** Renoise `x_source` back to σ_i with a
  fresh noise draw; the model does not return to the identical x0 from that renoised state,
  so every draw is a fresh, correctly-targeted sample on the deployment manifold. Sample
  **both** families: *on-policy* states from the natural k2 trajectory (what deployment
  visits) and *renoised* states (cheap, decorrelated, resamplable). Weight on-policy higher;
  report both.

### Modules

**`chunk_states.py` — the AR-geometry sampler.** Per source clip: read a window, VAE-encode,
freeze `CTX_LATENT_FRAMES` via `VideoConditionByLatentIndex(strength=1.0, latent_idx=1)`
(mirroring the carryover in the refine script, including its frame-0 keyframe caveat), noise
the next `n_new ∈ {1,2,3}` latent frames to σ_i, build the state.

```python
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_pipelines.utils.blocks import _build_state
from ltx_pipelines.utils.types import ModalitySpec

def make_state(l_init, ctx_latent, sigma, video_tools, seed, device):
    noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seed))
    conds = [VideoConditionByLatentIndex(latent=ctx_latent, strength=1.0, latent_idx=1)]
    return _build_state(
        ModalitySpec(context=video_context, conditionings=conds,
                     noise_scale=float(sigma), initial_latent=l_init),
        video_tools, noiser, DTYPE, device)
```
Persist states to disk (~1–4 MB each) so **every estimator sees the identical calibration
set** — this is what makes the §7 comparison fair. Split clips 40 calib / 12 held-out,
**by clip, never by window**.

**`teacher.py`** — cache source-target on-policy student states alongside renoised states:

```python
x_source = video_tools.patchify(video_tools.create_initial_state(device, DTYPE, l_enc)).clean_latent
```

**`losses.py`** — always x0 space, always masked to fresh tokens:

```python
def x0_loss(pred_x0, x0_star, state):
    m = state.denoise_mask                      # 1 on fresh chunk tokens, 0 on frozen ctx
    return ((pred_x0.float() - x0_star.float()) ** 2 * m).sum() / m.sum().clamp(min=1)

def rel_l2(pred_x0, x0_star, state):            # the T0 metric
    m = state.denoise_mask
    return (((pred_x0 - x0_star) ** 2 * m).sum() / ((x0_star ** 2 * m).sum() + 1e-8)).sqrt()
```

**`metrics.py`** — four tiers:
- **T0 (latent)** `rel_l2` vs `x0*` on fresh tokens. Cheap; runs inside search loops.
- **T1 (pixel, single chunk)** decode and compare vs the source clip — PSNR / SSIM / LPIPS.
- **T2 (AR rollout)** ≥200 sequential chunks, each chunk's own output feeding the next
  chunk's frozen context. Report PSNR-vs-source **as a function of chunk index** plus
  brightness/saturation drift. **This is the gate that matters** — a 1% per-chunk error
  invisible in isolation compounds over a 600-frame rollout, and no single-chunk metric
  catches it.
- **T3** side-by-side `source | source target | candidate` review artifacts on 5 clips, in **both**
  forms: a still frame grid (`t3_grid` → PNG) *and* a frame-aligned horizontally-concatenated
  **MP4** (`t3_video` → H.264 via `ffmpeg`, `-crf 18`, `yuv420p`). The video is not optional
  garnish: temporal artifacts are the failure mode this task is most exposed to — flicker,
  chunk-boundary seams, and the slow brightness/saturation drift T2 scores numerically are all
  invisible in a still grid. Every stage that decodes pixels for review (teacher validation,
  T1, T2 rollouts, and every pruned candidate in §7–§9) writes both, plus a `figures/INDEX.md`
  naming them, following the `scripts/ltx23_diag` figure conventions.

**Gate:** source-target cache written; T0/T1/T2 recorded for the unpruned student; the
unpruned model's own T2 rollout characterizes the **intrinsic drift floor** (not zero — the
sliding-window script's carryover design exists because of it); the 2.5 Euler-vs-ancestral
A/B (§4) is decided and recorded; and the T3 pair (grid PNG **and** MP4) is written for the
source-vs-student comparison, so the drift floor is inspectable and not only tabulated.

---

## 7. Phase 2 — attention-head pruning  (~3 days)

Every estimator sits behind one interface returning `scores[layer, attn_type, head]`; decide
empirically on the Phase 1 harness at matched sparsity.

```python
# scripts/prune/head_scores.py
def score(model, states, method: str) -> torch.Tensor:   # -> (num_layers, 2, num_heads)
```

### 2d. Coarse pre-pass — do this first, it is nearly free

Ablate whole self-attention layers with the existing STG machinery, no code change:

```python
from ltx_core.guidance.perturbations import Perturbation, PerturbationConfig, PerturbationType
cfg = PerturbationConfig([Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=[i])])
# pass through the denoiser's `perturbations` argument; measure T0 per ablated block i
```
48 runs. Any layer whose *complete* removal barely moves T0 is a strong prior that its heads
are near-free, and it bounds what the fine-grained search should find. Same for whole-`attn2`
removal per layer (a small skip shim) — with a constant prompt, expect several layers' text
cross-attention to be nearly inert.

### 2a. Head output-contribution norm  (zeroth-order, cheap)

Score each head by how much it actually moves the block output:

`C_h = E_x ‖ g_h(x) · (A_h V_h) · W_out[:, hD:(h+1)D] ‖₂`, normalized per layer.

This subsumes raw attention statistics — it accounts for the value projection, the output
projection and the learned gate, all of which attention scores ignore. One instrumented
forward sweep, no backward.

```python
@torch.no_grad()
def contribution_scores(model, states, denoiser, sigmas):
    acc = {}
    def mk(name, attn):
        Wo = attn.to_out[0].weight                  # (dim, H*D)
        def hook(mod, args):
            (x,) = args                              # (B,T,H*D) post-gate, pre-projection
            b, t, _ = x.shape
            xh = x.view(b, t, attn.heads, attn.dim_head).float()
            Wh = Wo.view(-1, attn.heads, attn.dim_head).float()
            contrib = torch.einsum("bthd,ohd->btho", xh, Wh)   # per-head block-output delta
            acc[name] = acc.get(name, 0) + contrib.pow(2).sum(dim=(0, 1, 3)).cpu()
        return attn.to_out[0].register_forward_pre_hook(hook)
    ...   # attach for every (layer, attn1/attn2), run states, normalize per layer
```
Memory: the einsum materializes `(B,T,H,dim)`. Fine at 1024 tokens × 32 heads; chunk over
heads at 16k tokens. Prefer the exact form at the AR geometry, which is small by construction.

Report the same sweep's diagnostics (gate mean per head, attention entropy, context-vs-chunk
mass, text-vs-padding mass for `attn2`) **as interpretation only**, never as a selector.

### 2b. Michel et al. (2019) — mask-gradient importance

`I_h = E_x |∂L/∂ξ_h|` at ξ = 1: the first-order Taylor estimate of the loss change from
zeroing head h. One backward scores all 3072 heads at once. Apply the paper's per-layer
normalization `I_h ← I_h / ‖I_layer‖₂`; without it layer-scale differences dominate.

This is a **backward pass, not training** — no optimizer, no weight update, nothing
persisted but a score tensor. It belongs in this plan.

The loss is the §6 construction: `L = ‖D_θ(z_σ, σ) − x0*‖²` on fresh tokens, `x0*` from the
deeper teacher schedule, `z_σ` drawn from both the on-policy trajectory and renoised draws.
Nonzero at ξ = 1 by construction — the naive self-distillation loss is not.

```python
def michel_scores(model, states, sigmas):
    for p in model.parameters():
        p.requires_grad_(False)                 # only xi carries grad -> no weight-grad buffers
    xis = attach_head_masks(model)              # dict[name -> (H,) tensor, requires_grad=True]
    acc = {k: torch.zeros_like(v) for k, v in xis.items()}
    for st, sigma, step_idx, x0_star in states:
        for v in xis.values():
            v.grad = None
        res, _ = denoiser(wrapped, st, None, sigmas, step_idx)
        x0_loss(res.denoised, x0_star, st).backward()
        for k, v in xis.items():
            acc[k] += v.grad.abs().detach()      # |dL/dxi| at xi = 1
    return normalize_per_layer(acc)
```
Feasibility: weights frozen, so backward stores activations only — small at chunk geometry.
The video-only build (§5) supplies the headroom.

**Variant worth computing alongside** — the Gauss-Newton term `‖∂f/∂ξ_h‖²` via K ≈ 8–32
random-projection backwards (`(u * f).sum().backward()` for random unit `u`). It measures
how much the *output* moves, independent of any target choice. The two should broadly agree;
disagreement is itself a finding.

**Validation:** exact leave-one-out ablation (`ξ_h = 0`, measure ΔT0) on ~200 randomly
sampled heads; report Spearman correlation against each estimator. If ρ < ~0.6 the estimator
is not usable at that sparsity and the iterative schedule below has to carry more of the load.

### 2c. Handling redundancy without training — iterate and re-solve

The known weakness of any independent ranking is redundancy: two heads that duplicate each
other both score high, and removing both is catastrophic even though removing either is free.
Since the whole thesis is that the refine task makes heads redundant, this must be addressed.
Two training-free mechanisms, used together:

1. **Iterative prune-and-rescore** (Michel et al.'s own iterative variant). Remove the budget
   in ~6 increments, recomputing scores after each. Once one of a redundant pair is gone the
   other's score rises on its own, so the schedule self-corrects. Costs 6× the scoring pass
   — cheap next to a single forward-heavy sweep, and it is the single biggest quality lever
   available without training.
2. **Least-squares re-solve of `to_out[0]` on the surviving heads.** Exactly the same
   closed-form trick §8 applies to the FFN, applied to attention: after choosing the kept
   head set `K`, re-solve the output projection so the surviving heads best reproduce the
   *full* attention output on calibration data. This directly compensates for removed
   redundant heads by re-mixing what remains, and needs no gradients.

```python
# X: (H*D, N) pre-projection activations, Y = W_out @ X (the full block-attention output)
idx = head_index(keep, D)                        # kept row indices
XK  = X[idx]                                     # (k, N)
G   = XK @ XK.T; G += lam * torch.eye(len(idx), device=G.device) * G.diagonal().mean()
W_new = torch.linalg.solve(G, XK @ Y.T).T        # (dim, k) -- ridge, fp32
attn.to_out[0].weight.data = W_new.to(dtype)     # bias untouched (per-output-dim)
```
Accumulate `G` and `XK @ Yᵀ` **streaming** over calibration batches; never materialize `X`.

**Ceiling, stated honestly:** these two recover much of what learned gates would buy, but
not all of it — gates optimize the selection jointly rather than greedily. Expect the
training-free ceiling around **25–40%** of video-branch parameters at the quality gates.
Beyond that, switch to the companion training plan.

**Gate:** a head mask at the target budget whose T0 is within threshold and whose T2 rollout
does not drift; plus a written comparison of 2a vs 2b (and their iterative variants) at
matched sparsity — the deliverable that answers "which strategy is better".

---

## 8. Phase 3 — FFN pruning  (~2 days)

The FFN is **6.44 B of the 12.88 B executed params** — the biggest prize. Method:
**structured intermediate-channel pruning with least-squares reconstruction.** Unstructured
(Wanda-style) sparsity is not used: it gives zero wall-clock gain on A6000 in bf16, since no
dense kernel skips zeros. 2:4 semi-structured remains available as an *orthogonal* layer
later (§11), not as the method.

Channel `j` contributes `gate ⊙ (W_out[:, j] · a_j(x))`. Dropping it removes one row of
`ff.net.0.proj` and one column of `ff.net.2` — a real FLOP and bandwidth cut, no sparse
kernels.

**1. Score** (activation-aware, output-aware — Wanda's idea lifted to whole channels):
`S_j = sqrt(E_x[a_j(x)²]) · ‖W_out[:, j]‖₂`, over Phase 1 states **at the refine sigmas only**.

```python
@torch.no_grad()
def channel_stats(model, states):
    sq, n = {}, {}
    def mk(name, ff):
        def hook(mod, args):
            (a,) = args                                    # (B,T,inner) post-GELU
            sq[name] = sq.get(name, 0) + a.float().pow(2).sum(dim=(0, 1)).cpu()
            n[name]  = n.get(name, 0) + a.shape[0] * a.shape[1]
        return ff.net[2].register_forward_pre_hook(hook)
    ...
    return {k: (sq[k] / n[k]).sqrt() for k in sq}            # RMS activation per channel

def channel_scores(rms, ff):
    return rms * ff.net[2].weight.float().norm(dim=0).cpu()
```

**2. Per-layer budget allocation.** Measure each block's FFN branch contribution ratio
`E‖gate ⊙ ff(x)‖ / E‖x‖` at the refine sigmas — `vgate_mlp` is σ-dependent, so a layer whose
FFN is barely gated in at low σ should be pruned harder. Allocate the global budget
proportionally, not uniformly.

**3. Least-squares output reconstruction — the step that makes this work.** After choosing
the kept set `K`, re-solve the output projection on the kept channels:

```python
# A: (inner, N) calibration activations; Y = W_out @ A (the full-FFN output, (dim, N))
A_K = A[keep]                                       # (k, N)
G   = A_K @ A_K.T                                   # (k, k)
G  += lam * torch.eye(G.shape[0], device=G.device) * G.diagonal().mean()
W_new = torch.linalg.solve(G, A_K @ Y.T).T          # (dim, k) -- ridge, fp32
ff.net[2].weight.data = W_new.to(dtype)
```
Accumulate `G` and `A_K @ Yᵀ` **streaming** over calibration batches — never materialize `A`
in full. At `k ≈ 8192` the solve is seconds per layer in fp32. This is the same routine §7.2c
uses for attention; write it once in `scripts/prune/lstsq.py` and call it from both.

**4. Iterate** 3–4 rounds of (score → drop a quarter of the budget → reconstruct →
recollect activations). Activation statistics shift after each round; one shot leaves quality
on the table.

**5. Biases.** Prune `ff.net.0.proj.bias` entries for dropped channels on 2.3;
**2.5 has no FFN biases at all** (`ff_bias=false`) — branch on `caps.ff_bias`, and note that
`ff.net.2.bias` is per-output-dim and never sliced.

### Interaction with head pruning
Alternate, do not run independently: removing heads changes the FFN input distribution and
vice versa. Schedule: heads 50% of budget → FFN 50% → recollect → heads rest → FFN rest →
final reconstruction of both.

**Gate:** T0/T1 within threshold at the FFN budget; reconstruction demonstrably beats naive
masking (report both numbers).

---

## 9. Phase 4 — export and integration  (~2 days)

The only part requiring real `ltx-core` changes, because per-layer heterogeneous head counts
and FFN widths are not currently expressible:

1. `FeedForward.__init__` — accept an explicit `inner_dim` (keep `mult` as the default path).
2. `TransformerConfig` / `BasicAVTransformerBlock` — separate head counts for `attn1` and
   `attn2` (today one `heads` field feeds both) plus an explicit `ff_inner_dim`.
3. `LTXModel.__init__` — accept per-layer lists where it takes scalars, defaulting to
   `[scalar] * num_layers`.
4. `LTXModelConfigurator` / `LTXVideoOnlyModelConfigurator` — read new optional config keys
   (`per_layer_video_attn1_heads`, `per_layer_video_attn2_heads`, `per_layer_ff_inner_dim`)
   with backwards-compatible defaults so existing checkpoints are untouched.
5. **`scripts/prune/export_pruned.py`** — surgery in the *checkpoint* key namespace (§2),
   writing a new safetensors with an updated `config` blob (preserve `model_version`, add a
   `pruning` provenance block: model key, teacher hash, budgets, method, calibration-set hash).
6. **Verify the RoPE head-count assumption** (§2) before trusting any exported model —
   note 2.5 sets `frequencies_precision=float64`, so check both precision paths.
7. Check `LTXV_MODEL_COMFY_RENAMING_MAP` and the block-streaming / quantization / compile
   paths do not assume uniform shapes.
8. **`vae_refine_sliding_window.py`**: `--model {2.3,2.5}` via the registry (plus its
   per-component overrides), `--sampler`, `--video-only`, `--pruned-checkpoint`. Mostly done
   in Phase 0; what remains is accepting a pruned transformer and validating that pruned
   `ModelCaps` round-trips. Keep the carryover / overlap semantics untouched — they are
   load-bearing and heavily commented — beyond replacing the literal `8` with the probed
   temporal scale factor.

### Coding guidance — the surgery, and the test that must pass

```python
P = "model.diffusion_model.transformer_blocks"

def prune_heads(sd, layer, kind, keep, D):
    """keep: LongTensor of kept head indices. Per-head slices are contiguous."""
    idx = (keep[:, None] * D + torch.arange(D)[None, :]).reshape(-1)
    b = f"{P}.{layer}.{kind}"
    for proj in ("to_q", "to_k", "to_v"):
        sd[f"{b}.{proj}.weight"] = sd[f"{b}.{proj}.weight"][idx, :]
        sd[f"{b}.{proj}.bias"]   = sd[f"{b}.{proj}.bias"][idx]
    for nrm in ("q_norm", "k_norm"):
        sd[f"{b}.{nrm}.weight"]  = sd[f"{b}.{nrm}.weight"][idx]
    sd[f"{b}.to_out.0.weight"]   = sd[f"{b}.to_out.0.weight"][:, idx]   # columns
    # to_out.0.bias is per-output-dim -> untouched
    if f"{b}.to_gate_logits.weight" in sd:                              # rows are per-head
        sd[f"{b}.to_gate_logits.weight"] = sd[f"{b}.to_gate_logits.weight"][keep, :]
        sd[f"{b}.to_gate_logits.bias"]   = sd[f"{b}.to_gate_logits.bias"][keep]

def prune_ffn(sd, layer, keep):
    b = f"{P}.{layer}.ff"
    sd[f"{b}.net.0.proj.weight"] = sd[f"{b}.net.0.proj.weight"][keep, :]
    if f"{b}.net.0.proj.bias" in sd:                # 2.3 only; 2.5 has ff_bias=false
        sd[f"{b}.net.0.proj.bias"] = sd[f"{b}.net.0.proj.bias"][keep]
    sd[f"{b}.net.2.weight"] = sd[f"{b}.net.2.weight"][:, keep]          # bias untouched
```

**The non-negotiable equivalence test**: the exported checkpoint, loaded fresh, must match
the hook-masked in-memory model **bit-for-bit** on a fixed state:

```python
masked   = run_with_hooks(full_model, mask, state)   # xi in {0,1} via the to_out[0] hook
exported = run(load(pruned_path), state)
assert torch.equal(masked, exported)
```
Run it after *every* exporter change, and note it only holds **before** any least-squares
re-solve (which deliberately changes the weights). Sequence: verify exactness on a
mask-only export first, then apply reconstruction and fall back to a T0 tolerance check.

---

## 10. Targets and go/no-go gates

| Metric | Gate |
|---|---|
| T0 latent rel-L2 vs source (held-out chunks) | Report baseline and candidate delta |
| T1 PSNR vs source decode | Report baseline and candidate delta |
| T2 200-chunk AR rollout | PSNR-vs-source slope; no monotone brightness/saturation drift beyond the unpruned model's own floor |
| T3 | no reviewer-visible difference on 5 clips, judged on the **MP4s** (temporal artifacts do not show in the still grid); PNG grid + MP4 both written |
| Speed | measured ms/step at chunk ∈ {1,2,3}; **≥1.4× at 30% params off** |

**Target: 30% of executed (video-branch) parameters removed**, training-free, with the gates
held. Realistic training-free band is 25–40% (§7.2c); anything past that is the companion
training plan's job. Report the achieved fraction and the gate margins, not just pass/fail —
the margin is what tells the training plan how much headroom it is starting from.

**Reality check on "real-time":** structured pruning alone buys ≤2×. Realtime AR refinement
at 1024×1024 needs the stack — pruning × FP8 (`ltx-kernels` blockwise GEMM, ~2× on the
GEMMs) × step reduction (k2→k1) × `torch.compile` + CUDA graphs × resolution × the
prompt/KV caches from Phase 0. Measure each independently; prune **before** quantizing and
re-run calibration afterwards.

### Coding guidance

One `scripts/prune/gates.py` emitting a single JSON verdict, so "did it pass" is never an ad
hoc judgement:

```bash
conda run -n ltx python -m scripts.prune.gates --model 2.5 \
    --checkpoint expr/refiner_prune/2.5/<run>/pruned.safetensors --rollout-chunks 200
# -> {"T0": 0.017, "T1_psnr": 39.4, "T2_slope_db_per_100": -0.2, "speedup": 1.52, "pass": true}
```

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Self-distillation loss makes the mask-gradient estimator identically zero | §6 uses the VAE-encoded source target + renoised states; documented in §7.2b |
| Independent head ranking ignores redundancy | Iterative prune-and-rescore + least-squares `to_out[0]` re-solve (§7.2c); training-free ceiling stated honestly, with the companion plan beyond it |
| AR drift over long rollouts | T2 gate mandatory before any export; ≥200 chunks |
| Least-squares re-solve overfits the calibration set | Ridge term, held-out clip split, and T0 reported on held-out clips only |
| Gradient scoring OOM at 21 B on 49 GB | Video-only build (−11 GB) + chunk geometry + frozen weights (no grad buffers) + `ltx_core.block_streaming` fallback |
| Speedup fails to materialize at chunk = 1 (MFU drop) | Measure in Phase 0 before committing to a budget; the parameter-count win survives either way, the multiplier may not |
| A 2.3-derived mask applied to 2.5 (or vice versa) | Artifacts namespaced by model key; key + transformer hash in provenance; loader refuses a mismatch |
| 2.5's ancestral sampler silently changes refine quality | Registry owns the sampler; `--sampler {euler,ancestral,auto}` with a one-off A/B in Phase 1 |
| 2.5 decode cost dominates the calibration loop (diffusion decoder) | Download the conv VAE variant (§3.3); until then keep T1/T2 decodes off the inner loop |
| Per-layer shapes break streaming / quantization / compile | Equivalence test in §9 plus a pass over those three code paths |
| Pruned model reused for general generation | Name it `*-refiner-pruned-*`, stamp `pruning` provenance, document loudly |

---

## 12. Deliverables

```
scripts/prune/
  refine_task.py      # the deployment conditions, single source of truth (§1)
  model_registry.py   # --model {2.3,2.5} -> ModelPaths + sigmas + sampler + probed ModelCaps (§4)
  preflight.py        # path/caps/GPU validation, called by every script (§3)
  hooks.py            # head/FFN masks + activation collection on to_out[0] and ff.net[2] (§2)
  timing.py           # StageTimer + FLOP counting (§5)
  chunk_states.py     # AR-geometry calibration states (frozen ctx + 1-3 fresh latent frames) (§6)
  teacher.py          # source-latent targets + on-policy/renoised state families (§6)
  losses.py           # masked x0-space loss and rel_l2 (§6)
  metrics.py          # T0 latent / T1 pixel / T2 AR-rollout / T3 grids + MP4s (§6)
  bench_refiner.py    # Phase 0 latency + FLOP baseline table (§5)
  head_scores.py      # 2d block ablation, 2a contribution norm, 2b Michel + Gauss-Newton (§7)
  ffn_scores.py       # channel scores + per-layer budget allocation (§8)
  lstsq.py            # streaming ridge re-solve, shared by attention (§7.2c) and FFN (§8)
  prune_schedule.py   # the alternating, iterative head/FFN budget schedule (§7.2c, §8)
  export_pruned.py    # checkpoint surgery + provenance + bit-exactness test (§9)
  gates.py            # one JSON verdict per candidate (§10)
  README.md           # how to run the whole thing, in order
expr/refiner_prune/<model-key>/<run-id>/   # scores, masks, metrics -- per generation
expr/refiner_prune/<model-key>/<run-id>/figures/   # PNG plots/grids, T3 MP4s, INDEX.md
plans/2026-08-26-refiner-head-ffn-pruning.md      # this file
plans/2026-08-27-refiner-pruning-training.md      # the training follow-on
```

Findings report at `expr/refiner_prune/<model-key>/FINDINGS.md`, following the
`scripts/ltx23_diag/README.md` § "Writing the report" conventions (numbers and figures
produced mechanically; prose written from `analysis_summary.json` + `figures/INDEX.md`).

## 13. Schedule (single-owner, 8 GPUs available)

| Phase | Days |
|---|---|
| 0 registry (§4) + free wins + baseline (§5) | 1 |
| 1 harness (§6) | 1 |
| 2 head pruning — 2d → 2a → 2b → iterate + re-solve (§7) | 3 |
| 3 FFN pruning (§8) | 2 |
| 4 export + integration (§9) | 2 |
| **total** | **~9 days** |

Phase 2's estimators parallelize across GPUs (one method or one budget per card). Phase 4's
`ltx-core` changes can start during Phase 2 — they do not depend on the scores. The training
follow-on adds ~4 days on top, and only after this plan's gates are measured.
