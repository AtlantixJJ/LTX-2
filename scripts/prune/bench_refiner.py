"""Phase 0 baseline: ms/step, peak memory and FLOPs per geometry, per generation.

This table is the denominator for every later pruning speedup claim (plan §5 item
5), so it measures the geometries the refiner actually runs in:

* **isolated** (``--ctx-latent-frames 0``) -- a whole window denoised at once, what
  today's sliding-window refiner does.
* **AR** (``--ctx-latent-frames 4``) -- ``ctx`` already-refined latent frames frozen
  via ``VideoConditionByLatentIndex(strength=1.0)`` with ``n_new`` fresh frames
  after them, what the upcoming autoregressive refiner does (§3.1, §6). This
  matters: at ``n_new = 1`` the AR forward still attends over ``ctx + 1`` frames of
  keys, so an isolated 1-frame row understates the deployed cost and would inflate
  every speedup ratio computed against it.

Measured on a synthetic (randomly-initialized) latent rather than a real source
clip: transformer cost depends on token count, not latent content, and skipping
the VAE encode keeps this to "build the transformer once, sweep geometries".
The transformer weights are built **once per (compile, video_only) stage** and
reused across geometries -- positional embeddings come from the per-call state,
not from the build -- instead of paying ~20 s per row.

``torch.compile`` is a separate axis so a later pruning gain is never confounded
with a compilation gain, and the per-sigma cross-attention K/V cache (§5 item 4)
is measured as its own axis on top of the uncached baseline.

    conda run -n ltx python -m scripts.prune.bench_refiner --model 2.5 --gpu-id 2
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.model.transformer import LTXModelConfigurator, LTXVideoOnlyModelConfigurator
from ltx_core.tools import VideoLatentTools
from ltx_pipelines.utils.blocks import DiffusionStage, _build_state
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.samplers import _step_state
from ltx_pipelines.utils.types import ModalitySpec

from scripts.prune import artifacts, cross_kv_cache, geometry, model_registry, provenance, refine_task, session
from scripts.prune.model_registry import RefinerModel, WORKSPACE_ROOT
from scripts.prune.session import DTYPE
from scripts.prune.timing import StageTimer, count_flops

DEFAULT_CHUNK_LATENT_FRAMES = (1, 2, 3, 4, 16)
DEFAULT_OUT_DIR = artifacts.OUT_ROOT


def analytic_flops(caps, tokens_video: int, tokens_ctx: int) -> dict:
    """Closed-form FLOP breakdown for one video-branch forward.

    Exists because the plan's §2 roofline ("2 x 12.9e9 x 4096 ~ 105 TFLOP") counts
    only the per-video-token GEMMs and then compares that against a measured
    wall-clock, which silently attributes the context-side work to poor MFU. The
    text context is 1024 tokens on both generations, and ``attn2``'s K/V
    projections run over *all* of it on every forward at every sigma -- a cost that
    is constant in video tokens and therefore dominates at small AR chunks. Split
    the terms so the pruning target (``video_gemm``) is never confused with the
    part only the K/V cache can remove (``context_gemm``).
    """
    d = caps.num_heads * caps.head_dim
    layers = caps.num_layers
    ff = caps.ff_inner_dim

    per_token_params = (
        4 * d * d  # attn1 q,k,v,out
        + 2 * d * d  # attn2 q,out  (k,v run on the context, counted below)
        + 2 * d * ff  # ff up, down
    )
    if caps.apply_gated_attention:
        per_token_params += 2 * d * caps.num_heads  # attn1 + attn2 gate logits

    video_gemm = 2 * per_token_params * tokens_video * layers
    context_gemm = 2 * (2 * d * d) * tokens_ctx * layers  # attn2 to_k + to_v
    self_attn = 4 * tokens_video * tokens_video * d * layers  # qk^T + av
    cross_attn = 4 * tokens_video * tokens_ctx * d * layers
    total = video_gemm + context_gemm + self_attn + cross_attn
    return {
        "video_gemm": video_gemm,
        "context_gemm": context_gemm,
        "self_attn": self_attn,
        "cross_attn": cross_attn,
        "total": total,
    }


def build_state(
    model: RefinerModel,
    video_context: torch.Tensor,
    n_new: int,
    ctx_frames: int,
    height: int,
    width: int,
    fps: float,
    sigma0: float,
    device: torch.device,
    seed: int,
):
    """AR-geometry state: ``ctx_frames`` frozen latent frames + ``n_new`` fresh ones.

    ``ctx_frames = 0`` gives the isolated-window geometry. The frozen prefix is
    injected at ``latent_idx=0``; Phase 1's ``chunk_states.py`` uses ``latent_idx=1``
    to dodge the frame-0 keyframe caveat the refine script documents, which shifts
    which latent frame is frozen but not the token count -- so it changes nothing
    that this benchmark measures.
    """
    total_frames = ctx_frames + n_new
    pixel_shape, latent_shape = geometry.latent_shape_for(
        total_frames, height, width, fps, model.scale_factors, model.caps.latent_channels
    )
    video_tools = VideoLatentTools(
        VideoLatentPatchifier(patch_size=1), latent_shape, fps, scale_factors=model.scale_factors
    )

    conditionings = []
    if ctx_frames:
        ctx_latent = torch.randn(
            (1, model.caps.latent_channels, ctx_frames, latent_shape.height, latent_shape.width),
            generator=torch.Generator(device=device).manual_seed(seed + 1),
            device=device,
            dtype=DTYPE,
        )
        conditionings.append(VideoConditionByLatentIndex(latent=ctx_latent, strength=1.0, latent_idx=0))

    noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seed))
    state = _build_state(
        ModalitySpec(context=video_context, conditionings=conditionings, noise_scale=sigma0),
        video_tools,
        noiser,
        DTYPE,
        device,
    )
    return video_tools, state, latent_shape, pixel_shape


def bench_one(
    model: RefinerModel,
    transformer,
    video_context: torch.Tensor,
    *,
    n_new: int,
    ctx_frames: int,
    height: int,
    width: int,
    fps: float,
    device: torch.device,
    kv_cache: bool,
    warmup_steps: int,
    seed: int,
) -> dict:
    sigmas_list = refine_task.schedule_for(model.sigmas, refine_task.K_STEP)
    sigmas = torch.tensor(sigmas_list, dtype=torch.float32, device=device)
    stepper = EulerDiffusionStep()

    video_tools, state, latent_shape, pixel_shape = build_state(
        model, video_context, n_new, ctx_frames, height, width, fps, sigmas_list[0], device, seed
    )
    denoiser = SimpleDenoiser(video_context, None)
    tokens_video = latent_shape.token_count()
    tokens_ctx = video_context.shape[1]

    row: dict = {
        "model": model.key,
        "n_new_latent_frames": n_new,
        "ctx_latent_frames": ctx_frames,
        "geometry": "ar" if ctx_frames else "isolated",
        "latent_frames": latent_shape.frames,
        "pixel_frames": pixel_shape.frames,
        "tokens_video": tokens_video,
        "tokens_context": tokens_ctx,
        "fresh_tokens": int(state.denoise_mask.sum().item()),
        "height": height,
        "width": width,
        "kv_cache": kv_cache,
    }

    with ExitStack() as stack:
        cache = None
        if kv_cache:
            cache = stack.enter_context(cross_kv_cache.CrossKVCache(transformer))

        with torch.no_grad():
            for _ in range(max(warmup_steps, 1)):
                cross_kv_cache.run_schedule(denoiser, transformer, state, sigmas, stepper, _step_state, cache)

            fwd_s, peak_alloc_gb = [], 0.0
            local = state
            for step_idx in range(len(sigmas_list) - 1):
                if cache is not None:
                    cache.set_sigma(float(sigmas[step_idx]))
                with StageTimer(f"fwd_{step_idx}", device) as t_fwd:
                    result, _ = denoiser(transformer, local, None, sigmas, step_idx)
                fwd_s.append(t_fwd.elapsed_s)
                peak_alloc_gb = max(peak_alloc_gb, t_fwd.peak_alloc_gb)
                local = _step_state(local, result.denoised, stepper, sigmas, step_idx)

            row["fwd_s_per_step"] = fwd_s
            row["ms_per_fwd"] = 1000 * sum(fwd_s) / len(fwd_s)
            row["ms_per_k2_window"] = 1000 * sum(fwd_s)
            row["peak_alloc_gb"] = peak_alloc_gb

            if cache is not None:
                row["kv_cache_hits"] = cache.hits
                row["kv_cache_misses"] = cache.misses
                row["kv_cache_mb"] = round(cache.cached_bytes / 1e6, 1)
                cache.set_sigma(float(sigmas[0]))

            measured = count_flops(lambda: denoiser(transformer, state, None, sigmas, 0))

    row["flops_per_fwd_measured"] = measured
    row["flops_per_fwd_analytic"] = analytic_flops(model.caps, tokens_video, tokens_ctx)
    row["tflops_achieved"] = measured / (row["ms_per_fwd"] / 1000) / 1e12
    return row


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    session.add_model_args(ap)  # --model, --gpu-id, --seed
    ap.add_argument("--sampler", default="euler", choices=model_registry.SAMPLER_CHOICES)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--chunk-latent-frames", type=int, nargs="+", default=list(DEFAULT_CHUNK_LATENT_FRAMES))
    ap.add_argument(
        "--ctx-latent-frames", type=int, nargs="+", default=[0, refine_task.CTX_LATENT_FRAMES],
        help="Frozen AR context sizes to sweep. 0 = isolated window (today's refiner).",
    )
    ap.add_argument("--video-only", dest="video_only", action="store_true", default=True)
    ap.add_argument("--no-video-only", dest="video_only", action="store_false")
    ap.add_argument(
        "--compile-chunks", type=int, nargs="*", default=None,
        help="Chunk sizes to ALSO measure compiled (each variant gets its own build). Omitted = eager only.",
    )
    ap.add_argument(
        "--compile-variants", nargs="+", default=["compile", "compile+cudagraphs"],
        choices=["compile", "compile+cudagraphs"],
        help="Which compiled variants to measure when --compile-chunks is given.",
    )
    ap.add_argument("--no-kv-cache", dest="kv_cache", action="store_false", default=True,
                    help="Skip the cross-attention K/V cache axis.")
    ap.add_argument("--warmup-steps", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--tag", default="baseline",
        help="Names the output file (bench_<tag>.json) in both the run dir and the stable "
        "per-generation pointer, so a partial sweep (e.g. --tag compile) never clobbers the "
        "canonical baseline table.",
    )
    args = ap.parse_args()

    s = session.open_session(args, script="bench_refiner", sampler=args.sampler)
    model, device, video_context = s.model, s.device, s.context
    if model.stepper_kind == "ancestral":
        # EulerAncestralDiffusionStep.step() needs a per-step noise draw (eta=1.0 renoises
        # after every step); the plain _step_state() call below does not supply one. Plan §4
        # decision 1 defaults the refiner to Euler on both generations and defers the ancestral
        # A/B to Phase 1's noise-injecting loop (mirroring
        # ltx_pipelines.utils.samplers._ancestral_euler_denoising_loop) -- raise here rather
        # than silently running EulerAncestralDiffusionStep without noise.
        raise SystemExit(
            "--sampler ancestral is not wired into bench_refiner.py's step loop yet (needs the "
            "noise-injecting ancestral loop, a Phase 1 item per plan §4 decision 1). Use "
            "--sampler euler (the Phase 0 default) or auto on a 2.3 checkpoint."
        )

    configurator = LTXVideoOnlyModelConfigurator if args.video_only else LTXModelConfigurator

    compile_chunks = set(args.compile_chunks or [])
    kv_axis = [False, True] if args.kv_cache else [False]

    # (label, CompilationConfig | None). `capture=True` is the CUDA-graph variant: per-block
    # compile plus one graph over the block loop, which the plan asks to be measured separately
    # from plain compilation so neither gets folded into a later pruning number.
    variants: list[tuple[str, object]] = [("eager", None)]
    if compile_chunks:
        from ltx_core.model.transformer.compiling import CompilationConfig

        for name in args.compile_variants:
            variants.append((name, CompilationConfig(capture=(name == "compile+cudagraphs"))))

    rows: list[dict] = []
    builds: list[dict] = []

    for variant, compilation_config in variants:
        compiled = variant != "eager"
        chunks = sorted(compile_chunks) if compiled else args.chunk_latent_frames

        try:
            stage = DiffusionStage.from_checkpoint(
                model.paths.transformer(),
                DTYPE,
                device,
                model_configurator=configurator,
                compilation_config=compilation_config,
                scale_factors=model.scale_factors,
            )
        except ValueError as exc:
            # A variant this build path cannot serve is a *result*, not a crash: on the plain
            # single-GPU builder `capture=True` raises because CUDA graphs need GPU-resident
            # weights. Record the refusal so the report can cite it, and keep the variants
            # that do run.
            print(f"[bench] variant={variant} unsupported on this build path: {exc}", flush=True)
            builds.append({"variant": variant, "unsupported": str(exc)})
            continue
        # video_tools is only consulted by the tiled-data-parallel builder; the standard
        # builder ignores it and positional embeddings are computed per call from the
        # state, so one build serves every geometry below.
        probe_tools, _, _, _ = build_state(
            model, video_context, 1, 0, args.height, args.width, args.fps, 1.0, device, args.seed
        )
        with torch.no_grad(), StageTimer("transformer_build", device) as t_build:
            ctx = stage._transformer_ctx(video_tools=probe_tools)
            transformer = ctx.__enter__()
        builds.append(
            {
                "variant": variant,
                "compile": compiled,
                "video_only": args.video_only,
                "build_s": t_build.elapsed_s,
                "build_peak_alloc_gb": t_build.peak_alloc_gb,
                "resident_alloc_gb": torch.cuda.memory_allocated(device) / 1e9,
            }
        )
        print(
            f"[bench] built transformer (variant={variant}, video_only={args.video_only}) in "
            f"{t_build.elapsed_s:.1f}s, build peak {t_build.peak_alloc_gb:.1f} GB, "
            f"resident {torch.cuda.memory_allocated(device) / 1e9:.1f} GB",
            flush=True,
        )

        try:
            for chunk in chunks:
                for ctx_frames in args.ctx_latent_frames:
                    for kv in kv_axis:
                        label = (
                            f"model={model.key} n_new={chunk} ctx={ctx_frames} "
                            f"kv_cache={kv} variant={variant}"
                        )
                        print(f"[bench] {label} ...", flush=True)
                        row = bench_one(
                            model,
                            transformer,
                            video_context,
                            n_new=chunk,
                            ctx_frames=ctx_frames,
                            height=args.height,
                            width=args.width,
                            fps=args.fps,
                            device=device,
                            kv_cache=kv,
                            warmup_steps=args.warmup_steps,
                            seed=args.seed,
                        )
                        row["variant"] = variant
                        row["compile"] = compiled
                        row["video_only"] = args.video_only
                        rows.append(row)
                        print(
                            f"  -> {row['ms_per_fwd']:.1f} ms/fwd, {row['tflops_achieved']:.1f} TFLOPS, "
                            f"{row['tokens_video']} video tokens, peak {row['peak_alloc_gb']:.1f} GB",
                            flush=True,
                        )
        finally:
            ctx.__exit__(None, None, None)
            del transformer, stage
            torch.cuda.empty_cache()

    out_dir = args.out_dir or (DEFAULT_OUT_DIR / model.key / provenance.run_id(f"bench-{args.tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": s.stamp(),
        "config": {
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "k_step": refine_task.K_STEP,
            "sigmas": refine_task.schedule_for(model.sigmas, refine_task.K_STEP),
            "chunk_latent_frames": args.chunk_latent_frames,
            "ctx_latent_frames": args.ctx_latent_frames,
            "compile_chunks": sorted(compile_chunks),
            "compile_variants": [v for v, _ in variants],
            "video_only": args.video_only,
        },
        "builds": builds,
        "rows": rows,
    }
    out_path = out_dir / f"bench_{args.tag}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    # Also refresh the stable per-generation pointer so downstream scripts do not
    # have to know the run id.
    artifacts.bench(model.key, args.tag).write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
