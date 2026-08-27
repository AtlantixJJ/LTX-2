"""Per-sigma text cross-attention K/V cache (plan §5 item 4).

``attn2``'s keys and values depend only on the (constant) prompt context and the
sigma-dependent AdaLN modulation applied to it, never on the video latent -- so on
a k2 schedule they are 2 tensors per projection per layer (2 forwards), reused
across every window, every chunk and every AR step. With 48 blocks and a
1024-token context that is 48 x 2 x (1024 x 4096) GEMM per forward eliminated.

Why not the plan's sketch. §5 proposes ``attn.to_k = lambda c: ...``; that raises
``TypeError: cannot assign 'function' as child module 'to_k'`` -- ``nn.Module.
__setattr__`` refuses to replace a registered submodule with a non-Module. It also
keys the memo on ``id(context)``, which is a freshly-allocated modulated tensor on
every call (so it would never hit) and whose id is reusable after free (so if it
did hit, it could hit *wrongly*). This module instead swaps in a real
``nn.Module`` wrapper and keys on the sigma the caller declares.

Correctness rests on one assumption -- that nothing else varies K/V within a sigma
-- so ``verify()`` asserts a cached run is **bit-for-bit** identical to an uncached
one before any measurement trusts it, which is the Phase 0 gate's wording.

    with CrossKVCache(transformer) as cache:
        for step_idx in range(len(sigmas) - 1):
            cache.set_sigma(float(sigmas[step_idx]))
            result, _ = denoiser(transformer, state, None, sigmas, step_idx)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


class _CachedProjection(nn.Module):
    """Wraps one ``to_k``/``to_v`` Linear, memoizing its output per sigma key.

    Registered as a child module (so it survives ``nn.Module.__setattr__``) and
    holding the original as *its* child, so parameters, dtype/device moves and
    ``state_dict`` keys are unchanged apart from one extra ``.inner`` level --
    which is why :meth:`CrossKVCache.detach` restores the originals rather than
    leaving the wrapper in place.
    """

    def __init__(self, inner: nn.Module, owner: "CrossKVCache") -> None:
        super().__init__()
        self.inner = inner
        self._owner = [owner]  # list, so the owner is not registered as a submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        owner = self._owner[0]
        key = owner.sigma_key
        if key is None:
            return self.inner(x)
        hit = self._cache.get(key)
        if hit is not None:
            owner.hits += 1
            return hit
        out = self.inner(x)
        self._cache[key] = out
        owner.misses += 1
        return out

    # Per-instance cache, created lazily so it is never part of the module state.
    @property
    def _cache(self) -> dict:
        cache = self.__dict__.get("_kv_cache")
        if cache is None:
            cache = {}
            self.__dict__["_kv_cache"] = cache
        return cache

    def clear(self) -> None:
        self.__dict__["_kv_cache"] = {}


@dataclass
class CrossKVCache:
    """Attach/detach per-sigma K/V memoization on every block's ``attn2``.

    ``transformer`` is the built model (an ``X0Model``); the block list is found
    via ``transformer_blocks`` on it or on its wrapped inner model.
    """

    transformer: object
    sigma_key: float | None = None
    hits: int = 0
    misses: int = 0
    _wrapped: list[tuple[nn.Module, str, nn.Module]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._blocks = _transformer_blocks(self.transformer)

    def attach(self) -> "CrossKVCache":
        if self._wrapped:
            raise RuntimeError("CrossKVCache is already attached")
        for block in self._blocks:
            attn = block.attn2
            for name in ("to_k", "to_v"):
                original = getattr(attn, name)
                self._wrapped.append((attn, name, original))
                setattr(attn, name, _CachedProjection(original, self))
        return self

    def detach(self) -> None:
        for attn, name, original in reversed(self._wrapped):
            setattr(attn, name, original)
        self._wrapped.clear()
        self.sigma_key = None

    def set_sigma(self, sigma: float | torch.Tensor | None) -> None:
        """Declare which sigma the next forward runs at (``None`` disables caching)."""
        self.sigma_key = None if sigma is None else float(sigma)

    def clear(self) -> None:
        for attn, name, _ in self._wrapped:
            getattr(attn, name).clear()
        self.hits = self.misses = 0

    @property
    def cached_tensors(self) -> int:
        return sum(len(getattr(attn, name)._cache) for attn, name, _ in self._wrapped)

    @property
    def cached_bytes(self) -> int:
        total = 0
        for attn, name, _ in self._wrapped:
            for t in getattr(attn, name)._cache.values():
                total += t.numel() * t.element_size()
        return total

    def __enter__(self) -> "CrossKVCache":
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.detach()


def _transformer_blocks(transformer: object) -> list[nn.Module]:
    """Find the block list on an ``X0Model`` / wrapper / bare ``LTXModel``.

    ``DiffusionStage`` hands out an ``X0Model``, which holds the real transformer
    as ``velocity_model``; ``BatchSplitAdapter`` and the streaming wrapper nest one
    level further. Walk the known wrapper attributes rather than assuming a depth.
    """
    seen: set[int] = set()
    stack = [transformer]
    while stack:
        holder = stack.pop()
        if holder is None or id(holder) in seen:
            continue
        seen.add(id(holder))
        blocks = getattr(holder, "transformer_blocks", None)
        if blocks is not None:
            return list(blocks)
        for attr in ("velocity_model", "model", "inner", "module", "wrapped"):
            stack.append(getattr(holder, attr, None))
    raise AttributeError(f"Could not find `transformer_blocks` on {type(transformer).__name__}")


def run_schedule(denoiser, transformer, state, sigmas, stepper, step_state, cache: CrossKVCache | None):
    """Run the full k-step tail, returning every step's ``denoised`` prediction.

    Shared by :func:`verify` and the benchmark so the cached and uncached paths
    are literally the same code with a different ``cache`` argument.
    """
    outs = []
    local = state
    for step_idx in range(len(sigmas) - 1):
        if cache is not None:
            cache.set_sigma(float(sigmas[step_idx]))
        result, _ = denoiser(transformer, local, None, sigmas, step_idx)
        outs.append(result.denoised)
        local = step_state(local, result.denoised, stepper, sigmas, step_idx)
    return outs


def verify(denoiser, transformer, state, sigmas, stepper, step_state) -> dict:
    """Assert the cached path is bit-for-bit identical to the uncached path.

    Runs the whole k-step tail three times from the same state: cache off, cache on
    (every projection a miss, so the cache is populated), and cache on again (every
    projection a **hit**). The third pass is the one that matters -- the first
    cached pass only proves the wrapper forwards to the original module, which is
    trivially true; deployment reuses the cache across windows and AR chunks, so
    the hit path is what every later measurement actually runs on.

    Bit-for-bit, not a tolerance: the cache is supposed to return *the same tensor*
    the projection would have produced, so any difference means the K/V depend on
    something the sigma key does not capture.
    """
    with torch.no_grad():
        uncached = [t.clone() for t in run_schedule(denoiser, transformer, state, sigmas, stepper, step_state, None)]
        with CrossKVCache(transformer) as cache:
            fill = [t.clone() for t in run_schedule(denoiser, transformer, state, sigmas, stepper, step_state, cache)]
            fill_stats = {"hits": cache.hits, "misses": cache.misses}
            cache.hits = cache.misses = 0
            reuse = [t.clone() for t in run_schedule(denoiser, transformer, state, sigmas, stepper, step_state, cache)]
            stats = {
                "fill_pass": fill_stats,
                "reuse_pass": {"hits": cache.hits, "misses": cache.misses},
                "cached_tensors": cache.cached_tensors,
                "cached_mb": round(cache.cached_bytes / 1e6, 1),
            }

    if stats["reuse_pass"]["misses"]:
        raise AssertionError(
            f"reuse pass still took {stats['reuse_pass']['misses']} cache misses -- the sigma key is "
            "not stable across identical schedules, so the cache is not doing what it claims."
        )

    steps = []
    all_equal = True
    for i, (a, b, c) in enumerate(zip(uncached, fill, reuse)):
        fill_equal = torch.equal(a, b)
        reuse_equal = torch.equal(a, c)
        all_equal &= fill_equal and reuse_equal
        steps.append(
            {
                "step": i,
                "sigma": float(sigmas[i]),
                "bit_exact_fill": fill_equal,
                "bit_exact_reuse": reuse_equal,
                "max_abs_diff_reuse": float((a.float() - c.float()).abs().max()),
            }
        )
    return {"bit_exact": all_equal, "steps": steps, **stats}


def main() -> int:
    """Phase 0 gate: run the k2 tail with and without the cache on a real AR-geometry
    state and assert the two are bit-for-bit identical."""
    import argparse
    import json

    import torch as _torch

    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.model.transformer import LTXVideoOnlyModelConfigurator
    from ltx_pipelines.utils.blocks import DiffusionStage
    from ltx_pipelines.utils.denoisers import SimpleDenoiser
    from ltx_pipelines.utils.samplers import _step_state

    from scripts.prune import bench_refiner, model_registry, preflight, prompt_cache, provenance, refine_task

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--model", default="2.5", choices=model_registry.SUPPORTED_MODELS)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--n-new", type=int, default=2)
    ap.add_argument("--ctx-latent-frames", type=int, default=refine_task.CTX_LATENT_FRAMES)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = preflight.check(args.model, gpu_id=args.gpu_id)
    device = _torch.device(f"cuda:{args.gpu_id}")
    dtype = _torch.bfloat16
    video_context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, dtype, device)

    sigmas_list = refine_task.schedule_for(model.sigmas, refine_task.K_STEP)
    sigmas = _torch.tensor(sigmas_list, dtype=_torch.float32, device=device)
    video_tools, state, _, _ = bench_refiner.build_state(
        model, video_context, args.n_new, args.ctx_latent_frames, args.height, args.width,
        24.0, sigmas_list[0], device, args.seed,
    )
    stage = DiffusionStage.from_checkpoint(
        model.paths.transformer(), dtype, device,
        model_configurator=LTXVideoOnlyModelConfigurator, scale_factors=model.scale_factors,
    )
    with stage._transformer_ctx(video_tools=video_tools) as transformer:
        report = verify(
            SimpleDenoiser(video_context, None), transformer, state, sigmas, EulerDiffusionStep(), _step_state
        )

    report["geometry"] = {
        "n_new_latent_frames": args.n_new,
        "ctx_latent_frames": args.ctx_latent_frames,
        "height": args.height,
        "width": args.width,
        "context_tokens": video_context.shape[1],
    }
    out_dir = model_registry.WORKSPACE_ROOT / "expr" / "refiner_prune" / model.key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kv_cache_check.json"
    out_path.write_text(json.dumps({"provenance": provenance.stamp(model, device, script="cross_kv_cache"), **report}, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}. Overall: {'PASS' if report['bit_exact'] else 'FAIL'}")
    return 0 if report["bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
