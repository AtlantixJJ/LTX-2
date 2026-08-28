"""Precompute + cache the refiner's constant text conditioning to disk.

The refine task uses exactly one prompt (``refine_task.REFINE_PROMPT``), so
encoding it once per (model, prompt) and caching the tensor removes the text
encoder -- 26 GB on 2.5 -- and the embeddings connector from every calibration
run and from deployment entirely. See plans/2026-08-26-refiner-head-ffn-pruning.md
§5 item 3.

    conda run -n ltx python -m scripts.prune.data.prompt_cache --model 2.5 --gpu-id 0 --verify

Cache key is (model key, prompt hash): the model key because video-encoding
dim/values differ across generations (gemma3 vs gemma4, different connector),
the prompt hash so changing ``REFINE_PROMPT`` invalidates stale caches instead
of silently reusing them.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from ltx_pipelines.utils.blocks import PromptEncoder
from scripts.prune.core import artifacts, model_registry, preflight, provenance, refine_task
from scripts.prune.core.model_registry import RefinerModel

DEFAULT_CACHE_DIR = artifacts.OUT_ROOT / "prompt_cache"


def cache_path(model_key: str, prompt: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    digest = hashlib.sha1(prompt.encode()).hexdigest()[:8]
    return cache_dir / f"prompt_ctx_{model_key}_{digest}.pt"


def get_or_build(
    model: RefinerModel,
    prompt: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> torch.Tensor:
    """Return the cached ``video_encoding`` tensor for *prompt* under *model*,
    building (and caching) it via a real text-encoder pass if not already cached.
    """
    path = cache_path(model.key, prompt, cache_dir)
    if path.exists() and not force:
        return torch.load(path, map_location=device).to(dtype=dtype, device=device)

    with torch.no_grad():
        prompt_encoder = PromptEncoder(model.paths, dtype, device)
        (ctx,) = prompt_encoder([prompt])
        video_encoding = ctx.video_encoding.detach()
    del prompt_encoder
    gc.collect()
    torch.cuda.empty_cache()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(video_encoding.cpu(), path)
    return video_encoding.to(dtype=dtype, device=device)


def verify(
    model: RefinerModel,
    prompt: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict:
    """Assert the on-disk cache is bit-for-bit what the text encoder produces.

    The Phase 0 gate (plan §5) asks for exactly this: the cache is only a free win
    if it is *indistinguishable* from running the encoder, and the whole point of
    the cache is that the encoder is never loaded again to notice otherwise. Loads
    the 26 GB text encoder once, re-encodes, and compares with ``torch.equal`` --
    a tolerance would be the wrong test, since a deterministic re-encode of a
    constant string on the same device must reproduce identical bits.
    """
    path = cache_path(model.key, prompt, cache_dir)
    if not path.exists():
        raise SystemExit(f"No prompt cache to verify at {path}; run get_or_build() first.")
    cached = torch.load(path, map_location=device).to(dtype=dtype, device=device)
    fresh = get_or_build(model, prompt, dtype, device, cache_dir=cache_dir, force=True)
    equal = torch.equal(cached, fresh)
    return {
        "cache_path": str(path),
        "sha256": provenance.file_sha256(path),
        "shape": list(cached.shape),
        "dtype": str(cached.dtype),
        "bit_exact": equal,
        "max_abs_diff": float((cached.float() - fresh.float()).abs().max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="2.5", choices=model_registry.SUPPORTED_MODELS)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="Re-run the text encoder and assert bit-exactness.")
    args = ap.parse_args()

    model = preflight.check(args.model, gpu_id=args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    ctx = get_or_build(model, refine_task.REFINE_PROMPT, torch.bfloat16, device)
    print(f"prompt context {tuple(ctx.shape)} {ctx.dtype} -> {cache_path(model.key, refine_task.REFINE_PROMPT)}")

    if not args.verify:
        return 0

    report = verify(model, refine_task.REFINE_PROMPT, torch.bfloat16, device)
    out_path = artifacts.gate(model.key, "prompt_cache_check")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({**report, "provenance": provenance.stamp(model, device)}, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    return 0 if report["bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
