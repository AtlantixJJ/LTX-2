"""Cache text conditioning by model and prompt; verify cached tensors on request."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from ltx_pipelines.utils.blocks import PromptEncoder
from scripts.prune.core import artifacts, model_registry, preflight, provenance
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

    Re-encode the selected prompt and compare with ``torch.equal``. A tolerance
    would hide changed conditioning bytes in a supposedly identical cached input.
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

    from scripts.prune.core.session import DEFAULT_PROMPT, DTYPE

    ctx = get_or_build(model, DEFAULT_PROMPT, DTYPE, device)
    print(f"prompt context {tuple(ctx.shape)} {ctx.dtype} -> {cache_path(model.key, DEFAULT_PROMPT)}")

    if not args.verify:
        return 0

    report = verify(model, DEFAULT_PROMPT, torch.bfloat16, device)
    out_path = artifacts.gate(model.key, "prompt_cache_check")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({**report, "provenance": provenance.stamp(model, device)}, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    return 0 if report["bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
