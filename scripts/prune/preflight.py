"""Path/caps/GPU validation, called by every scripts/prune/* script at start-up.

Fails in ~1s on a wrong env or half-downloaded checkpoint pack instead of ~25s
into a real run. See plans/2026-08-26-refiner-head-ffn-pruning.md §3.

    conda run -n ltx python -m scripts.prune.preflight --model 2.5
    conda run -n ltx python -m scripts.prune.preflight --model 2.5 --dump-caps

``--dump-caps`` writes ``expr/refiner_prune/<key>/caps.json`` -- the Phase 0 gate's
"``ModelCaps`` dumped to JSON per generation" (plan §5). Printing to stdout is not
the same thing: later phases key budgets off ``num_heads`` / ``ff_inner_dim``, and
those have to be on disk next to the scores that used them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import torch

from scripts.prune import artifacts, model_registry, provenance
from scripts.prune.model_registry import RefinerModel


def free_gpus(min_free_gb: float = 4.0) -> list[dict]:
    """Per-device free/total memory (bytes -> GB) for CUDA devices with at least
    ``min_free_gb`` unallocated. Uses ``torch.cuda.mem_get_info`` directly rather
    than shelling out to ``nvidia-smi``, so it works wherever torch/CUDA does.
    """
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        free_gb, total_gb = free_b / 1e9, total_b / 1e9
        if free_gb >= min_free_gb:
            out.append({"index": i, "free_gb": round(free_gb, 1), "total_gb": round(total_gb, 1)})
    return out


def check(
    key: str,
    *,
    sampler: str = "euler",
    min_free_gb: float = 4.0,
    gpu_id: int | None = None,
    transformer_path: str | Path | None = None,
    text_encoder_path: str | Path | None = None,
    video_vae_path: str | Path | None = None,
) -> RefinerModel:
    """Resolve *key* to a :class:`RefinerModel`, asserting the distilled sigma
    schedule and a usable GPU exist. Raises ``SystemExit`` with an actionable
    message (missing file, busy or nonexistent device) rather than a bare
    traceback. Call this at the top of every scripts/prune/* entry point.

    When *gpu_id* is given it is checked directly -- a caller that passes
    ``--gpu-id 3`` cares whether device 3 is free, not whether *some* device is,
    and the old "any free GPU" check happily green-lit a run that then OOMed on
    the busy device it was actually pointed at.
    """
    model = model_registry.resolve(
        key,
        sampler=sampler,
        transformer_path=transformer_path,
        text_encoder_path=text_encoder_path,
        video_vae_path=video_vae_path,
    )

    if len(model.sigmas) != 9 or model.sigmas[0] != 1.0 or model.sigmas[-1] != 0.0:
        raise SystemExit(
            f"Unexpected distilled sigma schedule for --model {key}: {model.sigmas} "
            "(expected the 9-value DISTILLED_SIGMA_VALUES schedule, sigma[0]=1.0, sigma[-1]=0.0)."
        )

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA available. Are you in the `ltx` conda env on a GPU host?")

    if gpu_id is not None:
        if gpu_id >= torch.cuda.device_count():
            raise SystemExit(
                f"--gpu-id {gpu_id} does not exist (torch sees {torch.cuda.device_count()} device(s); "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})."
            )
        free_gb = torch.cuda.mem_get_info(gpu_id)[0] / 1e9
        if free_gb < min_free_gb:
            others = [g["index"] for g in free_gpus(min_free_gb)]
            raise SystemExit(
                f"--gpu-id {gpu_id} has only {free_gb:.1f} GB free (need >= {min_free_gb}). "
                f"Free devices right now: {others or 'none'}. `nvidia-smi` to see what's busy."
            )
        # Make the requested device *current*, not just the one tensors are allocated on.
        # Triton launches kernels on the current device's context, so the 2.5 diffusion VAE
        # decoder's neighborhood-attention fallback dies with "Pointer argument (at 0) cannot
        # be accessed from Triton (cpu tensor?)" when the tensors live on cuda:N while cuda:0
        # is still current. Setting it once here fixes it for every script.
        torch.cuda.set_device(gpu_id)
    elif not free_gpus(min_free_gb):
        raise SystemExit(f"No CUDA device with >= {min_free_gb} GB free. `nvidia-smi` to see what's busy.")

    return model


def dump_caps(model: RefinerModel) -> Path:
    """Write ``expr/refiner_prune/<key>/caps.json`` (plan §5 gate)."""
    path = artifacts.gate(model.key, "caps")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "caps": dataclasses.asdict(model.caps),
                "sigmas": model.sigmas,
                "stepper_kind": model.stepper_kind,
                "provenance": provenance.stamp(model),
            },
            indent=2,
        )
    )
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="2.5", choices=model_registry.SUPPORTED_MODELS)
    ap.add_argument("--sampler", default="euler", choices=model_registry.SAMPLER_CHOICES)
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--dump-caps", action="store_true", help="Write expr/refiner_prune/<key>/caps.json.")
    args = ap.parse_args()

    model = check(args.model, sampler=args.sampler, min_free_gb=args.min_free_gb, gpu_id=args.gpu_id)

    print(f"model:            {model.key} (version {model.version})")
    print(f"transformer:      {model.paths.transformer()}")
    print(f"text_encoder:     {model.paths.text_encoder_path}")
    print(f"video_vae:        {model.paths.video_vae()}")
    print(f"stepper:          {model.stepper_kind}")
    print(f"scale_factors:    {tuple(model.scale_factors)} (from {model.scale_factors_source})")
    print(f"sigmas:           {model.sigmas}")
    print("caps:")
    print(json.dumps(dataclasses.asdict(model.caps), indent=2))
    print("free GPUs (>= {:.0f} GB):".format(args.min_free_gb))
    print(json.dumps(free_gpus(args.min_free_gb), indent=2))
    if args.dump_caps:
        print(f"wrote {dump_caps(model)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
