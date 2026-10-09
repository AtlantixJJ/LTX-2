"""Render saved rollout tensors and decoder controls without loading a transformer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from ltx_trainer.video_utils import save_video
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.media import decode, decode_key, frame, native_decoder_settings, open_decoder_session
from scripts.prune.core.session import add_model_args


def differences(left: torch.Tensor, right: torch.Tensor) -> dict:
    """Keep float RGB measurements independent from PNG quantization and MP4 compression."""
    count = min(len(left), len(right))
    all_pixels, foreground = [], []
    maximum = 0.0
    for index in range(count):
        a, b = left[index].float(), right[index].float()
        if a.max() > 1.5:
            a = a / 255
        if b.max() > 1.5:
            b = b / 255
        diff = (a - b).abs()
        mask = (a.min(dim=0).values < 0.9) | (b.min(dim=0).values < 0.9)
        all_pixels.append(float(diff.mean()))
        foreground.append(float(diff.mean(dim=0)[mask].mean()) if mask.any() else 0.0)
        maximum = max(maximum, float(diff.max()))
    return {
        "frames": count,
        "max_abs": maximum,
        "all_pixel_mae_per_frame": all_pixels,
        "foreground_mae_per_frame": foreground,
        "foreground_mask": "union of comparison RGB min-channel < 0.9; derived QA mask, not training mask",
    }


def reusable(cached: dict | None, job: dict, output: Path, source_hash: str, key: str) -> bool:
    """Require decoder identity and every presentation artifact before reuse."""
    if (
        not cached
        or job.get("comparison_required")
        or cached.get("input") != job
        or cached.get("source_code_sha256") != source_hash
        or cached.get("decode_key") != key
    ):
        return False
    try:
        software.check_current(cached.get('software'))
    except ValueError:
        return False
    artifacts = [{"file": cached.get("video"), "sha256": cached.get("video_sha256")}]
    artifacts.extend(cached.get("samples", []))
    return all(
        isinstance(row.get("file"), str)
        and (output / row["file"]).is_file()
        and sha256(output / row["file"]) == row.get("sha256")
        for row in artifacts
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:  # noqa: PLR0915 -- sequential saved-job decoding and comparison emission
    args = parse_args()
    producer_software = software.capture('decoding')
    jobs = json.loads(args.jobs.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(Path(__file__))
    manifest_path = args.output / "manifest.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"jobs": [], "comparisons": []}
    completed = {row["id"]: row for row in prior["jobs"]}
    for job in jobs['jobs']:
        cached = completed.get(job['id'])
        if cached:
            try:
                software.check_current(cached.get('software'))
            except ValueError as error:
                raise ValueError('historical decoder software differs; preserve outputs and use a fresh destination') from error
    software.check_current(producer_software)
    session = open_decoder_session(args.model, args.gpu_id, script="onestep_avatar.decode_saved")
    vae_path = Path(session.model.paths.video_vae())
    vae_hash = sha256(vae_path)
    rendered = {}
    with session.decoder() as decoder, torch.inference_mode():
        for job in jobs["jobs"]:
            source = Path(job["latent"])
            digest = sha256(source)
            if digest != job["sha256"]:
                raise ValueError(f"Latent hash changed: {source}")
            destination = args.output / (job["id"] + ".mp4")
            cached = completed.get(job["id"])
            latent = torch.load(source, map_location="cpu", weights_only=True)
            if job.get("latent_frames"):
                latent = latent[:, :, : job["latent_frames"]]
            settings = native_decoder_settings()
            key = decode_key(
                digest, vae_hash, list(latent.shape), "native_decode_video", job.get("seed", args.seed), settings
            )
            if reusable(cached, job, args.output, source_hash, key):
                continue
            started = time.perf_counter()
            pixels = decode(session, latent, decoder, job.get("seed", args.seed)).cpu()
            seconds = time.perf_counter() - started
            save_video(pixels, destination, fps=job.get("fps", 30), video_format="FCHW")
            samples = []
            for index in job.get("sample_frames", [0, 32, 48, 60, 64, 65]):
                if index >= len(pixels):
                    continue
                output = args.output / (job["id"] + f"_f{index:03d}.png")
                frame(pixels, index).save(output)
                samples.append(
                    {
                        "frame": index,
                        "file": output.name,
                        "sha256": sha256(output),
                        "status": "lossless PNG of quantized decoder RGB",
                    }
                )
            row = {
                "id": job["id"],
                "input": job,
                "video": destination.name,
                "video_sha256": sha256(destination),
                "frames": len(pixels),
                "seconds": seconds,
                "source_code_sha256": source_hash,
                "software": producer_software,
                "samples": samples,
                "status": "new_decode_saved_latent",
                "decoder": {
                    "model": args.model,
                    "vae_path": str(vae_path),
                    "vae_sha256": vae_hash,
                    "torch": torch.__version__,
                    "seed": job.get("seed", args.seed),
                    "fresh_generator_per_decode": True,
                },
            }
            completed[job["id"]] = row
            row["decode_key"] = key
            row["decoder"]["settings"] = settings
            if job.get("comparison_required"):
                rendered[job["id"]] = pixels
            prior["jobs"] = list(completed.values())
            software.check_current(producer_software)
            manifest_path.write_text(json.dumps(prior, indent=2) + "\n")
            print(json.dumps({"id": job["id"], "frames": len(pixels), "seconds": seconds}), flush=True)  # noqa: T201 -- CLI progress
        comparisons = []
        for comparison in jobs.get("comparisons", []):
            left, right = rendered[comparison["left"]], rendered[comparison["right"]]
            record = {**comparison, **differences(left, right), "difference_gain": 20, "samples": [],
                      "software": producer_software}
            for index in [0, 32, 48, 60, 64]:
                if index >= min(len(left), len(right)):
                    continue
                a, b = left[index].float(), right[index].float()
                diff = (a - b).abs().mul(20).clamp(0, 1)
                path = args.output / (comparison["id"] + f"_diff20_f{index:03d}.png")
                Image.fromarray(diff.mul(255).round().byte().permute(1, 2, 0).numpy()).save(path)
                record["samples"].append({"frame": index, "file": path.name, "sha256": sha256(path)})
            comparisons.append(record)
        prior["comparisons"] = comparisons
        software.check_current(producer_software)
        manifest_path.write_text(json.dumps(prior, indent=2) + "\n")


if __name__ == "__main__":
    main()
