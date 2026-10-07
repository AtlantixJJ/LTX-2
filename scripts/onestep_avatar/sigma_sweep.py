"""Decode and score saved historical sigma-sweep tensors; see doc/sigma_sweep.md."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.onestep_avatar import dataset, evaluate, media, sigma_sweep_results, software
from scripts.onestep_avatar.hashing import sha256

LEVELS = {0.421875: ("one_step", 1), 0.725: ("official", 2),
          0.909375: ("official", 3), 1.0: ("official", 8)}
SAMPLES = (0, 16, 17, 48, 49, 63, 64, 65, 66, 96, 97, 128)


def prepare(  # noqa: PLR0912 -- ordered historical/result-bound preflight gates
    spec_path: Path, output: Path, *, require_fresh_output: bool = True
) -> tuple[dict, dict, dict, dict]:
    """Check every saved input before opening a decoder or output directory."""
    from scripts.prune.core import model_registry  # noqa: PLC0415 -- metadata resolution only

    spec = json.loads(spec_path.read_text())
    if (spec.get("schema_version") not in (1, 2) or spec.get("model") != "2.5"
            or spec.get("decode_seed") != 42 or spec.get("fps") != 30):
        raise ValueError("sigma sweep requires schema one or two, model 2.5, decode seed 42 and fps 30")
    if require_fresh_output and (output.exists() or output.is_symlink()):
        raise ValueError("sigma sweep requires a fresh output directory")
    identities = [(cell.get("sigma"), cell.get("arm")) for cell in spec.get("cells", [])]
    if (any(type(cell.get("sigma")) not in (int, float) for cell in spec.get("cells", []))
            or identities != [(sigma, arm) for sigma in LEVELS for arm in ("d0", "d1")]):
        raise ValueError("sigma sweep requires the ordered exact four-level by two-arm cell inventory")
    tensors, hashes = {}, {str(spec_path.resolve()): sha256(spec_path)}
    if spec["schema_version"] == 2:
        sigma_sweep_results.resolve_cells(spec, hashes, LEVELS)
    for role in ("capture", "guide"):
        entry = spec[role]
        path = Path(entry["path"])
        hashes[str(path.resolve())] = sha256(path)
        if hashes[str(path.resolve())] != entry["sha256"]:
            raise ValueError(f"sigma sweep {role} master bytes changed")
        master, fps = dataset.load_training_master(path)
        if fps != 30 or master.shape[1] < 17:
            raise ValueError("sigma sweep master fps or coverage differs")
        tensors[role] = master[:, :17].unsqueeze(0)
    for cell in spec["cells"]:
        path = Path(cell["path"])
        hashes[str(path.resolve())] = sha256(path)
        if hashes[str(path.resolve())] != cell["sha256"]:
            raise ValueError("sigma sweep generated tensor bytes changed")
        tensors[f"sigma{cell['sigma']:.6f}_{cell['arm']}"] = torch.load(path, map_location="cpu", weights_only=True)
    shape = tensors["capture"].shape
    if any(not isinstance(value, torch.Tensor) or value.shape != shape or value.ndim != 5
           or value.shape[0] != 1 or value.shape[2] != 17 or value.dtype != torch.bfloat16
           or not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("sigma sweep requires matching finite bf16 17-frame encodings")
    model = model_registry.resolve(spec["model"])
    if shape[1] != model.caps.latent_channels:
        raise ValueError("sigma sweep channels differ from the registered VAE")
    vae_path = Path(model.paths.video_vae()).resolve()
    if vae_path != Path(spec["vae"]["path"]).resolve() or sha256(vae_path) != spec["vae"]["sha256"]:
        raise ValueError("sigma sweep VAE identity differs from the requested decoder")
    hashes[str(vae_path)] = spec["vae"]["sha256"]
    sources = {str(Path(module.__file__).resolve()): sha256(Path(module.__file__))
               for module in (media, evaluate)}
    sources[str(Path(__file__).resolve())] = sha256(Path(__file__))
    if spec["schema_version"] == 2:
        sources[str(Path(sigma_sweep_results.__file__).resolve())] = sha256(Path(sigma_sweep_results.__file__))
    return spec, tensors, hashes, sources


def execute(spec_path: Path, output: Path, *, gpu_id: int) -> dict:
    """Decode matched saved outputs, then publish scored media and final provenance."""
    from ltx_trainer.video_utils import save_video  # noqa: PLC0415 -- presentation writer

    producer_software = software.capture('decoding')
    spec, tensors, inputs, sources = prepare(spec_path, output)
    software.check_current(producer_software)
    session = media.open_decoder_session(spec["model"], gpu_id, script="onestep_avatar.sigma_sweep")
    videos = {}
    with session.decoder() as decoder, torch.inference_mode():
        for name, tensor in tensors.items():
            pixels = media.decode(session, tensor, decoder, spec["decode_seed"]).float().cpu()
            if pixels.max() > 1.5:
                pixels = pixels / 255
            videos[name] = pixels.permute(0, 2, 3, 1).numpy()
    reference = videos["capture"]
    if (reference.shape[0] != 129 or reference.shape[-1] != 3
            or any(value.shape != reference.shape or not np.isfinite(value).all()
                   or value.min() < 0 or value.max() > 1 for value in videos.values())):
        raise ValueError("sigma sweep decoder returned mismatched or invalid RGB coverage")
    mask = np.any([value.min(-1) < 0.9 for value in videos.values()], axis=0)
    cells = []
    reference_c0 = tensors["sigma1.000000_d0"][:, :, :1]
    for cell in spec["cells"]:
        name = f"sigma{cell['sigma']:.6f}_{cell['arm']}"
        kind, calls = LEVELS[cell["sigma"]]
        cells.append({"sigma": cell["sigma"], "arm": cell["arm"], "calls_per_block": calls, "path": kind,
                      "c0_equal": torch.equal(tensors[name][:, :, :1], reference_c0),
                      **evaluate.sigma_sweep_boundary_metrics(videos[name], reference, videos["guide"], mask)})
    d0, d1 = tensors["sigma1.000000_d0"], tensors["sigma1.000000_d1"]
    metrics = {"tag": spec["tag"], "decode_seed": 42, "boundaries": [17, 33, 49, 65, 81, 97, 113],
               "post_eviction": [81, 97, 113], "cells": cells,
               "capture_motion_ref": float(evaluate.masked_rgb_transition_steps(reference, mask)[16:].mean()),
               "sigma1_d0_equals_d1": torch.equal(d0, d1),
               "sigma1_d0_vs_d1_max_abs": float((d0.float() - d1.float()).abs().max())}
    if any(sha256(Path(path)) != digest for path, digest in {**inputs, **sources}.items()):
        raise ValueError("sigma sweep inputs or producers changed during decoding")
    software.check_current(producer_software)
    output.mkdir(parents=True, exist_ok=False)
    (output / "videos").mkdir()
    (output / "frames").mkdir()
    rendered = {}
    for name, value in videos.items():
        pixels = torch.from_numpy(value).permute(0, 3, 1, 2)
        movie = output / "videos" / f"{name}.mp4"
        save_video(pixels, movie, fps=30, video_format="FCHW")
        geometry = media.verify_saved_video(movie, 129, 30)
        height, width = value.shape[1:3]
        if geometry["width"] != width or geometry["height"] != height:
            raise ValueError("sigma sweep saved video geometry differs from decoded RGB")
        samples = []
        for index in SAMPLES:
            path = output / "frames" / f"{name}_f{index:03d}.png"
            media.frame(pixels, index).save(path)
            media.verify_saved_png(path, width, height)
            samples.append({"frame": index, "path": str(path.resolve()), "sha256": sha256(path)})
        rendered[name] = {"video": str(movie.resolve()), "sha256": sha256(movie), "samples": samples,
                          "pixel_sha256": evaluate.tensor_sha256(pixels), "frames": 129, "fps": 30,
                          "width": width, "height": height}
    metrics_path = output / "metrics.json"
    dataset.atomic_write(metrics_path,
                         lambda path: path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n"))
    if any(sha256(Path(path)) != digest for path, digest in {**inputs, **sources}.items()):
        raise ValueError("sigma sweep inputs or producers changed during publication")
    software.check_current(producer_software)
    manifest = {"schema_version": 2, "kind": "onestep_avatar.saved_sigma_sweep", "spec": spec,
                "software": producer_software,
                "spec_path": str(spec_path.resolve()),
                "input_file_hashes": inputs, "producer_source_hashes": sources,
                "decode_settings": media.native_decoder_settings(), "outputs": rendered,
                "metrics": {"path": str(metrics_path.resolve()), "sha256": sha256(metrics_path)},
                "status": "complete_saved_decoding"}
    dataset.atomic_write(output / "manifest.json",
                         lambda path: path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n"))
    return manifest


def verify_completion(spec_path: Path, output: Path) -> dict:  # noqa: PLR0912 -- ordered saved-artifact gates
    """Verify exact saved sweep inputs and media without model or decoder sessions."""
    spec, tensors, inputs, sources = prepare(spec_path, output, require_fresh_output=False)
    manifest = json.loads((output / "manifest.json").read_text())
    software.check_current(manifest.get('software'))
    if (manifest.get("schema_version") != 2 or manifest.get("kind") != "onestep_avatar.saved_sigma_sweep"
            or manifest.get("status") != "complete_saved_decoding" or manifest.get("spec") != spec
            or manifest.get("spec_path") != str(spec_path.resolve())
            or manifest.get("input_file_hashes") != inputs or manifest.get("producer_source_hashes") != sources
            or manifest.get("decode_settings") != media.native_decoder_settings()):
        raise ValueError("sigma sweep completion identities differ from requested inputs")
    rows = manifest.get("outputs", {})
    if set(rows) != set(tensors):
        raise ValueError("sigma sweep completion output inventory differs")

    def checked_file(recorded: str, expected: Path, digest: str) -> Path:
        path = Path(recorded)
        if (path.resolve() != expected.resolve() or not path.resolve().is_relative_to(output.resolve())
                or not path.is_file() or sha256(path) != digest):
            raise ValueError("sigma sweep saved media path or bytes changed")
        return path

    for name, row in rows.items():
        if row.get("frames") != 129 or row.get("fps") != 30:
            raise ValueError("sigma sweep saved media timebase changed")
        digest = row.get("pixel_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("sigma sweep saved pixel digest is invalid")
        movie = checked_file(row["video"], output / "videos" / f"{name}.mp4", row["sha256"])
        geometry = media.verify_saved_video(movie, 129, 30)
        if geometry["width"] != row.get("width") or geometry["height"] != row.get("height"):
            raise ValueError("sigma sweep saved video geometry changed")
        samples = row.get("samples", [])
        if (any(type(entry.get("frame")) is not int for entry in samples)
                or [entry.get("frame") for entry in samples] != list(SAMPLES)):
            raise ValueError("sigma sweep saved sample inventory changed")
        for entry in samples:
            path = checked_file(entry["path"], output / "frames" / f"{name}_f{entry['frame']:03d}.png", entry["sha256"])
            media.verify_saved_png(path, row["width"], row["height"])
    metrics_record = manifest.get("metrics", {})
    metrics_path = checked_file(metrics_record["path"], output / "metrics.json", metrics_record["sha256"])
    metrics = json.loads(metrics_path.read_text())

    def finite_values(value: object) -> bool:
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite_values(item) for item in value.values())
        if isinstance(value, list):
            return all(finite_values(item) for item in value)
        return True

    if not finite_values(metrics):
        raise ValueError("sigma sweep metrics contain nonfinite values")
    if (metrics.get("tag") != spec["tag"] or metrics.get("decode_seed") != 42
            or metrics.get("boundaries") != [17, 33, 49, 65, 81, 97, 113]
            or metrics.get("post_eviction") != [81, 97, 113]
            or [(cell.get("sigma"), cell.get("arm")) for cell in metrics.get("cells", [])]
            != [(cell["sigma"], cell["arm"]) for cell in spec["cells"]]):
        raise ValueError("sigma sweep metrics inventory differs")
    d0, d1 = tensors["sigma1.000000_d0"], tensors["sigma1.000000_d1"]
    if (metrics.get("sigma1_d0_equals_d1") != torch.equal(d0, d1)
            or metrics.get("sigma1_d0_vs_d1_max_abs") != float((d0.float() - d1.float()).abs().max())):
        raise ValueError("sigma sweep metrics sigma-one control differs")
    for cell in metrics["cells"]:
        name = f"sigma{cell['sigma']:.6f}_{cell['arm']}"
        kind, calls = LEVELS[cell["sigma"]]
        if (cell.get("path") != kind or cell.get("calls_per_block") != calls
                or cell.get("c0_equal") != torch.equal(tensors[name][:, :, :1], d0[:, :, :1])):
            raise ValueError("sigma sweep metrics cell controls differ")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse shared CLI/queue settings without input or model access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify and args.gpu_id is not None:
        parser.error("saved verification rejects --gpu-id")
    return args


def main(argv: list[str] | None = None) -> None:
    """Execute decoding only through the package CLI."""
    args = parse_args(argv)
    if args.verify:
        verify_completion(args.spec, args.output)
        return
    if args.gpu_id is None:
        argparse.ArgumentParser(description=__doc__).error("saved decoding requires --gpu-id")
    execute(args.spec, args.output, gpu_id=args.gpu_id)


if __name__ == "__main__":
    main()
