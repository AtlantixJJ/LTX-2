"""Decode saved D1 raw latents once per complete sequence for visual review.

Run from the LTX-2 root in the ltx conda environment. No transformer is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from PIL import Image

from scripts.onestep_avatar import visualize_d0
from scripts.onestep_avatar.visualize_d1 import _load_chain
from scripts.prune.core import provenance
from scripts.prune.core.session import add_model_args, open_session


def pixel_boundaries(blocks: list[list[int]], temporal_scale: int) -> list[int]:
    """The first pixel frame emitted by each block after block zero."""
    return [(start - 1) * temporal_scale + 1 for start, _ in blocks[1:]]


def enumerate_latents(runs: list[Path], model: str, seed: int) -> list[dict]:
    """Enumerate each D1 artifact once, even when a view has several sigma entries."""
    entries = []
    seen = set()
    for run in runs:
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["model"]["model_key"] != model or manifest["seed"] != seed:
            raise ValueError(f"model or seed mismatch: {run}")
        for video in manifest["videos"]:
            artifacts = video["artifacts"]
            for latent in artifacts["latents"]:
                if latent["arm"] != "d1" or latent["sigma"] != video["sigma"]:
                    continue
                path = run / latent["path"]
                key = (path.resolve(), video["sigma"], artifacts["view"])
                if key in seen:
                    continue
                seen.add(key)
                entries.append({"run": run, "manifest": manifest, "manifest_path": manifest_path,
                                "video": video, "latent": latent, "path": path})
    return entries


def write_video(frames: torch.Tensor, path: Path, fps: float) -> None:
    frames = frames.float().clamp(0, 1)
    _, _, height, width = frames.shape
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            rgb = frame[:3].permute(1, 2, 0).mul(255).round().byte().numpy()
            process.stdin.write(rgb.tobytes())
        process.stdin.close()
        error = process.stderr.read()
        code = process.wait()
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
    if code:
        raise RuntimeError(f"ffmpeg failed for {path}: {error.decode(errors='replace')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    add_model_args(parser)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"decode output directory must be fresh: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    entries = enumerate_latents(args.runs, args.model, args.seed)
    session = open_session(args, script="onestep_avatar.decode_saved_d1")
    decoder_path = Path(session.model.paths.video_vae()).resolve()
    decoder_fingerprint = provenance.checkpoint_fingerprint(decoder_path)
    for entry in entries:
        manifest = entry["manifest"]
        recorded = Path(manifest["model"]["video_vae_path"])
        expected = manifest["model"]["video_vae_fingerprint"]
        if provenance.checkpoint_fingerprint(recorded) != expected or decoder_fingerprint != expected:
            raise ValueError(f"resolved decoder does not match generation VAE: {entry['run']}")
        artifacts = entry["video"]["artifacts"]
        if provenance.file_sha256(Path(artifacts["capture"])) != artifacts["capture_sha256"]:
            raise ValueError(f"capture hash mismatch: {artifacts['capture']}")
        if provenance.file_sha256(entry["path"]) != entry["latent"]["sha256"]:
            raise ValueError(f"latent hash mismatch: {entry['path']}")
    decoded = []
    with session.decoder() as decoder:
        for entry in entries:
            run, manifest, video, path = (entry[key] for key in ("run", "manifest", "video", "path"))
            view = Path(video["artifacts"]["view"])
            actor = view.parent.parent.name
            count = video["blocks"][-1][1]
            source_key = hashlib.sha256(str(view.resolve()).encode()).hexdigest()[:10]
            tag = (f"{run.name}_{actor}_{view.name}_{source_key}_sigma{video['sigma']:.6f}_"
                   f"{manifest['objective']}_{count}lat_d1_{entry['latent']['sha256'][:12]}")
            target_key = (f"{actor}_{view.name}_{source_key}_{manifest['objective']}_{count}lat_"
                          f"vae{decoder_fingerprint}_seed{args.seed}_{video['artifacts']['capture_sha256'][:12]}")
            target_name = f"capture_{target_key}"
            target_path = args.output / f"{target_name}.mp4"
            if not target_path.exists():
                chain = _load_chain(view, manifest["objective"])
                target = chain.z_y.unsqueeze(0)[:, :, :count]
                target_frames = visualize_d0._decode(session, target, decoder, args.seed)
                write_video(target_frames, target_path, chain.fps)
                print(f"decoded {target_name}", flush=True)
            else:
                chain = None
            latent = torch.load(path, map_location="cpu", weights_only=True)
            frames = visualize_d0._decode(session, latent, decoder, args.seed)
            if chain is None:
                # FPS is recorded at generation time, avoiding a second source read.
                fps = video["artifacts"]["fps"]
            else:
                fps = chain.fps
            stem = tag
            out = args.output / f"{stem}.mp4"
            if out.exists():
                raise FileExistsError(f"refusing to overwrite decoded video: {out}")
            write_video(frames, out, fps)
            boundaries = pixel_boundaries(video["blocks"], manifest["geometry"]["scale_factors"][0])
            # Shared checkpoints permit cross-run review; only `boundaries` are actual seams.
            checkpoints = set(boundaries) | {17, 33, 49, 65, 81}
            frame_indices = {i for point in checkpoints for i in (point - 1, point, point + 1)}
            frame_indices.update(range(57, 74))  # first continuation after a 65-frame block
            for frame_index in sorted(frame_indices):
                if frame_index >= len(frames):
                    continue
                rgb = frames[frame_index, :3].float().clamp(0, 1).permute(1, 2, 0)
                image = Image.fromarray(rgb.mul(255).round().byte().numpy())
                image.save(args.output / f"{stem}_f{frame_index:03d}.png")
            decoded.append({"run": str(run.resolve()), "actor": actor, "view": str(view.resolve()),
                            "history_mode": manifest["history_mode"], "history_policy": manifest["history_policy"],
                            "sigma": video["sigma"], "objective": manifest["objective"], "arm": "d1",
                            "video": out.name, "capture_video": target_path.name,
                            "source_latent": str(path.resolve()), "source_latent_sha256": entry["latent"]["sha256"],
                            "source_manifest_sha256": provenance.file_sha256(entry["manifest_path"]),
                            "capture_sha256": video["artifacts"]["capture_sha256"],
                            "decoder_path": str(decoder_path), "decoder_fingerprint": decoder_fingerprint,
                            "decode_seed": args.seed, "blocks": video["blocks"], "boundaries": boundaries,
                            "frames": len(frames), "fps": fps})
            print(f"decoded {stem}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(decoded, indent=2) + "\n")


if __name__ == "__main__":
    main()
