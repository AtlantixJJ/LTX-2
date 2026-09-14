"""Write the artifact-backed completion report for the D0 GT-renoise sanity arm.

The report deliberately checks the actual checkpoint, rank logs, and decoded MP4s before it
writes anything.  It is a handoff record, not an interpretation of visual quality: D0 is a
capacity control and its loss is not comparable to guide-conditioned arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


# This run's own training schedule (trained before sigma=0.0 was disallowed -- see
# train.py's training_sigmas). Kept distinct from PROBE_SIGMAS: the two need not agree, and
# after training_sigmas started refusing 0.0 they never will again for a new run.
TRAIN_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
# visualize_d0.py's fixed evaluation grid. Excludes 0.0 -- see that module's PROBE_SIGMAS
# comment for why sigma=0.0 cannot be run through the probe's step schedule at all.
PROBE_SIGMAS = (0.909375, 0.725, 0.421875)


def _records(run: Path) -> list[dict]:
    rows = []
    for path in sorted(run.glob("metrics_rank*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows:
        raise SystemExit(f"{run}: no nonempty metrics_rank*.jsonl")
    return rows


def _mean_by_step(rows: list[dict]) -> dict[int, float]:
    values: dict[int, list[float]] = {}
    for row in rows:
        values.setdefault(int(row["step"]), []).append(float(row["mse"]))
    return {step: sum(value) / len(value) for step, value in values.items()}


def _video_metadata(path: Path) -> dict:
    """Fail closed unless FFmpeg can read a nonempty video stream."""
    try:
        result = subprocess.run(
            (
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_type,width,height,avg_frame_rate:format=duration",
                "-of", "json", str(path),
            ),
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot validate probe video {path}: {exc}") from exc
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    if len(streams) != 1 or streams[0].get("codec_type") != "video":
        raise SystemExit(f"{path}: no readable primary video stream")
    stream = streams[0]
    duration = float(info.get("format", {}).get("duration", 0.0))
    if int(stream.get("width", 0)) <= 0 or int(stream.get("height", 0)) <= 0 or duration <= 0:
        raise SystemExit(f"{path}: invalid video dimensions or duration")
    return {"width": int(stream["width"]), "height": int(stream["height"]), "duration": duration}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=200)
    parser.add_argument(
        "--gpus",
        default="unspecified",
        help="Physical GPU IDs used for this launch (recorded because CUDA_VISIBLE_DEVICES remaps ranks).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads((args.run / "config.json").read_text())
    if config.get("guide_mode") != "d0":
        raise SystemExit(f"{args.run}: expected guide_mode=d0, got {config.get('guide_mode')!r}")
    if config.get("steps") != args.expected_steps:
        raise SystemExit(f"{args.run}: expected {args.expected_steps} steps, got {config.get('steps')}")
    if tuple(config.get("sigma_levels", (config.get("sigma0"),))) != TRAIN_SIGMAS:
        raise SystemExit(f"{args.run}: expected multilevel D0 schedule {TRAIN_SIGMAS}, got {config.get('sigma_levels')}")

    checkpoint = args.run / "checkpoints" / f"lora_weights_step_{args.expected_steps:05d}.safetensors"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise SystemExit(f"missing final checkpoint: {checkpoint}")
    probe_dir = args.run / "probes" / f"step_{args.expected_steps:05d}"
    manifest_path = probe_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing probe manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if tuple(manifest.get("sigmas", ())) != PROBE_SIGMAS:
        raise SystemExit(f"{manifest_path}: sigma sequence is not the distilled probe levels")
    expected = [f"step_{args.expected_steps:05d}_sigma_{sigma:.6f}.mp4" for sigma in PROBE_SIGMAS]
    missing = [name for name in expected if not (probe_dir / name).is_file() or (probe_dir / name).stat().st_size == 0]
    if missing:
        raise SystemExit(f"missing or empty optimized probe video(s): {missing}")
    base_expected = [f"frozen_base_sigma_{sigma:.6f}.mp4" for sigma in PROBE_SIGMAS]
    missing_base = [name for name in base_expected if not (probe_dir / name).is_file() or (probe_dir / name).stat().st_size == 0]
    if missing_base:
        raise SystemExit(f"missing or empty frozen-base probe video(s): {missing_base}")
    video_metadata = {name: _video_metadata(probe_dir / name) for name in (*base_expected, *expected)}

    rows = _records(args.run)
    means = _mean_by_step(rows)
    if max(means) != args.expected_steps:
        raise SystemExit(f"rank logs stop at step {max(means)}, expected {args.expected_steps}")
    first, final = means[min(means)], means[max(means)]
    ranks = sorted({int(row["rank"]) for row in rows})
    gradients = [float(row["grad_norm"]) for row in rows if row.get("grad_norm") is not None]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join((
        "# D0 GT-renoise sanity run",
        "",
        "**Status:** completed artifact check.",
        "",
        "D0 is a capacity-control experiment, not a deployable guide-conditioned arm: it noises the",
        "ground-truth capture latent itself. Its MSE must therefore not be compared directly with D1/D2.",
        "",
        "## Training",
        "",
        f"- GPUs: {args.gpus} (FSDP world size {config.get('world_size')}; rank-{config.get('lora_rank')} {config.get('lora_target')} LoRA).",
        f"- Objective: D0, cyclic sigma levels {list(TRAIN_SIGMAS)}, union loss mask, {args.expected_steps} steps.",
        f"- Checkpoint: `{checkpoint}` ({checkpoint.stat().st_size / 2**20:.1f} MiB).",
        f"- Rank-mean masked x0 MSE: {first:.6f} at step {min(means)} → {final:.6f} at step {max(means)} "
        f"({(final / first - 1) * 100:.1f}%).",
        f"- Minimum rank-mean MSE: {min(means.values()):.6f} at step {min(means, key=means.get)}.",
        f"- Logged ranks: {ranks}; gradient norm range: {min(gradients):.6f}–{max(gradients):.6f}.",
        "",
        "## Fixed D0 visual probe",
        "",
        f"Probe source: `{manifest['source']}`, windows {manifest['windows']}, seed {manifest['seed']}, {manifest['fps']} fps.",
        "Each optimized MP4 is frame-aligned `ground-truth capture | frozen base | step-200 D0`.",
        "The base copies are the before-optimization reference; the step-200 copies are after optimization.",
        f"All {2 * len(PROBE_SIGMAS)} MP4s were FFprobe-validated for a readable video stream, nonzero duration, and positive dimensions.",
        "",
        *[
            f"- σ={sigma:.6f}: `{probe_dir / f'step_{args.expected_steps:05d}_sigma_{sigma:.6f}.mp4'}` "
            f"({video_metadata[f'step_{args.expected_steps:05d}_sigma_{sigma:.6f}.mp4']['width']}×"
            f"{video_metadata[f'step_{args.expected_steps:05d}_sigma_{sigma:.6f}.mp4']['height']}, "
            f"{video_metadata[f'step_{args.expected_steps:05d}_sigma_{sigma:.6f}.mp4']['duration']:.2f}s)"
            for sigma in PROBE_SIGMAS
        ],
        "",
        "The corresponding `frozen_base_sigma_*.mp4` files and `manifest.json` are in the same directory.",
        "",
        "## Interpretation boundary",
        "",
        "This verifies that the model trained and that the requested fixed-noise visual evidence was produced. "
        "It answers D0's capacity question only; it does not establish guide-to-capture adaptation quality or deployment performance.",
        "",
    )) )
    print(args.report)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
