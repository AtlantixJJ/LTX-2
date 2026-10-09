"""Measure preserved saved-probe encodings without model sessions; see doc/experiments/saved_probe_metrics.md."""

import argparse
import json
from pathlib import Path

import torch

from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.corpus.dataset import atomic_write
from scripts.onestep_avatar.hashing import sha256


def saved_latent_metrics(
    output: torch.Tensor, capture: torch.Tensor, guide: torch.Tensor, *, long: bool = False
) -> dict:
    """Preserve historical generated-frame measurements separately from training loss."""
    if (
        output.ndim != 4
        or output.shape != capture.shape
        or output.shape != guide.shape
        or output.shape[1] < 3
        or output.shape[1] % 2 != 1
        or min(output.shape[2:]) < 2
        or any(not torch.isfinite(value).all() for value in (output, capture, guide))
    ):
        raise ValueError("saved metrics require finite matching C,F,H,W complete two-frame blocks")
    output, capture, guide = (value.float() for value in (output, capture, guide))
    frames = output.shape[1]
    if not long and frames != 17:
        raise ValueError("short saved metrics require exactly 17 encoded frames")
    blocks = [(1, 3)] + [(start, start + 2) for start in range(3, frames, 2)]

    def detail(value: torch.Tensor) -> float:
        return float(
            (value[:, :, 1:] - value[:, :, :-1]).abs().mean() + (value[:, :, :, 1:] - value[:, :, :, :-1]).abs().mean()
        )

    def ratio(numerator: float, denominator: float) -> float:
        if denominator == 0:
            raise ValueError("saved metric ratio has a zero denominator")
        return numerator / denominator

    result = {
        "c0_exact": bool(torch.equal(output[:, 0], capture[:, 0])),
        "per_block_mse": [float((output[:, a:b] - capture[:, a:b]).square().mean()) for a, b in blocks],
    }
    if long:
        return {
            **result,
            "latent_frames": frames,
            "per_block_guide_mse": [float((guide[:, a:b] - capture[:, a:b]).square().mean()) for a, b in blocks],
            "per_block_detail_ratio": [ratio(detail(output[:, a:b]), detail(capture[:, a:b])) for a, b in blocks],
        }
    boundaries = [end for _, end in blocks[:-1]]

    def seam(value: torch.Tensor) -> float:
        steps = (value[:, 1:] - value[:, :-1]).square().mean(dim=(0, 2, 3))
        across = torch.stack([steps[end - 1] for end in boundaries]).mean()
        inside = torch.stack([steps[t - 1] for t in range(2, frames) if t not in boundaries]).mean()
        if float(inside) == 0:
            raise ValueError("saved metric ratio has a zero denominator")
        return float(across / inside)

    def motion(value: torch.Tensor) -> float:
        return float((value[:, 2:] - value[:, 1:-1]).abs().mean())

    return {
        **result,
        "capture_mse": float((output[:, 1:] - capture[:, 1:]).square().mean()),
        "guide_mse": float((guide[:, 1:] - capture[:, 1:]).square().mean()),
        "motion_ratio": ratio(motion(output), motion(capture)),
        "detail_ratio": ratio(detail(output[:, 1:]), detail(capture[:, 1:])),
        "guide_detail_ratio": ratio(detail(guide[:, 1:]), detail(capture[:, 1:])),
        "seam_ratio": seam(output),
        "capture_seam_ratio": seam(capture),
    }


def measure_saved_probe(directory: Path, *, long: bool = False) -> dict:
    """Measure verified saved historical encodings, without any model session."""
    manifest = json.loads((directory / "manifest.json").read_text())
    rows, seen = [], set()
    for video in manifest["videos"]:
        artifacts = video["artifacts"]
        seed = artifacts.get("seed", manifest.get("seed"))
        key = (artifacts["view"], seed)
        if key in seen:
            continue
        seen.add(key)
        capture, _ = dataset.load_training_master(Path(artifacts["capture"]))
        guide, _ = dataset.load_training_master(Path(artifacts["guide"]))
        for latent in artifacts["latents"]:
            path = directory / latent["path"]
            if sha256(path) != latent["sha256"]:
                raise ValueError(f"saved encoding content changed: {path}")
            output = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(output, torch.Tensor) or output.ndim != 5 or output.shape[0] != 1:
                raise ValueError("saved output must be a single B,C,F,H,W tensor")
            frames = output.shape[2]
            view = Path(artifacts["view"])
            row = {
                "view": f"{view.parent.parent.parent.name}/{view.parent.parent.name}/{view.name}",
                "seed": seed,
                **saved_latent_metrics(output[0], capture[:, :frames], guide[:, :frames], long=long),
            }
            if not long:
                row.update(
                    sigma=latent["sigma"],
                    arm=latent["arm"],
                    latent_sha256=latent["sha256"],
                    epsilon_sha256=artifacts["epsilon_sha256"],
                )
            rows.append(row)
    result = {"probe": str(directory), "checkpoint": manifest.get("checkpoint"), "rows": rows}
    if not long:
        result.update(
            off_condition=manifest.get("off_condition", False),
            model_variant=manifest.get("model_variant"),
            schedule=manifest["videos"][0]["schedule"] if manifest["videos"] else None,
        )
    atomic_write(
        directory / ("metrics_long.json" if long else "metrics.json"),
        lambda temporary: temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Read saved directories directly; never dispatch model work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-metrics", nargs="+", type=Path, required=True)
    parser.add_argument("--long-metrics", action="store_true")
    args = parser.parse_args(argv)
    for directory in args.saved_metrics:
        measure_saved_probe(directory, long=args.long_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
