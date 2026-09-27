"""Measure D0/D1 reconstruction and temporal residuals directly on saved probe latents.

Run from LTX-2 with ``conda run -n ltx python``. Each input is a fresh
``visualize_d1.py`` output directory. Metrics are diagnostic latent-space quantities;
they do not substitute for motion-compensated appearance or pose evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.onestep_avatar.train import _load_training_master


def measure(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    records = []
    retention_end = manifest["geometry"]["sink_latent_frames"] + manifest["geometry"]["context_latent_frames"]
    for video in manifest["videos"]:
        artifacts = video["artifacts"]
        ground_truth, _ = _load_training_master(Path(artifacts["capture"]))
        starts = {start for start, _ in video["blocks"][1:]}
        for item in artifacts["latents"]:
            if item["sigma"] != video["sigma"]:
                continue
            predicted = torch.load(run / item["path"], map_location="cpu", weights_only=True).float()[0]
            target = ground_truth[:, : predicted.shape[1]].float()
            if predicted.shape != target.shape:
                raise ValueError(f"{item['path']}: predicted {predicted.shape} != target {target.shape}")
            residual = (predicted[:, 1:] - predicted[:, :-1]) - (target[:, 1:] - target[:, :-1])
            per_transition = residual.abs().mean(dim=(0, 2, 3))
            boundary = torch.tensor([frame - 1 for frame in starts], dtype=torch.long)
            early = torch.tensor([frame - 1 for frame in starts if frame <= retention_end], dtype=torch.long)
            late = torch.tensor([frame - 1 for frame in starts if frame > retention_end], dtype=torch.long)
            interior = torch.tensor([i - 1 for i in range(1, predicted.shape[1]) if i not in starts], dtype=torch.long)
            interior_after_first = interior[interior != 0]  # frame 0 -> 1 is the causal-VAE keyframe transition
            records.append(
                {
                    "view": artifacts["view"],
                    "sigma": item["sigma"],
                    "arm": item["arm"],
                    "latent_frames": predicted.shape[1],
                    "boundary_frames": sorted(starts),
                    "mse": ((predicted - target) ** 2).mean().item(),
                    "boundary_temporal_residual_mae": per_transition[boundary].mean().item()
                    if boundary.numel()
                    else None,
                    "pre_eviction_boundary_mae": per_transition[early].mean().item() if early.numel() else None,
                    "post_eviction_boundary_mae": per_transition[late].mean().item() if late.numel() else None,
                    "interior_temporal_residual_mae": per_transition[interior].mean().item()
                    if interior.numel()
                    else None,
                    "interior_excluding_first_transition_mae": per_transition[interior_after_first].mean().item()
                    if interior_after_first.numel()
                    else None,
                }
            )
    return {
        "run": str(run.resolve()),
        "checkpoint": manifest["checkpoint"],
        "transformer_fingerprint": manifest["model"]["transformer_fingerprint"],
        "history_policy": manifest["history_policy"],
        "history_mode": manifest["history_mode"],
        "geometry": manifest["geometry"],
        "measurements": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"metric": "uncompensated raw latent residual; exploratory", "runs": [measure(run) for run in args.runs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
