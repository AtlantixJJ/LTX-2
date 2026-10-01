"""Benchmark matched whole-clip D0 one-step forwards for baseline and compact export."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from ltx_core.model.transformer.modality import Modality
from scripts.onestep_avatar import causal_core
from scripts.prune.core import provenance, session
from scripts.prune.data import whole_clip


def _one_arm(current: session.Session, path: Path, expected: Path,
             grid: causal_core.ClipGrid, modality: Modality, c0: torch.Tensor,
             *, warmup: int, repeats: int, label: str) -> dict:
    times = []
    with current.transformer(path) as transformer:
        def forward() -> torch.Tensor:
            prediction, _ = transformer(video=modality, audio=None, perturbations=None)
            return causal_core.with_clean_prefix(prediction, c0)

        with torch.no_grad():
            output = forward()
            torch.cuda.synchronize(current.device)
            recorded = torch.load(expected, map_location="cpu", weights_only=True)
            measured = grid.unpatchify_block(output, grid.latent_frames).cpu()
            difference = (recorded.float() - measured.float()).abs()
            max_abs = float(difference.max())
            rel_l2 = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(recorded.float()))
            if max_abs > 0.02:
                raise ValueError(f"{label} forward differs from saved rollout: max_abs={max_abs:.5f}")
            del output, recorded, measured, difference
            torch.cuda.reset_peak_memory_stats(current.device)
            for index in range(warmup + repeats):
                torch.cuda.synchronize(current.device)
                start = time.perf_counter()
                output = forward()
                torch.cuda.synchronize(current.device)
                if index >= warmup:
                    times.append((time.perf_counter() - start) * 1000)
                del output
            peak = torch.cuda.max_memory_allocated(current.device)
    return {
        "arm": label,
        "checkpoint": str(path.resolve()),
        "checkpoint_fingerprint": provenance.checkpoint_fingerprint(path),
        "saved_rollout_max_abs": max_abs,
        "saved_rollout_relative_l2": rel_l2,
        "times_ms": times,
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_cuda_memory_bytes": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--sigma", type=float, default=0.909375)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--order", choices=("ABA", "BAB"), default="ABA",
                        help="A=baseline, B=candidate; repeat the first arm to expose run-order drift")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 3:
        parser.error("use at least one warmup and three timed repetitions")
    base = whole_clip.load_manifest(args.baseline)
    pruned = whole_clip.load_manifest(args.candidate)
    whole_clip.verify_candidate(base, pruned)
    if (args.view, args.sigma) not in whole_clip.records(base):
        parser.error("view/sigma pair is absent from the matched manifests")
    if args.gpu_id >= torch.cuda.device_count():
        parser.error(f"GPU {args.gpu_id} does not exist")
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < 44 * 2**30:
        parser.error(f"GPU {args.gpu_id} needs at least 44 GiB free for the 1024px whole clip")
    model_args = argparse.Namespace(model="2.5", gpu_id=args.gpu_id, seed=base["seed"])
    current = session.open_session(model_args, script="prune.evaluate.bench_whole_clip_d0",
                                   prompt=base["text_context"]["prompt"])
    b, p = whole_clip.records(base)[(args.view, args.sigma)], whole_clip.records(pruned)[(args.view, args.sigma)]
    whole_clip.verify_saved_noise(args.baseline, args.candidate, b, p)
    grid, modality, c0, _ = whole_clip.build_input(
        args.baseline, base, view=args.view, sigma=args.sigma, current=current,
    )
    base_path = Path(base["model"]["transformer_path"])
    candidate_path = Path(pruned["model"]["transformer_path"])
    for path, manifest in ((base_path, base), (candidate_path, pruned)):
        if provenance.checkpoint_fingerprint(path) != manifest["model"]["transformer_fingerprint"]:
            raise ValueError(f"checkpoint changed since saved rollout: {path}")
    arm_inputs = {
        "A": (base_path, whole_clip.latent_path(args.baseline, b)),
        "B": (candidate_path, whole_clip.latent_path(args.candidate, p)),
    }
    arms = []
    for index, code in enumerate(args.order):
        path, expected = arm_inputs[code]
        label = f"{'baseline' if code == 'A' else 'candidate'}_{index + 1}"
        row = _one_arm(current, path, expected, grid, modality, c0,
                       warmup=args.warmup, repeats=args.repeats, label=label)
        arms.append(row)
        print(f"{label}: {row['median_ms']:.2f} ms; saved max_abs={row['saved_rollout_max_abs']:.5f}", flush=True)
    baseline_ms = statistics.mean(row["median_ms"] for row in arms if row["arm"].startswith("baseline"))
    candidate_ms = statistics.mean(row["median_ms"] for row in arms if row["arm"].startswith("candidate"))
    result = {
        "task": whole_clip.TASK,
        "method": "same-GPU wall-clock, synchronized around one transformer forward; excludes load, noising, VAE",
        "view": args.view, "sigma": args.sigma, "schedule": [args.sigma, 0.0],
        "seed": base["seed"], "prompt_sha256": base["text_context"]["prompt_sha256"],
        "source_sha256": b["artifacts"]["capture_sha256"],
        "epsilon_sha256": b["artifacts"]["epsilon_sha256"],
        "attention": base["attention"], "geometry": base["geometry"],
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "warmup": args.warmup, "repeats": args.repeats, "order": args.order,
        "arms": arms,
        "baseline_median_mean_ms": baseline_ms,
        "candidate_median_mean_ms": candidate_ms,
        "compact_over_baseline_time_ratio": candidate_ms / baseline_ms,
        "compact_speedup": baseline_ms / candidate_ms,
        "bracket_arm_drift_percent": 100 * (arms[2]["median_ms"] - arms[0]["median_ms"]) / arms[0]["median_ms"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
