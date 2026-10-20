"""Benchmark matched whole-clip D0 one-step forwards for baseline and compact export."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from scripts.prune.core import numerics, provenance, session
from scripts.prune.data import whole_clip
from scripts.prune.evaluate import timing


def main() -> None:  # noqa: PLR0915 -- preserve the ordered bracket and conditional drift retry
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--sigma", type=float, default=0.909375)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--order",
        choices=("ABA", "BAB"),
        default="ABA",
        help="A=baseline, B=candidate; repeat the first arm to expose run-order drift",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    numerics.deterministic_policy()
    if args.warmup < 1 or args.repeats < 3:
        parser.error("use at least one warmup and three timed repetitions")
    base = whole_clip.load_manifest(args.baseline)
    pruned = whole_clip.load_manifest(args.candidate)
    learning = whole_clip.verify_candidate(base, pruned)
    if (args.view, args.sigma) not in whole_clip.records(base):
        parser.error("view/sigma pair is absent from the matched manifests")
    if args.gpu_id >= torch.cuda.device_count():
        parser.error(f"GPU {args.gpu_id} does not exist")
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < 44 * 2**30:
        parser.error(f"GPU {args.gpu_id} needs at least 44 GiB free for the 1024px whole clip")
    current = session.open_model(
        "2.5",
        args.gpu_id,
        script="prune.evaluate.bench_whole_clip_d0",
        transformer_path=Path(base["model"]["transformer_path"]),
        prompt=base["text_context"]["prompt"],
    )
    b, p = whole_clip.records(base)[(args.view, args.sigma)], whole_clip.records(pruned)[(args.view, args.sigma)]
    whole_clip.verify_saved_noise(args.baseline, args.candidate, b, p)
    grid, modality, c0, _ = whole_clip.build_input(
        args.baseline,
        base,
        view=args.view,
        sigma=args.sigma,
        current=current,
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
        row = timing.measure_arm(
            current, path, expected, grid, modality, c0, warmup=args.warmup, repeats=args.repeats, label=label
        )
        arms.append(row)
        print(f"{label}: {row['median_ms']:.2f} ms; saved max_abs={row['saved_rollout_max_abs']:.5f}", flush=True)
    initial_arms = None
    if abs(arms[2]["median_ms"] - arms[0]["median_ms"]) / arms[0]["median_ms"] > 0.02 and args.repeats < 20:
        initial_arms, arms = arms, []
        args.repeats = 20
        for index, code in enumerate(args.order):
            path, expected = arm_inputs[code]
            label = f"{'baseline' if code == 'A' else 'candidate'}_{index + 1}"
            arms.append(
                timing.measure_arm(
                    current, path, expected, grid, modality, c0, warmup=args.warmup, repeats=20, label=label
                )
            )
    baseline_ms = statistics.mean(row["median_ms"] for row in arms if row["arm"].startswith("baseline"))
    candidate_ms = statistics.mean(row["median_ms"] for row in arms if row["arm"].startswith("candidate"))
    result = {
        "task": whole_clip.TASK,
        "method": "same-GPU wall-clock, synchronized around one transformer forward; excludes load, noising, VAE",
        "view": args.view,
        "sigma": args.sigma,
        "schedule": [args.sigma, 0.0],
        "seed": base["seed"],
        "prompt_sha256": base["text_context"]["prompt_sha256"],
        "source_sha256": b["artifacts"]["capture_sha256"],
        "epsilon_sha256": b["artifacts"]["epsilon_sha256"],
        "attention": base["attention"],
        "geometry": base["geometry"],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "order": args.order,
        "arms": arms,
        "initial_arms_before_drift_repeat": initial_arms,
        "candidate_provenance": learning,
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
