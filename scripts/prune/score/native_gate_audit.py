"""Independently verify a completed native-gate prefix from saved CPU artifacts.

This reader never opens a model, imports torch or changes a completed milestone.
It checks source draws, input receipts, worker ownership and exact shard merging.
The audit receipt belongs in monitoring/, separate from immutable scientific data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from scripts.prune.score.native_gate_accumulator import Accumulator, atomic_json, digest

HEAD_SHAPE = (48, 32)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_draw(receipt: dict, draw: dict, study: dict, members: dict) -> None:
    index = receipt["draw_index"]
    group = study["source_sigma_indices"].index(draw["sigma_index"])
    member = members[draw["view"]]
    expected = {
        "step": index // 4 + 1,
        "logical_rank": index % 4,
        "source_sigma_index": draw["sigma_index"],
        "sigma_index": group,
        "sigma": study["sigmas"][group],
        "view": draw["view"],
        "noise_seed": draw["noise_seed"],
        "actor": member["actor"],
        "capture_sha256": member["capture_latent_sha256"],
        "context_sha256": study["context"]["tensor_sha256"],
        "frames": 18,
        "tokens_per_frame": member["shape"][-1] * member["shape"][-2],
    }
    require(draw["microbatch"] == index, "source draw index differs")
    require(all(receipt[k] == v for k, v in expected.items()), f"draw {index}: source identity differs")
    window = receipt["training_window"]
    start = random.Random(f"uniform_f18:{draw['noise_seed']}").randrange(member["n_latent_frames"] - 17)
    require(
        window["latent_start"] == start
        and window["latent_stop"] == start + 18
        and window["position_policy"] == "window_local_first_keyframe"
        and window["noise_seed"] == draw["noise_seed"]
        and window["source_latent_frames"] == member["n_latent_frames"]
        and window["tokens_per_latent_frame"] == receipt["tokens_per_frame"],
        f"draw {index}: continuous training crop differs",
    )
    manifest_path = Path(receipt["input_manifest"])
    require(sha256(manifest_path) == receipt["input_sha256"], f"draw {index}: input manifest changed")
    manifest = read_json(manifest_path)
    require(manifest["sigmas"] == study["source"]["config"]["sigmas"], "original ten-level input grid differs")
    require(manifest["seed"] == draw["noise_seed"], f"draw {index}: input seed differs")
    require(manifest["text_context"]["sha256"] == study["context"]["bytes_sha256"], "input context bytes differ")
    artifact = manifest["videos"][0]["artifacts"]
    require(artifact["epsilon_sha256"] == receipt["epsilon_sha256"], f"draw {index}: epsilon pin differs")
    require(artifact["capture_sha256"] == receipt["capture_sha256"], f"draw {index}: capture pin differs")
    historical = Path(study["source"]["root"]) / f"training/updates/{receipt['step']:04d}.json"
    if receipt["historical_training_match"]:
        training = read_json(historical)["draws"][receipt["logical_rank"]]
        require(all(training[k] == v for k, v in draw.items()), f"draw {index}: training draw changed")
        require(training["training_window"] == window, f"draw {index}: training crop/noise hashes differ")
        require(training["input_sha256"] == receipt["input_sha256"], f"draw {index}: training input differs")


def verify_statistics(acc: Accumulator) -> None:
    required = {
        p + s
        for p in ("gate_", "logit_", "clean_gate_", "clean_logit_")
        for s in ("sum", "square_sum", "minimum", "maximum")
    }
    required.update(("histogram", "threshold_counts", "clean_threshold_counts", "frame_gate_sum", "frame_high_count"))
    require(acc.arrays.keys() == required, "required sufficient statistic inventory differs")
    require(acc.arrays["histogram"].shape == (4, *HEAD_SHAPE, 2048), "statistics head/bin geometry differs")
    acc.validate_counts()
    per_frame = {r["tokens_per_frame"] for r in acc.receipts}
    require(len(per_frame) == 1, "draw token counts differ across workers")
    expected_clean = acc.counts * next(iter(per_frame))
    require(np.array_equal(acc.clean_tokens, expected_clean), "clean token count differs")
    require(np.array_equal(acc.tokens, expected_clean * 17), "generated token count differs")
    for key, value in acc.arrays.items():
        require(np.isfinite(value).all(), f"nonfinite statistic: {key}")
        require(
            value.dtype == (np.int64 if "count" in key or key == "histogram" else np.float64),
            f"statistic dtype differs: {key}",
        )
    hist = acc.arrays["histogram"]
    require((hist >= 0).all(), "negative histogram counts")
    centers = (np.arange(2048) + 0.5) / 1024
    histogram_mean = (hist @ centers) / acc.tokens[:, None, None]
    exact_mean = acc.arrays["gate_sum"] / acc.tokens[:, None, None]
    require(
        np.all(np.abs(histogram_mean - exact_mean) <= 1 / 2048 + 1e-12), "histogram and exact gate moments disagree"
    )
    frame_sum = acc.arrays["frame_gate_sum"][:, :, 1:].sum(axis=2)
    require(
        np.allclose(frame_sum, acc.arrays["gate_sum"], rtol=1e-12, atol=1e-9), "frame and generated-token sums disagree"
    )
    for prefix, counts in (("", acc.tokens), ("clean_", acc.clean_tokens)):
        fractions = acc.arrays[prefix + "threshold_counts"] / counts[:, None, None, None]
        require(((fractions >= 0) & (fractions <= 1)).all(), "threshold count outside token population")
        require((fractions[:, :, 0] <= fractions[:, :, 1]).all(), "lower threshold ordering differs")
        require((fractions[:, :, 3] <= fractions[:, :, 2]).all(), "upper threshold ordering differs")


def same_state(actual: Accumulator, rebuilt: Accumulator) -> None:
    require(actual.receipts == rebuilt.receipts, "rebuilt receipt order differs")
    for name in ("counts", "tokens", "clean_tokens"):
        require(np.array_equal(getattr(actual, name), getattr(rebuilt, name)), f"rebuilt {name} differs")
    require(actual.arrays.keys() == rebuilt.arrays.keys(), "rebuilt statistic inventory differs")
    for key in actual.arrays:
        require(np.array_equal(actual.arrays[key], rebuilt.arrays[key]), f"rebuilt statistic differs: {key}")


def audit_milestone(root: Path, step: int) -> dict:
    study = read_json(root / "configs/frozen_study.json")
    identity = study["identity"]
    require(digest({k: v for k, v in study.items() if k != "identity"}) == identity, "frozen study identity differs")
    require(step % 100 == 0 and 100 <= step <= study["steps"], "audit needs a complete source milestone")
    for relative, expected in study["files"].items():
        require(sha256(root / relative) == expected, f"frozen evidence changed: {relative}")
    draws = read_json(root / "configs/draws.json")
    expected = [i for i in range(step * 4) if draws[i]["sigma_index"] in study["source_sigma_indices"]]
    require(expected == [i for i in study["selected_draw_indices"] if i < step * 4], "selected source prefix differs")
    selected = set(expected)
    require(
        [i for i in study["skipped_draw_indices"] if i < step * 4] == [i for i in range(step * 4) if i not in selected],
        "skipped source prefix differs",
    )
    directory = root / f"milestones/step_{step:04d}"
    complete = read_json(directory / "complete.json")
    require(complete["study"] == identity and complete["step"] == step, "completed milestone identity differs")
    for name in ("window", "cumulative"):
        require(sha256(directory / f"{name}.npz") == complete[f"{name}_sha256"], f"{name} archive changed")
    cumulative = Accumulator.load(directory / "cumulative.npz", identity)
    cumulative.require_indices(expected)
    verify_statistics(cumulative)
    require(np.array_equal(cumulative.counts, np.full(4, step * 2 // 5)), "per-sigma cumulative draws differ")
    require(
        complete["per_sigma"] == cumulative.counts.tolist()
        and complete["draws"] == len(expected)
        and complete["scheduled_draws"] == step * 4
        and complete["skipped_draws"] == step * 4 - len(expected),
        "published progress counts differ",
    )
    require(digest(cumulative.receipts) == complete["receipt_chain_sha256"], "receipt chain differs")
    members = {m["relative_dir"]: m for m in study["source"]["splits"]["learning"]}
    for receipt in cumulative.receipts:
        index = receipt["draw_index"]
        require(read_json(root / f"draws/{index:05d}.json") == receipt, f"published draw {index} changed")
        verify_draw(receipt, draws[index], study, members)
    window = Accumulator.load(directory / "window.npz", identity)
    verify_statistics(window)
    rebuilt = Accumulator(4, identity)
    owners = []
    for slot, gpu in enumerate(study["gpus"]):
        path = root / f"workers/gpu_{gpu}/step_{step:04d}.npz"
        pin = read_json(path.with_suffix(".json"))
        require(pin["study"] == identity and pin["sha256"] == sha256(path), "worker shard changed")
        indices = [i for i in expected if (step - 100) * 4 <= i and i % len(study["gpus"]) == slot]
        require(pin["indices"] == indices and pin["gpu"] == gpu and pin["step"] == step, "worker ownership differs")
        shard = Accumulator.load(path, identity)
        shard.require_indices(indices)
        rebuilt.merge(shard)
        owners.extend({"draw_index": i, "gpu": gpu, "shard_sha256": pin["sha256"]} for i in indices)
    same_state(window, rebuilt)
    if step == 100:
        same_state(cumulative, rebuilt)
    else:
        prior_dir = root / f"milestones/step_{step - 100:04d}"
        prior_pin = read_json(prior_dir / "complete.json")
        require(sha256(prior_dir / "cumulative.npz") == prior_pin["cumulative_sha256"], "predecessor changed")
        prior = Accumulator.load(prior_dir / "cumulative.npz", identity)
        prior.merge(window)
        same_state(cumulative, prior)
    return {
        "passed": True,
        "study": identity,
        "step": step,
        "observed_draws": len(expected),
        "scheduled_draws": step * 4,
        "skipped_draws": step * 4 - len(expected),
        "per_sigma": cumulative.counts.tolist(),
        "exact_worker_merge": True,
        "exact_predecessor_merge": True,
        "input_manifests_and_training_receipts": True,
        "historical_training_matches": sum(r["historical_training_match"] for r in cumulative.receipts),
        "fresh_input_receipts": sum(not r["historical_training_match"] for r in cumulative.receipts),
        "window_worker_provenance": sorted(owners, key=lambda r: r["draw_index"]),
        "limits": "Saved evidence audit; does not repeat native forwards or independently replay every token.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--step", required=True, type=int)
    args = parser.parse_args()
    result = audit_milestone(args.root.resolve(), args.step)
    atomic_json(args.root / f"monitoring/audits/step_{args.step:04d}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "window_worker_provenance"}, indent=2))


if __name__ == "__main__":
    main()
