"""Measure saved guide-study outputs on CPU; see the mirrored design document."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from scripts.onestep_avatar import hashing
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.experiments.prepare_guide_study import write_json
from scripts.onestep_avatar.model import backbone, causal


def read_output(record: dict) -> torch.Tensor:
    """Check output identity before and after reading."""
    path = Path(record["output"]["path"])
    digest = hashing.sha256(path)
    if digest != record["output"]["sha256"]:
        raise ValueError("saved output hash mismatch")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        list(value.shape) != record["output"]["shape"]
        or value.ndim != 5
        or not torch.isfinite(value).all()
        or hashing.sha256(path) != digest
    ):
        raise ValueError("saved output geometry, values or bytes changed")
    return value


def reconstructed_ranges(record: dict, scale_factors) -> list:  # noqa: ANN001
    """Call the canonical planner; these are not saved executed ranges."""
    settings = record["mode_settings"]
    geometry = causal.CausalGeometry(scale_factors, settings["block_latent_frames"], settings["context_latent_frames"])
    ranges = geometry.plan(record["frames"])
    counts = record["call_counts"]
    if (
        not ranges
        or ranges[-1][1] != record["frames"]
        or counts["denoise_calls"] != len(ranges) * (len(record["schedule"]) - 1)
        or counts["refresh_calls"] != len(ranges)
        or counts["prime_calls"] != 0
    ):
        raise ValueError("saved frames and calls disagree with canonical block planner")
    return ranges


def measures(output: torch.Tensor, capture: torch.Tensor, guide: torch.Tensor, ranges: list) -> dict:
    """Exclude c0 from each aggregate encoded error."""
    output, capture, guide = output.float(), capture.float(), guide.float()
    if output.shape != capture.shape or output.shape != guide.shape or output.shape[2] < 2:
        raise ValueError("metric tensors require identical geometry and at least two frames")
    per_frame = (output - capture).square().mean(dim=(0, 1, 3, 4))
    starts = [span[0] for span in ranges[1:]]
    seams = [
        ((output[:, :, t] - output[:, :, t - 1]) - (capture[:, :, t] - capture[:, :, t - 1])).abs().mean().item()
        for t in starts
    ]
    return {
        "capture_mse": per_frame[1:].mean().item(),
        "guide_mse": (output[:, :, 1:] - guide[:, :, 1:]).square().mean().item(),
        "guide_capture_mse": (guide[:, :, 1:] - capture[:, :, 1:]).square().mean().item(),
        "per_frame_capture_mse": per_frame.tolist(),
        "boundary_temporal_residual_mae": sum(seams) / len(seams) if seams else None,
        "boundary_starts": starts,
        "boundary_ranges": [list(span) for span in ranges],
        "boundary_evidence": "canonical_planner_reconstruction" if ranges else None,
    }


def equal_control(left: torch.Tensor, right: torch.Tensor, span: tuple | None = None) -> bool:
    if span is not None:
        left, right = left[:, :, slice(*span)], right[:, :, slice(*span)]
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)


def load_results(root: Path, allow_partial: bool) -> tuple:
    """Check completed cases against content-pinned current study inputs."""
    plan = json.loads((root / "inputs/plan.json").read_text())
    store = dataset.ClipStore(subset.load_membership(root / "inputs/membership.json"))
    store.verify(require_guide=True)
    rows, missing = [], []
    for case in plan["cases"]:
        if case["kind"] != "evaluate":
            continue
        path = root / "runs" / case["id"] / "case_0000/variant_000/result.json"
        if not path.is_file():
            missing.append(case["id"])
            continue
        digest = hashing.sha256(path)
        record = json.loads(path.read_text())
        if record.get("state") != "complete" or record["schedule"] != case["schedule"]:
            raise ValueError("saved result incomplete or schedule differs")
        if (
            record["source"] != f"Part_1/{case['actor']}/views/view00_cam51"
            or record["guide_mode"] != case["arm"]
            or record["prompt"] != case["prompt"]
            or record["seed"] != 42
            or record["frames"] != (17 if case["history"] else 18)
        ):
            raise ValueError("saved scientific case differs from planned inputs")
        source = next(r for r in plan["input_records"] if r["actor"] == case["actor"])
        noise = source["noise"][str(record["frames"])]
        if record["noise_sha256"] != noise["tensor_sha256"] or record["input_file_hashes"] != (
            source["source_hashes"] if case["arm"] == "d1" else {"capture": source["source_hashes"]["capture"]}
        ):
            raise ValueError("saved noise or source hashes differ from preparation")
        if hashing.sha256(Path(noise["path"])) != noise["sha256"]:
            raise ValueError("saved input noise file changed")
        expected = (
            {"cfg": 3.0, "stg": 1.0, "stg_blocks": [28], "rescale": 0.7}
            if case["variant"] == "dev"
            else {"cfg": 1.0, "stg": 0.0, "stg_blocks": [], "rescale": 0.0}
        )
        if any(record["guidance"].get(k) != v for k, v in expected.items()):
            raise ValueError("saved guidance differs from planned recipe")
        ranges = (
            reconstructed_ranges(record, backbone.resolve("2.5", case["variant"]).scale_factors)
            if case["history"]
            else []
        )
        if case["history"] and (
            record.get("history_mode") != "cache"
            or record.get("kv_source") != "refresh"
            or record["mode_settings"]["teacher_forcing"] != (case["history"] == "gt")
        ):
            raise ValueError("saved history differs from plan")
        counts = record["call_counts"]
        if counts["model_calls"] != (3 if case["variant"] == "dev" else 1) * (
            counts["denoise_calls"] + counts["refresh_calls"]
        ):
            raise ValueError("measured forwards disagree with guidance and refresh")
        output = read_output(record)
        if hashing.sha256(path) != digest:
            raise ValueError("result record changed while reading")
        rows.append((case, record, output, ranges, {"path": str(path), "sha256": digest}))
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} results missing; measurements never generate them")
    return plan, rows, missing


def measure(root: Path, allow_partial: bool) -> dict:
    plan, rows, missing = load_results(root, allow_partial)
    store = dataset.ClipStore(subset.load_membership(root / "inputs/membership.json"))
    videos, results = {}, []
    for case, record, output, ranges, identity in rows:
        if case["actor"] not in videos:
            videos[case["actor"]] = store.load(record["source"], require_guide=True)
        video = videos[case["actor"]]
        frames = record["frames"]
        results.append(
            {
                "id": case["id"],
                "case": case,
                "result": identity,
                **measures(output, video.z_y[:, :frames].unsqueeze(0), video.z_g[:, :frames].unsqueeze(0), ranges),
            }
        )
    return {
        "schema_version": 1,
        "results": results,
        "missing": missing,
        "complete": not missing,
        "boundary_evidence": plan["boundary_evidence"],
    }


def controls(root: Path, allow_partial: bool) -> dict:  # noqa: PLR0912 -- distinct recorded proofs
    _plan, rows, missing = load_results(root, allow_partial)
    sigma_groups, history_groups, fixed_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    schedules = []
    for row in rows:
        case, record, _value, _ranges, identity = row
        key = (case["actor"], case["variant"], tuple(case["schedule"]), case["prompt"])
        if case["sigma"] == 1 and not case["history"] and any(r["prompt"] == "P0" for r in case["roles"]):
            sigma_groups[key].append(row)
        if case["history"]:
            history_groups[(*key, case["arm"])].append(row)
        fixed_groups[(case["actor"], bool(case["history"]))].append(row)
        schedules.append(
            {
                "id": case["id"],
                "requested": case["schedule"],
                "executed": record["schedule"],
                "call_counts": record["call_counts"],
                "result": identity,
                "passed": True,
            }
        )
    checks, incomplete_pairs = [], []
    for kind, groups in (("sigma_one", sigma_groups), ("first_block", history_groups)):
        for key, group in groups.items():
            if len(group) != 2:
                if allow_partial:
                    incomplete_pairs.append({"kind": kind, "key": list(map(str, key))})
                    continue
                raise ValueError("control pair inventory incomplete")
            left, right = group
            span = left[3][0] if kind == "first_block" else None
            if span is not None and left[3] != right[3]:
                raise ValueError("history control ranges differ")
            checks.append(
                {
                    "kind": kind,
                    "ids": [left[0]["id"], right[0]["id"]],
                    "passed": equal_control(left[2], right[2], span),
                    "first_block_range": list(span) if span else None,
                    "boundary_evidence": "canonical_planner_reconstruction" if span else None,
                    "results": [left[4], right[4]],
                }
            )
    matched = []
    for key, group in fixed_groups.items():
        for field in ("noise_sha256", "capture_sha256", "c0_sha256", "source"):
            if len({json.dumps(row[1][field], sort_keys=True) for row in group}) != 1:
                raise ValueError(f"fixed study input {field} differs within {key}")
        for arm in ("d0", "d1"):
            selected = [r for r in group if r[0]["arm"] == arm]
            for field in ("guide_sha256", "input_file_hashes"):
                if len({json.dumps(r[1][field], sort_keys=True) for r in selected}) > 1:
                    raise ValueError("role-specific source evidence differs")
        for variant in ("distilled", "dev"):
            if len({r[1]["conditions"]["model"]["base_sha256"] for r in group if r[0]["variant"] == variant}) > 1:
                raise ValueError("weights differ within declared model variant")
        texts = defaultdict(set)
        for row in group:
            texts[row[0]["prompt"]].add(row[1]["text_sha256"])
        if any(len(values) != 1 for values in texts.values()):
            raise ValueError("identical prompts produced different text conditioning")
        matched.append(
            {
                "actor": key[0],
                "causal": key[1],
                "ids": [r[0]["id"] for r in group],
                "passed": True,
                "declared_exceptions": ["prompt text", "model variant"],
            }
        )
    return {
        "schema_version": 1,
        "passed": all(c["passed"] for c in checks),
        "complete": not missing and not incomplete_pairs,
        "checks": checks,
        "missing": missing,
        "incomplete_pairs": incomplete_pairs,
        "matched_inputs": matched,
        "executed_schedules": schedules,
        "boundary_evidence": "canonical_planner_reconstruction",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("measure", "controls"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    result = measure(root, args.allow_partial) if args.operation == "measure" else controls(root, args.allow_partial)
    if not args.dry_run:
        write_json(root / f"analysis/{'latent_metrics' if args.operation == 'measure' else 'controls'}.json", result)
        if args.operation == "controls":
            for name in ("matched_inputs", "executed_schedules"):
                write_json(root / f"analysis/{name}.json", result[name])
    print(json.dumps({"operation": args.operation, "complete": result["complete"], "missing": len(result["missing"])}))  # noqa: T201
    return 2 if result.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
