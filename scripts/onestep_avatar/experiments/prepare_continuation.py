"""Prepare fresh E3 inputs and seal protocols from verified ordinary results.

See doc/experiments/prepare_continuation.md. Both operations are CPU-only;
the shared queue executes all emitted jobs. Existing evidence is never repaired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from scripts.onestep_avatar import evaluate, hashing
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.model import backbone
from scripts.onestep_avatar.training import resources

ENTRY = "scripts/onestep_avatar/experiments/prepare_continuation.py"
SIGMAS = (1.0, 0.909375)
BOUNDARIES = (3, 11)
ROLES = ("cached", "cached_repeat", "recompute")
FRAMES = 17
TOKENS_PER_FRAME = 1024
CHANNELS = 128


def write_json(path: Path, value: dict) -> None:
    dataset.atomic_write(
        path, lambda temporary: temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    )


def identity(path: Path, value: torch.Tensor | None = None) -> dict:
    record = {"path": str(path.resolve()), "sha256": hashing.sha256(path)}
    if value is not None:
        record.update(tensor_sha256=hashing.tensor_sha256(value), shape=list(value.shape), dtype=str(value.dtype))
    return record


def fixed_noise(seed: int, *, tokens_per_frame: int = TOKENS_PER_FRAME, channels: int = CHANNELS) -> dict:
    """Use one CPU draw; negate only the later region for each intervention."""
    original = torch.randn(
        (1, FRAMES * tokens_per_frame, channels),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.bfloat16,
    )
    values = {"original": original}
    for boundary in BOUNDARIES:
        changed = original.clone()
        changed[:, boundary * tokens_per_frame :] *= -1
        if torch.equal(changed, original):
            raise ValueError("future-noise intervention must change later bytes")
        values[f"changed_{boundary}"] = changed
    return values


def cohort_sources(membership: dict, sources: list[str], seeds: list[int], cohort: str) -> list[dict]:
    subset.validate_membership(membership)
    if membership["objective"] != "white" or len(set(sources)) != len(sources):
        raise ValueError("E3 requires white membership and unique source views")
    available = {row["relative_dir"]: row for row in membership["sources"]}
    if not sources or any(source not in available for source in sources):
        raise ValueError("E3 source selection is empty or absent from membership")
    selected = [available[source] for source in sources]
    if len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("E3 seeds must be distinct nonnegative integers")
    if cohort == "held_out":
        if (
            len(selected) != 3
            or len({row["actor"] for row in selected}) != 3
            or len(seeds) != 2
            or any(row["split"] != "held_out" for row in selected)
        ):
            raise ValueError("held-out E3 requires three people, one view each and two seeds")
    elif cohort != "first_view" or len(selected) != 1 or len(seeds) != 1:
        raise ValueError("first-view E3 requires one view and one seed")
    return selected


def evaluation_arguments(case: dict, membership: Path, sigma: float, role: str, output: Path) -> list[str]:
    arguments = [
        "--mode",
        "causal",
        "--subset",
        str(membership.resolve()),
        "--source",
        case["source"],
        "--split",
        case["split"],
        "--model",
        "2.5",
        "--variant",
        "dev",
        "--guide-mode",
        "d1",
        "--include-base",
        "--span-latent-frames",
        "17",
        "--block-latent-frames",
        "2",
        "--context-latent-frames",
        "8",
        "--seed",
        str(case["seed"]),
        "--schedule",
        str(sigma),
        "0",
        "--history-mode",
        "recompute" if role == "recompute" else "cache",
        "--kv-source",
        "refresh",
        "--noise-file",
        case["noise"]["original"]["path"],
        "--output",
        str(output.resolve()),
    ]
    if case.get("prompt") is not None:
        arguments += ["--prompt", case["prompt"]]
    return arguments


def queue_fields(budget: dict) -> dict:
    return {
        "numerical_policy": "deterministic",
        "resource_budget": budget["path"],
        "resource_budget_sha256": budget["sha256"],
    }


def baseline_job(case: dict, membership: Path, sigma: float, role: str, root: Path, budget: dict) -> dict:
    name = f"{case['id']}_sigma_{str(sigma).replace('.', '_')}_{role}"
    output = root / "baselines" / name
    return {
        "id": name,
        "kind": "evaluate",
        "arguments": evaluation_arguments(case, membership, sigma, role, output),
        "output": str(output),
        "completion": {"records": [str(output / "case_0000/variant_000/result.json")]},
        "dependencies": [],
        **queue_fields(budget),
    }


def plan(args: argparse.Namespace) -> dict:
    """Verify current producer inputs before creating any preparation artifacts."""
    output = args.output.resolve()
    if output.exists():
        raise ValueError("E3 preparation output must be fresh")
    membership_path = args.membership.resolve()
    membership_hash = hashing.sha256(membership_path)
    membership = json.loads(membership_path.read_text())
    selected = cohort_sources(membership, args.source, args.seeds, args.cohort)
    budget = resources.read_budget(args.resource_budget)
    if budget is None or budget["wall_seconds_per_phase"] != 1800:
        raise ValueError("E3 preparation requires its explicit 1800-second phase budget")
    store = dataset.ClipStore(membership)
    store.verify(require_guide=True)
    specification = backbone.resolve("2.5", "dev")
    from scripts.onestep_avatar.corpus.precompute import file_fingerprint  # noqa: PLC0415 -- producer identity only

    vae = file_fingerprint(Path(specification.paths.video_vae()))
    videos = []
    for row in selected:
        video = store.load(row["relative_dir"], require_guide=True)
        if (
            video.fps != 30
            or tuple(video.z_y.shape[:1]) != (CHANNELS,)
            or video.z_y.shape[1] < FRAMES
            or tuple(video.z_y.shape[2:]) != (32, 32)
            or video.z_g is None
            or video.z_g.shape != video.z_y.shape
            or any(video.encode_records[role]["vae_fingerprint"] != vae for role in ("capture", "guide"))
        ):
            raise ValueError("E3 needs matched native 1024-square, 30-fps, 17-frame LTX-2.5 masters")
        videos.append(video)
    producer = software.capture(
        "evaluation", "causal", extra_sources=(ENTRY, "scripts/onestep_avatar/experiments/__init__.py")
    )
    if hashing.sha256(membership_path) != membership_hash:
        raise ValueError("E3 membership changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    cases, jobs = [], []
    patch = VideoLatentPatchifier(1)
    for video in videos:
        capture = patch.patchify(video.z_y[:, :FRAMES].unsqueeze(0).to(torch.bfloat16))
        guide = patch.patchify(video.z_g[:, :FRAMES].unsqueeze(0).to(torch.bfloat16))
        for seed in args.seeds:
            name = f"actor_{video.actor}_{hashing.tensor_sha256(capture)[:8]}_seed_{seed}"
            directory = output / "inputs" / name
            directory.mkdir(parents=True)
            noise = {}
            for role, value in fixed_noise(seed).items():
                path = directory / f"noise_{role}.pt"
                dataset.atomic_write(path, lambda temporary, value=value: torch.save(value, temporary))
                noise[role] = identity(path, value)
            case = {
                "id": name,
                "source": video.source,
                "actor": video.actor,
                "split": video.split,
                "seed": seed,
                "prompt": args.prompt,
                "noise": noise,
                "input_file_hashes": video.hashes,
                "capture_sha256": hashing.tensor_sha256(capture),
                "guide_sha256": hashing.tensor_sha256(guide),
                "c0_sha256": hashing.tensor_sha256(capture[:, :TOKENS_PER_FRAME]),
            }
            case_jobs = [
                baseline_job(case, membership_path, sigma, role, output, budget) for sigma in SIGMAS for role in ROLES
            ]
            case["baseline_jobs"] = case_jobs
            cases.append(case)
            jobs.extend(case_jobs)
    record = {
        "schema_version": 1,
        "kind": "onestep_avatar.e3_preparation",
        "cohort": args.cohort,
        "membership": identity(membership_path),
        "membership_sha256": membership["sha256"],
        "software": producer,
        "resource_budget": budget,
        "cases": cases,
        "state": "prepared_not_run",
        "required_separate_evidence": ["c0", "prefix", "quality", "warmed_cost"],
    }
    record["sha256"] = subset.record_hash(record)
    write_json(output / "plan.json", record)
    write_json(output / "baseline_jobs.json", {"schema_version": 1, "jobs": jobs})
    return record


def read_plan(path: Path) -> dict:
    record = json.loads(path.read_text())
    if (
        record.get("schema_version") != 1
        or record.get("kind") != "onestep_avatar.e3_preparation"
        or subset.record_hash(record) != record.get("sha256")
    ):
        raise ValueError("E3 preparation plan content changed")
    software.check_current(record["software"])
    membership_path = Path(record["membership"]["path"])
    if identity(membership_path) != record["membership"]:
        raise ValueError("E3 preparation membership bytes changed")
    membership = json.loads(membership_path.read_text())
    cohort_sources(
        membership,
        list(dict.fromkeys(case["source"] for case in record["cases"])),
        list(dict.fromkeys(case["seed"] for case in record["cases"])),
        record["cohort"],
    )
    sources = {case["source"] for case in record["cases"]}
    seeds = {case["seed"] for case in record["cases"]}
    visits = [(case["source"], case["seed"]) for case in record["cases"]]
    if len(set(visits)) != len(visits) or set(visits) != {(source, seed) for source in sources for seed in seeds}:
        raise ValueError("E3 preparation must retain every source and seed case exactly once")
    if resources.read_budget(Path(record["resource_budget"]["path"])) != record["resource_budget"]:
        raise ValueError("E3 preparation budget bytes changed")
    dataset.ClipStore(membership).verify(require_guide=True)
    for case in record["cases"]:
        values = {}
        for row in case["noise"].values():
            value = torch.load(row["path"], map_location="cpu", weights_only=True)
            if identity(Path(row["path"]), value) != row:
                raise ValueError("E3 preparation noise bytes or tensor changed")
        for role, row in case["noise"].items():
            values[role] = torch.load(row["path"], map_location="cpu", weights_only=True)
        original = values["original"]
        if (
            original.dtype != torch.bfloat16
            or tuple(original.shape) != (1, FRAMES * TOKENS_PER_FRAME, CHANNELS)
            or not torch.isfinite(original).all()
        ):
            raise ValueError("E3 preparation needs finite native full-length noise")
        for boundary in BOUNDARIES:
            changed = values[f"changed_{boundary}"]
            cut = boundary * TOKENS_PER_FRAME
            if (
                changed.shape != original.shape
                or changed.dtype != original.dtype
                or not torch.equal(changed[:, :cut], original[:, :cut])
                or not torch.equal(changed[:, cut:], -original[:, cut:])
                or torch.equal(changed, original)
            ):
                raise ValueError("E3 future noise must preserve earlier bytes and negate only later bytes")
    return record


def verify_case(case: dict) -> list[dict]:
    """Delegate full scientific checks, then prove fixed inputs and exact repeats."""
    rows, outputs = [], {}
    text_hash = None
    for job in case["baseline_jobs"]:
        arguments = job["arguments"]
        path = Path(job["completion"]["records"][0])
        evaluate.verify_evaluation_conditions(arguments, [path])
        record = json.loads(path.read_text())
        sigma = float(arguments[arguments.index("--schedule") + 1])
        role = next(role for role in reversed(ROLES) if job["id"].endswith("_" + role))
        fixed = {
            key: case[key] for key in ("source", "input_file_hashes", "capture_sha256", "guide_sha256", "c0_sha256")
        }
        fixed["noise_sha256"] = case["noise"]["original"]["tensor_sha256"]
        if any(record.get(key) != value for key, value in fixed.items()):
            raise ValueError("E3 ordinary baseline fixed inputs differ from the preparation")
        text_hash = record["text_sha256"] if text_hash is None else text_hash
        if record["text_sha256"] != text_hash:
            raise ValueError("E3 ordinary baseline text differs between roles or sigmas")
        latent = torch.load(record["output"]["path"], map_location="cpu", weights_only=True)
        if role in ("cached", "cached_repeat"):
            outputs[(sigma, role)] = latent
        rows.append(
            {
                "id": job["id"],
                "sigma": sigma,
                "role": role,
                "history": "recompute" if role == "recompute" else "cache",
                "arguments": arguments,
                "result": identity(path),
            }
        )
    if {(row["sigma"], row["role"]) for row in rows} != {(sigma, role) for sigma in SIGMAS for role in ROLES}:
        raise ValueError("E3 baseline inventory is incomplete")
    for sigma in SIGMAS:
        if not torch.equal(outputs[(sigma, "cached")], outputs[(sigma, "cached_repeat")]):
            raise ValueError("E3 cached same-input repeat is not bit-identical")
    return rows


def experiment_job(
    name: str, spec: Path, arguments: list[str], output: Path, budget: dict, dependencies: list[str]
) -> dict:
    write_json(spec, {"schema_version": 1, "protocol": "continuation_check", "arguments": arguments})
    return {
        "id": name,
        "kind": "experiment",
        "experiment": "continuation_check",
        "spec": str(spec),
        "spec_sha256": hashing.sha256(spec),
        "arguments": ["--spec", str(spec), "--output", str(output)],
        "output": str(output),
        "completion": {"manifest": str(output / "manifest.json")},
        "dependencies": dependencies,
        **queue_fields(budget),
    }


def seal(args: argparse.Namespace) -> dict:
    """Publish protocols only after every ordinary baseline and repeat is verified."""
    if args.output.exists():
        raise ValueError("sealed E3 output must be fresh")
    record = read_plan(args.plan)
    checked = [(case, verify_case(case)) for case in record["cases"]]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    jobs, protocols = [], []
    for case, baselines in checked:
        directory = root / case["id"]
        directory.mkdir()
        (directory / "runs").mkdir()
        protocol = {
            "scope": "fresh E3 base generated-history case",
            "membership": record["membership"]["path"],
            "membership_sha256": record["membership"]["sha256"],
            "source": case["source"],
            "model": "2.5",
            "variant": "dev",
            "adapter": None,
            "guide_mode": "d1",
            "objective": "white",
            "latent_frames": 17,
            "rgb_frames": 129,
            "geometry": {"B": 2, "D": 8, "sink": 1, "height": 1024, "width": 1024},
            "noise": {**case["noise"]["original"], "seed": case["seed"], "generator_device": "cpu"},
            "input_file_hashes": case["input_file_hashes"],
            "jobs": baselines,
            "identical_repeat_required": True,
            "c0_exact_required": True,
            "wall_seconds_per_case": 1800,
            "observed_gpu_memory_limit_mib": 48800,
        }
        protocol_path = directory / "protocol.json"
        write_json(protocol_path, protocol)
        controls = []
        for row in baselines:
            if row["role"] == "cached_repeat":
                continue
            for boundary in BOUNDARIES:
                name = f"sigma_{row['sigma']}_{row['role']}_future_{boundary}"
                arguments = row["arguments"] + [
                    "--changed-noise-file",
                    case["noise"][f"changed_{boundary}"]["path"],
                    "--future-noise-start",
                    str(boundary),
                ]
                controls.append(
                    {
                        "id": name,
                        "arguments": arguments,
                        "source": case["source"],
                        "schedule": [row["sigma"], 0],
                        "history_mode": row["history"],
                        "changed_variable": f"noise from encoded frame {boundary}",
                        "state": "not_run",
                    }
                )
        control_path = directory / "controls.json"
        write_json(
            control_path,
            {
                "original_protocol_sha256": hashing.sha256(protocol_path),
                "controls": controls,
                "fixed_noise_files": case["noise"],
                "state": "not_run",
            },
        )
        for sigma in SIGMAS:
            baseline = next(row for row in baselines if row["sigma"] == sigma and row["role"] == "cached")
            name = f"{case['id']}_sigma_{str(sigma).replace('.', '_')}_generated_capture"
            destination = directory / "runs" / name
            specs = directory / "specs"
            specs.mkdir(exist_ok=True)
            jobs.append(
                experiment_job(
                    name,
                    specs / f"{name}.json",
                    [
                        "--phase",
                        "capture",
                        "--protocol",
                        str(protocol_path),
                        "--sigma",
                        str(sigma),
                        "--history",
                        "generated",
                    ],
                    destination,
                    record["resource_budget"],
                    [baseline["id"]],
                )
            )
            reference = name.replace("_capture", "_reference")
            jobs.append(
                experiment_job(
                    reference,
                    specs / f"{reference}.json",
                    ["--phase", "reference", "--snapshot", str(destination / "continuation.json")],
                    directory / "runs" / reference,
                    record["resource_budget"],
                    [name],
                )
            )
        for control in controls:
            name = f"{case['id']}_{control['id'].replace('.', '_')}"
            role = "cached" if control["history_mode"] == "cache" else "recompute"
            baseline = next(row for row in baselines if row["sigma"] == control["schedule"][0] and row["role"] == role)
            jobs.append(
                experiment_job(
                    name,
                    specs / f"{name}.json",
                    [
                        "--phase",
                        "control",
                        "--protocol",
                        str(protocol_path),
                        "--control-plan",
                        str(control_path),
                        "--control-id",
                        control["id"],
                    ],
                    directory / "runs" / name,
                    record["resource_budget"],
                    [baseline["id"]],
                )
            )
        protocols.append(identity(protocol_path))
    full_jobs = [job for case in record["cases"] for job in case["baseline_jobs"]] + jobs
    write_json(root / "continuation_jobs.json", {"schema_version": 1, "jobs": full_jobs})
    receipt = {
        "schema_version": 1,
        "preparation": identity(args.plan),
        "protocols": protocols,
        "baseline_jobs": len(full_jobs) - len(jobs),
        "diagnostic_jobs": len(jobs),
        "required_separate_evidence": record["required_separate_evidence"],
        "state": "sealed_not_run",
    }
    write_json(root / "sealed.json", receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    prepare = operations.add_parser("plan")
    prepare.add_argument("--membership", type=Path, required=True)
    prepare.add_argument("--source", action="append", required=True)
    prepare.add_argument("--seeds", nargs="+", type=int, required=True)
    prepare.add_argument("--cohort", choices=("first_view", "held_out"), required=True)
    prepare.add_argument("--resource-budget", type=Path, required=True)
    prepare.add_argument("--prompt")
    publish = operations.add_parser("seal")
    publish.add_argument("--plan", type=Path, required=True)
    for command in (prepare, publish):
        command.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = plan(args) if args.operation == "plan" else seal(args)
    print(json.dumps({"state": result["state"], "output": str(args.output.resolve())}))  # noqa: T201 -- CLI summary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
