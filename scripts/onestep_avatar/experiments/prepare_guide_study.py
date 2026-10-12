"""Prepare CPU-only S1-S3 guide-study inputs; see the mirrored design document."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from scripts.onestep_avatar import hashing
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.model import backbone, sampling
from scripts.onestep_avatar.training.config import (
    BidirectionalSettings,
    CausalSettings,
    RunSettings,
    build_frame_plan,
)

ACTORS = ("0008_01", "0012_09", "0025_11")
SIGMAS = (0.421875, 0.725, 0.909375, 0.975, 1.0)
COUNTS = (1, 2, 4, 6, 8, 10, 12, 16, 20, 25)
PHASES = ("s1", "s3_distilled", "s2_full", "s3_dev", "s2_steps", "s2_bench")


def write_json(path: Path, value: object) -> None:
    """Publish JSON using the corpus atomic writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.atomic_write(path, lambda temporary: temporary.write_text(json.dumps(value, indent=2) + "\n"))


def prompt_rows(path: Path) -> dict:
    """Verify literal text before retaining only the study's approved white rows."""
    record = json.loads(path.read_text())
    rows = {}
    for row in record["prompts"]:
        if row["actor"] in ACTORS and row["objective"] == "white" and row["view"] == "view00_cam51":
            digest = hashlib.sha256(row["text"].encode()).hexdigest()
            if digest != row["prompt_sha256"]:
                raise ValueError("prompt text hash mismatch")
            key = (row["actor"], row["variant"])
            if key in rows:
                raise ValueError("duplicate study prompt")
            rows[key] = row
    if set(rows) != {(actor, f"P{i}") for actor in ACTORS for i in range(5)}:
        raise ValueError("study requires all three actors and P0-P4")
    return rows


def tail(sigma: float, variant: str) -> list[float]:
    """Get exact official levels from their public owners."""
    if variant == "dev":
        return list(sampling.truncated_schedule(sigma, 30))
    grid = backbone.resolve("2.5", "distilled").sigmas
    return list(sampling.validate_schedule([sigma, *[v for v in grid if v < sigma]]))


def inventory(prompts: dict) -> list[dict]:  # noqa: PLR0912 -- explicit scientific inventory
    """Build unique evaluations; retain every sub-question role as an alias."""
    cases = {}

    def add(
        actor: str,
        arm: str,
        sigma: float,
        schedule: list[float],
        variant: str,
        phase: str,
        question: str,
        prompt: str = "P0",
        history: str | None = None,
        kind: str = "evaluate",
    ) -> None:
        text = prompts[(actor, prompt)]["text"]
        key = (actor, arm, tuple(schedule), variant, text, history, kind)
        role = {"question": question, "prompt": prompt}
        if key in cases:
            if role not in cases[key]["roles"]:
                cases[key]["roles"].append(role)
            return
        identifier = f"{phase}_{actor}_{arm}_{sigma:g}_{len(schedule) - 1}steps_{prompt}_{history or 'whole'}"
        cases[key] = {
            "id": identifier.replace(".", "_"),
            "actor": actor,
            "arm": arm,
            "sigma": sigma,
            "schedule": schedule,
            "variant": variant,
            "phase": phase,
            "prompt": text,
            "history": history,
            "kind": kind,
            "roles": [role],
        }

    for actor in ACTORS:
        for arm in ("d0", "d1"):
            for sigma in SIGMAS:
                add(actor, arm, sigma, [sigma, 0.0], "distilled", "s1", "1a" if arm == "d0" else "1b")
                add(actor, arm, sigma, [sigma, 0.0], "distilled", "s1", "2a")
                if arm == "d0":
                    add(actor, arm, sigma, [sigma, 0.0], "distilled", "s1", "1b")
                if sigma != SIGMAS[0]:
                    add(actor, arm, sigma, [sigma, 0.0], "distilled", "s1", "1c" if arm == "d0" else "1d")
                    add(actor, arm, sigma, tail(sigma, "distilled"), "distilled", "s1", "1c" if arm == "d0" else "1d")
                add(actor, arm, sigma, tail(sigma, "dev"), "dev", "s2_full", "2a")
            for sigma in (*SIGMAS[:3], 0.97, *SIGMAS[3:]):
                maximum = len(tail(sigma, "dev")) - 1
                for count in (*[n for n in COUNTS if n < maximum], maximum):
                    schedule = list(sampling.thinned_truncated_schedule(sigma, 30, count))
                    add(actor, arm, sigma, schedule, "dev", "s2_steps", "2b" if arm == "d0" else "2c")
            for variant, levels in (("distilled", SIGMAS[:-1]), ("dev", SIGMAS[:3])):
                for sigma in levels:
                    for history in ("gt", "generated"):
                        add(
                            actor,
                            arm,
                            sigma,
                            tail(sigma, variant),
                            variant,
                            "s3_distilled" if variant == "distilled" else "s3_dev",
                            ("3a" if arm == "d0" else "3b") if variant == "distilled" else "3c",
                            history=history,
                        )
        for sigma in SIGMAS[1:]:
            for prompt in ("P0", "P1", "P2", "P3", "P4"):
                for schedule in ([sigma, 0.0], tail(sigma, "distilled")):
                    add(actor, "d1", sigma, schedule, "distilled", "s1", "1e", prompt)
        for sigma in SIGMAS[2:]:
            for prompt in ("P0", "P1", "P2", "P3", "P4"):
                add(actor, "d1", sigma, tail(sigma, "dev"), "dev", "s2_full", "2d", prompt)
    for sigma in SIGMAS:
        add(ACTORS[0], "d1", sigma, tail(sigma, "dev"), "dev", "s2_bench", "2e", kind="bench")
    values = list(cases.values())
    counts = Counter("S3" if c["history"] else "S1" if c["variant"] == "distilled" else "S2" for c in values)
    if counts != {"S1": 142, "S2": 338, "S3": 84}:
        raise ValueError(f"study inventory differs from required counts: {counts}")
    return values


def balanced(items: list[dict], owners: int) -> list[list[dict]]:
    """Balance estimated denoise work, with model-loading cost per process."""
    bins, costs = [[] for _ in range(owners)], [0] * owners
    for item in sorted(items, key=lambda c: -len(c.get("schedule", []))):
        index = min(range(owners), key=lambda i: (costs[i], i))
        bins[index].append(item)
        costs[index] += 3 + (len(item.get("schedule", [])) - 1) * (8 if item.get("history") else 1)
    return bins


def evaluation_job(case: dict, root: Path) -> dict:
    """Express each case through the existing ordinary CLI only."""
    mode = "causal" if case["history"] else "bidirectional"
    output = root / "runs" / case["id"]
    args = [
        "--mode",
        mode,
        "--subset",
        str(root / "inputs/membership.json"),
        "--frame-plan",
        str(root / f"inputs/{mode}_frame_plan.json"),
        "--split",
        "test",
        "--source",
        f"Part_1/{case['actor']}/views/view00_cam51",
        "--model",
        "2.5",
        "--variant",
        case["variant"],
        "--guide-mode",
        case["arm"],
        "--seed",
        "42",
        "--noise-file",
        str(root / f"inputs/{case['actor']}/noise_{17 if case['history'] else 18}.pt"),
        "--prompt",
        case["prompt"],
        "--schedule",
        *map(str, case["schedule"]),
        "--output",
        str(output),
    ]
    if mode == "causal":
        args += [
            "--output-latent-frames",
            "17",
            "--block-latent-frames",
            "2",
            "--context-latent-frames",
            "8",
            "--history-mode",
            "cache",
            "--kv-source",
            "refresh",
        ]
        if case["history"] == "gt":
            args += ["--teacher-forcing"]
    args += (
        ["--cfg", "3", "--stg", "1", "--stg-blocks", "28", "--rescale", "0.7"]
        if case["variant"] == "dev"
        else ["--cfg", "1", "--stg", "0", "--rescale", "0"]
    )
    if case["kind"] == "bench":
        args += ["--repetitions", "3", "--warmup", "1"]
    return {
        "id": case["id"],
        "kind": case["kind"],
        "arguments": args,
        "output": str(output),
        "dependencies": [],
        "numerical_policy": "deterministic",
        "completion": {"records": [str(output / "case_0000/variant_000/result.json")]},
        **({"resource_budget": str(root / "inputs/bench_budget.json")} if case["kind"] == "bench" else {}),
    }


def comparison_specs(cases: list[dict], root: Path, corpus: Path) -> list[tuple[dict, dict]]:
    """One render per actor/question; small comparison grids cover every output."""
    groups = defaultdict(list)
    for case in cases:
        if case["kind"] != "evaluate":
            continue
        for question in sorted({r["question"] for r in case["roles"]}):
            groups[(case["actor"], question)].append(case)
    specs = []
    for (actor, question), selected in sorted(groups.items()):
        name = f"{question}_{actor}"
        view = corpus / f"Part_1/{actor}/views/view00_cam51"
        causal = question.startswith("3")
        comparisons = []
        # Keep sigma and schedule families separate so every group has readable panel titles.
        sections = defaultdict(list)
        for case in selected:
            family = (
                case["sigma"],
                len(case["schedule"]) if question in ("1e", "2d") else 0,
                case["arm"] if question in ("2a", "3c") else "both",
            )
            sections[family].append(case)
        for family, rows in sorted(sections.items()):
            for start in range(0, len(rows), 4):
                chunk = rows[start : start + 4]
                panels = [
                    {
                        "title": "Capture (VAE)",
                        "role": "capture",
                        "latent": f"capture:{view / 'ltx_vae_latent_white.pt'}",
                    },
                    {
                        "title": "Guide (VAE)",
                        "role": "guide",
                        "latent": f"guide:{view / 'argavatar_ltx_vae_latent_white.pt'}",
                    },
                ]
                for case in chunk:
                    labels = "/".join(sorted({r["prompt"] for r in case["roles"] if r["question"] == question}))
                    title = f"{case['arm'].upper()} {case['variant']} {len(case['schedule']) - 1} steps {labels}"
                    if case["history"]:
                        title += " " + case["history"]
                    panels.append(
                        {
                            "title": title,
                            "role": case["id"],
                            "latent": str(root / "runs" / case["id"] / "case_0000/variant_000/generated.pt"),
                        }
                    )
                comparisons.append(
                    {
                        "name": f"{name}_sigma{family[0]:g}_family{family[1]}_{family[2]}_part{start // 4}".replace(
                            ".", "_"
                        ),
                        "question": f"{question}, actor {actor}, sigma {family[0]:g}",
                        "span": 17 if causal else 18,
                        "fps": 30,
                        "view": str(view),
                        "poster_frame": 24,
                        "panels": panels,
                    }
                )
        spec_path = root / "queues/specs" / f"{name}.json"
        output = root / "renders" / name
        job = {
            "id": "render_" + name,
            "kind": "render",
            "output": str(output),
            "arguments": [
                "--render-saved-comparisons",
                str(spec_path),
                "--seed",
                "42",
                "--output",
                str(output),
            ],
            "dependencies": [c["id"] for c in selected],
            "numerical_policy": "deterministic",
            "completion": {"manifest": str(output / "render_manifest.json")},
            "resource_budget": str(root / "inputs/render_budget.json"),
        }
        specs.append(({"comparisons": comparisons}, job))
    return specs


def fixed_noise(seed: int, frames: int = 18, tokens_per_frame: int = 1024, channels: int = 128) -> dict:
    """Draw once, then preserve exactly the first seventeen frames for causal work."""
    value = torch.randn(
        (1, frames * tokens_per_frame, channels),
        dtype=torch.bfloat16,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    return {18: value, 17: value[:, : 17 * tokens_per_frame].clone()}


def plan(args: argparse.Namespace) -> dict:  # noqa: PLR0912 -- ordered prepublication gates
    """Verify inputs before publishing a fresh preparation."""
    root = args.root.resolve()
    if any((root / name).exists() for name in ("inputs", "queues")):
        raise ValueError("guide-study inputs and queues must be fresh")
    if args.owners < 1:
        raise ValueError("owners must be positive")
    prompts = prompt_rows(args.prompts)
    cases = inventory(prompts)
    sources = [f"Part_1/{actor}/views/view00_cam51" for actor in ACTORS]
    membership = subset.build_membership(
        args.corpus_root,
        objective="white",
        require_guide=True,
        source_ids=sources,
        splits={
            "train": [],
            "held_out": [],
            "validation": [],
            "test": [str(int(actor.split("_")[0])) for actor in ACTORS],
        },
    )
    store = dataset.ClipStore(membership)
    videos = [store.load(source, require_guide=True) for source in sources]
    spec = backbone.resolve("2.5", "distilled")
    from scripts.onestep_avatar.corpus.precompute import file_fingerprint  # noqa: PLC0415

    for video in videos:
        if tuple(video.z_y.shape) != (128, 18, 32, 32) or video.z_g.shape != video.z_y.shape or video.fps != 30:
            raise ValueError("study requires matched 18-frame, 1024-square, 30-fps masters")
        if any(
            video.encode_records[role]["vae_fingerprint"] != file_fingerprint(Path(spec.paths.video_vae()))
            for role in ("capture", "guide")
        ):
            raise ValueError("study source VAE differs from selected model")
    summary = {
        "schema_version": 1,
        "counts": {"S1": 142, "S2": 338, "S3": 84},
        "owners": args.owners,
        "cases": cases,
        "state": "checked_not_written" if args.dry_run else "prepared_not_run",
    }
    if args.dry_run:
        return summary
    (root / "inputs").mkdir(parents=True, exist_ok=False)
    (root / "queues").mkdir(exist_ok=False)
    for kind in ("bench", "render"):
        write_json(
            root / f"inputs/{kind}_budget.json",
            {"wall_seconds_per_phase": 7200, "memory_limit_allocated_bytes": 48_000_000_000},
        )
    membership = subset.summarize_exclusions(membership)
    subset.write_membership(root / "inputs/membership.json", membership)
    for mode, mode_settings in (
        ("bidirectional", BidirectionalSettings()),
        ("causal", CausalSettings(span_latent_frames=17)),
    ):
        settings = RunSettings(
            mode, root / "inputs/membership.json", root / "runs", mode_settings, objective="white", split="test"
        )
        write_json(root / f"inputs/{mode}_frame_plan.json", build_frame_plan(settings, membership, spec.scale_factors))
    input_records = []
    for video in videos:
        actor = Path(video.source).parts[1]
        directory = root / "inputs" / actor
        directory.mkdir()
        noise = fixed_noise(42)
        identities = {}
        for frames in (18, 17):
            value = noise[frames]
            path = directory / f"noise_{frames}.pt"
            dataset.atomic_write(path, lambda temporary, value=value: torch.save(value, temporary))
            identities[str(frames)] = {
                "path": str(path),
                "sha256": hashing.sha256(path),
                "tensor_sha256": hashing.tensor_sha256(value),
                "shape": list(value.shape),
            }
        input_records.append(
            {
                "actor": actor,
                "source": video.source,
                "source_hashes": video.hashes,
                "encode_records": video.encode_records,
                "noise": identities,
                "slice_verified": True,
            }
        )
    write_json(
        root / "inputs/prompts.json",
        {
            "source": str(args.prompts.resolve()),
            "source_sha256": hashing.sha256(args.prompts),
            "prompts": list(prompts.values()),
        },
    )
    jobs = {case["id"]: evaluation_job(case, root) for case in cases}
    for phase in PHASES:
        selected = [case for case in cases if case["phase"] == phase]
        for i, owner_cases in enumerate(balanced(selected, args.owners)):
            write_json(
                root / f"queues/{phase}_owner{i}.json",
                {"schema_version": 1, "jobs": [jobs[c["id"]] for c in owner_cases]},
            )
    renders = []
    for specification, job in comparison_specs(cases, root, args.corpus_root.resolve()):
        path = Path(job["arguments"][1])
        write_json(path, specification)
        deps = [jobs[identifier] for identifier in job["dependencies"]]
        write_json(root / f"queues/{job['id']}.json", {"schema_version": 1, "jobs": [*deps, job]})
        renders.append(job)
    summary.update(
        input_records=input_records,
        jobs=list(jobs.values()),
        render_jobs=renders,
        boundary_evidence="canonical_planner_reconstruction",
    )
    write_json(root / "inputs/plan.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan",))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--owners", type=int, default=2)
    parser.add_argument("--corpus-root", type=Path, default=dataset.DEFAULT_CORPUS_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = plan(parse_args(argv))
    print(json.dumps({key: result[key] for key in ("state", "counts", "owners")}))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
