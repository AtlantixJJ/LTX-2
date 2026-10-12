"""Select changed-file CPU gates; see doc/tests/select.md for the contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.onestep_avatar import LTX_ROOT, PACKAGE_ROOT
from scripts.onestep_avatar.tests.support.imports import targets

PREFIX = "scripts/onestep_avatar/"
TESTS = PREFIX + "tests/"
BOUNDARY = {
    "test_experiment_import_boundary.py",
    "test_profile_coverage.py",
    "test_execution_ownership.py",
    "test_corpus_model_boundaries.py",
    "test_roots.py",
    "test_output_owner_split.py",
    "test_software.py",
    "experiments/test_checker_entrypoints.py",
}
AREAS = {
    "model": {
        "causal_core",
        "causal_train_sample",
        "bidirectional",
        "sampling",
        "prompt_and_whole_clip",
        "adapters",
        "adapter_only_export",
        "consumer_trace",
        "update_state",
        "numerics",
        "applied_runtime",
    },
    "corpus": {
        "dataset",
        "geometry",
        "mask_video",
        "motion",
        "qa",
        "hashing",
        "build_guidance",
        "precompute",
        "subset",
    },
    "training": {
        "training_config",
        "training_helpers",
        "training_preflight",
        "training_runtime",
        "training_completion",
        "training_launch_binding",
        "checkpoint_contract",
        "resources",
        "plot_training",
        "required_inputs",
        "numerics",
        "applied_runtime",
        "consumer_trace",
        "update_state",
    },
    "support": {
        "evaluate",
        "evaluate_coverage",
        "evaluation_completion",
        "infer",
        "product_coverage",
        "prepare_inputs",
        "prepare_preview",
        "previews",
        "lpips_frame_scores",
        "rgb_measurements",
        "ordinary_queue_completion",
        "bench",
    },
    "media": {"media", "saved_comparisons", "saved_comparison_references", "checkpoint_visualization", "decode_saved"},
}
ROOT_AREAS = {
    "train.py": "training",
    "plot_training.py": "training",
    **dict.fromkeys(
        ("evaluate.py", "infer.py", "previews.py", "prepare_inputs.py", "bench.py", "metrics.py"), "support"
    ),
    **dict.fromkeys(("media.py", "comparisons.py", "decode_saved.py"), "media"),
}
DISPATCH = {"execution/jobs.py", "execution/completion.py", "execution/software.py"}
PRUNE = {"corpus/dataset.py", "corpus/subset.py", "media.py"}


def select(changed: list[str], tier: str, tests: dict[str, str], *, imports_changed: bool = False) -> dict:  # noqa: PLR0912 -- the explicit area and escalation policy
    """Return deterministic paths and escalation without executing production imports."""
    selected: set[str] = set()
    boundary = imports_changed
    prune = False
    for path in changed:
        if not path.startswith(PREFIX):
            if path.startswith(("scripts/prune/", "packages/")) and path.endswith(".py"):
                tier = "T2"
            continue
        relative = path[len(PREFIX) :]
        if not relative.endswith(".py"):
            continue
        boundary |= relative in DISPATCH or relative.startswith("tests/blocked_imports/")
        if (
            relative in {"hashing.py", "execution/software.py", "tests/conftest.py"}
            or relative.endswith("__init__.py")
            or relative.startswith("tests/support/")
        ):
            tier = "T2"
        if relative.startswith("tests/"):
            if path in tests:
                selected.add(path)
            continue
        area = ROOT_AREAS.get(relative, relative.split("/")[0])
        if area in AREAS:
            selected.update(TESTS + "test_" + name + ".py" for name in AREAS[area])
        elif area == "execution":
            selected.update(
                name
                for name in tests
                if Path(name).name.startswith(
                    ("test_queue", "test_guarded", "test_process_registry", "test_supervision")
                )
            )
        elif area == "experiments":
            selected.add(TESTS + "experiments/test_" + Path(relative).stem + ".py")
        else:
            tier = "T2"
        owner = path[:-3].replace("/", ".")
        for name, source in tests.items():
            imported = targets(source, name[:-3].replace("/", "."))
            if any(target == owner or target.startswith(owner + ".") for target in imported):
                selected.add(name)
        prune |= area == "model" or relative in PRUNE
    if tier == "T2":
        selected = set(tests)
        prune = True
    elif tier == "T3":
        selected = {TESTS + name for name in BOUNDARY if TESTS + name in tests and not name.startswith("experiments/")}
        boundary = True
    return {
        "tier": tier,
        "files": sorted(selected.intersection(tests)),
        "prune": prune and tier in {"T1", "T2"},
        "boundary": boundary,
        "lint_files": sorted(path for path in changed if path.endswith(".py") and (LTX_ROOT / path).is_file())
        if tier == "T2"
        else [],
    }


def commands(plan: dict) -> list[tuple[list[str], dict[str, str]]]:
    rows = []
    base = [sys.executable, "-m", "pytest"]
    if plan["files"] and plan["tier"] != "T3":
        rows.append((base + plan["files"] + ["-q"] + (["-m", "not subprocess"] if plan["tier"] == "T0" else []), {}))
    if plan["prune"]:
        rows.append(([*base, "scripts/prune/tests", "-q"], {"CUDA_VISIBLE_DEVICES": ""}))
    if plan.get("lint_files"):
        rows.append((["uv", "run", "ruff", "check", *plan["lint_files"]], {}))
    if plan["boundary"]:
        rows.append(([*base, TESTS.rstrip("/"), "-q", "-m", "boundary"], {"ONESTEP_AVATAR_BLOCK_EXPERIMENTS": "1"}))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD")
    parser.add_argument("--tier", choices=["T0", "T1", "T2", "T3"], default="T0")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=LTX_ROOT, text=True)

    changed = git("diff", "--name-only", args.base).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard", PREFIX).splitlines()
    diff = git("diff", "--unified=0", args.base, "--", PREFIX)
    imports_changed = any(
        line[:1] in {"+", "-"} and line[1:].lstrip().startswith(("import ", "from "))
        for line in diff.splitlines()
        if not line.startswith(("+++", "---"))
    )
    imports_changed |= any(
        targets((LTX_ROOT / path).read_text(), path[:-3].replace("/", "."))
        for path in untracked
        if path.endswith(".py")
    )
    tests = {str(path.relative_to(LTX_ROOT)): path.read_text() for path in (PACKAGE_ROOT / "tests").rglob("test_*.py")}
    plan = select(changed + untracked, args.tier, tests, imports_changed=imports_changed)
    rows = commands(plan)
    print(  # noqa: T201 -- selection is a reviewable CLI output
        json.dumps({**plan, "commands": [{"argv": command, "environment": env} for command, env in rows]}, indent=2),
        flush=True,
    )
    if args.run:
        for command, env in rows:
            status = subprocess.call(command, cwd=LTX_ROOT, env={**os.environ, **env})
            if status:
                return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
