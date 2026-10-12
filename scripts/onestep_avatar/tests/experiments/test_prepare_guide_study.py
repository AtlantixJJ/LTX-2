"""Guide-study inventory and exact CLI contracts."""

from collections import Counter
from pathlib import Path

import pytest
import torch

from scripts.onestep_avatar import comparisons
from scripts.onestep_avatar.experiments import prepare_guide_study as study
from scripts.onestep_avatar.model import sampling


@pytest.fixture
def prompts() -> dict:
    return {
        (a, f"P{i}"): {"text": f"{a}_P{2 if a == '0012_09' and i == 3 else i}" if i else "generic"}
        for a in study.ACTORS
        for i in range(5)
    }


def test_inventory_and_schedules(prompts: dict) -> None:
    cases = study.inventory(prompts)
    assert Counter(c["phase"] for c in cases) == {
        "s1": 142,
        "s3_distilled": 48,
        "s2_full": 63,
        "s3_dev": 36,
        "s2_steps": 270,
        "s2_bench": 5,
    }
    assert sum(c["sigma"] == 1 and c["arm"] == "d0" and c["variant"] == "dev" for c in cases) == 33
    for case in cases:
        if case["variant"] == "dev":
            assert case["schedule"] == list(
                sampling.thinned_truncated_schedule(case["sigma"], 30, len(case["schedule"]) - 1)
            )
    shared = [c for c in cases if c["actor"] == "0012_09" and any(r["prompt"] == "P3" for r in c["roles"])]
    assert shared
    assert all(any(r["prompt"] == "P2" for r in c["roles"]) for c in shared)


def test_arguments_panels_balance(prompts: dict, tmp_path: Path) -> None:
    cases = study.inventory(prompts)
    for case in cases:
        args = study.evaluation_job(case, tmp_path)["arguments"]
        assert "--checkpoint" not in args
        assert "--gpu-id" not in args
        assert args[args.index("--prompt") + 1] == case["prompt"]
        if case["history"]:
            assert args[args.index("--output-latent-frames") + 1] == "17"
            assert ("--teacher-forcing" in args) == (case["history"] == "gt")
        assert args[args.index("--cfg") + 1] == ("3" if case["variant"] == "dev" else "1")
    for phase in study.PHASES:
        selected = [c for c in cases if c["phase"] == phase]
        bins = study.balanced(selected, 2)
        assert {c["id"] for b in bins for c in b} == {c["id"] for c in selected}
        assert abs(len(bins[0]) - len(bins[1])) <= 2
    for spec, job in study.comparison_specs(cases, tmp_path, Path("/corpus")):
        for comparison in spec["comparisons"]:
            assert comparison["view"]
            assert comparison["panels"][0]["latent"].startswith("capture:")
            assert len(comparison["panels"]) <= 6
        comparisons.parse_saved_comparison_args(job["arguments"])
        assert job["resource_budget"].endswith("inputs/render_budget.json")
        assert job["dependencies"]


def test_existing_preparation_refused(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir()
    args = study.parse_args(["plan", "--root", str(tmp_path), "--prompts", "/missing", "--dry-run"])
    with pytest.raises(ValueError, match="fresh"):
        study.plan(args)


def test_causal_noise_is_exact_slice() -> None:
    values = study.fixed_noise(42, tokens_per_frame=4, channels=2)
    assert values[18].dtype == torch.bfloat16
    assert values[17].shape == (1, 68, 2)
    assert torch.equal(values[18][:, :68], values[17])
    assert torch.equal(values[18], study.fixed_noise(42, tokens_per_frame=4, channels=2)[18])
