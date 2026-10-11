"""E3 preparation keeps cohort, full-length noise and verified repeats explicit."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.onestep_avatar.corpus import subset
from scripts.onestep_avatar.experiments import prepare_continuation as prepare


def membership() -> dict:
    sources = [
        {"relative_dir": f"Part_1/{actor}/views/view00", "actor": actor, "split": "held_out"}
        for actor in ("8", "12", "25")
    ]
    record = {
        "schema_version": 2,
        "kind": subset.KIND,
        "objective": "white",
        "corpus_root": "/checked/corpus",
        "sources": sources,
        "splits": {"held_out": ["8", "12", "25"]},
    }
    record["sha256"] = subset.membership_hash(record)
    return record


def test_held_out_cohort_requires_three_people_and_both_seeds() -> None:
    record = membership()
    ids = [row["relative_dir"] for row in record["sources"]]
    assert len(prepare.cohort_sources(record, ids, [42, 43], "held_out")) == 3
    for selected, seeds in ((ids[:2], [42, 43]), (ids, [42]), (ids, [42, 42]), (ids + ids[:1], [42, 43])):
        with pytest.raises(ValueError, match=r"E3 requires|E3 seeds|E3 requires white"):
            prepare.cohort_sources(record, selected, seeds, "held_out")
    changed = copy.deepcopy(record)
    changed["sources"][0]["split"] = "train"
    changed["splits"] = {"train": ["8"], "held_out": ["12", "25"]}
    changed["sha256"] = subset.membership_hash(changed)
    with pytest.raises(ValueError, match="three people"):
        prepare.cohort_sources(changed, ids, [42, 43], "held_out")
    assert prepare.cohort_sources(changed, ids[:1], [42], "first_view")[0]["actor"] == "8"


def test_seeded_noise_is_repeatable_and_changes_only_the_declared_future() -> None:
    values = prepare.fixed_noise(42, tokens_per_frame=4, channels=2)
    other = prepare.fixed_noise(43, tokens_per_frame=4, channels=2)
    assert torch.equal(values["original"], prepare.fixed_noise(42, tokens_per_frame=4, channels=2)["original"])
    assert not torch.equal(values["original"], other["original"])
    assert values["original"].shape == (1, 68, 2)
    assert values["original"].dtype == torch.bfloat16
    for boundary in (3, 11):
        changed, original = values[f"changed_{boundary}"], values["original"]
        assert torch.equal(changed[:, : 4 * boundary], original[:, : 4 * boundary])
        assert torch.equal(changed[:, 4 * boundary :], -original[:, 4 * boundary :])
        assert not torch.equal(changed, original)
    assert not torch.cuda.is_initialized()


def case_and_jobs(tmp_path: Path) -> dict:
    case = {
        "id": "actor_8_seed_42",
        "source": "Part_1/8/views/view00",
        "split": "held_out",
        "seed": 42,
        "prompt": "same saved prompt",
        "noise": {"original": {"path": str(tmp_path / "noise.pt"), "tensor_sha256": "noise"}},
        "capture_sha256": "capture",
        "guide_sha256": "guide",
        "c0_sha256": "c0",
        "input_file_hashes": {"capture": "capture-file", "guide": "guide-file"},
    }
    budget = {"path": str(tmp_path / "budget.json"), "sha256": "budget"}
    case["baseline_jobs"] = [
        prepare.baseline_job(case, tmp_path / "membership.json", sigma, role, tmp_path, budget)
        for sigma in prepare.SIGMAS
        for role in prepare.ROLES
    ]
    return case


def test_baseline_jobs_hold_every_condition_and_never_choose_devices(tmp_path: Path) -> None:
    case = case_and_jobs(tmp_path)
    assert len({job["id"] for job in case["baseline_jobs"]}) == 6
    for job in case["baseline_jobs"]:
        arguments = job["arguments"]
        assert not {"--gpu-id", "--dry-run", "--checkpoint", "--teacher-forcing"}.intersection(arguments)
        args = prepare.evaluate.parse_args(arguments)
        assert args.mode == "causal"
        assert args.variant == "dev"
        assert args.guide_mode == "d1"
        assert args.span_latent_frames == 17
        assert args.mode_settings.block_latent_frames == 2
        assert args.mode_settings.context_latent_frames == 8
        assert not args.mode_settings.teacher_forcing
        assert args.schedule in ([1.0, 0.0], [0.909375, 0.0])
        assert args.seed == 42
        assert args.split == "held_out"
        assert args.history_mode == ("recompute" if job["id"].endswith("recompute") else "cache")
        assert args.kv_source == "refresh"
        assert args.prompt == "same saved prompt"
        assert job["numerical_policy"] == "deterministic"


def saved_baselines(case: dict, *, defect: str | None = None) -> None:
    for job in case["baseline_jobs"]:
        path = Path(job["completion"]["records"][0])
        path.parent.mkdir(parents=True)
        latent = torch.arange(17, dtype=torch.bfloat16).reshape(1, 1, 17, 1, 1)
        if defect == "repeat" and job["id"].endswith("cached_repeat"):
            latent[:, :, 3] += 1
        generated = path.parent / "generated.pt"
        torch.save(latent, generated)
        record = {
            key: case[key] for key in ("source", "input_file_hashes", "capture_sha256", "guide_sha256", "c0_sha256")
        }
        record.update(noise_sha256="noise", text_sha256="text", output={"path": str(generated)})
        if defect == "text" and job["id"].endswith("recompute"):
            record["text_sha256"] = "changed"
        if defect == "c0" and job["id"].endswith("cached"):
            record["c0_sha256"] = "changed"
        path.write_text(json.dumps(record))


def test_sealing_requires_public_verification_and_actual_exact_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = case_and_jobs(tmp_path)
    saved_baselines(case)
    calls = []
    monkeypatch.setattr(
        prepare.evaluate, "verify_evaluation_conditions", lambda arguments, paths: calls.append((arguments, paths))
    )
    result = prepare.verify_case(case)
    assert len(calls) == len(result) == 6
    assert {(row["sigma"], row["role"]) for row in result} == {
        (sigma, role) for sigma in prepare.SIGMAS for role in prepare.ROLES
    }
    assert all(row["result"]["sha256"] for row in result)


@pytest.mark.parametrize("defect", ["repeat", "text", "c0"])
def test_changed_repeat_or_fixed_input_prevents_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    case = case_and_jobs(tmp_path)
    saved_baselines(case, defect=defect)
    monkeypatch.setattr(prepare.evaluate, "verify_evaluation_conditions", lambda *_args: None)
    with pytest.raises(ValueError, match=r"bit-identical|text differs|fixed inputs"):
        prepare.verify_case(case)


def test_missing_or_failed_public_baseline_is_never_repaired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = case_and_jobs(tmp_path)

    def reject(*_args: object) -> None:
        raise ValueError("public scientific baseline verification failed")

    monkeypatch.setattr(prepare.evaluate, "verify_evaluation_conditions", reject)
    with pytest.raises(ValueError, match="public scientific"):
        prepare.verify_case(case)
    assert list(tmp_path.iterdir()) == []
