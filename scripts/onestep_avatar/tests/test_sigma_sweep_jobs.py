"""Historical generation conversion preserves literal conditions and saved noise without execution."""

import json
from pathlib import Path

import pytest
import torch

from scripts.onestep_avatar import dataset, evaluate, queue, sigma_sweep_jobs, subset
from scripts.onestep_avatar.hashing import sha256

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "expr/onestep_avatar/two_mode_restructure_20261005"
STUDY = ROOT / "expr/onestep_avatar/d1_selfrollout_sigma_sweep_20260926"


def test_actual_prepared_jobs_preserve_all_historical_inputs_and_schedules() -> None:  # noqa: PLR0915 -- full case inventory
    cases = json.loads((STUDY / "configs/generation_cases.json").read_text())["cases"]
    prepared = EVIDENCE / "sigma_sweep_generation_v4"
    provenance = json.loads((prepared / "preparation.json").read_text())
    assert provenance["schema_version"] == 2
    assert provenance["producer_sha256"] == sha256(Path(sigma_sweep_jobs.__file__))
    assert len(provenance["derived_file_hashes"]) == 13
    for name, digest in provenance["derived_file_hashes"].items():
        assert sha256(prepared / name) == digest
    for name, digest in provenance["input_file_hashes"].items():
        assert sha256(Path(name)) == digest
    jobs = queue.prepare_jobs(prepared / "jobs.json")
    assert len(jobs) == 36
    for case in cases:
        official = json.loads(Path(case["official_manifest"]).read_text())
        membership = json.loads((prepared / f"{case['tag']}_membership.json").read_text())
        subset.validate_membership(membership)
        source = membership["sources"][0]
        art = official["videos"][0]["artifacts"]
        assert source["capture_latent_sha256"] == art["capture_sha256"]
        assert source["guide_latent_sha256"] == art["guide_sha256"]
        assert membership["original_probe_file_sha256"] == case["official_sha256"]
        assert membership["group_rule"] == "historical_replay_only_no_training_split"
        dataset.ClipStore(membership).verify(require_guide=True)
        original = torch.load(Path(case["official_manifest"]).parent / art["epsilon"], weights_only=True)
        noise = torch.load(prepared / f"{case['tag']}_noise.pt", weights_only=True)
        assert torch.equal(noise, original[:, :17 * source["shape"][2] * source["shape"][3]])
        selected = [job for job in jobs if job["id"].startswith(f"sweep_{case['tag']}_")]
        assert len(selected) == 8
        decoder = next(job for job in jobs if job["id"] == f"decode_{case['tag']}")
        assert decoder["kind"] == "sigma_sweep"
        assert decoder["dependencies"] == [job["id"] for job in selected]
        spec = json.loads((prepared / f"{case['tag']}_decode.json").read_text())
        assert spec["schema_version"] == 2
        assert [cell["evaluation_job"]["id"] for cell in spec["cells"]] == decoder["dependencies"]
        assert all("path" not in cell and "sha256" not in cell for cell in spec["cells"])
        assert {(evaluate.parse_args(job["arguments"]).schedule[0],
                 evaluate.parse_args(job["arguments"]).guide_mode) for job in selected} == {
            (sigma, arm) for sigma in (0.421875, 0.725, 0.909375, 1.0) for arm in ("d0", "d1")}
        for job in selected:
            args = evaluate.parse_args(job["arguments"])
            assert args.mode == "causal"
            assert args.variant == "distilled"
            assert args.seed == official["seed"]
            assert args.prompt == official["text_context"]["prompt"]
            assert args.source == [source["relative_dir"]]
            assert args.history_mode == "cache"
            assert args.kv_source == "refresh"
            assert args.mode_settings.block_latent_frames == 2
            assert args.mode_settings.blocks_per_sample == 8
            assert args.mode_settings.context_latent_frames == 8
            assert not args.mode_settings.teacher_forcing
            assert not args.checkpoint
            assert not args.research_override
            assert args.cfg == 1
            assert args.stg == 0
            assert args.rescale == 0
            kind = "one_step" if args.schedule[0] == 0.421875 else "official"
            manifest = json.loads(Path(case[f"{kind}_manifest"]).read_text())
            row = next(row for row in manifest["videos"] if row["sigma"] == args.schedule[0])
            assert args.schedule == row["schedule"]
            assert job["completion"]["records"] == [str(args.output / "case_0000/variant_000/result.json")]


def test_existing_generation_output_is_refused_before_input_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fresh output"):
        sigma_sweep_jobs.prepare(tmp_path / "missing.json", tmp_path)


def test_changed_manifest_pin_is_refused_without_output(tmp_path: Path) -> None:
    cases = json.loads((STUDY / "configs/generation_cases.json").read_text())
    cases["cases"][0]["official_sha256"] = "0" * 64
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="manifest bytes changed"):
        sigma_sweep_jobs.prepare(path, output)
    assert not output.exists()


def test_changed_guidance_source_is_refused_without_output(tmp_path: Path) -> None:
    cases = json.loads((STUDY / "configs/generation_cases.json").read_text())
    cases["cases"][0]["legacy_guidance"]["source_sha256"] = "0" * 64
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="guidance source bytes changed"):
        sigma_sweep_jobs.prepare(path, output)
    assert not output.exists()


def test_extra_historical_cell_is_refused_even_with_updated_manifest_hash(tmp_path: Path) -> None:
    cases = json.loads((STUDY / "configs/generation_cases.json").read_text())
    original = json.loads(Path(cases["cases"][0]["official_manifest"]).read_text())
    original["videos"].append(original["videos"][0])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(original))
    cases["cases"][0]["official_manifest"] = str(manifest)
    cases["cases"][0]["official_sha256"] = sha256(manifest)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="exact schedule inventory differs"):
        sigma_sweep_jobs.prepare(path, output)
    assert not output.exists()


def test_late_input_change_prevents_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cases = json.loads((STUDY / "configs/generation_cases.json").read_text())
    cases["cases"] = cases["cases"][:1]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    save = torch.save

    def change_after_save(*args, **kwargs) -> None:
        save(*args, **kwargs)
        path.write_text(path.read_text() + "\n")

    monkeypatch.setattr(torch, "save", change_after_save)
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="inputs changed during publication"):
        sigma_sweep_jobs.prepare(path, output)
    assert output.exists()
    assert not (output / "preparation.json").exists()
    assert not (output / "jobs.json").exists()


def test_replay_membership_preserves_source_and_refuses_conflicting_rows(tmp_path: Path) -> None:
    path = Path(json.loads((STUDY / "configs/generation_cases.json").read_text())["cases"][0]["official_manifest"])
    membership = subset.from_saved_probe(path)
    assert membership["sources"][0]["actor"] == "8"
    assert membership["sources"][0]["relative_dir"] == "Part_1/0008_01/views/view00_cam51"
    assert membership["original_probe_file_sha256"] == sha256(path)
    record = json.loads(path.read_text())
    record["videos"][1]["artifacts"]["guide_sha256"] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="different input sources"):
        subset.from_saved_probe(changed)
