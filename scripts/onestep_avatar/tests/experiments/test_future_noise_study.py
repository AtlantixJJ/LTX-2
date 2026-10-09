"""Historical noise conversion preserves original block bytes and experiment controls."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.onestep_avatar import WORKSPACE_ROOT, evaluate, hashing
from scripts.onestep_avatar.experiments import future_noise_study as study
from scripts.onestep_avatar.hashing import sha256


@pytest.fixture
def saved() -> tuple[dict, dict, dict]:
    generator = torch.Generator().manual_seed(42)
    eps = [torch.randn(1, size, 128, generator=generator).bfloat16() for size in (6, 4, 4, 4)]
    original = torch.cat([eps[0][:, 2:], *eps[1:]], dim=1)
    noises = {"A": original}
    for name in ("B", "B2", "B3"):
        noises[name] = original.clone()
        noises[name][:, 4:] = torch.randn(1, 12, 128, generator=generator).bfloat16()
    manifest = {"latent_frames": 9, "tokens_per_latent_frame": 2,
                "source": "Part_1/0012_09/views/view01_cam52", "seed": 42,
                "geometry": {"block_latent_frames": 2, "context_latent_frames": 8},
                "schedule": [1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0],
                "noise": {name: hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
                          for name, value in noises.items()}}
    return manifest, noises, {"seed": 42, "epsilons": eps}


def test_global_noise_exactly_reproduces_all_four_saved_blocks(saved: tuple) -> None:
    manifest, noises, blocks = saved
    converted = study.full_noise(manifest, noises, blocks)
    assert torch.equal(converted["A"], torch.cat(blocks["epsilons"], dim=1))
    for name, value in converted.items():
        assert torch.equal(value[:, :2], blocks["epsilons"][0][:, :2])
        assert torch.equal(value[:, 2:], noises[name])
        assert torch.equal(value[:, :6], converted["A"][:, :6])


@pytest.mark.parametrize("defect", ["hash", "early", "no_change", "dtype", "shape", "seed", "keys"])
def test_invalid_noise_cannot_enter_package_jobs(saved: tuple, defect: str) -> None:
    manifest, noises, blocks = copy.deepcopy(saved)
    if defect == "seed":
        blocks["seed"] = 43
    elif defect == "keys":
        noises.pop("B3")
    elif defect == "dtype":
        noises["B"] = noises["B"].float()
    elif defect == "shape":
        noises["B"] = noises["B"][:, :-1]
    else:
        if defect == "no_change":
            noises["B"] = noises["A"].clone()
        else:
            noises["B"][:, 0] += 1
        if defect != "hash":
            manifest["noise"]["B"] = hashlib.sha256(
                noises["B"].view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
    with pytest.raises(ValueError, match=r"noise|intervention|historical study|nine frames"):
        study.full_noise(manifest, noises, blocks)


def test_job_data_preserves_repeat_modes_boundary_and_seven_results(saved: tuple, tmp_path: Path) -> None:
    manifest, _, _ = saved
    record, outputs = study.job_data(manifest, tmp_path)
    assert set(outputs) == {"J-A", "J-A-repeat", "J-B", "J-B2", "J-B3", "C-A", "C-B"}
    assert len(record["jobs"]) == 5
    for job in record["jobs"]:
        args = evaluate.parse_args(job["arguments"])
        assert args.source == [manifest["source"]]
        assert args.schedule == manifest["schedule"]
        assert args.seed == 42
        assert args.span_latent_frames == 9
        assert args.variant == "distilled"
        assert not args.checkpoint
        if args.changed_noise_file:
            assert args.future_noise_start == 3
            assert len(job["completion"]["records"]) == 2
        else:
            assert job["completion"]["records"] == [outputs["J-A-repeat"]]
        if args.mode == "causal":
            assert args.mode_settings.block_latent_frames == 2
            assert args.mode_settings.context_latent_frames == 8
            assert args.mode_settings.blocks_per_sample == 4


def test_existing_output_is_refused_before_reading_or_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fresh output"):
        study.prepare(*(tmp_path / "missing" for _ in range(4)), tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", [False, True])
def test_preparation_publishes_only_after_unchanged_input_checks(
    saved: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: bool,
) -> None:
    manifest, noises, blocks = saved
    manifest_path, noise_path, block_path, subset_path = [tmp_path / name for name in
                                                        ("manifest.json", "noise.pt", "blocks.pt", "subset.json")]
    manifest_path.write_text(json.dumps(manifest))
    torch.save(noises, noise_path)
    torch.save(blocks, block_path)
    subset_path.write_text("{}")
    membership = {"objective": "bg", "sources": [{"relative_dir": manifest["source"]}]}

    def converted(*_args: object, **_kwargs: object) -> tuple[dict, dict]:
        if change:
            subset_path.write_text('{"changed": true}')
        return membership, {}

    monkeypatch.setattr(study.subset, "convert_legacy", converted)
    output = tmp_path / "converted"
    if change:
        with pytest.raises(ValueError, match="inputs changed"):
            study.prepare(manifest_path, noise_path, block_path, subset_path, output)
        assert not output.exists()
    else:
        record = study.prepare(manifest_path, noise_path, block_path, subset_path, output)
        assert len(record["outputs"]) == 7
        jobs = json.loads((output / "jobs.json").read_text())
        assert len(jobs["jobs"]) == 5
        assert torch.equal(torch.load(output / "noise_A.pt", weights_only=True),
                           torch.cat(blocks["epsilons"], dim=1))


def test_real_historical_noise_preserves_original_block_bytes() -> None:
    root = WORKSPACE_ROOT / "expr/onestep_avatar"
    source = root / "joint_future_noise_influence_20260926"
    prefix = root / "d1_diagnostic/ar_sigma_rollouts/artifacts/confirmations/base_distill_noise_prefixes_20260926"
    manifest = json.loads((source / "manifest.json").read_text())
    noises = torch.load(source / "noise_AB.pt", map_location="cpu", weights_only=True)
    blocks = torch.load(prefix / "block_epsilons.pt", map_location="cpu", weights_only=True)
    converted = study.full_noise(manifest, noises, blocks)
    assert torch.equal(converted["A"], torch.cat(blocks["epsilons"][:4], dim=1))


@pytest.fixture
def preparation_inputs(saved: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[Path], Path]:
    manifest, noises, blocks = saved
    paths = [tmp_path / name for name in ("manifest.json", "noise.pt", "blocks.pt", "subset.json")]
    paths[0].write_text(json.dumps(manifest))
    torch.save(noises, paths[1])
    torch.save(blocks, paths[2])
    paths[3].write_text("{}")
    monkeypatch.setattr(study.subset, "convert_legacy", lambda *_a, **_k: (
        {"objective": "bg", "sources": [{"relative_dir": manifest["source"]}]}, {"original_plan": True}))
    return paths, tmp_path / "preparation"


def test_late_input_change_leaves_partial_files_without_conversion_record(
    preparation_inputs: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, output = preparation_inputs
    original_save = torch.save

    def interrupted_save(value: object, destination: Path) -> None:
        original_save(value, destination)
        paths[0].write_text('{"changed_during_publication": true}')

    monkeypatch.setattr(study.torch, "save", interrupted_save)
    with pytest.raises(ValueError, match="during derived-file writes"):
        study.prepare(*paths, output)
    assert (output / "noise_A.pt").is_file()
    assert not (output / "conversion.json").exists()


def test_current_preparation_verifies_without_execution(preparation_inputs: tuple) -> None:
    paths, output = preparation_inputs
    result = study.prepare(*paths, output)
    assert result["schema_version"] == 2
    assert len(result["artifact_file_hashes"]) == 7
    assert study.verify_preparation(output) == result
    before = {path: sha256(path) for path in output.iterdir()}
    study.main(["--verify-preparation", str(output)])
    assert {path: sha256(path) for path in output.iterdir()} == before


@pytest.mark.parametrize("arguments", [[], ["--verify-preparation", "unused", "--output", "unused"]])
def test_cli_refuses_incomplete_or_mixed_operations(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        study.main(arguments)


@pytest.mark.parametrize("defect", ["input", "artifact", "inventory", "producer", "roles",
                                    "jobs_rehashed", "noise_rehashed", "membership_rehashed", "plan_rehashed"])
def test_preparation_verification_refuses_changed_bytes_and_semantics(
    preparation_inputs: tuple, defect: str,
) -> None:
    paths, output = preparation_inputs
    result = study.prepare(*paths, output)
    if defect == "input":
        paths[3].write_text('{"changed": true}')
    elif defect == "artifact":
        (output / "noise_A.pt").write_bytes(b"changed serialization")
    elif defect == "inventory":
        result["artifact_file_hashes"].pop("noise_A.pt")
    elif defect == "producer":
        result["producer_source_sha256"] = "0" * 64
    elif defect == "roles":
        result["outputs"]["J-A"] = result["outputs"]["C-A"]
    else:
        name = {"jobs_rehashed": "jobs.json", "noise_rehashed": "noise_A.pt",
                "membership_rehashed": "membership.json", "plan_rehashed": "frame_plan.json"}[defect]
        artifact = output / name
        if defect == "noise_rehashed":
            tensor = torch.load(artifact, weights_only=True) + 1
            torch.save(tensor, artifact)
            result["noise_tensor_sha256"]["A"] = hashing.tensor_sha256(tensor)
        else:
            data = json.loads(artifact.read_text())
            if defect == "jobs_rehashed":
                args = data["jobs"][0]["arguments"]
                args[args.index("--seed") + 1] = "43"
            elif defect == "membership_rehashed":
                data["objective"] = "white"
            else:
                data["original_plan"] = False
            artifact.write_text(json.dumps(data))
        result["artifact_file_hashes"][name] = sha256(artifact)
    (output / "conversion.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match=r"future-noise preparation"):
        study.verify_preparation(output)
