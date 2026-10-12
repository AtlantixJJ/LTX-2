"""Scientific queue completion uses real saved masters/results and controlled CPU execution."""

import json
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.onestep_avatar import evaluate, hashing
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.execution import completion as queue_completion
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.support.contracts import A, B, _contract
from scripts.onestep_avatar.tests.support.fixtures_evaluation_completion import (
    completed as completed,  # noqa: PLC0414 -- explicit pytest fixture re-export
)
from scripts.onestep_avatar.tests.support.fixtures_evaluation_completion import controlled_scientific_receipt
from scripts.onestep_avatar.tests.support.tree import restore, snapshot
from scripts.onestep_avatar.training import checkpoints
from scripts.onestep_avatar.training.checkpoints import CONTRACT_KEY
from scripts.prune.core import preflight
from scripts.prune.data import prompt_cache


def test_completed_science_and_normal_fresh_output_gate(completed: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    execute, arguments, _, _, _ = completed
    job, paths = execute()
    assert len(paths) == 2
    with pytest.raises(ValueError, match="already used"):
        evaluate.prepare_evaluation(evaluate.parse_args(arguments))
    assert queue_completion.verify_completion(job)
    receipt = controlled_scientific_receipt(job, monkeypatch)
    assert len(receipt["evidence"]) == 7
    assert queue_completion.verify_receipt(job, receipt)


def test_saved_scalar_text_refuses_before_native_session(completed: tuple) -> None:
    execute, _, settings, _, calls = completed
    job, _ = execute()
    torch.save(torch.tensor(1.0, dtype=torch.bfloat16), settings.output / "text.pt")
    before = list(calls)
    with pytest.raises(ValueError, match="batch-one context"):
        queue_completion.verify_completion(job)
    assert calls == before


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--split", "train", "--source", "unknown/video"], "unknown videos"),
        (["--split", "validation"], "selection is empty"),
    ],
)
def test_split_selection_rejects_invalid_inventory_before_sessions(
    completed: tuple, extra: list[str], message: str
) -> None:
    _, arguments, _, _, calls = completed
    with pytest.raises(ValueError, match=message):
        evaluate.prepare_evaluation(evaluate.parse_args(arguments + extra))
    assert not calls


def test_split_selection_limits_results_and_rejects_cross_split_source(completed: tuple) -> None:
    execute, arguments, _, membership, calls = completed
    held_out = next(source["relative_dir"] for source in membership["sources"] if source["split"] == "held_out")
    with pytest.raises(ValueError, match="differs from the requested fixed-video split"):
        evaluate.prepare_evaluation(evaluate.parse_args([*arguments, "--split", "train", "--source", held_out]))
    assert not calls
    job, paths = execute(["--split", "held_out"])
    assert len(paths) == 1
    assert json.loads(paths[0].read_text())["source"] == held_out
    assert queue_completion.verify_completion(job)
    job["arguments"][job["arguments"].index("--split") + 1] = "train"
    with pytest.raises(ValueError, match="scientific settings or input evidence"):
        queue_completion.verify_completion(job)


@pytest.mark.parametrize(("mode", "expected_calls"), [("bidirectional", 1), ("causal", 6)])
def test_both_modes_publish_verifiable_real_small_transformer_outputs(
    completed: tuple, mode: str, expected_calls: int
) -> None:
    execute, _, _, _, _ = completed
    job, paths = execute(mode=mode)
    assert queue_completion.verify_completion(job)
    assert all(json.loads(path.read_text())["call_counts"]["model_calls"] == expected_calls for path in paths)
    if mode == "causal":
        job["arguments"].append("--teacher-forcing")
        with pytest.raises(ValueError, match="scientific settings or input evidence"):
            queue_completion.verify_completion(job)


def test_causal_physical_prefix_roundtrip_preserves_null_span_c0_calls_and_saved_completion(completed: tuple) -> None:
    execute, _arguments, settings, membership, calls = completed
    # Extend checked continuous masters; the selected generation still covers only [0,7).
    for source in membership["sources"]:
        path = Path(membership["corpus_root"]) / source["relative_dir"] / dataset.capture_bundle_name("white")
        bundle = torch.load(path, weights_only=True)
        bundle["master"] = torch.arange(144).reshape(2, 18, 2, 2).float()
        torch.save(bundle, path)
        source.update(n_latent_frames=18, shape=[2, 18, 2, 2], capture_latent_sha256=sha256(path))
    membership["sha256"] = subset.membership_hash(membership)
    settings.subset.write_text(json.dumps(membership))
    job, paths = execute(["--output-latent-frames", "7"], mode="causal")
    for path in paths:
        record = json.loads(path.read_text())
        assert record["frames"] == 7
        assert record["mode_settings"]["span_latent_frames"] is None
        assert record["conditions"]["mode_settings"]["span_latent_frames"] is None
        assert record["conditions"]["shape"]["frames"] == 7
        assert record["call_counts"]["model_calls"] == 6
        generated = torch.load(path.parent / "generated.pt", weights_only=True)
        assert generated.shape == (1, 2, 7, 2, 2)
        assert hashing.tensor_sha256(generated[:, :, :1].permute(0, 2, 3, 4, 1).reshape(1, 4, 2)) == record["c0_sha256"]
    before = list(calls)
    assert queue_completion.verify_completion(job)
    assert calls == before
    job["arguments"][job["arguments"].index("--output-latent-frames") + 1] = "9"
    with pytest.raises(ValueError, match="saved noise differs from requested input"):
        queue_completion.verify_completion(job)
    assert calls == before


def test_result_cannot_claim_other_scientific_conditions(subtests: object, tmp_path: object, completed: tuple) -> None:
    execute, _, _, _, _ = completed
    job, paths = execute()
    saved_job = json.dumps(job)
    pristine = snapshot(tmp_path)
    for case in [
        ("seed", 43),
        ("schedule", [0.5, 0]),
        ("source", "other/video"),
        ("fps", 24),
        ("frames", 5),
        ("mode_settings", {"span_latent_frames": 5}),
        ("conditions", {}),
        ("input_file_hashes", {}),
        ("membership_sha256", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        ("adapter", "/other/adapter"),
        ("application_method", "unmerged"),
        ("prompt", "other prompt"),
        ("negative_prompt", "other negative"),
        ("guidance", {}),
        ("text_sha256", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        ("capture_sha256", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        ("guide_sha256", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        ("c0_sha256", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        ("noise_sha256", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        ("producer_source_sha256", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
    ]:
        with subtests.test(case=case):
            restore(tmp_path, pristine)
            field, value = case
            job = json.loads(saved_job)
            record = json.loads(paths[0].read_text())
            record[field] = value
            paths[0].write_text(json.dumps(record))
            with pytest.raises(ValueError, match="scientific settings or input evidence"):
                queue_completion.verify_completion(job)

    restore(tmp_path, pristine)


def test_result_cannot_complete_different_requested_job(subtests: object, tmp_path: object, completed: tuple) -> None:
    execute, _, _, _, _ = completed
    job, paths = execute()  # noqa: RUF059 -- pytest hook or fixture interface
    saved_job = json.dumps(job)
    pristine = snapshot(tmp_path)
    for case in [("--seed", "43"), ("--guide-mode", "d1"), ("--variant", "distilled")]:
        with subtests.test(case=case):
            restore(tmp_path, pristine)
            option, value = case
            job = json.loads(saved_job)
            job["arguments"][job["arguments"].index(option) + 1] = value
            with pytest.raises((ValueError, FileNotFoundError)):
                queue_completion.verify_completion(job)

    restore(tmp_path, pristine)


def test_exact_case_inventory_is_required(subtests: object, tmp_path: object, completed: tuple) -> None:
    execute, _, _, _, _ = completed
    job, paths = execute()
    saved_job = json.dumps(job)
    pristine = snapshot(tmp_path)
    for case in ["missing", "duplicate", "extra", "wrong_path"]:
        with subtests.test(case=case):
            restore(tmp_path, pristine)
            defect = case
            job = json.loads(saved_job)
            if defect == "missing":
                job["completion"]["records"].pop()
            elif defect == "duplicate":
                job["completion"]["records"].append(str(paths[0]))
            else:
                alias = paths[0].parent / "unrequested.json"
                alias.write_bytes(paths[0].read_bytes())
                if defect == "extra":
                    job["completion"]["records"].append(str(alias))
                else:
                    job["completion"]["records"][0] = str(alias)
            with pytest.raises(ValueError, match="inventory differs"):
                queue_completion.verify_completion(job)

    restore(tmp_path, pristine)


def test_saved_evidence_changes_fail_even_with_updated_output_hash(
    subtests: object, tmp_path: object, completed: tuple
) -> None:
    execute, _, _, _, _ = completed
    job, paths = execute()
    saved_job = json.dumps(job)
    pristine = snapshot(tmp_path)
    for case in ["text", "noise", "dtype", "first_frame", "shape", "base"]:
        with subtests.test(case=case):
            restore(tmp_path, pristine)
            asset = case
            execute, _, settings, _, _ = completed
            job = json.loads(saved_job)
            record = json.loads(paths[0].read_text())
            if asset == "base":
                evaluate.backbone.resolve("2.5", "dev").paths.transformer().write_bytes(b"changed base weights")
            elif asset in ("text", "noise"):
                path = settings.output / "text.pt" if asset == "text" else settings.output / "case_0000/noise.pt"
                value = torch.load(path, weights_only=True)
                torch.save(value + 1, path)
            else:
                path = Path(record["output"]["path"])
                value = torch.load(path, weights_only=True)
                if asset == "dtype":
                    value = value.float()
                if asset == "first_frame":
                    value[:, :, :1] += 1
                if asset == "shape":
                    value = value[:, :, :5].contiguous()
                torch.save(value, path)
                record["output"].update(sha256=sha256(path), shape=list(value.shape))
                paths[0].write_text(json.dumps(record))
            with pytest.raises(ValueError):  # noqa: PT011 -- mutation controls check refusal type and preserved bytes across different validators
                queue_completion.verify_completion(job)

    restore(tmp_path, pristine)


def test_guidance_saves_actual_negative_text_and_binds_prompt(completed: tuple) -> None:
    execute, _, settings, _, _ = completed
    job, _ = execute(["--cfg", "3", "--negative-prompt", "fixed negative text"])
    assert (settings.output / "negative_text.pt").is_file()
    job["arguments"][-1] = "different negative text"
    with pytest.raises(ValueError, match="scientific settings or input evidence"):
        queue_completion.verify_completion(job)


def test_checked_adapter_bytes_path_and_complete_variant_inventory(
    subtests: object, tmp_path: object, completed: tuple
) -> None:

    pristine = snapshot(tmp_path)
    for case in ["bytes", "path", "missing_variant"]:
        with subtests.test(case=case):
            restore(tmp_path, pristine)
            defect = case
            execute, _, settings, _, _ = completed
            contract = _contract()
            contract["model"]["base_sha256"] = sha256(evaluate.backbone.resolve("2.5", "dev").paths.transformer())
            contract["shape"]["channels"] = 2
            contract["task"]["guide_mode"] = "d0"
            contract["training"]["sigma_levels"] = [0.725]
            contract["training"]["schedules"] = [[0.725, 0]]
            checkpoints.validate_contract(contract)
            adapter = settings.output.parent / "adapter.safetensors"
            metadata = {checkpoints.CONTRACT_KEY: json.dumps(contract)}
            save_file(
                {A: torch.ones(2, 4, dtype=torch.bfloat16), B: torch.zeros(4, 2, dtype=torch.bfloat16)},
                adapter,
                metadata=metadata,
            )
            job, paths = execute(["--span-latent-frames", "7", "--checkpoint", str(adapter), "--include-base"])
            assert len(paths) == 4
            if defect == "bytes":
                save_file(
                    {A: torch.ones(2, 4, dtype=torch.bfloat16), B: torch.ones(4, 2, dtype=torch.bfloat16)},
                    adapter,
                    metadata=metadata,
                )
            elif defect == "path":
                other = adapter.with_name("other.safetensors")
                other.write_bytes(adapter.read_bytes())
                job["arguments"][job["arguments"].index("--checkpoint") + 1] = str(other)
            else:
                job["completion"]["records"].pop()
            with pytest.raises(ValueError, match=r"scientific settings or input evidence|inventory differs"):
                queue_completion.verify_completion(job)

    restore(tmp_path, pristine)


@pytest.mark.parametrize("asset", ["text", "noise"])
def test_receipts_pin_input_file_bytes_even_when_tensor_content_is_identical(
    completed: tuple, asset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    execute, _, settings, _, _ = completed
    job, _ = execute()
    receipt = controlled_scientific_receipt(job, monkeypatch)
    path = settings.output / "text.pt" if asset == "text" else settings.output / "case_0000/noise.pt"
    before = sha256(path)
    value = torch.load(path, weights_only=True)
    torch.save(value, path, _use_new_zipfile_serialization=False)
    assert sha256(path) != before
    assert queue_completion.verify_completion(job), "scientific tensor values changed"
    with pytest.raises(ValueError, match="receipt evidence changed"):
        queue_completion.verify_receipt(job, receipt)


@pytest.mark.parametrize("changed", [None, "guide", "sidecar"])
def test_d1_checks_paired_guide_provenance_and_capture_first_frame(completed: tuple, changed: str | None) -> None:
    execute, _, settings, membership, _ = completed
    for source in membership["sources"]:
        view = Path(membership["corpus_root"]) / source["relative_dir"]
        render = view / dataset.render_name("white")
        render.write_bytes(b"controlled guide render bytes")
        sidecar = view / dataset.render_metadata_name("white")
        sidecar.write_text(json.dumps({"objective": "white", "compositing_version": dataset.GUIDE_COMPOSITING_VERSION}))
        capture = torch.load(view / dataset.capture_bundle_name("white"), weights_only=True)
        encoding = {**source["capture_encode_record"], "input_fingerprint": sha256(render)}
        guide_path = view / dataset.guide_bundle_name("white")
        torch.save({**capture, **encoding, "master": capture["master"] + 1}, guide_path)
        source.update(
            guide_latent_sha256=sha256(guide_path), guide_encode_record=encoding, guide_sidecar_sha256=sha256(sidecar)
        )
    membership["sha256"] = subset.membership_hash(membership)
    settings.subset.write_text(json.dumps(membership))
    job, paths = execute(guide_mode="d1")
    assert all(json.loads(path.read_text())["guide_sha256"] for path in paths)
    if changed:
        source = membership["sources"][0]
        view = Path(membership["corpus_root"]) / source["relative_dir"]
        path = view / (
            dataset.guide_bundle_name("white") if changed == "guide" else dataset.render_metadata_name("white")
        )
        path.write_bytes(b"changed evidence")
        with pytest.raises(ValueError, match="content changed"):
            queue_completion.verify_completion(job)


@pytest.mark.parametrize(
    ("history", "kv", "field"),
    [("recompute", "refresh", "history_mode"), ("joint", "refresh", "history_mode"), ("cache", "denoise", "kv_source")],
)
def test_causal_diagnostic_adapter_preflight_before_sessions(
    checked_settings: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, history: str, kv: str, field: str
) -> None:

    from scripts.prune.core import (  # noqa: PLC0415 -- defer loader or fixture imports until preflight and test hooks are installed
        session,
    )

    settings, _membership = checked_settings
    contract = _contract("causal")
    contract["shape"]["channels"] = 2
    contract["task"]["guide_mode"] = "d0"
    adapter = tmp_path / "adapter.safetensors"
    save_file({A: torch.ones(2, 4), B: torch.zeros(4, 2)}, adapter, metadata={CONTRACT_KEY: json.dumps(contract)})
    arguments = [
        "--mode",
        "causal",
        "--subset",
        str(settings.subset),
        "--output",
        str(settings.output),
        "--variant",
        "dev",
        "--guide-mode",
        "d0",
        "--schedule",
        "0.725",
        "0",
        "--history-mode",
        history,
        "--kv-source",
        kv,
        "--checkpoint",
        str(adapter),
    ]

    def forbidden(*_a: object, **_kw: object) -> None:
        pytest.fail("off-condition adapter reached a model/text session")

    monkeypatch.setattr(preflight, "check", forbidden)
    monkeypatch.setattr(session, "Session", forbidden)
    monkeypatch.setattr(prompt_cache, "get_or_build", forbidden)
    with pytest.raises(ValueError, match="incompatible.*" + field):
        evaluate.execute_evaluation(evaluate.parse_args(arguments))
    assert not settings.output.exists()
    _, _, cases, _ = evaluate.prepare_evaluation(evaluate.parse_args([*arguments, "--research-override"]))
    for _, _, requested, adapters in cases:
        assert requested["history_mode"] == history
        assert requested["kv_source"] == kv
        assert len(adapters[0]["overrides"]) == 1
        assert field in adapters[0]["overrides"][0]
        assert "cached_refresh_global_sigma0" in adapters[0]["overrides"][0]
    # Base-only diagnostics retain their explicit computation without calibration overrides.
    base_arguments = arguments[:-2]
    _, _, cases, _ = evaluate.prepare_evaluation(evaluate.parse_args(base_arguments))
    assert all(adapters[0]["overrides"] == [] for _, _, _, adapters in cases)
    assert all(requested["history_mode"] == history and requested["kv_source"] == kv for _, _, requested, _ in cases)
    assert not settings.output.exists()


def test_evaluation_releases_each_resident_model_before_the_next_case(
    completed: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    execute, _, _, _, _ = completed
    references = []

    @contextmanager
    def resident(session: object, *_args: object, **_kwargs: object) -> Iterator[torch.nn.Module]:
        assert all(reference() is None for reference in references), "prior full model remains resident"
        with session.transformer() as model:
            references.append(weakref.ref(model))
            yield model

    monkeypatch.setattr(evaluate.adapter_loader, "inference_transformer", resident)
    execute()
    assert len(references) == 2
    assert all(reference() is None for reference in references)


def test_model_owner_change_blocks_current_completion_without_entry_change(
    completed: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    execute, _, _, _, _ = completed
    job, paths = execute()
    record = json.loads(paths[0].read_text())
    digest = record["producer_source_sha256"]
    original = software.sha256
    monkeypatch.setattr(software, "sha256", lambda p: "f" * 64 if p.name == "common.py" else original(p))
    assert sha256(Path(evaluate.__file__)) == digest
    software.validate(record["software"])
    with pytest.raises(ValueError, match=r"software.*changed since preflight"):
        queue_completion.verify_completion(job)


def test_changed_model_owner_during_sampling_prevents_publication(
    completed: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    execute, _, settings, _, _ = completed
    original_hash = software.sha256
    original_sample = evaluate.sample_case

    def sample(*a: object, **kw: object) -> object:
        result = original_sample(*a, **kw)
        monkeypatch.setattr(software, "sha256", lambda p: "f" * 64 if p.name == "common.py" else original_hash(p))
        return result

    monkeypatch.setattr(evaluate, "sample_case", sample)
    with pytest.raises(ValueError, match=r"software.*changed since preflight"):
        execute()
    assert not list(settings.output.glob("case_*/variant_*/result.json"))
