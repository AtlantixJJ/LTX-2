"""D0 learning review keeps exact adapter inventory, source conditions and matched media."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import media, metrics
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import capacity_review as review
from scripts.onestep_avatar.experiments import conditioning_check as owner
from scripts.onestep_avatar.training.config import BidirectionalSettings, CausalSettings


def test_capacity_pins_borrowed_checkpoint_checker_without_changing_unrelated_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = review.adapter_effect_check.ENTRY
    capacity = software.capture("evaluation", "causal", decoder=True, extra_sources=review.EXTRAS)
    assert helper in capacity["extra_sources"]
    profiles = [
        ("training", "causal", False, ()),
        ("evaluation", "causal", True, owner.EXTRAS),
        ("decoding", None, False, review.reference_inputs.EXTRAS),
    ]
    unrelated = [software.capture(profile, mode, decoder=decode, extra_sources=extras)
                 for profile, mode, decode, extras in profiles]
    assert all(helper not in record["sources"] for record in unrelated)
    real_hash = software.sha256
    helper_path = (software.LTX_ROOT / helper).resolve()
    monkeypatch.setattr(software, "sha256", lambda path: "0" * 64
                        if Path(path).resolve() == helper_path else real_hash(path))
    with pytest.raises(ValueError, match="owners or runtime changed"):
        software.check_current(capacity)
    for record in unrelated:
        software.check_current(record)


def saved_states(mode: str) -> tuple:
    capture = torch.zeros(1, 8, 7, 2, 2, dtype=torch.bfloat16)
    outputs = {state: capture + value for state, value in zip(review.STATES, (1, 1, 2, 3), strict=True)}
    counts = {"model_calls": 1 if mode == "bidirectional" else 6,
              "denoise_calls": 1 if mode == "bidirectional" else 3,
              "prime_calls": 0, "refresh_calls": 0 if mode == "bidirectional" else 3}
    settings = BidirectionalSettings(7) if mode == "bidirectional" else CausalSettings(span_latent_frames=7)
    records = []
    for index, state in enumerate(review.STATES):
        record = {"capture_sha256": "C", "guide_sha256": None, "c0_sha256": "I", "noise_sha256": "N",
                  "text_sha256": "T", "source": "target", "fps": 30, "frames": 7, "mode": mode,
                  "mode_settings": asdict(settings), "guide_mode": "d0", "schedule": [.725, 0],
                  "conditions": {"task": {"guide_mode": "d0"}},
                  "application_method": "base" if index == 0 else "unmerged_fp32",
                  "adapter": None if index == 0 else state, "adapter_sha256": None if index == 0 else state,
                  "call_counts": counts.copy(), "metrics": metrics.encoded_metrics(outputs[state], capture)}
        if mode == "causal":
            record.update(history_mode="cache", kv_source="refresh")
        records.append(record)
    return capture, outputs, records


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
def test_learning_requires_exact_four_states_same_inputs_zero_noop_and_native_counts(mode: str) -> None:
    capture, outputs, records = saved_states(mode)
    review.check_outputs(mode, outputs, records, capture)


@pytest.mark.parametrize("change", ["missing_state", "zero_effect", "noise", "metrics", "calls", "bool_calls"])
def test_learning_rejects_changed_native_claim(change: str) -> None:
    capture, outputs, records = saved_states("causal")
    if change == "missing_state":
        outputs.pop("step60")
    elif change == "zero_effect":
        outputs["zero"] += 1
    elif change == "noise":
        records[2]["noise_sha256"] = "different"
    elif change == "metrics":
        records[3]["metrics"] = {"mse": 0}
    elif change == "bool_calls":
        records[0]["call_counts"]["prime_calls"] = False
    else:
        records[0]["call_counts"]["model_calls"] = 5
    with pytest.raises(ValueError, match=r"capacity review|comparison changes"):
        review.check_outputs("causal", outputs, records, capture)


@pytest.mark.parametrize("state", ["step20", "step60"])
def test_capacity_layout_preserves_coordinate_evidence_and_adjacent_changed_pair(state: str) -> None:
    mapping = tuple(range(49))
    references = [media.Panel(role, role, torch.zeros(49, 3, 4, 4), mapping)
                  for role in ("recorded", "decoded", "guide")]
    pixels = {name: torch.zeros(49, 3, 4, 4) for name in review.STATES}
    variants = review.layouts(review.panels_for(references, pixels, state))
    for name, panels, layout, width in variants:
        _rgb, record = media.render_panels(panels, question=review.QUESTION, layout=layout, fps=30,
                                          viewing_width=width, poster_frame=24)
        positions = {row["role"]: (row["row"], row["column"]) for row in record["panels"]}
        assert positions["baseline"][0] == positions["changed"][0]
        assert positions["changed"][1] == positions["baseline"][1] + 1
        assert record["source_frames"] == list(range(49))
        assert record["font_size"] * width / record["display_size"][0] >= 16
        assert record["padding"] == "aspect-preserving contain with neutral padding"
        if name == "narrow":
            gap = next(row for row in record["panels"] if row["role"] == "control_gap")
            assert gap["missing_reason"] == "No second zero"


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    args = SimpleNamespace(output=tmp_path / "review", mode="bidirectional", evaluation_list=tmp_path / "jobs.json")
    args.evaluation_list.write_text("{}")
    member = tmp_path / "membership.json"
    member.write_text("{}")
    paths = {state: tmp_path / f"{state}.safetensors" for state in review.STATES[1:]}
    arguments = ["--mode", "bidirectional", "--subset", str(member), "--source", "target", "--output",
                 str(tmp_path / "evaluate"), "--variant", "dev", "--guide-mode", "d0", "--schedule", ".725", "0",
                 "--span-latent-frames", "7", "--include-base"]
    for path in paths.values():
        arguments += ["--checkpoint", str(path)]
    job = {"kind": "evaluate", "arguments": arguments, "output": str(tmp_path / "evaluate")}
    monkeypatch.setattr(review.queue, "prepare_jobs", lambda *_a: [job])
    monkeypatch.setattr(review.software, "capture", lambda *_a, **_k: {"current": "software"})
    monkeypatch.setattr(review.subset, "verify_prepared_membership", lambda *_a: None)
    contracts = {state: {"task": {"guide_mode": "d0"}, "adapter": {"step": step}}
                 for state, step in zip(review.STATES[1:], (0, 20, 60), strict=True)}
    monkeypatch.setattr(review.adapter_effect_check, "checkpoint_selection", lambda *_a, **_k: (paths, contracts))
    capture, outputs, records = saved_states("bidirectional")
    video = dataset.EncodedVideo("target", "8", "held_out", capture[0], None, 30,
                                 {"capture": {"edge": 4}}, {"capture": "C"})
    variants = [None, *paths.values()]
    membership = {"objective": "white"}

    def prepare_evaluation(ordinary, **_kwargs):  # noqa: ANN001, ANN202 -- controlled ordinary preflight
        ordinary.saved_noise = torch.zeros(1, 28, 8, dtype=torch.bfloat16)
        return object(), variants, [(video, 7, {}, [])], membership

    monkeypatch.setattr(review.evaluate, "prepare_evaluation", prepare_evaluation)
    monkeypatch.setattr(review.evaluate, "verify_evaluation_conditions", lambda *_a: None)
    for index, (state, record) in enumerate(zip(review.STATES, records, strict=True)):
        folder = tmp_path / f"evaluate/case_0000/variant_{index:03d}"
        folder.mkdir(parents=True)
        record["contract"] = None if index == 0 else contracts[state]
        (folder / "result.json").write_text(json.dumps(record))
        torch.save(outputs[state], folder / "generated.pt")
    return args, contracts, variants


def test_prepare_uses_actual_saved_d0_state_inventory_without_native_work(prepared: tuple,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    args, _contracts, _variants = prepared
    monkeypatch.setattr(media, "open_decoder_session", lambda *_a, **_k: pytest.fail("prepare opened VAE"))
    result = review.prepare(args)
    assert list(result["cases"][0]["outputs"]) == list(review.STATES)
    assert result["cases"][0]["video"].encode_records == {"capture": {"edge": 4}}


@pytest.mark.parametrize("change", ["D1_training", "order", "record_step", "missing_output"])
def test_prepare_refuses_invalid_capacity_evidence(prepared: tuple, change: str) -> None:
    args, contracts, variants = prepared
    if change == "D1_training":
        contracts["step20"]["task"]["guide_mode"] = "d1"
    elif change == "order":
        variants[2], variants[3] = variants[3], variants[2]
    elif change == "record_step":
        path = args.evaluation_list.parent / "evaluate/case_0000/variant_003/result.json"
        record = json.loads(path.read_text())
        record["contract"]["adapter"]["step"] = 59
        path.write_text(json.dumps(record))
    else:
        (args.evaluation_list.parent / "evaluate/case_0000/variant_003/generated.pt").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        review.prepare(args)


def test_current_selector_dispatches_capacity_review_and_declares_all_saved_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification = tmp_path / "spec.json"
    evaluation_list = tmp_path / "evaluation.json"
    evaluation_list.write_text("{}")
    specification.write_text(json.dumps({"schema_version": 1, "protocol": "conditioning_check",
        "operation": "capacity_review", "arguments": ["--mode", "causal", "--evaluation-list", str(evaluation_list)]}))
    paths = [tmp_path / "case1", tmp_path / "case2"]
    monkeypatch.setattr(review.queue, "prepare_jobs", lambda *_a: [{"output": str(path)} for path in paths])
    args = owner.read_spec(specification, tmp_path / "out")
    assert args.operation == "capacity_review"
    assert args.mode == "causal"
    assert owner.dependency_outputs(specification) == paths
