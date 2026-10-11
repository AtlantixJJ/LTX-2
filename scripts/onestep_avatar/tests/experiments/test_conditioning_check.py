"""Conditioning controls preserve the actual changed input and reject false saved claims."""

import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import evaluate, hashing, media, metrics
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.experiments import conditioning_check as check
from scripts.onestep_avatar.model import common
from scripts.onestep_avatar.tests.test_causal_core import _geometry, _grid, _model
from scripts.onestep_avatar.training.config import BidirectionalSettings, CausalSettings
from scripts.prune.core import session
from scripts.prune.data import prompt_cache


@pytest.mark.parametrize("control", ["guide", "text", "sigma_one", "capture_oracle"])
def test_controls_change_exactly_the_named_input(control: str) -> None:
    text, guide = torch.zeros(1, 3, 16), torch.zeros(1, 28, 8)
    arms = check.control_inputs(control, text, guide, foreign_guide=guide + 1, foreign_text=text + 1)
    assert tuple(arm[0] for arm in arms) == check.ARMS[control]
    assert (arms[0][1] is arms[1][1]) == (control != "text")
    assert (arms[0][2] is arms[1][2]) == (control != "guide")
    assert (arms[0][3] == arms[1][3]) == (control not in ("sigma_one", "capture_oracle"))


@pytest.mark.parametrize("control", ["guide", "text"])
@pytest.mark.parametrize("invalid", ["identical", "shape", "dtype", "nan", "absent"])
def test_invalid_interventions_fail_before_sampling(control: str, invalid: str) -> None:
    text, guide = torch.zeros(1, 3, 16), torch.zeros(1, 28, 8)
    original = guide if control == "guide" else text
    changed = original + 1
    if invalid == "identical":
        changed = original.clone()
    elif invalid == "shape":
        changed = changed[:, :1]
    elif invalid == "dtype":
        changed = changed.double()
    elif invalid == "nan":
        changed[0, 0, 0] = float("nan")
    else:
        changed = None
    with pytest.raises(ValueError, match="foreign"):
        check.control_inputs(control, text, guide, **{
            "foreign_guide" if control == "guide" else "foreign_text": changed})


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
def test_sigma_one_real_shared_sampler_matches_and_preserves_c0(mode: str) -> None:
    model, grid = _model(), _grid(_geometry())
    capture = torch.ones(1, 28, 8)
    guide, noise = capture + 3, torch.zeros_like(capture)
    context = torch.zeros(1, 3, 16)
    settings = BidirectionalSettings(7) if mode == "bidirectional" else CausalSettings()
    outputs = []
    for arm in ("d0", "d1"):
        output, record = evaluate.sample_case(
            model, context, grid, capture, guide, noise, mode=mode, mode_settings=settings,
            guide_mode=arm, schedule=[1, 0], seed=42,
            predict_x0=common.denoised_from_velocity_model(model))
        outputs.append(output)
        assert record["schema_version"] == 2
        assert record["c0_sha256"] == hashing.tensor_sha256(capture[:, :4])
        assert record["call_counts"] == {
            "model_calls": 1 if mode == "bidirectional" else 6,
            "denoise_calls": 1 if mode == "bidirectional" else 3,
            "prime_calls": 0,
            "refresh_calls": 0 if mode == "bidirectional" else 3}
    assert check.summarize("sigma_one", outputs)["state"] == "passed"
    outputs[1][:, :, 1:] += 1
    assert check.summarize("sigma_one", outputs)["state"] == "failed"


@pytest.fixture
def saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    args = SimpleNamespace(output=tmp_path, control="guide", mode="bidirectional", prompt="matched",
                           wrong_prompt=None, checkpoint=None, seed=42,
                           subset=tmp_path / "subset.json", noise_file=tmp_path / "noise.pt", decode=False)
    args.subset.write_text("{}")
    records = {"capture": {"source": "target"}, "guide": {"source": "target"}}
    video = dataset.EncodedVideo("target", "8", "held_out", torch.zeros(8, 7, 2, 2),
                                 torch.ones(8, 7, 2, 2), 30, records, {"capture": "C", "guide": "G"})
    donor = dataset.EncodedVideo("donor", "12", "held_out", torch.zeros(8, 7, 2, 2),
                                 torch.full((8, 7, 2, 2), 2), 30,
                                 {"guide": {"source": "donor"}}, {"capture": "D", "guide": "F"})
    ordinary = SimpleNamespace(mode_settings=BidirectionalSettings(7), schedule=[.725, 0])
    prepared = {"video": video, "donor": donor, "ordinary": ordinary,
                "membership": {"sha256": "M"}, "requested": {"task": {"guide_mode": "d1"}},
                "checks": {name: {"adapter": None, "application_method": "base", "overrides": []}
                           for name in check.ARMS["guide"]}}
    grid = SimpleNamespace(tokens_per_latent_frame=4,
                           unpatchify_block=lambda _tokens, _frames: video.z_y.unsqueeze(0).to(torch.bfloat16))
    capture = torch.zeros(1, 28, 8, dtype=torch.bfloat16)
    guide, foreign, noise = capture + 1, capture + 2, capture + 3
    context = torch.zeros(1, 3, 16, dtype=torch.bfloat16)
    producer = {"saved": "software"}
    monkeypatch.setattr(check, "prepare", lambda *_a, **_k: prepared)
    monkeypatch.setattr(check, "input_tokens", lambda *_a: (grid, capture, guide, foreign, noise))
    monkeypatch.setattr(check.software, "capture", lambda *_a, **_k: producer)
    monkeypatch.setattr(check.software, "check_current", lambda *_a: None)
    def blocked(*_a, **_k) -> None:
        pytest.fail("saved completion opened native execution")

    monkeypatch.setattr(session, "Session", blocked)
    monkeypatch.setattr(prompt_cache, "get_or_build", blocked)
    monkeypatch.setattr(check.adapters, "inference_transformer", blocked)
    torch.save(context, tmp_path / "text.pt")
    torch.save(noise, tmp_path / "noise.pt")
    outputs = []
    for name, text, selected_guide, guide_mode in check.control_inputs(
            "guide", context, guide, foreign_guide=foreign):
        output = torch.ones(1, 8, 7, 2, 2, dtype=torch.bfloat16)
        output[:, :, :1] = 0
        if name == "foreign_guide":
            output[:, :, 1:] += 1
        outputs.append(output)
        record = {**check.arm_provenance(args, prepared, name, guide_mode, args.prompt, producer),
                  "schema_version": 2, "mode": "bidirectional", "mode_settings": asdict(ordinary.mode_settings),
                  "guide_mode": guide_mode, "frames": 7, "schedule": [.725, 0], "seed": 42,
                  "capture_sha256": hashing.tensor_sha256(capture),
                  "guide_sha256": hashing.tensor_sha256(selected_guide),
                  "c0_sha256": hashing.tensor_sha256(capture[:, :4]),
                  "noise_sha256": hashing.tensor_sha256(noise), "text_sha256": hashing.tensor_sha256(text),
                  "call_counts": {"model_calls": 1, "denoise_calls": 1,
                                  "prime_calls": 0, "refresh_calls": 0},
                  "metrics": metrics.encoded_metrics(output, grid.unpatchify_block(capture, 7))}
        evaluate.save_case(output, record, tmp_path / name)
    (tmp_path / "result.json").write_text(json.dumps(check.summarize("guide", outputs)))
    return args, tmp_path


def test_saved_verifier_reconstructs_actual_donor_with_native_execution_blocked(saved: tuple) -> None:
    args, root = saved
    assert check.verify_scientific(args)["interpretation"] == "sensitive_to_changed_input"
    baseline = json.loads((root / "matched/result.json").read_text())
    changed = json.loads((root / "foreign_guide/result.json").read_text())
    assert baseline["input_file_hashes"] == changed["input_file_hashes"]
    assert baseline["guide_input_file_hashes"] != changed["guide_input_file_hashes"]
    assert baseline["guide_source"] == "target"
    assert changed["guide_source"] == "donor"


@pytest.mark.parametrize("field", ["guide_sha256", "source", "guide_source", "prompt", "noise_sha256",
                                 "conditions", "c0_sha256", "call_counts", "output", "schema_version", "metrics"])
def test_saved_verifier_rejects_changed_scientific_claim(saved: tuple, field: str) -> None:
    args, root = saved
    path = root / "foreign_guide/result.json"
    record = json.loads(path.read_text())
    record[field] = {"model_calls": 99} if field == "call_counts" else "changed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="conditioning"):
        check.verify_scientific(args)


@pytest.mark.parametrize("change", ["missing", "extra", "non_integer"])
def test_saved_verifier_requires_exact_native_call_inventory(saved: tuple, change: str) -> None:
    args, root = saved
    path = root / "foreign_guide/result.json"
    record = json.loads(path.read_text())
    counts = record["call_counts"]
    if change == "missing":
        counts.pop("refresh_calls")
    elif change == "extra":
        counts["backward_calls"] = 0
    else:
        counts["denoise_calls"] = True
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="native call counts"):
        check.verify_scientific(args)


def test_saved_verifier_rejects_changed_summary_and_first_image(saved: tuple) -> None:
    args, root = saved
    (root / "result.json").write_text(json.dumps({"state": "passed"}))
    with pytest.raises(ValueError, match="summary"):
        check.verify_scientific(args)
    path = root / "foreign_guide/generated.pt"
    output = torch.load(path, weights_only=True)
    output[:, :, :1] = 2
    torch.save(output, path)
    with pytest.raises(ValueError, match="clean first image"):
        check.verify_scientific(args)


def test_saved_verifier_rejects_changed_fixed_input_bytes(saved: tuple) -> None:
    args, _root = saved
    args.subset.write_text("{\"changed\": true}")
    with pytest.raises(ValueError, match="actual branch evidence"):
        check.verify_scientific(args)


def test_evidence_inventory_uses_one_prior_verification(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(check, "verify_completion", lambda *_a: calls.append("verify"))
    monkeypatch.setattr(check, "read_spec", lambda *_a: SimpleNamespace())
    monkeypatch.setattr(check.spec_protocol, "inspect", lambda *_a: {"artifacts": {"result.json": "hash"}})
    paths = check.evidence_paths(tmp_path / "spec.json", tmp_path)
    assert calls == ["verify"]
    assert paths == [tmp_path / "manifest.json", tmp_path / "result.json"]
    check.evidence_paths(tmp_path / "spec.json", tmp_path, verified=True)
    assert calls == ["verify"]


@pytest.fixture
def decoded_saved(saved: tuple, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Use real RGB layout/serialization with controlled decoder and video-file doubles."""
    args, root = saved
    args.decode, args.source, args.donor_source = True, "target", "donor"
    prepared = check.prepare(args)
    prepared["video"].encode_records["capture"]["edge"] = 4
    prepared["membership"]["sources"] = [
        {"relative_dir": "target", "actor": "8", "capture_latent_sha256": "c" * 64, "guide_sha256": "a" * 64},
        {"relative_dir": "donor", "actor": "12"}]
    store = SimpleNamespace(root=root, sources={row["relative_dir"]: row
                                              for row in prepared["membership"]["sources"]})
    monkeypatch.setattr(check.dataset, "ClipStore", lambda *_a: store)
    monkeypatch.setattr(check.reference_inputs, "checked_source", lambda selected, source:
                        (selected.sources[source], {"controlled_crop": source}))
    vae = root / "vae.safetensors"
    vae.write_bytes(b"controlled VAE identity")
    prepared["specification"] = SimpleNamespace(paths=SimpleNamespace(video_vae=lambda: str(vae)))
    producer = {"saved": "software"}
    outputs, records = [], []
    for name in check.ARMS[args.control]:
        record_path = root / name / "result.json"
        record = json.loads(record_path.read_text())
        record["target_encode_records"] = prepared["video"].encode_records
        record_path.write_text(json.dumps(record))
        records.append(record)
        outputs.append(torch.load(root / name / "generated.pt", weights_only=True))
    mapping = tuple(range(49))
    rgb = {"recorded": torch.zeros(49, 3, 4, 4, dtype=torch.uint8),
           "decoded": torch.full((49, 3, 4, 4), .1),
           "guide": torch.ones(49, 3, 4, 4, dtype=torch.uint8),
           "donor_guide": torch.full((49, 3, 4, 4), 2, dtype=torch.uint8)}
    refs = [media.Panel(name, name, rgb[name], mapping) for name in ("recorded", "decoded", "guide")]
    ref_record = {"source": args.source, "objective": "white", "fps": 30,
                  "capture_encoding_sha256": "c" * 64, "guide_rgb_sha256": "a" * 64,
                  "source_frames": list(mapping),
                  "vae_sha256": check.sha256(vae), "decode_seed": 42,
                  "decoder_settings": media.native_decoder_settings(),
                  "decode_key": media.decode_key("c" * 64, check.sha256(vae), list(outputs[0].shape),
                                                "native_decode_video", 42, media.native_decoder_settings()),
                  "panels": [{"role": panel.role, "pixels_sha256": hashing.tensor_sha256(panel.pixels)}
                             for panel in refs]}
    monkeypatch.setattr(media, "prepare_training_references", lambda *_a, **_k: (refs, ref_record))
    monkeypatch.setattr(media, "recorded_capture_rgb", lambda *_a: rgb["recorded"])
    monkeypatch.setattr(media, "recorded_guide_rgb", lambda source, *_a:
                        rgb["guide"] if source["relative_dir"] == "target" else rgb["donor_guide"])
    monkeypatch.setattr(media, "decode", lambda _session, latent, *_a: torch.full((49, 3, 4, 4),
                        .2 if torch.equal(latent, outputs[0]) else .3))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_a: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_a: 200)

    def save_render(pixels: torch.Tensor, record: dict, folder: Path) -> dict:
        folder.mkdir(parents=True)
        (folder / "comparison.mp4").write_bytes(b"controlled encoded video")
        media.frame(pixels, record["poster_frame"]).save(folder / "poster.png")
        record = {**record, "outputs": {role: {"path": str(folder / filename),
                                             "sha256": check.sha256(folder / filename)}
                  for role, filename in (("video", "comparison.mp4"), ("poster", "poster.png"))}}
        (folder / "rendering.json").write_text(json.dumps(record))
        return record

    monkeypatch.setattr(media, "save_render", save_render)
    monkeypatch.setattr(media, "verify_saved_video", lambda path, *_a: dict(zip(("width", "height"),
                        json.loads(path.with_name("rendering.json").read_text())["display_size"], strict=True)))
    native_session = SimpleNamespace(device=torch.device("cpu"), decoder=lambda: nullcontext(None))
    check.render_review(args, prepared, native_session, outputs, records, producer)
    monkeypatch.setattr(media, "decode", lambda *_a, **_k: pytest.fail("saved verification decoded again"))
    monkeypatch.setattr(media, "prepare_training_references", lambda *_a, **_k:
                        pytest.fail("saved verification prepared references again"))
    return args, root


def test_saved_decoded_review_matches_rgb_coordinates_and_readable_layouts(decoded_saved: tuple) -> None:
    args, root = decoded_saved
    assert check.verify_scientific(args)["state"] == "passed"
    full = json.loads((root / "review/full/rendering.json").read_text())
    narrow = json.loads((root / "review/narrow/rendering.json").read_text())
    assert [(panel["row"], panel["column"]) for panel in full["panels"]] == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert narrow["viewing_width"] == 480
    assert narrow["font_size"] * 480 / narrow["display_size"][0] >= 16
    positions = {panel["role"]: (panel["row"], panel["column"]) for panel in narrow["panels"]}
    assert positions["baseline"] == (2, 0)
    assert positions["changed"] == (2, 1)
    assert full["padding"] == "aspect-preserving contain with neutral padding"
    assert full["source_frames"] == narrow["source_frames"] == list(range(49))
    assert full["panels"][-1]["title"] == "Donor guide"
    assert full["panels"][-1]["value"] == "RGB; person 12"


@pytest.mark.parametrize("field", ["settings", "decoder", "reference", "layout", "poster", "donor", "cost"])
def test_saved_decoded_review_rejects_changed_evidence(decoded_saved: tuple, field: str) -> None:
    args, root = decoded_saved
    path = root / "review.json"
    review = json.loads(path.read_text())
    if field == "settings":
        review["settings"]["prompts"]["foreign_guide"] = "different text"
    elif field == "decoder":
        review["decoded_outputs"]["matched"] = "changed"
    elif field == "cost":
        review["seconds_decode_and_render"] = -1
    elif field == "reference":
        reference_path = Path(review["reference_bundle"])
        reference = json.loads(reference_path.read_text())
        reference["producer"]["objective"] = "bg"
        reference_path.write_text(json.dumps(reference))
    elif field == "layout":
        rendering_path = root / "review/narrow/rendering.json"
        rendering = json.loads(rendering_path.read_text())
        rendering["panels"][3]["title"] = "Wrong role"
        rendering_path.write_text(json.dumps(rendering))
    elif field == "poster":
        from PIL import Image  # noqa: PLC0415 -- controlled saved poster damage

        Image.new("RGB", (10, 10)).save(root / "review/full/poster.png")
    else:
        donor_path = Path(review["panels"]["donor_guide"]["path"])
        value = torch.zeros(49, 3, 4, 4, dtype=torch.uint8)
        torch.save(value, donor_path)
        review["panels"]["donor_guide"].update(sha256=check.sha256(donor_path),
                                             tensor_sha256=hashing.tensor_sha256(value))
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="conditioning"):
        check.verify_scientific(args)


def test_capture_oracle_cannot_claim_learning_capacity() -> None:
    summary = check.summarize("capture_oracle", [torch.zeros(1, 2), torch.ones(1, 2)])
    assert summary["state"] == "passed"
    assert summary["interpretation"] == "target-provided capture oracle; not D0 learning capacity"
    assert summary["quality_acceptance"] is None
