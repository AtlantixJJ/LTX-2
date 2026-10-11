"""Original membership survives crop-schema adaptation; reference completion stays saved-only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.experiments import conditioning_check as owner
from scripts.onestep_avatar.experiments import reference_inputs as refs


@pytest.fixture
def store(tmp_path: Path) -> SimpleNamespace:
    source = "Part_1/0008_01/views/view00_cam51"
    box = [0.0, 2.0, 8.0, 10.0]
    manifest = tmp_path / dataset.CAPTURE_MANIFEST_NAME
    manifest.write_text(json.dumps({"pad_factor": 1.0, "edge": 4,
                                  "windows": [{"bundle": source + "/capture.pt", "box_xyxy": box}]}))
    record = {"source": source, "objective": "white", "box_xyxy": box, "edge": 4}
    return SimpleNamespace(root=tmp_path, objective="white", membership={"sha256": "original identity",
        "capture_manifest": {"sha256": refs.sha256(manifest)}},
        sources={source: {"relative_dir": source, "capture_encode_record": copy.deepcopy(record),
                          "guide_encode_record": copy.deepcopy(record)}})


def test_crop_adapter_checks_actual_saved_manifest_and_never_changes_original_membership(
    store: SimpleNamespace,
) -> None:
    original = copy.deepcopy(store.sources)
    source, evidence = refs.checked_source(store, next(iter(store.sources)))
    assert store.sources == original
    assert "box_xyxy" not in next(iter(store.sources.values()))
    assert source["box_xyxy"] == source["capture_encode_record"]["box_xyxy"]
    assert evidence["original_membership_sha256"] == "original identity"
    source["capture_encode_record"]["box_xyxy"][0] = 50
    assert store.sources == original


@pytest.mark.parametrize("change", ["manifest_bytes", "capture_crop", "guide_crop", "edge", "objective", "top_crop"])
def test_crop_adapter_rejects_any_presentation_crop_substitution(store: SimpleNamespace, change: str) -> None:
    source_id = next(iter(store.sources))
    source = store.sources[source_id]
    if change == "manifest_bytes":
        (store.root / dataset.CAPTURE_MANIFEST_NAME).write_text("{}")
    elif change in ("capture_crop", "guide_crop"):
        source[change.split("_", maxsplit=1)[0]+"_encode_record"]["box_xyxy"][0] = 1
    elif change == "top_crop":
        source["box_xyxy"] = [0, 0, 2, 2]
    else:
        source["capture_encode_record"][change] = 1 if change == "edge" else "bg"
    with pytest.raises(ValueError, match="reference"):
        refs.checked_source(store, source_id)


def test_unused_d0_guide_still_requires_exact_capture_crop(store: SimpleNamespace) -> None:
    source_id = next(iter(store.sources))
    store.sources[source_id].pop("guide_encode_record")
    source, _record = refs.checked_source(store, source_id)
    assert source["box_xyxy"] == source["capture_encode_record"]["box_xyxy"]


def test_current_selector_reads_preparation_operation_and_declares_no_native_dependencies(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema_version": 1, "protocol": "conditioning_check",
        "operation": "prepare_references", "arguments": ["--subset", str(tmp_path / "original.json"),
        "--source", "Part_1/0008_01/views/view00_cam51", "--encoded-frames", "17", "--seed", "42"]}))
    args = owner.read_spec(spec, tmp_path / "out")
    assert args.operation == "prepare_references"
    assert args.guide_mode == "d1"
    assert args.encoded_frames == 17
    assert owner.dependency_outputs(spec) == []


def test_saved_preparation_completion_dispatches_without_decoder_or_native_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema_version": 1, "protocol": "conditioning_check",
        "operation": "prepare_references", "arguments": ["--subset", str(tmp_path / "original.json"),
        "--source", "target", "--encoded-frames", "7", "--seed", "42"]}))
    monkeypatch.setattr(owner.spec_protocol, "inspect", lambda *_a: {})
    monkeypatch.setattr(refs, "verify", lambda _args: {"state": "passed", "scope": "saved references"})
    monkeypatch.setattr(refs.media, "open_decoder_session", lambda *_a, **_k: pytest.fail("completion opened VAE"))
    assert owner.verify_completion(spec, tmp_path / "out")["state"] == "passed"


@pytest.fixture
def saved_references(store: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    import torch  # noqa: PLC0415 -- controlled saved tensor fixture

    source_id = next(iter(store.sources))
    row, crop = refs.checked_source(store, source_id)
    row.update(shape=[2, 7, 1, 1], fps=30, n_latent_frames=7,
               capture_latent_sha256="c" * 64, guide_sha256="g" * 64)
    subset_file = store.root / "original.json"
    subset_file.write_text("original membership bytes")
    vae = store.root / "vae.safetensors"
    vae.write_bytes(b"controlled VAE identity")
    args = SimpleNamespace(output=store.root / "references", subset=subset_file,
                           source=source_id, encoded_frames=7, seed=42)
    producer_software = {"controlled": "current"}
    prepared = {"source": row, "store": store, "crop": crop,
                "input_files": {str(subset_file): refs.sha256(subset_file)}, "software": producer_software,
                "specification": SimpleNamespace(paths=SimpleNamespace(video_vae=lambda: str(vae)))}
    monkeypatch.setattr(refs, "prepare", lambda *_a, **_k: prepared)
    monkeypatch.setattr(refs.software, "check_current", lambda *_a: None)
    pixels = {"recorded": torch.zeros(49, 3, 4, 4, dtype=torch.uint8),
              "decoded": torch.full((49, 3, 4, 4), .2), "guide": torch.ones(49, 3, 4, 4, dtype=torch.uint8)}
    panels = [refs.media.Panel(role, role, value, tuple(range(49))) for role, value in pixels.items()]
    vae_hash = refs.sha256(vae)
    producer = {"source": source_id, "objective": "white", "fps": 30, "source_frames": list(range(49)),
        "capture_encoding_sha256": "c" * 64, "guide_rgb_sha256": "g" * 64,
        "vae_sha256": vae_hash, "decode_seed": 42, "decoder_settings": refs.media.native_decoder_settings(),
        "decode_key": refs.media.decode_key("c" * 64, vae_hash, [1, 2, 7, 1, 1], "native_decode_video", 42,
                                            refs.media.native_decoder_settings()),
        "membership_file_sha256": refs.sha256(subset_file), "membership_sha256": store.membership["sha256"],
        "source_crop_adapter": crop, "software": producer_software,
        "panels": [{"role": panel.role, "pixels_sha256": refs.media._pixel_identity(panel.pixels)}
                   for panel in panels]}
    refs.media.save_training_references(panels, producer, args.output)
    (args.output / "source_crop.json").write_text(json.dumps({"crop_adapter": crop,
        "input_files": prepared["input_files"], "software": producer_software}))
    monkeypatch.setattr(refs.media, "recorded_capture_rgb", lambda *_a: pixels["recorded"])
    monkeypatch.setattr(refs.media, "recorded_guide_rgb", lambda *_a: pixels["guide"])
    monkeypatch.setattr(refs.media, "open_decoder_session",
                        lambda *_a, **_k: pytest.fail("saved completion opened VAE"))
    return args


def test_saved_reference_verifier_preserves_original_producer_identity(saved_references: SimpleNamespace) -> None:
    assert refs.verify(saved_references)["state"] == "passed"


@pytest.mark.parametrize("change", ["crop", "membership", "decoder", "source_pixels"])
def test_saved_reference_verifier_rejects_false_coordinate_or_decoder_claim(
    saved_references: SimpleNamespace, change: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch  # noqa: PLC0415 -- saved RGB damage

    root = saved_references.output
    if change == "crop":
        path = root / "source_crop.json"
        value = json.loads(path.read_text())
        value["crop_adapter"]["box_xyxy"][0] = 50
    elif change == "source_pixels":
        monkeypatch.setattr(refs.media, "recorded_capture_rgb", lambda *_a: torch.ones(49, 3, 4, 4))
    else:
        path = root / "references.json"
        value = json.loads(path.read_text())
        value["producer"]["membership_sha256" if change == "membership" else "decode_key"] = "different"
    if change != "source_pixels":
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="saved reference"):
        refs.verify(saved_references)
