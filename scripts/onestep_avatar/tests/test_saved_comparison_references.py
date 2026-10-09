"""Prepared RGB stays separate from VAE decoding and binds to original output records."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import comparisons, hashing, media
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model.common import VideoLatentPatchifier


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict, list[int]]:
    from scripts.prune.core import model_registry  # noqa: PLC0415 -- controlled paths only

    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"checked VAE")
    monkeypatch.setattr(model_registry, "resolve", lambda *_args: SimpleNamespace(
        paths=SimpleNamespace(video_vae=lambda: vae),
        scale_factors=SimpleNamespace(time=8, height=32, width=32)))
    frames = tuple(range(9))
    pixels = [torch.full((9, 3, 64, 64), index * 50, dtype=torch.uint8) for index in range(3)]
    roles = ("recorded", "decoded", "guide")
    refs = [media.Panel(role, role, value, frames) for role, value in zip(roles, pixels, strict=True)]
    producer = {"source": "actor/view", "objective": "white", "fps": 30,
                "source_frames": list(frames), "capture_encoding_sha256": "a" * 64,
                "guide_rgb_sha256": "b" * 64, "membership_sha256": "c" * 64,
                "vae_sha256": sha256(vae), "decode_seed": 42,
                "decoder_settings": media.native_decoder_settings(), "software": software.capture("decoding"),
                "panels": [{"role": panel.role, "pixels_sha256": media._pixel_identity(panel.pixels)}
                           for panel in refs]}
    media.save_training_references(refs, producer, tmp_path / "refs")
    panels = [{"role": role, "title": title, "reference_role": role}
              for role, title in zip(roles, ("RGB", "VAE", "Guide"), strict=True)]
    for index, history in enumerate(("cache", "recompute")):
        latent = torch.zeros(1, 3, 2, 2, 2)
        latent[:, :, 1:] = (index + 1) / 10
        path = tmp_path / f"{history}.pt"
        torch.save(latent, path)
        record = {"state": "complete", "source": "actor/view", "fps": 30, "frames": 2,
                  "capture_sha256": "d" * 64, "guide_sha256": "e" * 64,
                  "c0_sha256": hashing.tensor_sha256(VideoLatentPatchifier(patch_size=1).patchify(latent[:, :, :1])),
                  "noise_sha256": "f" * 64,
                  "text_sha256": "1" * 64, "mode": "causal", "mode_settings": {"teacher_forcing": False},
                  "guide_mode": "d1", "schedule": [1.0, 0], "history_mode": history, "kv_source": "refresh",
                  "conditions": {"task": {"objective": "white", "guide_mode": "d1"}, "history_mode": history},
                  "application_method": "base", "adapter": None, "membership_sha256": "c" * 64,
                  "input_file_hashes": {"capture": "a" * 64, "render": "b" * 64},
                  "software": software.capture("evaluation", "causal"),
                  "output": {"path": str(path), "sha256": sha256(path), "shape": list(latent.shape)}}
        (tmp_path / f"{history}.json").write_text(json.dumps(record))
        panels.append({"role": "baseline" if index == 0 else "changed", "title": "Cache" if index == 0 else "Recalc",
                       "latent": f"{history}.pt", "result": f"{history}.json"})
    comparison = {"name": "history", "question": "Compare history", "span": 2, "fps": 30,
                  "panel_size": [80, 80], "viewing_width": 288, "reference_bundle": "refs/references.json",
                  "changed_factor": "history_mode", "panels": panels}
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"model": "2.5", "comparisons": [comparison]}))
    calls = []

    def decode(_session: object, latent: torch.Tensor, _decoder: object, seed: int) -> torch.Tensor:
        calls.append(seed)
        return torch.full((9, 3, 64, 64), float(latent[:, :, 1:].mean()))

    monkeypatch.setattr(media, "decode", decode)
    monkeypatch.setattr(media, "open_decoder_session", lambda *_args, **_kwargs: SimpleNamespace(
        device=torch.device("cpu"), decoder=lambda: nullcontext(None)))
    return spec, comparison, calls


def test_saved_rgb_references_render_without_another_reference_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _comparison, calls = fixture(tmp_path, monkeypatch)
    result = comparisons.render_saved_comparisons(spec, tmp_path / "out", gpu_id=0)
    row = result["comparisons"][0]
    assert calls == [42, 42]
    assert [item["role"] for item in row["rendering"]["panels"]] == [
        "recorded", "decoded", "guide", "baseline", "changed", "unused"]
    assert [item.get("kind") for item in row["inputs"]] == ["rgb_reference"] * 3 + [None, None]
    manifest = json.loads((tmp_path / "refs/references.json").read_text())
    assert [item["pixels_sha256"] for item in row["rendering"]["panels"][:3]] == [
        item["pixels_sha256"] for item in manifest["panels"]]
    monkeypatch.setattr(media, "open_decoder_session", lambda *_a, **_k: pytest.fail("completion opened VAE"))
    assert comparisons.verify_saved_comparison_completion(spec, tmp_path / "out")
    assert calls == [42, 42]


@pytest.mark.parametrize("changed", ["fps", "mapping", "seed", "vae", "settings", "missing_pixel",
                                     "output_source", "output_master", "output_guide", "output_membership",
                                     "output_shape", "output_c0", "second_factor", "result_missing", "role"])
def test_reference_or_output_mismatch_refuses_before_decoder(  # noqa: PLR0912 -- one-field negative cases
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    spec, comparison, _calls = fixture(tmp_path, monkeypatch)
    refs_path = tmp_path / "refs/references.json"
    refs = json.loads(refs_path.read_text())
    producer = refs["producer"]
    if changed == "fps":
        producer["fps"] = 24
    if changed == "mapping":
        producer["source_frames"] = list(range(1, 10))
        for panel in refs["panels"]:
            panel["source_frames"] = producer["source_frames"]
    if changed == "seed":
        producer["decode_seed"] = 43
    if changed == "vae":
        producer["vae_sha256"] = "0" * 64
    if changed == "settings":
        producer["decoder_settings"]["dtype"] = "float32"
    if changed == "missing_pixel":
        (tmp_path / "refs/guide.pt").unlink()
        assert not comparisons.saved_comparison_inputs_ready(spec)
    refs_path.write_text(json.dumps(refs))
    result_path = tmp_path / "recompute.json"
    record = json.loads(result_path.read_text())
    if changed == "output_source":
        record["source"] = "other/view"
    if changed == "output_master":
        record["input_file_hashes"]["capture"] = "0" * 64
    if changed == "output_guide":
        record["input_file_hashes"]["render"] = "0" * 64
    if changed == "output_membership":
        record["membership_sha256"] = "0" * 64
    if changed == "output_shape":
        record["output"]["shape"][2] = 3
    if changed == "output_c0":
        record["c0_sha256"] = "0" * 64
    if changed == "second_factor":
        record["noise_sha256"] = "0" * 64
    result_path.write_text(json.dumps(record))
    if changed == "result_missing":
        del comparison["panels"][-1]["result"]
    if changed == "role":
        comparison["panels"][0]["role"] = "pretend_capture"
    spec.write_text(json.dumps({"model": "2.5", "comparisons": [comparison]}))
    monkeypatch.setattr(media, "open_decoder_session", lambda *_a, **_k: pytest.fail("bad references opened VAE"))
    with pytest.raises((ValueError, FileNotFoundError)):
        comparisons.render_saved_comparisons(spec, tmp_path / "out", gpu_id=0)
    assert not (tmp_path / "out").exists()


def test_changed_original_result_prevents_manifest_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _comparison, _calls = fixture(tmp_path, monkeypatch)
    original = media.decode

    def changed_result(*args: object, **kwargs: object) -> torch.Tensor:
        (tmp_path / "recompute.json").write_text("changed during output decode")
        return original(*args, **kwargs)

    monkeypatch.setattr(media, "decode", changed_result)
    with pytest.raises(ValueError, match="inputs changed before publication"):
        comparisons.render_saved_comparisons(spec, tmp_path / "out", gpu_id=0)
    assert not (tmp_path / "out/render_manifest.json").exists()


@pytest.mark.parametrize("changed", ["reference", "result", "pixels"])
def test_completion_rejects_changed_reference_and_result_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    spec, _comparison, _calls = fixture(tmp_path, monkeypatch)
    comparisons.render_saved_comparisons(spec, tmp_path / "out", gpu_id=0)
    if changed == "reference":
        path = tmp_path / "refs/references.json"
        row = json.loads(path.read_text())
        row["extra"] = "changed bytes"
        path.write_text(json.dumps(row))
    if changed == "result":
        path = tmp_path / "cache.json"
        row = json.loads(path.read_text())
        row["extra"] = "changed bytes"
        path.write_text(json.dumps(row))
    if changed == "pixels":
        torch.save(torch.ones(9, 3, 64, 64), tmp_path / "refs/recorded.pt")
    monkeypatch.setattr(media, "open_decoder_session", lambda *_a, **_k: pytest.fail("completion opened VAE"))
    with pytest.raises(ValueError, match=r"input inventory differs|pixel file changed"):
        comparisons.verify_saved_comparison_completion(spec, tmp_path / "out")
