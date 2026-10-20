import pytest
import torch

from scripts.prune.evaluate.learned_head_media import panels
from scripts.prune.evaluate.learned_heads import capture_metrics


def test_gt_metrics_exclude_clean_frame_and_need_no_teacher():
    capture = torch.zeros(1, 3, 2)
    epsilon = torch.ones_like(capture)
    clean = capture.clone()
    clean[:, 1:] = 2
    velocity = epsilon.clone()
    result = capture_metrics(velocity, clean, capture, epsilon, prefix_tokens=1)
    assert result == {
        "capture_mse": 4.0,
        "official_velocity_mse": 0.0,
        "clean_frame_equal": True,
        "reference": "gt_capture",
    }
    clean[:, :1] = 3
    with pytest.raises(ValueError, match="clean frame"):
        capture_metrics(velocity, clean, capture, epsilon, prefix_tokens=1)


def test_gt_comparison_has_capture_and_candidates_without_teacher():
    pixels = torch.zeros(3, 3, 8, 8, dtype=torch.uint8)
    result = panels(pixels, pixels, None, {"learned_joint": pixels, "energy": pixels})
    assert len(result) == 4


def test_deferred_source_audit_checks_pins_without_foreground_decode(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.prune.core.provenance import file_sha256
    from scripts.prune.data import learned_head_study as owner

    view = tmp_path / "corpus/Part_1/0001_01/views/view00"
    view.mkdir(parents=True)
    (view / "rgb.mp4").write_bytes(b"pinned RGB source")
    (view.parent.parent / "meta.json").write_text("{}")
    torch.save(torch.zeros(1), view / "ltx_vae_latent_white.pt")
    source = {
        "relative_dir": "Part_1/0001_01/views/view00",
        "actor": "1",
        "shape": [1, 2, 1, 1],
        "capture_encode_record": {"box_xyxy": [0, 0, 32, 32]},
        "capture_latent_sha256": file_sha256(view / "ltx_vae_latent_white.pt"),
        "rgb_sha256": file_sha256(view / "rgb.mp4"),
        "clip_metadata_sha256": file_sha256(view.parent.parent / "meta.json"),
    }
    monkeypatch.setattr(owner, "load_membership", lambda _: {"corpus_root": str(tmp_path / "corpus")})
    monkeypatch.setattr(
        owner,
        "ClipStore",
        lambda _: SimpleNamespace(
            load=lambda *a, **k: SimpleNamespace(z_y=torch.zeros(1, 2, 1, 1, dtype=torch.bfloat16), fps=30)
        ),
    )
    monkeypatch.setattr(owner.CaptureManifest, "load", lambda _: SimpleNamespace(box_for=lambda _: [0, 0, 32, 32]))
    monkeypatch.setattr(owner.whole_clip, "actor_identity", lambda _: "dna_actor:1")

    def forbidden(*args, **kwargs):
        raise AssertionError("deferred audit must not decode foreground")

    monkeypatch.setattr(owner.foreground, "person_coverage", forbidden)
    monkeypatch.setattr(owner.media, "recorded_capture_rgb", forbidden)
    result = owner.prepare_view(("membership", source, True, str(tmp_path / "out"), {"status": "not_verified"}, False))
    assert result["foreground_audit"] == "deferred"
    assert result["statistics"] is None
    assert len(result["files"]) == 3
    (view / "rgb.mp4").write_bytes(b"changed RGB")
    with pytest.raises(ValueError, match="cached source audit input changed"):
        owner.prepare_view(("membership", source, True, str(tmp_path / "out"), {"status": "not_verified"}, False))
