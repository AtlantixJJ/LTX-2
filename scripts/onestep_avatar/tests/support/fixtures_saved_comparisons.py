"""Shared saved-input builders and fixtures; never a collected test module."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest
import torch

import scripts.prune.core.session as sessions
from scripts.onestep_avatar import comparisons, evaluate, media
from scripts.onestep_avatar.execution import software
from scripts.prune.core import model_registry


def test_saved_study_spec_renders_master_and_output_with_report_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = torch.full((3, 4, 2, 2), 0.2)
    output = capture[:, :2].unsqueeze(0) + 0.1
    torch.save({"master": capture}, tmp_path / "capture.pt")
    torch.save(output, tmp_path / "generated.pt")
    spec = tmp_path / "spec.json"
    comparison = {
        "name": "matched",
        "view": "view",
        "span": 2,
        "fps": 30,
        "caption": "Matched source frames",
        "panel_size": [80, 80],
        "viewing_width": 192,
        "panels": [{"title": "Capture", "latent": "capture:capture.pt"}, {"title": "Output", "latent": "generated.pt"}],
    }
    spec.write_text(json.dumps({"comparisons": [comparison]}))
    loaded = []

    def load_master(path: Path) -> tuple[torch.Tensor, int]:
        loaded.append(path)
        return torch.load(path, weights_only=True)["master"], 30

    decoded = []

    def decode(session: object, latent: torch.Tensor, decoder: object, seed: int) -> torch.Tensor:
        del decoder, session
        decoded.append((latent.clone(), seed))
        return latent[0, :, :1].permute(1, 0, 2, 3).repeat(9, 1, 1, 1)

    class Perceptual:
        def to(self, device: object) -> Self:
            del device
            return self

        def eval(self) -> Self:
            return self

        def __call__(self, left: torch.Tensor, right: torch.Tensor) -> object:
            return (left - right).square().mean(dim=(1, 2, 3), keepdim=True)

    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"controlled decoder identity")
    monkeypatch.setattr(
        model_registry,
        "resolve",
        lambda *_args: SimpleNamespace(
            paths=SimpleNamespace(video_vae=lambda: vae), scale_factors=SimpleNamespace(time=8)
        ),
    )

    monkeypatch.setattr(evaluate.dataset, "load_training_master", load_master)
    monkeypatch.setattr(media, "decode", decode)
    monkeypatch.setattr(sessions, "open_session", lambda *_a, **_k: pytest.fail("saved render prepared text"))
    monkeypatch.setattr(
        media,
        "open_decoder_session",
        lambda *_args, **_kwargs: SimpleNamespace(device=torch.device("cpu"), decoder=lambda: nullcontext(None)),
    )
    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=lambda **_kwargs: Perceptual()))
    result = comparisons.render_saved_comparisons(spec, tmp_path / "rendered", gpu_id=0, seed=99)
    row = result["results"][0]
    assert loaded == [tmp_path / "capture.pt"]
    assert decoded[0][0].shape == (1, 3, 2, 2, 2)
    assert [seed for _, seed in decoded] == [99, 99]
    assert row["frames"] == 9
    assert row["caption"] == comparison["caption"]
    assert row["metrics"][0]["psnr_full"] is None
    assert row["metrics"][0]["lpips"] == 0
    assert row["metrics"][1]["psnr_full"] == pytest.approx(20, abs=1e-5)
    assert row["metrics"][1]["lpips"] == pytest.approx(0.04, abs=1e-6)
    assert row["rendering"]["poster_frame"] == 8
    assert row["rendering"]["layout"] == "comparison"
    assert len(row["inputs"]) == 2
    assert all(record["sha256"] for record in row["inputs"])
    assert (tmp_path / "rendered" / row["video"]).is_file()
    assert (tmp_path / "rendered" / row["poster"]).is_file()
    assert comparisons.verify_saved_comparison_completion(spec, tmp_path / "rendered", seed=99)

    original = software.sha256
    decoder_owner = software.LTX_ROOT / "packages/ltx-core/src/ltx_core/model/video_vae/conv_video_decoder.py"
    assert str(decoder_owner.relative_to(software.LTX_ROOT)) in result["software"]["sources"]
    with monkeypatch.context() as changed_owner:
        changed_owner.setattr(software, "sha256", lambda p: "f" * 64 if p == decoder_owner else original(p))
        software.validate(result["software"])
        with pytest.raises(ValueError, match="completion identity differs"):
            comparisons.verify_saved_comparison_completion(spec, tmp_path / "rendered", seed=99)
