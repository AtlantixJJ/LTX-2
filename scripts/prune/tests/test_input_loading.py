"""The case loader uses exactly the capture/noise bytes it checked, once per case."""

from dataclasses import fields
from pathlib import Path

import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.data import whole_clip
from scripts.prune.tests.test_whole_clip_baseline import _Session, _vae, _views


def test_case_uses_one_verified_load_per_tensor_and_matches_native_input(tmp_path, monkeypatch):
    current = _Session(_vae(tmp_path))
    view = _views(tmp_path / "corpus")[0]
    path = view / "ltx_vae_latent_white.pt"
    bundle = torch.load(path, weights_only=True)
    source = dict(
        capture=str(path),
        capture_latent_sha256=file_sha256(path),
        shape=list(bundle["master"].shape),
        fps=bundle["fps"],
        capture_encode_record={"schema_version": bundle["schema_version"]},
        resolved_dir=str(view),
    )
    inputs = tmp_path / "inputs"
    manifest = whole_clip.prepare_score_inputs(inputs, source, current=current, sigmas=[0.9], seed=42)
    expected = whole_clip.build_input(inputs, manifest, view=str(view), sigma=0.9, current=current)
    calls, original = [], whole_clip._load_verified

    def checked(path, *args):
        calls.append(Path(path))
        return original(path, *args)

    monkeypatch.setattr(whole_clip, "_load_verified", checked)
    actual = whole_clip.load_case(inputs, source, current=current, sigmas=[0.9], seed=42, sigma=0.9)
    assert calls == [path, inputs / "epsilon.pt"]
    for field in fields(expected[1]):
        left, right = getattr(actual[1], field.name), getattr(expected[1], field.name)
        assert torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], actual[0].patchify(bundle["master"].unsqueeze(0)))
    grid, capture = whole_clip.source_grid(source, current)
    assert grid.latent_frames == actual[0].latent_frames and torch.equal(capture, bundle["master"])


def test_deserialization_uses_hashed_buffer_even_if_file_is_replaced(tmp_path, monkeypatch):
    path = tmp_path / "tensor.pt"
    torch.save(torch.ones(4), path)
    expected = file_sha256(path)
    original = Path.read_bytes

    def replace_after_read(self):
        raw = original(self)
        if self == path:
            torch.save(torch.zeros(4), self)
        return raw

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    assert torch.equal(whole_clip._load_verified(path, expected, "changed"), torch.ones(4))
    assert torch.equal(torch.load(path, weights_only=True), torch.zeros(4))
