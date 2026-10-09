"""Saved presentation reuse must bind decoder identity and all artifacts."""

from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.decode_saved import reusable
from scripts.onestep_avatar.hashing import sha256


def test_reuse_requires_decoder_and_all_artifacts(tmp_path: Path) -> None:
    video, poster = tmp_path / "case.mp4", tmp_path / "case.png"
    video.write_bytes(b"saved video")
    poster.write_bytes(b"saved poster")
    job = {"id": "case", "latent": "/saved/output.pt"}
    row = {
        "input": job,
        "source_code_sha256": "producer",
        "software": software.capture('decoding'),
        "decode_key": "decoder",
        "video": video.name,
        "video_sha256": sha256(video),
        "samples": [{"file": poster.name, "sha256": sha256(poster)}],
    }
    assert reusable(row, job, tmp_path, "producer", "decoder")
    assert not reusable({k: v for k, v in row.items() if k != 'software'}, job, tmp_path, 'producer', 'decoder')
    assert not reusable(row, job, tmp_path, "producer", "other VAE/settings")
    assert not reusable({k: v for k, v in row.items() if k != "decode_key"}, job, tmp_path, "producer", "decoder")
    assert not reusable(row, {**job, "comparison_required": True}, tmp_path, "producer", "decoder")
    poster.write_bytes(b"changed poster")
    assert not reusable(row, job, tmp_path, "producer", "decoder")
    poster.unlink()
    assert not reusable(row, job, tmp_path, "producer", "decoder")
    row["samples"] = []
    video.write_bytes(b"changed video")
    assert not reusable(row, job, tmp_path, "producer", "decoder")


def test_old_decoder_software_preserved_before_session(tmp_path, monkeypatch):
    from scripts.onestep_avatar import decode_saved

    jobs = tmp_path/'jobs.json'
    jobs.write_text(json.dumps({'jobs': [{'id': 'old'}]}))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'jobs': [{'id': 'old', 'producer': 'historical'}]}))
    original = manifest.read_bytes()
    monkeypatch.setattr(decode_saved, 'parse_args', lambda: SimpleNamespace(jobs=jobs, output=tmp_path))
    monkeypatch.setattr(decode_saved, 'open_decoder_session', lambda *_a, **_k: pytest.fail('stale output opened decoder'))
    with pytest.raises(ValueError, match='fresh destination'):
        decode_saved.main()
    assert manifest.read_bytes() == original
