"""Experiment interfaces accept only pinned specifications and refuse unbound inputs before work."""

import json
from pathlib import Path

import pytest

from scripts.onestep_avatar.experiments import causality, fusion_parity, sigma_sweep


@pytest.mark.parametrize("module", [causality, fusion_parity])
def test_experiment_requires_pinned_spec(module) -> None:  # noqa: ANN001 -- experiment module
    with pytest.raises(SystemExit):
        module.parse_args(["--checkpoint", "/not-read", "--output", "/not-written"])


def test_unbound_sigma_tensor_spec_fails_before_decoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema_version": 1, "model": "2.5", "decode_seed": 42, "fps": 30}))
    monkeypatch.setattr(sigma_sweep.media, "open_decoder_session", lambda *_a: pytest.fail("opened decoder"))
    with pytest.raises(ValueError, match="result-bound schema two"):
        sigma_sweep.prepare(spec, tmp_path / "output")
    assert not (tmp_path / "output").exists()
