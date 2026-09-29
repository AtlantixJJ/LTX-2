from __future__ import annotations

import json

import pytest

from scripts.prune.score import hooks


@pytest.fixture
def report(tmp_path):
    path = tmp_path / "head_scores.json"
    path.write_text(json.dumps({
        "provenance": {"model_key": "2.5", "transformer_fingerprint": "abc"},
        "iterative": {"masks": {"0.attn1": [1, 0], "0.attn2": [0, 1]}},
    }))
    return path


def test_valid_report_has_content_identity(report):
    masks, digest = hooks.read_mask_artifact(
        report, model_key="2.5", fingerprint="abc",
        widths={"0.attn1": 2, "0.attn2": 2, "0.ff": 4},
    )
    assert masks["0.attn1"] == [1, 0] and len(digest) == 64


@pytest.mark.parametrize("change", ["fingerprint", "partial", "nonbinary", "empty", "wrong_width"])
def test_invalid_report_fails_before_mask_install(report, change):
    payload = json.loads(report.read_text())
    masks = payload["iterative"]["masks"]
    if change == "fingerprint":
        payload["provenance"]["transformer_fingerprint"] = "other"
    elif change == "partial":
        del masks["0.attn2"]
    elif change == "nonbinary":
        masks["0.attn1"] = [1, 0.5]
    elif change == "empty":
        masks["0.attn1"] = [0, 0]
    else:
        masks["0.attn1"] = [1]
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        hooks.read_mask_artifact(report, model_key="2.5", fingerprint="abc",
                                 widths={"0.attn1": 2, "0.attn2": 2, "0.ff": 4})
