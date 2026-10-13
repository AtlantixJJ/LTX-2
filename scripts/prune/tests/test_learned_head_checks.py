import json

import pytest
import torch

from scripts.prune.checks.learned_heads import compare, validate_envelope


def test_fail_closed_and_nonfinite(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps({"format": "whole_clip_d0_learned_heads_v1", "family": "width_recovered", "external_adapters": True})
    )
    with pytest.raises(ValueError, match="external"):
        validate_envelope(path)
    a = torch.zeros(1, 5, 3)
    with pytest.raises(ValueError, match="finite"):
        compare(a, a * float("nan"))
    b = a.clone()
    b[:, 0] = 100
    assert compare(a, b, prefix_tokens=1)["passed"]
    b[:, 2] = 0.1
    assert not compare(a, b, prefix_tokens=1)["passed"]
