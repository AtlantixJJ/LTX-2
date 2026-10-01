from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest
import torch

from scripts.prune.core import session
from scripts.prune.score import hooks


def test_dtype_is_declared_exactly_once():
    subpackages = ("core", "data", "score", "evaluate", "checks")
    hits = [
        f.relative_to("scripts/prune").as_posix()
        for sub in subpackages
        for f in Path("scripts/prune", sub).glob("*.py")
        if re.search(r"^DTYPE\s*=", f.read_text(), re.M)
    ]
    assert hits == ["core/session.py"]


@pytest.mark.gpu
def test_transformer_context_frees_its_memory():
    args = argparse.Namespace(model="2.5", gpu_id=0, seed=42)
    s = session.open_session(args, script="test")
    before = torch.cuda.memory_allocated(s.device)
    with s.transformer() as t:
        assert next(iter(hooks.iter_video_attention(t)))[0] == "0.attn1"
        assert len(list(hooks.iter_video_attention(t))) == 96  # 48 layers x 2
    assert torch.cuda.memory_allocated(s.device) <= before + (1 << 20)
