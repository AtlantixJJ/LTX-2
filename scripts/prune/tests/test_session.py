from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest
import torch

from scripts.prune import hooks, session


def test_record_args_default_split_is_overridable():
    parser = argparse.ArgumentParser()
    session.add_record_args(parser, default_split="held_out")
    assert parser.parse_args([]).split == "held_out"
    assert parser.parse_args(["--split", "calibration"]).split == "calibration"


def test_max_records_defaults_to_all():
    parser = argparse.ArgumentParser()
    session.add_record_args(parser)
    assert parser.parse_args([]).max_records is None


def test_dtype_is_declared_exactly_once():
    hits = [f.name for f in Path("scripts/prune").glob("*.py") if re.search(r"^DTYPE\s*=", f.read_text(), re.M)]
    assert hits == ["session.py"]


@pytest.mark.gpu
def test_open_session_yields_the_deployed_schedule_on_the_requested_device():
    args = argparse.Namespace(model="2.5", gpu_id=0, seed=42)
    s = session.open_session(args, script="test")
    assert s.sigmas.dtype == torch.float32 and s.sigmas.device.index == 0
    # approx, not ==: 0.725 is not exactly representable in float32, so the tensor's
    # round-trip through .tolist() is 0.7250000238418579, not the raw Python float.
    assert s.sigmas.tolist() == pytest.approx([0.725, 0.421875, 0.0])
    assert s.geometry().as_dict()["chunk_latent_frames"] == 2


@pytest.mark.gpu
def test_transformer_context_frees_its_memory():
    args = argparse.Namespace(model="2.5", gpu_id=0, seed=42)
    s = session.open_session(args, script="test")
    before = torch.cuda.memory_allocated(s.device)
    with s.transformer() as t:
        assert next(iter(hooks.iter_video_attention(t)))[0] == "0.attn1"
        assert len(list(hooks.iter_video_attention(t))) == 96  # 48 layers x 2
    assert torch.cuda.memory_allocated(s.device) <= before + (1 << 20)
