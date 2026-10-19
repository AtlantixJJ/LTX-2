"""Regression controls for original gate geometry and lossless split bundles."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from ltx_core.model.transformer.feed_forward import ShapeFaithfulLinear
from ltx_core.model.transformer.ops import PytorchGatedAttention
from scripts.prune.learn import adapters
from scripts.prune.score import export_joint_split
from scripts.prune.tests.test_export_pruned import _checkpoint


def test_compact_native_gate_keeps_original_gemm_width_and_original_ids():
    torch.manual_seed(118)
    x = torch.randn(2, 7, 6)
    full = torch.nn.Linear(6, 4)
    keep = [0, 2, 3]
    small = torch.nn.Linear(6, 3)
    with torch.no_grad():
        small.weight.copy_(full.weight[keep])
        small.bias.copy_(full.bias[keep])
    attn = SimpleNamespace(qk_heads=4, heads=3, dim_head=2, rope_head_indices=keep, to_gate_logits=small)
    values = torch.randn(2, 7, 6)
    expected = (values.view(2, 7, 3, 2) * (2 * full(x)[..., keep].sigmoid()).unsqueeze(-1)).flatten(-2)
    linear = F.linear
    widths = []

    def capture(inp, weight, bias=None):
        widths.append(weight.shape[0])
        return linear(inp, weight, bias)

    with patch("torch.nn.functional.linear", side_effect=capture):
        actual = PytorchGatedAttention()(x, values, attn)
    assert widths == [4]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_split_adapter_restore_axes_keeps_native_operation_order(axis, dtype):
    torch.manual_seed(212)
    base = torch.nn.Linear(8, 12).to(dtype)
    ref = adapters.ProjectionAdapter(base, rank=3)
    with torch.no_grad():
        ref.B.normal_(std=0.01)
    ids = [0, 2, 5]
    compact = ShapeFaithfulLinear(8, 12, indices=ids, axis=axis, bias=True).to(dtype)
    with torch.no_grad():
        compact.weight.copy_(base.weight[ids] if axis == 0 else base.weight[:, ids])
        compact.bias.copy_(base.bias[ids] if axis == 0 else base.bias)
    a = ref.A.detach().clone() if axis == 0 else ref.A.detach()[:, ids].clone()
    b = ref.B.detach()[ids].clone() if axis == 0 else ref.B.detach().clone()
    candidate = export_joint_split.SplitAdapter(compact, a, b)
    x = torch.randn(2, 7, 8).to(dtype)
    if axis == 1:
        mask = torch.zeros(8, dtype=dtype)
        mask[ids] = 1
        x = x * mask
    expected = ref(x)
    if axis == 0:
        mask = torch.zeros(12, dtype=dtype)
        mask[ids] = 1
        expected = expected * mask
    assert torch.equal(candidate(x), expected)
    assert not any(p.requires_grad for p in candidate.parameters())


def test_bundle_pins_payloads_and_refuses_changed_adapter_file(tmp_path):
    source = tmp_path / "source.safetensors"
    _checkpoint(source)
    path = tmp_path / "candidate.json"
    masks = {"0.attn1": [1, 0], "0.attn2": [1, 1], "0.ff": [1, 0, 1]}
    export_joint_split.bundle(source, masks, {}, path, mode="compact")
    record = export_joint_split.read(path)
    assert record["execution_mode"] == "compact"
    assert record["compact_execution_qualified"] is False
    adapter_path = tmp_path / "candidate.adapters.safetensors"
    with adapter_path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="Changed bundle adapters"):
        export_joint_split.read(path)
    assert json.loads(path.read_text())["format"] == export_joint_split.FORMAT


def test_split_adapter_rejects_wrong_stored_axes():
    with pytest.raises(ValueError, match="stored base axes"):
        export_joint_split.SplitAdapter(torch.nn.Linear(4, 8), torch.zeros(2, 3), torch.zeros(8, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ffn_down_restores_original_reduction_coordinates(dtype):
    torch.manual_seed(654)
    original = torch.nn.Linear(8, 4).to(dtype)
    ref = adapters.ProjectionAdapter(original, rank=2)
    with torch.no_grad():
        ref.B.normal_(std=0.1)
    ids = [0, 2, 5, 6]
    small = ShapeFaithfulLinear(8, 4, indices=ids, axis=1, bias=True).to(dtype)
    with torch.no_grad():
        small.weight.copy_(original.weight[:, ids])
        small.bias.copy_(original.bias)
    down = export_joint_split.SplitAdapter(small, ref.A.detach()[:, ids].clone(), ref.B.detach().clone())
    deployed = export_joint_split.RestoreInput(down, 8, ids)
    x = torch.randn(1, 11, 8).to(dtype)
    mask = torch.zeros(8, dtype=dtype)
    mask[ids] = 1
    assert torch.equal(deployed(x[..., ids]), ref(x * mask))
