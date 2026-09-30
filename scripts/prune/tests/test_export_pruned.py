from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.feed_forward import ShapeFaithfulLinear
from scripts.prune.score import export_pruned


@pytest.mark.parametrize("axis", [0, 1])
def test_shape_faithful_linear_preserves_original_gemm_and_gradients(axis):
    torch.manual_seed(19)
    source = torch.nn.Linear(8, 12)
    indices = [1, 3, 5]
    candidate = ShapeFaithfulLinear(8, 12, indices=indices, axis=axis, bias=True)
    with torch.no_grad():
        candidate.weight.copy_(source.weight[indices] if axis == 0 else source.weight[:, indices])
        candidate.bias.copy_(source.bias[indices] if axis == 0 else source.bias)
    x = torch.randn(2, 7, 8, requires_grad=True)
    if axis == 0:
        expected = source(x)
        mask = torch.zeros(12)
        mask[indices] = 1
        expected = expected * mask
    else:
        mask = torch.zeros(8)
        mask[indices] = 1
        expected = source(x * mask)
    actual = candidate(x)
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert candidate.weight.grad is not None
    assert "retained_indices" not in candidate.state_dict()


def test_shape_faithful_export_stores_compact_tensors_with_original_execution_geometry(tmp_path):
    source, output = tmp_path / "source.safetensors", tmp_path / "faithful.safetensors"
    _checkpoint(source)
    export_pruned.export(source, {"0.attn1": [1, 0], "0.attn2": [0, 1], "0.ff": [1, 0, 1]},
                         output, model_key="2.5", mode="compact_faithful")
    with safe_open(output, framework="pt") as handle:
        config = json.loads(handle.metadata()["config"])["transformer"]
        assert config["video_pruning_shape_faithful"] is True
        assert config["per_layer_video_attn1_heads"] == [2]
        assert config["per_layer_ff_inner_dim"] == [3]
        assert config["per_layer_video_attn2_rope_head_indices"] == [None]
        assert config["video_pruning_select_active_heads"] is False
        assert handle.get_tensor(f"{export_pruned.PREFIX}.0.attn2.to_v.weight").shape == (2, 3)
        assert handle.get_tensor(f"{export_pruned.PREFIX}.0.ff.net.2.weight").shape == (3, 2)


def _checkpoint(path):
    prefix = export_pruned.PREFIX + ".0"
    tensors = {}
    for kind in ("attn1", "attn2"):
        base = f"{prefix}.{kind}"
        for projection in ("to_q", "to_k", "to_v"):
            tensors[f"{base}.{projection}.weight"] = torch.arange(12.0).reshape(4, 3).contiguous()
        for norm in ("q_norm", "k_norm"):
            tensors[f"{base}.{norm}.weight"] = torch.arange(4.0)
        tensors[f"{base}.to_out.0.weight"] = torch.arange(12.0).reshape(3, 4).contiguous()
    tensors[f"{prefix}.ff.net.0.proj.weight"] = torch.arange(9.0).reshape(3, 3).contiguous()
    tensors[f"{prefix}.ff.net.2.weight"] = torch.arange(9.0).reshape(3, 3).contiguous()
    save_file(tensors, str(path), metadata={"config": json.dumps({"transformer": {"num_layers": 1,
                                                                                  "attention_head_dim": 2}})})


def test_structural_export_slices_and_records_original_head_ids(tmp_path):
    source, output = tmp_path / "source.safetensors", tmp_path / "pruned.safetensors"
    _checkpoint(source)
    masks = {"0.attn1": [1, 0], "0.attn2": [0, 1], "0.ff": [1, 0, 1]}
    export_pruned.export(source, masks, output, model_key="2.5", provenance_block={"mask_sha256": "abc"}, mode="compact")
    with safe_open(output, framework="pt") as handle:
        config = json.loads(handle.metadata()["config"])["transformer"]
        assert config["per_layer_video_attn1_rope_head_indices"] == [[0]]
        assert config["per_layer_video_attn2_rope_head_indices"] == [[1]]
        assert config["per_layer_ff_inner_dim"] == [2]
        assert config["pruning"]["mask_sha256"] == "abc"
        assert config["video_pruning_preserve_qk_norm"] is True
        assert handle.get_tensor(f"{export_pruned.PREFIX}.0.attn2.to_q.weight").shape == (4, 3)
        assert handle.get_tensor(f"{export_pruned.PREFIX}.0.attn2.to_v.weight").shape == (2, 3)
        assert handle.get_tensor(f"{export_pruned.PREFIX}.0.ff.net.2.weight").shape == (3, 2)


def test_invalid_mask_or_reconstruction_fails_before_output(tmp_path):
    source, output = tmp_path / "source.safetensors", tmp_path / "pruned.safetensors"
    _checkpoint(source)
    with pytest.raises(ValueError, match="invalid binary mask"):
        export_pruned.export(source, {"0.attn1": [1]}, output, model_key="2.5")
    with pytest.raises(ValueError, match="reconstruction shape"):
        export_pruned.export(source, {"0.ff": [1, 0, 1]}, output, model_key="2.5",
                             reconstruction={"0.ff": torch.zeros(2, 2)}, mode="compact")
    assert not output.exists()


def test_full_width_qk_norm_matches_functional_head_mask():
    torch.manual_seed(3)
    source = Attention(query_dim=8, context_dim=8, heads=2, dim_head=4)
    exported = Attention(query_dim=8, context_dim=8, heads=1, dim_head=4, qk_heads=2)
    exported.rope_head_indices = [1]
    with torch.no_grad():
        for name in ("to_q", "to_k", "q_norm", "k_norm"):
            getattr(exported, name).load_state_dict(getattr(source, name).state_dict())
        exported.to_v.weight.copy_(source.to_v.weight[4:])
        exported.to_v.bias.copy_(source.to_v.bias[4:])
        exported.to_out[0].weight.copy_(source.to_out[0].weight[:, 4:])
        exported.to_out[0].bias.copy_(source.to_out[0].bias)
    x = torch.randn(2, 3, 8)
    handle = source.to_out[0].register_forward_pre_hook(
        lambda _module, args: (torch.cat([torch.zeros_like(args[0][..., :4]), args[0][..., 4:]], dim=-1),)
    )
    try:
        expected = source(x)
    finally:
        handle.remove()
    assert torch.allclose(exported(x), expected, atol=1e-5, rtol=1e-5)


def test_sparse_export_preserves_full_tensors_and_records_active_units(tmp_path):
    source, output = tmp_path / "source.safetensors", tmp_path / "sparse.safetensors"
    _checkpoint(source)
    export_pruned.export(source, {"0.attn1": [1, 0], "0.attn2": [0, 1], "0.ff": [1, 0, 1]},
                         output, model_key="2.5")
    with safe_open(output, framework="pt") as handle:
        config = json.loads(handle.metadata()["config"])["transformer"]
        assert config["pruning"]["mode"] == "masked_full"
        assert config["video_pruning_select_active_heads"] is False
        assert config["per_layer_video_attn1_rope_head_indices"] == [None]
        assert config["per_layer_video_attn2_rope_head_indices"] == [None]
        assert config["per_layer_video_attn1_active_head_indices"] == [[0]]
        assert config["per_layer_video_attn2_active_head_indices"] == [[1]]
        assert config["per_layer_video_ffn_active_channels"] == [[0, 2]]
        assert handle.get_tensor(f"{export_pruned.PREFIX}.0.attn2.to_v.weight").shape == (4, 3)
        assert handle.get_tensor(f"{export_pruned.PREFIX}.0.ff.net.2.weight").shape == (3, 3)


def test_compact_ffn_only_does_not_gather_identity_rope_heads(tmp_path):
    source, output = tmp_path / "source.safetensors", tmp_path / "compact_ffn.safetensors"
    _checkpoint(source)
    export_pruned.export(source, {"0.ff": [1, 0, 1]}, output, model_key="2.5", mode="compact")
    with safe_open(output, framework="pt") as handle:
        config = json.loads(handle.metadata()["config"])["transformer"]
        assert config["per_layer_video_attn1_rope_head_indices"] == [None]
        assert config["per_layer_video_attn2_rope_head_indices"] == [None]
        assert handle.get_tensor(f"{export_pruned.PREFIX}.0.attn2.to_v.weight").shape == (4, 3)
        assert handle.get_tensor(f"{export_pruned.PREFIX}.0.ff.net.2.weight").shape == (3, 2)


def test_sparse_attention_matches_functional_mask():
    torch.manual_seed(5)
    source = Attention(query_dim=8, context_dim=8, heads=2, dim_head=4)
    sparse = Attention(query_dim=8, context_dim=8, heads=2, dim_head=4, active_head_indices=[1])
    sparse.load_state_dict(source.state_dict())
    x = torch.randn(2, 3, 8)
    handle = source.to_out[0].register_forward_pre_hook(
        lambda _module, args: (torch.cat([torch.zeros_like(args[0][..., :4]), args[0][..., 4:]], dim=-1),)
    )
    try:
        expected = source(x)
    finally:
        handle.remove()
    assert torch.allclose(sparse(x), expected, atol=1e-5, rtol=1e-5)


def test_masked_full_attention_matches_functional_mask():
    torch.manual_seed(7)
    source = Attention(query_dim=8, context_dim=8, heads=2, dim_head=4)
    exported = Attention(query_dim=8, context_dim=8, heads=2, dim_head=4,
                         active_head_indices=[1], select_active_heads=False)
    exported.load_state_dict(source.state_dict())
    x = torch.randn(2, 3, 8)
    handle = source.to_out[0].register_forward_pre_hook(
        lambda _module, args: (torch.cat([torch.zeros_like(args[0][..., :4]), args[0][..., 4:]], dim=-1),)
    )
    try:
        expected = source(x)
    finally:
        handle.remove()
    assert torch.equal(exported(x), expected)


def test_sparse_ffn_matches_functional_mask():
    torch.manual_seed(6)
    source = FeedForward(dim=4, dim_out=4, inner_dim=8)
    sparse = FeedForward(dim=4, dim_out=4, inner_dim=8, active_channels=[0, 2, 4, 6])
    sparse.load_state_dict(source.state_dict())
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
    handle = source.net[2].register_forward_pre_hook(lambda _module, args: (args[0] * mask,))
    try:
        expected = source(x)
    finally:
        handle.remove()
    assert torch.equal(sparse(x), expected)
