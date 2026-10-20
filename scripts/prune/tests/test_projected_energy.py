"""Exact projected head energy against independent projections."""

import pytest
import torch

from scripts.prune.score.estimators import projected_energy
from scripts.prune.score.token_sampling import sample_indices


def test_covariance_and_direct_projection():
    torch.manual_seed(42)
    a, w = torch.randn(2, 19, 12), torch.randn(9, 12)
    indices = torch.tensor([2, 5, 7, 11, 18])
    result = projected_energy(a, w, 3, indices, token_chunk=2)
    expected = torch.stack(
        [
            (a[:, indices, h * 4 : (h + 1) * 4] @ w[:, h * 4 : (h + 1) * 4].T).double().square().sum(-1).mean()
            for h in range(3)
        ]
    )
    torch.testing.assert_close(result, expected, rtol=2e-6, atol=2e-6)
    a = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]])
    assert projected_energy(a, torch.ones(1, 2), 1, torch.arange(2)).item() == 0


@pytest.mark.parametrize("frames", [18, 28])
def test_complete_generated_tokens(frames):
    indices = sample_indices(frames * 1024, 32, 32, 1, torch.device("cpu"))
    assert torch.equal(indices, torch.arange(1024, frames * 1024))


def test_all_one_and_single_head_projection_with_bias(block):
    from scripts.prune.score import hooks

    attention = block.transformer_blocks[0].attn1
    x = torch.randn(1, 7, 32)
    dense = attention.to_out[0](x)
    with hooks.attach_head_masks(block, requires_grad=False):
        assert torch.equal(attention.to_out[0](x), dense)
    masks = {"0.attn1": torch.tensor([0.0, 1.0, 0.0, 0.0]), "0.attn2": torch.ones(4)}
    with hooks.attach_head_masks(block, masks, requires_grad=False):
        actual = attention.to_out[0](x)
    projection = attention.to_out[0]
    expected = x[..., 8:16] @ projection.weight[:, 8:16].T
    if projection.bias is not None:
        expected = expected + projection.bias
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch,heads,width,tokens,chunk", [(2, 3, 4, 19, 2), (1, 5, 8, 37, 7)])
def test_batched_products_match_scalar_fp32_reference(batch, heads, width, tokens, chunk):
    torch.manual_seed(142)
    a = torch.randn(batch, tokens, heads * width)
    w = torch.randn(17, heads * width)
    indices = torch.arange(1, tokens)
    expected = []
    for h in range(heads):
        local_w = w[:, h * width : (h + 1) * width].float()
        gram = local_w.T @ local_w
        total = torch.zeros((), dtype=torch.float64)
        for selected in indices.split(chunk):
            local_a = a[:, selected, h * width : (h + 1) * width].float()
            total += ((local_a @ gram) * local_a).sum(-1).double().sum()
        expected.append(total / (batch * indices.numel()))
    torch.testing.assert_close(
        projected_energy(a, w, heads, indices, token_chunk=chunk), torch.stack(expected), rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_deferred_invalid_energy_check(invalid):
    a = torch.ones(1, 7, 8)
    a[0, 6, 7] = invalid
    with pytest.raises(ValueError, match="nonfinite"):
        projected_energy(a, torch.ones(3, 8), 2, torch.arange(1, 7), token_chunk=2)


def test_reject_duplicate_tokens():
    with pytest.raises(ValueError):
        projected_energy(torch.ones(1, 3, 2), torch.ones(2, 2), 1, torch.tensor([1, 1]))
