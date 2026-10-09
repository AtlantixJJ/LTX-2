"""Fixtures backed by the deployed checkpoint and calibration cache."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def model():
    """The real 2.5 registry entry; resolution reads checkpoint metadata only."""
    from scripts.prune.core import model_registry

    try:
        return model_registry.resolve("2.5")
    except SystemExit as exc:
        pytest.skip(f"2.5 checkpoint not on disk: {exc}")


@pytest.fixture
def block():
    """A small real production transformer block, not a test double."""
    from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig

    torch.manual_seed(0)
    transformer_block = BasicAVTransformerBlock(
        video=TransformerConfig(dim=32, heads=4, d_head=8, context_dim=32)
    )

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([transformer_block])

    return Model()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Only declared native tests require visible CUDA; CPU gates load no weights."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="native GPU test requires visible CUDA; CPU gate hides CUDA")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip)
