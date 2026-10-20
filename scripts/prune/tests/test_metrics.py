from __future__ import annotations

import torch

from scripts.prune.evaluate import metrics


def test_as_bchw_accepts_every_decoder_layout() -> None:
    assert metrics._as_bchw(torch.rand(1, 3, 5, 8, 8)).shape == (5, 3, 8, 8)
    assert metrics._as_bchw(torch.rand(5, 8, 8, 3)).shape == (5, 3, 8, 8)
