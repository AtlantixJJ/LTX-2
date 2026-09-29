from __future__ import annotations

import torch

from scripts.prune.evaluate.phase1_gates import _stitch


def test_stitch_keeps_earlier_windows_overlap():
    first = torch.arange(25).reshape(25, 1, 1, 1)
    second = torch.arange(100, 125).reshape(25, 1, 1, 1)
    third = torch.arange(200, 225).reshape(25, 1, 1, 1)
    stitched = _stitch([first, second, third], [(0, 25), (16, 41), (32, 57)])
    expected = torch.cat([first, second[9:], third[9:]])
    assert torch.equal(stitched, expected)
    assert stitched.shape[0] == 57
