from __future__ import annotations

import torch

from scripts.prune.evaluate import metrics


def test_psnr_of_identical_frames_is_infinite():
    pixels = torch.rand(4, 3, 16, 16)
    assert metrics.psnr(pixels, pixels) == float("inf")
    assert abs(metrics.ssim_global(pixels, pixels) - 1.0) < 1e-4


def test_psnr_matches_closed_form_for_known_mse():
    x = torch.zeros(2, 3, 8, 8)
    y = torch.full_like(x, 0.1)
    assert abs(metrics.psnr(x, y) - 20.0) < 1e-4


def test_as_bchw_accepts_every_decoder_layout():
    assert metrics._as_bchw(torch.rand(1, 3, 5, 8, 8)).shape == (5, 3, 8, 8)
    assert metrics._as_bchw(torch.rand(5, 8, 8, 3)).shape == (5, 3, 8, 8)


def test_t2_slope_is_negative_for_degrading_rollout():
    # Start away from an exact match: PSNR(identical) is intentionally infinity,
    # which has no finite regression slope.
    rows = [{"chunk": index, "pred": torch.full((1, 3, 8, 8), 0.51 + 0.02 * index), "teacher": torch.full((1, 3, 8, 8), 0.5)} for index in range(6)]
    assert metrics.t2(rows)["psnr_slope_db_per_100_chunks"] < 0
