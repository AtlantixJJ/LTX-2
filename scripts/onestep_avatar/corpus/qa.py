"""Alignment metrics for the guidance-corpus review gate (SSB1 / SS4.5).

Pure and dependency-free (numpy only); see ``tests/test_qa.py``.
"""

from __future__ import annotations

import numpy as np


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray, threshold: int = 128) -> float:
    """Binary IoU between two single-channel masks, thresholded at ``threshold``.

    128 matches the dataset's own ``mask_source == "estimated"`` threshold (``meta.json``
    stage ``s6a``), so the render-alpha-vs-``mask.mp4`` IoU this drives is comparable to how
    the dataset's own mask was made. This is SSB1's decisive number: it tests whether
    ``K_raw`` (the pose's calibrated intrinsics) actually lands the render on the subject.
    """
    a = mask_a >= threshold
    b = mask_b >= threshold
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0  # both empty -- vacuously aligned
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(union)
