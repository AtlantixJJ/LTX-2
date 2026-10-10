"""Measure mask alignment for corpus review; see ``tests/test_qa.py``.

Read two single-channel masks in the same recorded crop. Threshold both at 128
by default, count their binary intersection and union, and return their ratio.
Two empty masks return one. The result measures overlap, not motion or image
quality, and never weights a training loss. This NumPy-only owner produces no
files and never changes the masks or their geometry.
"""

from __future__ import annotations

import numpy as np


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray, threshold: int = 128) -> float:
    """Binary IoU between two single-channel masks, thresholded at ``threshold``.

    128 matches the dataset's own ``mask_source == "estimated"`` threshold (``meta.json``
    stage ``s6a``), so the render-alpha-vs-``mask.mp4`` IoU this drives is comparable to how
    the dataset's own mask was made. This is the decisive alignment number: it tests whether
    ``K_raw`` (the pose's calibrated intrinsics) actually lands the render on the subject.
    """
    a = mask_a >= threshold
    b = mask_b >= threshold
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0  # both empty -- vacuously aligned
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(union)
