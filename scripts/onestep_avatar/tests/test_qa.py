"""CPU-only, no data/GPU needed. Run with: python -m unittest scripts.onestep_avatar.tests.test_qa"""

from __future__ import annotations

import unittest

import numpy as np

from scripts.onestep_avatar import qa


class TestMaskIoU(unittest.TestCase):
    def test_identical_masks_give_iou_one(self):
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        self.assertAlmostEqual(qa.mask_iou(mask, mask), 1.0)

    def test_disjoint_masks_give_iou_zero(self):
        a = np.array([[255, 0], [0, 0]], dtype=np.uint8)
        b = np.array([[0, 255], [0, 0]], dtype=np.uint8)
        self.assertAlmostEqual(qa.mask_iou(a, b), 0.0)

    def test_partial_overlap(self):
        a = np.array([1, 1, 1, 0], dtype=np.uint8) * 255
        b = np.array([1, 1, 0, 0], dtype=np.uint8) * 255
        self.assertAlmostEqual(qa.mask_iou(a, b), 2.0 / 3.0)

    def test_both_empty_is_vacuously_one(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        self.assertAlmostEqual(qa.mask_iou(mask, mask), 1.0)

    def test_threshold_matches_dataset_convention(self):
        a = np.array([127, 128], dtype=np.uint8)
        b = np.array([128, 128], dtype=np.uint8)
        # a's first pixel falls just below threshold=128, b's does not
        self.assertAlmostEqual(qa.mask_iou(a, b, threshold=128), 0.5)


if __name__ == "__main__":
    unittest.main()
