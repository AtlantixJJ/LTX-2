"""CPU-only, no data/GPU needed. Run with: python -m pytest scripts/onestep_avatar/tests"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

from scripts.onestep_avatar import geometry


class TestUnionBbox(unittest.TestCase):
    def test_ignores_invalid_and_nan_rows(self):
        xyxy = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [100.0, 100.0, 200.0, 200.0],  # invalid, should be ignored
                [np.nan, 0.0, 10.0, 10.0],  # NaN despite valid=True, should be ignored
                [5.0, 5.0, 20.0, 30.0],
            ]
        )
        valid = np.array([True, False, True, True])
        self.assertEqual(geometry.union_bbox(xyxy, valid), (0.0, 0.0, 20.0, 30.0))

    def test_raises_when_nothing_valid(self):
        xyxy = np.array([[np.nan, 0.0, 10.0, 10.0]])
        valid = np.array([True])
        with self.assertRaises(ValueError):
            geometry.union_bbox(xyxy, valid)


class TestSquareCropBox(unittest.TestCase):
    def test_square_and_centred(self):
        box = geometry.square_crop_box((0.0, 0.0, 100.0, 40.0), pad_factor=1.0)
        x0, y0, x1, y1 = box
        self.assertAlmostEqual(x1 - x0, y1 - y0)  # square
        self.assertAlmostEqual(x1 - x0, 100.0)  # side = max(w, h), no padding
        self.assertAlmostEqual((x0 + x1) / 2.0, 50.0)  # centred on the union's centre
        self.assertAlmostEqual((y0 + y1) / 2.0, 20.0)

    def test_padding_scales_the_side(self):
        box = geometry.square_crop_box((0.0, 0.0, 100.0, 100.0), pad_factor=1.20)
        x0, y0, x1, y1 = box
        self.assertAlmostEqual(x1 - x0, 120.0)


class TestCropWithPadding(unittest.TestCase):
    def test_pure_interior_crop_is_a_plain_slice(self):
        frame = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        out = geometry.crop_with_padding(frame, (10, 20, 30, 40))
        np.testing.assert_array_equal(out, frame[20:40, 10:30])

    def test_pads_outside_the_frame_with_pad_value(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        out = geometry.crop_with_padding(frame, (-10, -10, 40, 40), pad_value=255)
        self.assertEqual(out.shape, (50, 50, 3))
        # the top-left 10x10 corner is outside the source frame -> padded white
        np.testing.assert_array_equal(out[:10, :10], 255)
        # the rest overlaps the (all-zero) source frame
        np.testing.assert_array_equal(out[10:, 10:], 0)

    def test_box_entirely_outside_frame_is_all_pad(self):
        frame = np.zeros((50, 50), dtype=np.uint8)
        out = geometry.crop_with_padding(frame, (100, 100, 150, 150), pad_value=0)
        self.assertEqual(out.shape, (50, 50))
        self.assertTrue((out == 0).all())

    def test_single_channel_mask_shape_preserved(self):
        mask = np.full((20, 20), 200, dtype=np.uint8)
        out = geometry.crop_with_padding(mask, (5, 5, 15, 15), pad_value=0)
        self.assertEqual(out.shape, (10, 10))
        self.assertTrue((out == 200).all())


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize(
    ("union", "expected"),
    [
        # fits at 1.20
        ((500.0, 1000.0, 1400.0, 3000.0), (0.0, 800.0, 2400.0, 3200.0)),
        # wider than the canvas: capped to the short side
        ((100.0, 800.0, 2900.0, 3600.0), (0.0, 700.0, 3000.0, 3700.0)),
        # hard against the top-left corner: shifted, not clipped
        ((0.0, 0.0, 2400.0, 2400.0), (0.0, 0.0, 2880.0, 2880.0)),
        # hard against the bottom-right corner
        ((2600.0, 3600.0, 2990.0, 4090.0), (2412.0, 3508.0, 3000.0, 4096.0)),
    ],
)
def test_the_canonical_box_is_pinned_to_exact_values(union, expected):
    """A golden test, and the stakes are why: this box is what every capture latent on disk
    was encoded with. Changing the rule silently re-crops the corpus -- guides and targets
    would still *look* fine and would no longer be the same pixels.

    Until 2026-09-15 this test instead compared two transcriptions of the rule, because
    ``precompute.py`` lived in another tree and another conda env. It now calls this very
    function, so the comparison would be tautological; exact values are what still bites.
    """
    box = geometry.canonical_crop_box(np.array([union], dtype=np.float64), np.array([True]), 3000, 4096, 1.20)
    assert box == expected


def test_canonical_box_is_always_inside_the_canvas():
    """No synthetic padding ever reaches the loss target: whatever the bbox, the box is a
    square that lies wholly within the original camera canvas."""
    width, height = 3000, 4096
    for union in [(-500.0, -200.0, 400.0, 900.0), (2800.0, 3900.0, 3400.0, 4500.0)]:
        x0, y0, x1, y1 = geometry.canonical_crop_box(
            np.array([union], dtype=np.float64), np.array([True]), width, height
        )
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height
        assert (x1 - x0) == (y1 - y0)


def test_effective_pad_factor_flags_a_clipped_subject():
    """``< 1.0`` is the 0.4 % tail where the subject itself is wider than the canvas -- the
    box clips the person, and the view must be excluded rather than trained on."""
    width, height = 3000, 4096
    fits = np.array([[500.0, 1000.0, 1400.0, 3000.0]])
    too_wide = np.array([[0.0, 500.0, 3200.0, 3700.0]])
    valid = np.array([True])

    box = geometry.canonical_crop_box(fits, valid, width, height)
    assert geometry.effective_pad_factor(box, fits, valid) == pytest.approx(1.20, abs=1e-3)

    box = geometry.canonical_crop_box(too_wide, valid, width, height)
    assert geometry.effective_pad_factor(box, too_wide, valid) < 1.0
