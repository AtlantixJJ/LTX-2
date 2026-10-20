from __future__ import annotations

import pytest

from scripts.prune.core import geometry, model_registry


def test_pixel_frames_follow_the_temporal_grid(model: model_registry.RefinerModel) -> None:
    assert [geometry.pixel_frames_for(n, model.scale_factors) for n in (1, 2, 3, 4)] == [1, 9, 17, 25]
    with pytest.raises(ValueError):
        geometry.pixel_frames_for(0, model.scale_factors)
