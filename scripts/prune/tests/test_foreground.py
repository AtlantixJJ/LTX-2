import numpy as np
import pytest
import torch

from scripts.prune.data.foreground import region_coverage


def test_region_expand_crop_mapping_and_absence():
    mask = np.full((17, 256, 256), 255, dtype=np.uint8)
    record = {"box_xyxy": [100, 200, 1124, 1224], "edge": 1024}
    schema = {
        "format": "gt_region_boxes_v1",
        "coordinates": "source_pixels",
        "reviewed_on_gt": True,
        "producer_sha256": "x",
    }
    tracks = {"boxes": [[300, 400, 364, 464]] * 17, "valid": [True] * 17}
    weights, result = region_coverage(tracks, mask, record, schema=schema)
    assert weights.shape == (3, 32, 32) and weights[1:].sum() > 0
    assert result["crop_boxes"][0] == [184.0, 184.0, 280.0, 280.0]
    assert result["valid_frames"] == 16
    tracks["valid"] = [False] * 17
    empty, result = region_coverage(tracks, mask, record, schema=schema)
    assert torch.count_nonzero(empty) == 0 and result["absent"]
    with pytest.raises(ValueError, match="schema"):
        region_coverage(tracks, mask, record, schema={})
