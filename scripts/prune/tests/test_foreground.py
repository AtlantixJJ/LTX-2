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


def test_capture_only_repair_uses_canonical_crop_owner(tmp_path, monkeypatch):
    from scripts.onestep_avatar.corpus import dataset, mask_video, precompute
    from scripts.prune.data.foreground import repair_canonical_mask

    view = tmp_path / "view"
    view.mkdir()
    from types import SimpleNamespace

    monkeypatch.setattr(
        dataset.CaptureManifest, "load", lambda _: SimpleNamespace(box_for=lambda _: (0.0, 0.0, 64.0, 64.0))
    )
    expected = np.full((17, 256, 256), 255, dtype=np.uint8)
    calls = []
    monkeypatch.setattr(precompute, "read_cropped_capture_masks", lambda *args: calls.append(args) or expected)
    monkeypatch.setattr(
        mask_video, "write_mask_video", lambda pixels, path, **kwargs: path.write_bytes(pixels[:1].tobytes())
    )
    record = {"box_xyxy": [0.0, 0.0, 64.0, 64.0], "pixel_frames": 17, "fps": 30}
    path = repair_canonical_mask(view, record, tmp_path)
    assert path.name == dataset.CAPTURE_MASK_CROP_NAME and path.is_file() and len(calls) == 1
    repair_canonical_mask(view, record, tmp_path)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="crop"):
        repair_canonical_mask(view, {**record, "box_xyxy": [1, 2, 3, 4]}, tmp_path)
