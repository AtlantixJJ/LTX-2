"""Reusable RGB measurements validate evidence and weight frames consistently."""

import pytest
import torch

from scripts.onestep_avatar import metrics


def test_full_rgb_score_averages_error_before_logarithm():
    reference = torch.zeros(2, 3, 2, 2)
    prediction = reference.clone()
    prediction[1] = 1
    result = metrics.rgb_metrics(prediction, reference)
    assert result['per_frame_mse'] == [0, 1]
    assert result['mse'] == 0.5 and result['psnr'] == pytest.approx(3.0102999566)
    assert not result['all_exact_match']
    exact = metrics.rgb_metrics(reference, reference)
    assert exact['psnr'] is None and exact['all_exact_match']


def test_subject_mask_preserves_dilation_and_checks_coverage(tmp_path, monkeypatch):
    import numpy as np

    from scripts.onestep_avatar.corpus import mask_video
    path = tmp_path / 'mask.mp4'
    assert metrics.subject_mask(path, 1, 12, 16) is None
    path.write_bytes(b'controlled reader fixture')
    raw = np.zeros((1, 6, 8), dtype=np.uint8)
    raw[0, 3, 4] = 255
    monkeypatch.setattr(mask_video, 'read_mask_video', lambda path: raw)
    mask = metrics.subject_mask(path, 1, 12, 16)
    expected = torch.zeros(1, 12, 16, dtype=torch.bool)
    expected[:, 2:12, 4:14] = True
    assert torch.equal(mask, expected) and int(mask.sum()) == 100
    with pytest.raises(ValueError, match='does not cover'):
        metrics.subject_mask(path, 2, 12, 16)
    with pytest.raises(ValueError, match='positive'):
        metrics.subject_mask(path, 0, 12, 16)


def test_subject_measurement_selects_only_supplied_pixels():
    reference = torch.zeros(2, 3, 2, 3)
    prediction = reference.clone()
    prediction[:, :, :, 0] = 0.5
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    mask[:, :, 0] = True
    result = metrics.subject_rgb_metrics(prediction, reference, mask)
    assert result['mse'] == 0.25 and result['psnr'] == pytest.approx(6.020599913)
    assert result['selected_pixels'] == 4 and not result['exact_match']
    mask[:, :, 0] = False
    mask[:, :, 1] = True
    result = metrics.subject_rgb_metrics(prediction, reference, mask)
    assert result['mse'] == 0 and result['psnr'] is None and result['exact_match']


def test_perceptual_scores_weight_frames_and_transform_input_range():
    prediction = torch.tensor([0, 0.5, 1]).view(3, 1, 1, 1).expand(3, 3, 2, 2)
    reference = torch.zeros_like(prediction)
    sizes = []

    def model(left, right):
        sizes.append(len(left))
        assert torch.equal(right, torch.full_like(right, -1))
        assert not torch.is_grad_enabled()
        return (left - right).square().mean(dim=(1, 2, 3), keepdim=True)

    assert metrics.lpips_distance(model, prediction, reference, torch.device('cpu'), batch=2) == pytest.approx(5 / 3)
    assert sizes == [2, 1]


@pytest.mark.parametrize('invalid', ['empty', 'nan', 'range', 'batch'])
def test_perceptual_invalid_inputs_fail_before_model(invalid):
    pixels = torch.zeros(2, 3, 2, 2)
    if invalid == 'empty':
        pixels = pixels[:0]
    elif invalid == 'nan':
        pixels[0, 0, 0, 0] = float('nan')
    elif invalid == 'range':
        pixels[0, 0, 0, 0] = 2
    with pytest.raises(ValueError):
        metrics.lpips_distance(lambda *args: pytest.fail('invalid input reached model'), pixels, pixels,
                                torch.device('cpu'), batch=0 if invalid == 'batch' else 2)
