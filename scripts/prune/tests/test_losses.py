from __future__ import annotations

import torch
import pytest

from scripts.prune import chunk_states, losses


def test_zero_error_is_zero_loss(record_paths):
    state, x0, meta = chunk_states.load_record(record_paths[0])
    mask = chunk_states.chunk_token_mask(state, meta)
    assert float(losses.x0_loss(x0, x0, state, mask)) == 0.0
    assert float(losses.rel_l2(x0, x0, state, mask)) == 0.0


def test_masking_confines_loss_to_chunk(record_paths):
    state, x0, meta = chunk_states.load_record(record_paths[0])
    mask = chunk_states.chunk_token_mask(state, meta)
    keyframe = (mask[0, :, 0] == 0).nonzero()[0].item()
    assert keyframe == 0 and bool(state.denoise_mask[0, keyframe, 0])
    prediction = x0.clone()
    prediction[0, keyframe] += 100.0
    assert float(losses.x0_loss(prediction, x0, state, mask)) == 0.0
    assert float(losses.x0_loss(prediction, x0, state, None)) > 1000.0


def test_rel_l2_is_scale_relative(record_paths):
    state, x0, meta = chunk_states.load_record(record_paths[0])
    mask = chunk_states.chunk_token_mask(state, meta)
    a = float(losses.rel_l2(x0 * 1.10, x0, state, mask))
    b = float(losses.rel_l2(x0 * 2.20, x0 * 2.0, state, mask))
    assert abs(a - b) < 1e-5 and abs(a - 0.1) < 1e-3


def test_shape_mismatch_raises(record_paths):
    state, x0, _ = chunk_states.load_record(record_paths[0])
    with pytest.raises(ValueError, match="!= target"):
        losses.x0_loss(x0[:, :-1], x0, state, None)
