from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from scripts.prune.data import chunk_states


def test_real_record_token_layout(record_paths):
    state, x0, meta = chunk_states.load_record(record_paths[0])
    assert (meta.chunk_latent_frames, meta.context_latent_frames) == (1, 1)
    assert tuple(state.latent.shape) == (1, 1728, 128) and x0.shape == state.latent.shape
    assert int(state.denoise_mask.sum()) == 1152
    assert int(chunk_states.chunk_token_mask(state, meta).sum()) == 576


def test_keyframe_is_half_the_fresh_mass_at_n1(record_paths):
    state, _, meta = chunk_states.load_record(record_paths[0])
    assert int(chunk_states.chunk_token_mask(state, meta).sum()) * 2 == int(state.denoise_mask.sum())


def test_chunk_mask_is_a_strict_subset_of_denoise_mask(record_paths):
    for path in record_paths:
        state, _, meta = chunk_states.load_record(path)
        mask = chunk_states.chunk_token_mask(state, meta)
        assert torch.all(mask * state.denoise_mask.to(mask.dtype) == mask)
        assert 0 < int(mask.sum()) < int(state.denoise_mask.sum()) + 1


def test_save_load_round_trip_is_bit_exact(record_paths, tmp_path):
    state, x0, meta = chunk_states.load_record(record_paths[0])
    state_back, x0_back, meta_back = chunk_states.load_record(chunk_states.save_record(tmp_path / "r.pt", state, x0, meta))
    assert meta_back == meta and torch.equal(x0_back, x0)
    for field in ("latent", "denoise_mask", "positions", "clean_latent", "keyframes_mask"):
        assert torch.equal(getattr(state_back, field), getattr(state, field))


def test_format_1_record_is_refused_not_migrated(record_paths, tmp_path):
    payload = torch.load(record_paths[0], weights_only=True)
    payload["format"] = 1
    path = tmp_path / "old.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="format 1|cannot be migrated|expected 2"):
        chunk_states.load_record(path)


def test_save_refuses_a_state_with_no_fresh_tokens(record_paths, tmp_path):
    state, x0, meta = chunk_states.load_record(record_paths[0])
    empty = replace(state, denoise_mask=torch.zeros_like(state.denoise_mask))
    with pytest.raises(ValueError, match="no fresh tokens"):
        chunk_states.save_record(tmp_path / "bad.pt", empty, x0, meta)
