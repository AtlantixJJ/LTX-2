"""The historical reader must reject altered state and nonbinary endpoints."""

import json

import pytest
import torch

from scripts.prune.core import provenance, records
from scripts.prune.evaluate.historical_clip import monitor_tokens, read_checkpoint


def fixture(tmp_path):
    state = dict(
        format="head_learning_state_v1",
        identity=dict(study="old"),
        update=600,
        draw_cursor=2400,
        alpha=torch.tensor([[1.0, 0.0]]),
        adapters={"x": torch.ones(2)},
        progressive=dict(event_stage="post_reset", finalization_start=600, free=[[True, False]]),
    )
    path = tmp_path / "state.pt"
    save(path, state)
    return path, state


def save(path, state):
    torch.save(state, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            dict(
                sha256=provenance.file_sha256(path),
                identity=state["identity"],
                alpha_sha256=records.tensor_hash(state["alpha"]),
                tensor_hashes={k: records.tensor_hash(v) for k, v in state["adapters"].items()},
                progressive_sha256=records.digest(state["progressive"]),
            )
        )
    )


def test_binary_post_reset_endpoint(tmp_path):
    path, state = fixture(tmp_path)
    assert torch.equal(read_checkpoint(path, dict(identity="old"))["alpha"], state["alpha"])
    with pytest.raises(ValueError, match="identity"):
        read_checkpoint(path, dict(identity="another"))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        read_checkpoint(path, dict(identity="old"))


@pytest.mark.parametrize("field", ["update", "draw_cursor", "alpha", "policy"])
def test_rehashed_invalid_endpoint(tmp_path, field):
    path, state = fixture(tmp_path)
    if field == "alpha":
        state["alpha"][0, 0] = 0.9
    elif field == "policy":
        state["progressive"]["event_stage"] = "pre_reset"
    else:
        state[field] -= 1
    save(path, state)
    with pytest.raises(ValueError):
        read_checkpoint(path, dict(identity="old"))


def test_monitor_dense_and_token_layout_preserve_coordinates():
    dense = torch.arange(2 * 3 * 4 * 5).reshape(1, 2, 3, 4, 5)
    tokens = monitor_tokens(dense, [2, 3, 4, 5])
    assert tokens.shape == (1, 60, 2)
    for frame in range(3):
        for height in range(4):
            for width in range(5):
                for channel in range(2):
                    assert (
                        tokens[0, (frame * 4 + height) * 5 + width, channel] == dense[0, channel, frame, height, width]
                    )
    assert torch.equal(monitor_tokens(tokens, [2, 3, 4, 5]), tokens)
    with pytest.raises(ValueError, match="shape"):
        monitor_tokens(dense, [2, 3, 5, 4])
    with pytest.raises(ValueError, match="finite"):
        monitor_tokens(torch.full((1, 60, 2), float("nan")), [2, 3, 4, 5])
