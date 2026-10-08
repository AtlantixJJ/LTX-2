"""Real tiny native CPU attention/cache controls, without checkpoint or CUDA."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from scripts.onestep_avatar import continuation_check as check
from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar.model import causal, common
from scripts.onestep_avatar.tests.test_causal_core import CHANNELS, DEVICE, EDGE, FPS, SCALE, _context, _model
from scripts.onestep_avatar.training.config import CausalSettings


def inputs() -> tuple:
    geometry = causal.CausalGeometry(SCALE, 2, 8)
    grid = common.ClipGrid.build(17, EDGE, EDGE, FPS, geometry, device=DEVICE,
                                 dtype=torch.float32, latent_channels=CHANNELS)
    generator = torch.Generator().manual_seed(72)
    shape = (1, grid.latent_frames * grid.tokens_per_latent_frame, CHANNELS)
    capture, guide, noise = (torch.randn(shape, generator=generator) for _ in range(3))
    return grid, geometry, capture, guide, noise, _context()


@torch.no_grad()
def captured(root: Path, *, sigma: float = 1.0, teacher_forcing: bool = False,
             prompt_adaln: bool = True) -> tuple:
    model = _model(prompt_adaln=prompt_adaln)
    grid, geometry, capture, guide, noise, context = inputs()
    predict = common.denoised_from_velocity_model(model)
    recorder = check.CacheRecorder(predict, grid, geometry, root, sigma=sigma, capture=capture,
                                  guide=guide, noise=noise, context=context, teacher_forcing=teacher_forcing)
    arguments = {"transformer": model, "geometry": geometry, "schedule": [sigma, 0.0], "seed": 42,
                 "epsilon": noise, "teacher_tokens": capture, "teacher_forcing": teacher_forcing}
    output, counts = causal.sample(recorder, context, grid, guide,
                                   capture[:, :grid.tokens_per_latent_frame], **arguments)
    ordinary, ordinary_counts = causal.sample(predict, context, grid, guide,
                                              capture[:, :grid.tokens_per_latent_frame], **arguments)
    recorder.finish(len(model.transformer_blocks))
    assert torch.equal(output, ordinary)
    assert counts == ordinary_counts == {"denoise_calls": 8, "prime_calls": 0,
                                         "refresh_calls": 8, "model_calls": 16}
    assert not torch.cuda.is_initialized()
    return model, predict, grid, geometry, recorder, context


@pytest.mark.parametrize("sigma", [1.0, 0.909375])
@pytest.mark.parametrize("teacher_forcing", [False, True])
def test_native_capture_reference_preserves_rollout_and_exact_history(
    tmp_path: Path, sigma: float, teacher_forcing: bool,
) -> None:
    model, predict, grid, geometry, recorder, context = captured(
        tmp_path, sigma=sigma, teacher_forcing=teacher_forcing)
    assert [row["retained_frames"] for row in recorder.snapshots] == [[0, 1, 2], [0, *range(3, 11)]]
    assert [row["prefix_tokens"] for row in recorder.snapshots] == [12, 36]
    original = [(block.attn1.attention_function, block.attn1.masked_attention_function)
                for block in model.transformer_blocks]
    values = [check.load_inputs(tmp_path, row) for row in recorder.snapshots]
    assert torch.count_nonzero(values[0]["clean_history"][:, 12:]) == 0
    assert torch.count_nonzero(values[1]["clean_history"][:, 44:]) == 0
    assert torch.equal(values[0]["c0"], values[0]["capture"][:, :4])
    for row, saved in zip(recorder.snapshots, values, strict=True):
        if teacher_forcing:
            assert torch.equal(saved["clean_history"][:, :row["span"][0] * 4],
                               saved["capture"][:, :row["span"][0] * 4])
        for layer in row["layers"]:
            cached = torch.load(check.checked_path(tmp_path, layer["file"]), weights_only=True)
            assert cached["k"].shape == cached["v"].shape == (1, row["prefix_tokens"], model.inner_dim)
    reference_output = tmp_path / "reference"
    reference_output.mkdir()
    with evaluate.measure_calls(model) as counts:
        observations, blocks = check.reference_blocks(
            predict, model, grid, geometry, tmp_path, recorder.snapshots,
            context=context, sigma=sigma, output_root=reference_output)
    assert counts["model_calls"] == 2
    assert [(row["phase"], row["layer"]) for row in observations] == [
        ("before", 0), ("before", 1), ("after", 0), ("after", 1)]
    assert [row["input_tensors"]["clean_history"] for row in blocks] == [row["tensors"]["clean_history"]
                                                                       for row in recorder.snapshots]
    for block in blocks:
        output = torch.load(check.checked_path(reference_output, block["reference_prediction"]), weights_only=True)
        assert not output.requires_grad
        assert check.tensor_record(output) == block["reference_prediction_tensor"]
    assert all((block.attn1.attention_function, block.attn1.masked_attention_function) == previous
               for block, previous in zip(model.transformer_blocks, original, strict=True))
    assert not torch.cuda.is_initialized()


def test_observer_reads_real_post_normalization_and_rope_kernel_inputs(tmp_path: Path) -> None:
    model, predict, grid, geometry, recorder, context = captured(tmp_path)
    attention = model.transformer_blocks[0].attn1
    actual, raw, normalized, projected_v = [], [], [], []
    original = attention.masked_attention_function

    def operation(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int,
                  *arguments: object) -> torch.Tensor:
        actual.append((k.detach().clone(), v.detach().clone()))
        return original(q, k, v, heads, *arguments)

    attention.masked_attention_function = operation
    handles = [attention.to_k.register_forward_hook(lambda _m, _a, out: raw.append(out.detach().clone())),
               attention.k_norm.register_forward_hook(lambda _m, _a, out: normalized.append(out.detach().clone())),
               attention.to_v.register_forward_hook(lambda _m, _a, out: projected_v.append(out.detach().clone()))]
    try:
        observations, _ = check.reference_blocks(predict, model, grid, geometry, tmp_path,
                                                  recorder.snapshots, context=context, sigma=1.0)
    finally:
        attention.masked_attention_function = original
        for handle in handles:
            handle.remove()
    assert len(actual) == len(raw) == len(normalized) == len(projected_v) == 2
    for phase, (k, v), projection, norm, v_projection, snapshot in zip(
        ("before", "after"), actual, raw, normalized, projected_v, recorder.snapshots, strict=True,
    ):
        row = next(row for row in observations if row["phase"] == phase and row["layer"] == 0)
        prefix = snapshot["prefix_tokens"]
        assert row["k"]["reference_sha256"] == evaluate.tensor_sha256(k[:, :prefix])
        assert row["v"]["reference_sha256"] == evaluate.tensor_sha256(v[:, :prefix])
        assert not torch.equal(k[:, :prefix], projection[:, :prefix])
        assert not torch.equal(k[:, :prefix], norm[:, :prefix])
        assert torch.equal(v[:, :prefix], v_projection[:, :prefix])


def test_changed_snapshot_bytes_fail_before_native_forward(tmp_path: Path) -> None:
    model, predict, grid, geometry, recorder, context = captured(tmp_path)
    path = check.checked_path(tmp_path, recorder.snapshots[1]["layers"][1]["file"])
    path.write_bytes(path.read_bytes() + b"changed")
    with evaluate.measure_calls(model) as counts, pytest.raises(ValueError, match="missing, changed"):
        check.reference_blocks(predict, model, grid, geometry, tmp_path, recorder.snapshots,
                               context=context, sigma=1.0)
    assert counts["model_calls"] == 0


def test_changed_input_tensor_identity_fails_before_native_forward(tmp_path: Path) -> None:
    model, predict, grid, geometry, recorder, context = captured(tmp_path)
    recorder.snapshots[0]["tensors"]["clean_history"]["sha256"] = "0" * 64
    with evaluate.measure_calls(model) as counts, pytest.raises(ValueError, match="input tensors changed"):
        check.reference_blocks(predict, model, grid, geometry, tmp_path, recorder.snapshots,
                               context=context, sigma=1.0)
    assert counts["model_calls"] == 0


def test_hook_restoration_and_partial_observations_on_native_failure(tmp_path: Path) -> None:
    model, predict, grid, geometry, recorder, context = captured(tmp_path)
    first = model.transformer_blocks[0].attn1
    second = model.transformer_blocks[1].attn1
    second_original = second.masked_attention_function

    def fail(*_arguments: object) -> torch.Tensor:
        raise RuntimeError("controlled native operation failure")

    second.masked_attention_function = fail
    originals = [(block.attn1.attention_function, block.attn1.masked_attention_function)
                 for block in model.transformer_blocks]
    output = tmp_path / "reference"
    output.mkdir()
    try:
        with pytest.raises(RuntimeError, match="controlled native"):
            check.reference_blocks(predict, model, grid, geometry, tmp_path, recorder.snapshots,
                                   context=context, sigma=1.0, output_root=output)
        assert all((block.attn1.attention_function, block.attn1.masked_attention_function) == previous
                   for block, previous in zip(model.transformer_blocks, originals, strict=True))
        assert first.masked_attention_function is originals[0][1]
        import json  # noqa: PLC0415 -- read preserved failed progress
        partial = json.loads((output / "reference_progress.json").read_text())
        assert [row["layer"] for row in partial["layerwise"]] == [0, 1]
        assert "reference_blocks" not in partial
    finally:
        second.masked_attention_function = second_original


def test_chunk_metrics_hash_full_values_with_bounded_buffers() -> None:
    generator = torch.Generator().manual_seed(18)
    right = torch.randn((1, 1000, 256), generator=generator)
    left = right.clone()
    left[:, 750:, :4] += 0.125
    result = check.compare_chunks(left, right)
    delta = left.double() - right.double()
    assert result["cached_sha256"] == evaluate.tensor_sha256(left)
    assert result["reference_sha256"] == evaluate.tensor_sha256(right)
    assert result["rms_delta"] == pytest.approx(float(delta.square().mean().sqrt()))
    assert result["relative_l2"] == pytest.approx(float(delta.norm() / right.double().norm()))
    assert result["max_abs_delta"] == float(delta.abs().max())
    assert not result["bit_identical"]
    zero = check.compare_chunks(torch.zeros(1, 4, 8), torch.zeros(1, 4, 8))
    assert zero["bit_identical"]
    assert zero["reference_norm_zero"]
    assert zero["relative_l2"] is None


@pytest.mark.parametrize("mutate", [lambda tensor: tensor.double(),
                                   lambda tensor: tensor[:, :1],
                                   lambda tensor: tensor.fill_(float("nan"))])
def test_invalid_native_comparison_is_rejected(mutate: Callable) -> None:
    reference = torch.ones(1, 3, 8)
    with pytest.raises(ValueError, match=r"differs|nonfinite"):
        check.compare_chunks(mutate(reference.clone()), reference)


@pytest.mark.parametrize("sigma", [1.0, 0.909375])
@pytest.mark.parametrize("history_mode", ["cache", "recompute"])
@pytest.mark.parametrize("boundary", [3, 11])
def test_prepared_future_controls_use_native_public_sampler(
    sigma: float, history_mode: str, boundary: int,
) -> None:
    model = _model(prompt_adaln=True)
    grid, _geometry, capture, guide, noise, context = inputs()
    changed = noise.clone()
    changed[:, boundary * grid.tokens_per_latent_frame:] += 0.125
    with evaluate.measure_calls(model) as measured:
        _outputs, result = evaluate.probe_future_noise(
            model, context, grid, capture, guide, noise, changed, change_start_frame=boundary,
            mode="causal", mode_settings=CausalSettings(block_latent_frames=2, context_latent_frames=8),
            guide_mode="d1", schedule=[sigma, 0.0], seed=42, history_mode=history_mode,
            kv_source="refresh", predict_x0=common.denoised_from_velocity_model(model))
    ordinary = 16 if history_mode == "cache" else 8
    assert measured["model_calls"] == ordinary * 2
    assert all(row["call_counts"]["model_calls"] == ordinary for row in result["records"])
    assert result["earlier_output_bit_identical"]
    assert result["earlier_output_max_abs_delta"] == 0
    assert result["later_output_max_abs_delta"] > 0
    assert not torch.cuda.is_initialized()
