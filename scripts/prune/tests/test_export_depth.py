"""Real small LTX models exercise physical depth export without a GPU or model payload."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel, X0Model
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator, LTXVideoOnlyModelConfigurator
from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import causal_core
from scripts.prune.checks import export_parity
from scripts.prune.core import provenance
from scripts.prune.data import whole_clip
from scripts.prune.score import export_depth


def _config(*, per_layer: bool = False) -> dict:
    config = {
        "dropout": 0.0, "attention_bias": True, "num_vector_embeds": None,
        "activation_fn": "gelu-approximate", "num_embeds_ada_norm": 1000,
        "use_linear_projection": False, "only_cross_attention": False, "cross_attention_norm": True,
        "double_self_attention": False, "upcast_attention": False, "standardization_norm": "rms_norm",
        "norm_elementwise_affine": False, "qk_norm": "rms_norm", "positional_embedding_type": "rope",
        "use_audio_video_cross_attention": True, "share_ff": False, "av_cross_ada_norm": True,
        "use_middle_indices_grid": True, "num_attention_heads": 2, "attention_head_dim": 4,
        "num_layers": 4, "in_channels": 8, "out_channels": 8, "cross_attention_dim": 8,
        "audio_num_attention_heads": 2, "audio_attention_head_dim": 4, "audio_in_channels": 8,
        "audio_out_channels": 8, "audio_cross_attention_dim": 8, "caption_proj_before_connector": True,
        "cross_attention_adaln": True, "use_prompt_adaln_single": False,
        "apply_gated_attention": True, "use_keyframes_abs_pos_embedding": True, "ff_bias": False,
    }
    if per_layer:
        config.update({
            "per_layer_video_attn1_heads": [2, 2, 2, 2], "per_layer_video_attn2_heads": [2, 2, 2, 2],
            "per_layer_ff_inner_dim": [32, 24, 16, 40],
            "per_layer_video_attn1_rope_head_indices": [None] * 4,
            "per_layer_video_attn2_rope_head_indices": [None] * 4,
            "per_layer_video_attn1_active_head_indices": [None] * 4,
            "per_layer_video_attn2_active_head_indices": [None] * 4,
            "per_layer_video_ffn_active_channels": [None] * 4,
        })
    return {"transformer": config}


def _model(config: dict, *, av: bool = False) -> LTXModel:
    configurator = LTXModelConfigurator if av else LTXVideoOnlyModelConfigurator
    model = configurator.from_metadata({"config": config})
    generator = torch.Generator().manual_seed(4)
    # Several production scale-shift tables use torch.empty; initialize every parameter.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return model.eval()


def _setup(tmp_path: Path, removed: list[int], *, av: bool = False, per_layer: bool = False) -> tuple:
    source = tmp_path / "source.safetensors"
    config = _config(per_layer=per_layer)
    model = _model(config, av=av)
    tensors = {"model.diffusion_model." + name: value for name, value in model.state_dict().items()}
    save_file(tensors, str(source), metadata={"config": json.dumps(config), "license": "test license"})
    baseline = {
        "whole_clip": True, "trajectory_only": False, "attention": "full_bidirectional",
        "objective": "white", "text_context": {}, "geometry": {}, "seed": 42,
        "latent_dtype": "torch.bfloat16", "sigmas": [0.725],
        "guidance": {"cfg": 1, "stg": 0, "rescale": 0, "passes_per_step": 1},
        "model": {"model_key": "2.5", "transformer_path": str(source),
                  "transformer_fingerprint": provenance.checkpoint_fingerprint(source), "video_vae_fingerprint": "vae"},
        "videos": [{"view": "calibration/views/view00", "sigma": 0.725, "schedule": [0.725, 0],
                    "artifacts": {"capture_sha256": "capture", "epsilon_sha256": "epsilon",
                                  "fps": 30, "blocks": [[0, 7]]}}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(baseline))
    artifact = export_depth.create_artifact(tmp_path, ["calibration/views/view00"], [0.725], removed)
    artifact_path = tmp_path / "depth.json"
    artifact_path.write_text(json.dumps(artifact))
    return model, source, artifact_path, baseline, config


def _reload(path: Path, *, av: bool = False) -> LTXModel:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = {"config": json.loads(handle.metadata()["config"])}
        keys = handle.keys()
        tensors = {name.removeprefix("model.diffusion_model."): handle.get_tensor(name) for name in keys}
    configurator = LTXModelConfigurator if av else LTXVideoOnlyModelConfigurator
    with torch.device("meta"):
        model = configurator.from_metadata(metadata)
    model.load_state_dict(tensors, strict=True, assign=True)
    return model.eval()


def _inputs() -> tuple[causal_core.ClipGrid, Modality, torch.Tensor]:
    geometry = causal_core.CausalGeometry(
        scale_factors=SpatioTemporalScaleFactors(time=8, height=32, width=32),
        block_latent_frames=2, context_latent_frames=16,
    )
    grid = causal_core.ClipGrid.build(
        7, 64, 64, 30, geometry, device=torch.device("cpu"), dtype=torch.float32, latent_channels=8,
    )
    generator = torch.Generator().manual_seed(6)
    capture = torch.randn(1, 28, 8, generator=generator)
    noise = torch.randn(1, 28, 8, generator=generator)
    context = torch.randn(1, 3, 8, generator=generator)
    c0 = capture[:, :4]
    state = causal_core.with_clean_prefix(causal_core.mix_block_noise(capture, noise, 0.725), c0)
    modality = causal_core.block_modality(
        grid, state, context, 0.725, token_slices=[(0, 28)], clean_prefix_tokens=4,
    )
    return grid, modality, c0


@pytest.mark.parametrize("removed", [[], [0], [1, 2], [3], [0, 3]])
def test_real_video_bidirectional_reload_matches_retained_reference(tmp_path: Path, removed: list[int]) -> None:
    model, source, artifact_path, _, config = _setup(tmp_path, removed, per_layer=True)
    output = tmp_path / "compact.safetensors"
    result = export_depth.export(source, artifact_path, output, chunk_bytes=17)
    loaded = _reload(output)
    original = model.transformer_blocks
    calls = []
    handles = [block.register_forward_hook(lambda _m, _i, _o, index=index: calls.append(index))
               for index, block in enumerate(original)]
    grid, modality, c0 = _inputs()
    try:
        with export_depth.retained_blocks(model, removed), torch.no_grad():
            expected = export_parity._forward(
                X0Model(model), grid, modality, c0,
            )
        actual = export_parity._forward(X0Model(loaded), grid, modality, c0)
    finally:
        for handle in handles:
            handle.remove()
    assert model.transformer_blocks is original
    retained = [index for index in range(4) if index not in removed]
    assert calls == retained
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()
    assert torch.equal(grid.patchify(actual)[:, :4], c0)
    assert loaded.num_blocks == len(retained)
    assert result["exported_num_layers"] == len(retained)
    with safe_open(source, framework="pt", device="cpu") as before, safe_open(output, framework="pt") as after:
        metadata = json.loads(after.metadata()["config"])["transformer"]
        before_keys, after_keys = before.keys(), after.keys()
        assert after.metadata()["license"] == "test license"
        assert metadata["num_layers"] == len(retained)
        assert metadata["pruning"]["compact_to_original"] == retained
        for key in export_depth.PER_LAYER_FIELDS:
            assert metadata[key] == [config["transformer"][key][index] for index in retained]
        for compact_index, original_index in enumerate(retained):
            for key in before_keys:
                prefix = export_depth.PREFIX + str(original_index) + "."
                if key.startswith(prefix):
                    renamed = export_depth.PREFIX + str(compact_index) + "." + key[len(prefix):]
                    assert torch.equal(after.get_tensor(renamed), before.get_tensor(key))
        blocks = {int(key[len(export_depth.PREFIX):].split(".", 1)[0]) for key in after_keys
                  if key.startswith(export_depth.PREFIX)}
        assert blocks == set(range(len(retained)))
    export_depth.verify_export(source, artifact_path, output)


@pytest.mark.parametrize("removed", [[], [0, 2, 3]])
def test_real_av_strict_reload_and_both_streams(tmp_path: Path, removed: list[int]) -> None:
    model, source, artifact_path, _, _ = _setup(tmp_path, removed, av=True)
    output = tmp_path / "av_compact.safetensors"
    result = export_depth.export(source, artifact_path, output)
    loaded = _reload(output, av=True)
    _, video, _ = _inputs()
    time = torch.arange(5, dtype=torch.float32)
    audio = Modality(
        latent=torch.randn(1, 5, 8, generator=torch.Generator().manual_seed(7)),
        sigma=torch.tensor([0.725]), timesteps=torch.full((1, 5, 1), 0.725),
        positions=torch.stack((time, time + 1), dim=-1)[None, None], context=video.context,
    )
    with export_depth.retained_blocks(model, removed), torch.no_grad():
        expected = model(video=video, audio=audio, perturbations=None)
        actual = loaded(video=video, audio=audio, perturbations=None)
    for reference, measured in zip(expected, actual, strict=True):
        torch.testing.assert_close(reference, measured, atol=1e-6, rtol=1e-6)
        assert torch.isfinite(measured).all()
    counts = result["parameter_counts"]["source"]
    assert counts["resident_video_parameters"]["elements"] < counts["checkpoint_tensors"]["elements"]
    expected_video = _model(_config())
    assert counts["resident_video_parameters"]["elements"] == sum(value.numel()
                                                                 for value in expected_video.state_dict().values())


def test_reloaded_compact_cached_forward_matches_in_memory_and_causal_reference(tmp_path: Path) -> None:
    model, source, artifact_path, _, _ = _setup(tmp_path, [0, 2])
    output = tmp_path / "cached.safetensors"
    export_depth.export(source, artifact_path, output)
    loaded = _reload(output)
    grid, modality, _ = _inputs()
    geometry = causal_core.CausalGeometry(
        scale_factors=SpatioTemporalScaleFactors(time=8, height=32, width=32),
        block_latent_frames=2, context_latent_frames=16,
    )
    clean = torch.randn_like(modality.latent)
    plan = geometry.plan(grid.latent_frames)
    with export_depth.retained_blocks(model, [0, 2]), torch.no_grad():
        caches = [causal_core.BlockCache.allocate(
            grid, geometry, num_layers=subject.num_blocks, inner_dim=subject.inner_dim,
            device=torch.device("cpu"), dtype=torch.float32,
        ) for subject in (model, loaded)]
        forwards = [causal_core.denoised_from_velocity_model(subject) for subject in (model, loaded)]
        for index, span in enumerate(plan):
            lo, hi = grid.token_span(*span)
            outputs = [causal_core.denoise_block(
                forward, grid, cache, modality.latent[:, lo:hi], modality.context, 0.725, span,
            ) for forward, cache in zip(forwards, caches, strict=True)]
            assert torch.equal(outputs[0], outputs[1])
            sequence = torch.cat([clean[:, :lo], modality.latent[:, lo:hi]], dim=1)
            spans = [(start, end, order) for order, (start, end) in enumerate(plan[:index + 1])]
            ids = causal_core.block_ids_for(spans, grid.tokens_per_latent_frame)
            timesteps = torch.zeros(1, hi, 1)
            timesteps[:, lo:] = 0.725
            reference_modality = causal_core.block_modality(
                grid, sequence, modality.context, 0.725, token_slices=[(0, hi)],
                attention_mask=causal_core.block_causal_mask(ids),
            )
            reference = forwards[1](Modality(**{**reference_modality.__dict__, "timesteps": timesteps}))
            torch.testing.assert_close(outputs[1], reference[:, lo:], atol=2e-4, rtol=2e-4)
            for forward, cache in zip(forwards, caches, strict=True):
                causal_core.refresh_block(forward, grid, cache, clean[:, lo:hi], modality.context, span)
            assert all(len(cache.caches) == 2 for cache in caches)
            for before, after in zip(caches[0].caches, caches[1].caches, strict=True):
                assert before.length == after.length == hi
                assert torch.equal(before.k[:, :hi], after.k[:, :hi])
                assert torch.equal(before.v[:, :hi], after.v[:, :hi])


def test_compact_cache_eviction_preserves_reloaded_execution(tmp_path: Path) -> None:
    model, source, artifact_path, _, _ = _setup(tmp_path, [0, 2])
    output = tmp_path / "evicted.safetensors"
    export_depth.export(source, artifact_path, output)
    loaded = _reload(output)
    grid, modality, _ = _inputs()
    geometry = causal_core.CausalGeometry(
        scale_factors=SpatioTemporalScaleFactors(time=8, height=32, width=32),
        block_latent_frames=2, context_latent_frames=2,
    )
    evicted = False
    with export_depth.retained_blocks(model, [0, 2]), torch.no_grad():
        caches = [causal_core.BlockCache.allocate(
            grid, geometry, num_layers=subject.num_blocks, inner_dim=subject.inner_dim,
            device=torch.device("cpu"), dtype=torch.float32,
        ) for subject in (model, loaded)]
        forwards = [causal_core.denoised_from_velocity_model(subject) for subject in (model, loaded)]
        for span in geometry.plan(grid.latent_frames):
            lo, hi = grid.token_span(*span)
            outputs = [causal_core.denoise_block(
                forward, grid, cache, modality.latent[:, lo:hi], modality.context, 0.725, span,
            ) for forward, cache in zip(forwards, caches, strict=True)]
            assert torch.equal(outputs[0], outputs[1])
            for forward, cache in zip(forwards, caches, strict=True):
                written_length = cache.start + hi - lo
                causal_core.refresh_block(forward, grid, cache, outputs[0], modality.context, span)
                evicted = evicted or cache.start < written_length
            assert caches[0].start == caches[1].start
            for before, after in zip(caches[0].caches, caches[1].caches, strict=True):
                assert torch.equal(before.k[:, :before.length], after.k[:, :after.length])
                assert torch.equal(before.v[:, :before.length], after.v[:, :after.length])
    assert evicted


def test_retained_model_backward_matches_reloaded_gradients(tmp_path: Path) -> None:
    model, source, artifact_path, _, _ = _setup(tmp_path, [0, 2])
    output = tmp_path / "trainable.safetensors"
    export_depth.export(source, artifact_path, output)
    loaded = _reload(output)
    _, modality, _ = _inputs()
    original = model.transformer_blocks
    with export_depth.retained_blocks(model, [0, 2]):
        losses = []
        for subject in (model, loaded):
            subject.train()
            velocity, _ = subject(video=modality, audio=None, perturbations=None)
            loss = velocity[:, 4:].square().mean()
            loss.backward()
            losses.append(loss.detach())
        torch.testing.assert_close(losses[0], losses[1], atol=1e-7, rtol=1e-6)
        for before, after in zip(model.transformer_blocks, loaded.transformer_blocks, strict=True):
            reference = before.attn1.to_q.weight.grad
            measured = after.attn1.to_q.weight.grad
            assert torch.isfinite(measured).all()
            assert measured.abs().sum() > 0
            torch.testing.assert_close(reference, measured, atol=1e-7, rtol=1e-6)
    assert original[0].attn1.to_q.weight.grad is None
    assert original[2].attn1.to_q.weight.grad is None


@pytest.mark.parametrize("change", ["mapping", "duplicates", "out_of_range", "bool", "all", "family", "fingerprint"])
def test_invalid_artifact_is_rejected_without_output(tmp_path: Path, change: str) -> None:
    _, source, artifact_path, _, _ = _setup(tmp_path, [1])
    artifact = json.loads(artifact_path.read_text())
    if change == "mapping":
        artifact["original_to_compact"][1] = 0
    elif change == "family":
        artifact["family"] = "width"
    elif change == "fingerprint":
        artifact["provenance"]["transformer_fingerprint"] = "changed"
    else:
        artifact["removed_blocks"] = {"duplicates": [1, 1], "out_of_range": [4], "bool": [True], "all": [0, 1, 2, 3]}[
            change
        ]
    artifact_path.write_text(json.dumps(artifact))
    output = tmp_path / "invalid.safetensors"
    with pytest.raises(ValueError, match=r"depth artifact|removed_blocks|every transformer"):
        export_depth.export(source, artifact_path, output)
    assert not output.exists()


def test_mutated_native_manifest_is_rejected(tmp_path: Path) -> None:
    _, source, artifact_path, baseline, _ = _setup(tmp_path, [1])
    (tmp_path / "manifest.json").write_text(json.dumps({**baseline, "seed": 43}))
    with pytest.raises(ValueError, match="manifest content changed"):
        export_depth.export(source, artifact_path, tmp_path / "invalid.safetensors")


def _rewrite(path: Path, change) -> None:  # noqa: ANN001
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = handle.keys()
        tensors = {name: handle.get_tensor(name).clone() for name in keys}
        metadata = dict(handle.metadata())
    change(tensors, metadata)
    save_file(tensors, str(path), metadata=metadata)


def test_changed_source_and_already_pruned_sources_are_rejected(tmp_path: Path) -> None:
    _, source, artifact_path, _, _ = _setup(tmp_path, [1])
    _rewrite(source, lambda tensors, _metadata: tensors[next(iter(tensors))].add_(1))
    with pytest.raises(ValueError, match="fingerprint"):
        export_depth.export(source, artifact_path, tmp_path / "invalid.safetensors")
    def set_pruning(_tensors: dict, metadata: dict) -> None:
        config = json.loads(metadata["config"])
        config["transformer"]["pruning"] = {"family": "width"}
        metadata["config"] = json.dumps(config)
    _rewrite(source, set_pruning)
    with pytest.raises(ValueError, match="already-pruned"):
        export_depth.inspect_checkpoint(source)


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_source_overwrite_aliases_are_rejected(tmp_path: Path, alias: str) -> None:
    _, source, artifact_path, _, _ = _setup(tmp_path, [1])
    output = source if alias == "same" else tmp_path / "alias.safetensors"
    if alias == "symlink":
        output.symlink_to(source)
    elif alias == "hardlink":
        os.link(source, output)
    before = provenance.file_sha256(source)
    with pytest.raises(ValueError, match="overwrite source"):
        export_depth.export(source, artifact_path, output)
    assert provenance.file_sha256(source) == before


@pytest.mark.parametrize("failure", ["copy", "source_change"])
def test_failed_copy_leaves_no_checkpoint_or_temporary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     failure: str) -> None:
    _, source, artifact_path, _, _ = _setup(tmp_path, [1])
    original = export_depth._copy_tensor
    changed = False
    def fail_or_mutate(*args) -> None:
        nonlocal changed
        if failure == "copy":
            raise OSError("interrupted copy")
        original(*args)
        if not changed:
            source.touch()
            changed = True
    monkeypatch.setattr(export_depth, "_copy_tensor", fail_or_mutate)
    output = tmp_path / "failed.safetensors"
    with pytest.raises((ValueError, OSError), match=r"interrupted copy|changed during"):
        export_depth.export(source, artifact_path, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.safetensors.*.tmp"))


def test_native_candidate_dispatch_checks_depth_config_and_artifact(tmp_path: Path) -> None:
    _, source, artifact_path, baseline, _ = _setup(tmp_path, [1])
    output = tmp_path / "valid.safetensors"
    export_depth.export(source, artifact_path, output)
    candidate = copy.deepcopy(baseline)
    candidate["model"].update(transformer_path=str(output),
                              transformer_fingerprint=provenance.checkpoint_fingerprint(output))
    assert whole_clip.verify_candidate(baseline, candidate)["family"] == "depth"
    def alter_config(_tensors: dict, metadata: dict) -> None:
        config = json.loads(metadata["config"])
        config["transformer"]["timestep_scale_multiplier"] = 2000
        metadata["config"] = json.dumps(config)
    _rewrite(output, alter_config)
    candidate["model"]["transformer_fingerprint"] = provenance.checkpoint_fingerprint(output)
    with pytest.raises(ValueError, match="architecture"):
        whole_clip.verify_candidate(baseline, candidate)
    # Restore the file, then a content mutation of the external artifact must also fail.
    output.unlink()
    export_depth.export(source, artifact_path, output)
    candidate["model"]["transformer_fingerprint"] = provenance.checkpoint_fingerprint(output)
    artifact_path.write_text(artifact_path.read_text() + " ")
    with pytest.raises(ValueError, match="artifact content"):
        whole_clip.verify_candidate(baseline, candidate)


def test_parity_parser_preserves_masks_and_accepts_one_depth_artifact() -> None:
    common = ["--baseline", "baseline", "--exported-checkpoint", "export", "--view", "view",
              "--sigmas", "0.725", "--gpu-id", "0"]
    assert export_parity.argument_parser().parse_args([*common, "--masks", "masks"]).masks == Path("masks")
    depth_args = export_parity.argument_parser().parse_args([*common, "--depth-artifact", "depth"])
    assert depth_args.depth_artifact == Path("depth")
    with pytest.raises(SystemExit):
        export_parity.argument_parser().parse_args([*common, "--masks", "masks", "--depth-artifact", "depth"])
