"""Saved-noise generation works through both real small-model attention paths."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from scripts.onestep_avatar import evaluate


def test_saved_comparison_renderer_rejects_empty_spec_before_model_session(tmp_path):  # noqa: ANN001, ANN201
    spec = tmp_path / 'spec.json'
    spec.write_text('{"comparisons": []}')
    with pytest.raises(ValueError, match='specification is empty'):
        evaluate.render_saved_comparisons(spec, tmp_path / 'out', gpu_id=0)


def test_saved_comparison_renderer_rejects_missing_latent_before_model_session(tmp_path):  # noqa: ANN001, ANN201
    spec = tmp_path / 'spec.json'
    spec.write_text('{"comparisons": [{"name": "case", "panels": [{"title": "x", "latent": "missing.pt"}]}]}')
    with pytest.raises(ValueError, match='latent is missing'):
        evaluate.render_saved_comparisons(spec, tmp_path / 'out', gpu_id=0)
from scripts.onestep_avatar.model import common
from scripts.onestep_avatar.tests.test_causal_core import _geometry, _grid, _model
from scripts.onestep_avatar.tests.test_subset import old_subset  # noqa: F401 -- shared pytest fixture
from scripts.onestep_avatar.tests.test_training_preflight import checked_settings  # noqa: F401 -- shared pytest fixture
from scripts.onestep_avatar.training.config import BidirectionalSettings, CausalSettings
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.model.transformer.model import X0Model
from ltx_core.guidance.perturbations import BatchedPerturbationConfig, Perturbation, PerturbationConfig, PerturbationType


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
@pytest.mark.parametrize("cfg,stg,rescale", [(1.0, 0.0, 0.0), (3.0, 0.0, 0.0), (1.0, 1.0, 0.0), (3.0, 1.0, 0.5)])
def test_native_guidance_matches_explicit_calls_and_preserves_c0(mode, cfg, stg, rescale):
    model = X0Model(_model())
    grid = _grid(_geometry())
    generator = torch.Generator().manual_seed(83)
    capture = torch.randn(1, 28, 8, generator=generator)
    guide = torch.randn(1, 28, 8, generator=generator)
    noise = torch.randn(1, 28, 8, generator=generator)
    context = torch.randn(1, 3, 16, generator=generator)
    negative = torch.randn(1, 3, 16, generator=generator)
    guider = MultiModalGuider(
        params=MultiModalGuiderParams(cfg_scale=cfg, stg_scale=stg, stg_blocks=[0], rescale_scale=rescale),
        negative_context=negative,
    )
    settings = BidirectionalSettings() if mode == "bidirectional" else CausalSettings()
    kwargs = dict(mode=mode, mode_settings=settings, guide_mode="d1", schedule=[0.725, 0], seed=4)
    output, record = evaluate.sample_case(
        model, context, grid, capture, guide, noise,
        predict_x0=common.guided_denoised_from_x0_model(model, guider, negative), **kwargs,
    )

    def explicit_native(modality):
        conditional, _ = model(video=modality, audio=None, perturbations=None)
        unconditional = 0.0
        if guider.do_unconditional_generation():
            unconditional, _ = model(video=replace(modality, context=negative), audio=None, perturbations=None)
        perturbed = 0.0
        if guider.do_perturbed_generation():
            skipped = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=[0])
            perturbations = BatchedPerturbationConfig(
                [PerturbationConfig([skipped])], num_blocks=model.num_blocks,
                device=conditional.device, dtype=conditional.dtype,
            )
            perturbed, _ = model(video=modality, audio=None, perturbations=perturbations)
        return guider.calculate(conditional, unconditional, perturbed, 1.0)

    expected, explicit_record = evaluate.sample_case(
        model, context, grid, capture, guide, noise, predict_x0=explicit_native, **kwargs,
    )
    assert torch.equal(output, expected)
    assert record["call_counts"] == explicit_record["call_counts"]
    passes = 1 + int(cfg != 1) + int(stg != 0)
    assert record["call_counts"]["model_calls"] == (1 if mode == "bidirectional" else 6) * passes
    assert torch.equal(output[:, :, 0], grid.unpatchify_block(capture, 7)[:, :, 0])


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
def test_sample_saved_noise_clean_c0_and_actual_call_counts(mode, tmp_path):
    model = _model()
    grid = _grid(_geometry())
    generator = torch.Generator().manual_seed(83)
    capture = torch.randn(1, 28, 8, generator=generator)
    guide = torch.randn(1, 28, 8, generator=generator)
    noise = torch.randn(1, 28, 8, generator=generator)
    context = torch.randn(1, 3, 16, generator=generator)
    settings = BidirectionalSettings() if mode == "bidirectional" else CausalSettings()
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    output, record = evaluate.sample_case(
        model,
        context,
        grid,
        capture,
        guide,
        noise,
        mode=mode,
        mode_settings=settings,
        guide_mode="d1",
        schedule=[0.725, 0.0],
        seed=4,
        predict_x0=common.denoised_from_velocity_model(model),
    )
    hook.remove()
    assert record["call_counts"]["model_calls"] == len(calls) == (1 if mode == "bidirectional" else 6)
    assert record["call_counts"]["prime_calls"] == 0
    assert torch.equal(output[:, :, 0], grid.unpatchify_block(capture, 7)[:, :, 0])
    assert record["noise_sha256"] == evaluate.tensor_sha256(noise)
    assert record["metrics"]["per_frame_mse"][0] == 0
    saved = evaluate.save_case(output, record, tmp_path)
    assert json.loads((tmp_path / "result.json").read_text()) == saved
    assert saved["state"] == "complete"
    assert torch.equal(torch.load(tmp_path / "generated.pt", weights_only=True), output)


def test_encoded_mean_retains_first_frame():
    prediction = torch.tensor([0.0, 2.0, 2.0]).reshape(1, 1, 3, 1, 1)
    result = evaluate.encoded_metrics(prediction, torch.zeros_like(prediction))
    assert result["mse"] == pytest.approx(8 / 3)


def test_exact_rgb_match_json_has_no_infinity():
    pixels = torch.zeros(2, 3, 4, 4)
    metrics = evaluate.rgb_metrics(pixels, pixels)
    assert metrics["per_frame_psnr"] == [None, None]
    assert metrics["exact_match"] == [True, True]
    json.dumps(metrics, allow_nan=False)


def test_tensor_identity_includes_shape_dtype_and_bytes():
    value = torch.zeros(2, 3)
    identity = evaluate.tensor_sha256(value)
    assert identity != evaluate.tensor_sha256(value.reshape(3, 2))
    assert identity != evaluate.tensor_sha256(value.double())
    value[0, 0] = 1
    assert identity != evaluate.tensor_sha256(value)


@pytest.mark.parametrize("scale", ["nan", "inf", "-1"])
def test_invalid_cfg_fails_before_input_access(scale):
    with pytest.raises(SystemExit):
        evaluate.parse_args([
            "--mode", "bidirectional", "--subset", "unused", "--output", "unused",
            "--schedule", "0.725", "0", "--cfg", scale,
        ])


def test_cfg_parser_keeps_explicit_negative_text():
    args = evaluate.parse_args([
        "--mode", "causal", "--subset", "unused", "--output", "unused",
        "--schedule", "0.725", "0", "--cfg", "3", "--negative-prompt", "fixed negative text",
    ])
    assert args.cfg == 3
    assert args.negative_prompt == "fixed negative text"


@pytest.mark.parametrize("options", [
    ["--stg", "nan"], ["--rescale", "2"], ["--stg", "1"],
    ["--stg", "1", "--stg-blocks", "0", "0"], ["--stg", "1", "--stg-blocks", "-1"],
])
def test_invalid_stg_rescale_rejected_before_inputs(options):
    with pytest.raises(SystemExit):
        evaluate.parse_args([
            "--mode", "bidirectional", "--subset", "unused", "--output", "unused",
            "--schedule", "0.725", "0", *options,
        ])


def test_stg_block_limit_rejected_before_weights(checked_settings, monkeypatch):
    settings, _ = checked_settings
    specification = evaluate.backbone.resolve("2.5", "dev")
    specification.caps.num_layers = 1
    monkeypatch.setattr(evaluate.backbone, "identity", lambda *a, **k: pytest.fail("invalid STG reached weights"))
    with pytest.raises(ValueError, match="base layer count"):
        evaluate.main([
            "--mode", "bidirectional", "--subset", str(settings.subset), "--output", str(settings.output),
            "--variant", "dev", "--guide-mode", "d0", "--schedule", "0.725", "0",
            "--stg", "1", "--stg-blocks", "1", "--dry-run",
        ])
    assert not settings.output.exists()


def test_evaluation_dry_run_preserves_output_and_opens_no_session(checked_settings, monkeypatch, capsys):
    settings, _ = checked_settings
    result = evaluate.main(
        [
            "--mode",
            "bidirectional",
            "--subset",
            str(settings.subset),
            "--output",
            str(settings.output),
            "--variant",
            "dev",
            "--guide-mode",
            "d0",
            "--schedule",
            "0.725",
            "0",
            "--dry-run",
        ]
    )
    assert result == 0
    assert not settings.output.exists()
    record = json.loads(capsys.readouterr().out)
    assert len(record["cases"]) == 2
    assert record["variants"] == ["base"]


def test_evaluation_missing_guide_rejected_before_base_and_output(checked_settings, monkeypatch):
    settings, _ = checked_settings
    monkeypatch.setattr(evaluate.backbone, "resolve", lambda *args: pytest.fail("missing guide reached base"))
    with pytest.raises(ValueError, match="guide content hash"):
        evaluate.main(
            [
                "--mode",
                "bidirectional",
                "--subset",
                str(settings.subset),
                "--output",
                str(settings.output),
                "--schedule",
                "0.725",
                "0",
                "--dry-run",
            ]
        )
    assert not settings.output.exists()


def test_distilled_off_grid_rejected_before_weight_hash(checked_settings, monkeypatch):
    settings, _ = checked_settings
    monkeypatch.setattr(evaluate.backbone, "identity", lambda *a, **k: pytest.fail("invalid schedule reached weights"))
    with pytest.raises(ValueError, match="distilled base grid"):
        evaluate.main([
            "--mode", "bidirectional", "--subset", str(settings.subset),
            "--output", str(settings.output), "--variant", "distilled",
            "--guide-mode", "d0", "--schedule", "0.7", "0", "--dry-run",
        ])
    assert not settings.output.exists()


def test_output_file_refused_without_changes(checked_settings, monkeypatch):
    settings, _ = checked_settings
    settings.output.write_bytes(b"preserved output")
    monkeypatch.setattr(evaluate.backbone, "resolve", lambda *a: pytest.fail("used output reached base"))
    with pytest.raises(ValueError, match="output is already used"):
        evaluate.main([
            "--mode", "bidirectional", "--subset", str(settings.subset),
            "--output", str(settings.output), "--guide-mode", "d0",
            "--schedule", "0.725", "0", "--dry-run",
        ])
    assert settings.output.read_bytes() == b"preserved output"


def test_wrong_encoding_vae_rejected_before_weights(checked_settings, monkeypatch):
    from scripts.onestep_avatar import precompute

    settings, _ = checked_settings
    monkeypatch.setattr(precompute, "file_fingerprint", lambda path: "different VAE identity")
    monkeypatch.setattr(evaluate.backbone, "identity", lambda *a, **k: pytest.fail("wrong VAE reached weights"))
    with pytest.raises(ValueError, match="encoding VAE differs"):
        evaluate.main([
            "--mode", "bidirectional", "--subset", str(settings.subset), "--output", str(settings.output),
            "--variant", "dev", "--guide-mode", "d0", "--schedule", "0.725", "0", "--dry-run",
        ])
    assert not settings.output.exists()


@pytest.mark.parametrize("option", ["--block-latent-frames", "--blocks-per-sample", "--context-latent-frames"])
def test_evaluation_rejects_explicit_causal_options_for_bidirectional(option):
    with pytest.raises(SystemExit):
        evaluate.parse_args(
            [
                "--mode",
                "bidirectional",
                "--subset",
                "unused",
                "--output",
                "unused",
                "--schedule",
                "0.725",
                "0",
                option,
                "2",
            ]
        )


@pytest.mark.parametrize("history", ["cache", "recompute", "joint"])
def test_causal_diagnostic_history_cli(history):
    args = evaluate.parse_args([
        "--mode", "causal", "--subset", "unused", "--output", "unused",
        "--schedule", "0.725", "0", "--history-mode", history,
    ])
    assert args.history_mode == history
    assert args.kv_source == "refresh"


@pytest.mark.parametrize("options", [
    ["--history-mode", "joint", "--kv-source", "denoise"],
    ["--history-mode", "recompute", "--kv-source", "denoise"],
    ["--teacher-forcing", "--kv-source", "denoise"],
])
def test_invalid_diagnostic_history_rejected_before_inputs(options):
    with pytest.raises(SystemExit):
        evaluate.parse_args([
            "--mode", "causal", "--subset", "unused", "--output", "unused",
            "--schedule", "0.725", "0", *options,
        ])


@pytest.mark.parametrize("options", [["--history-mode", "cache"], ["--kv-source", "refresh"]])
def test_bidirectional_rejects_explicit_diagnostic_defaults(options):
    with pytest.raises(SystemExit):
        evaluate.parse_args([
            "--mode", "bidirectional", "--subset", "unused", "--output", "unused",
            "--schedule", "0.725", "0", *options,
        ])


def test_controlled_comparison_rejects_a_second_change_and_preserves_records():
    original = {
        **{key: "a" * 64 for key in ("capture_sha256", "guide_sha256", "c0_sha256", "noise_sha256", "text_sha256")},
        "source": "actor/view",
        "fps": 30,
        "frames": 7,
        "mode": "bidirectional",
        "mode_settings": {"attention": "full_bidirectional"},
        "guide_mode": "d1",
        "schedule": [0.725, 0],
        "conditions": {"task": {"guide_mode": "d1"}, "schedule": [0.725, 0]},
        "application_method": "fused_bf16",
        "adapter": "step100",
        "adapter_sha256": "b" * 64,
    }
    changed = deepcopy(original)
    changed.update(adapter="step200", adapter_sha256="c" * 64)
    evaluate.validate_comparison([original, changed], "adapter")
    before = deepcopy(original)
    d0 = deepcopy(original)
    d0["guide_mode"] = "d0"
    d0["conditions"]["task"]["guide_mode"] = "d0"
    evaluate.validate_comparison([original, d0], "guide_mode")
    assert original == before
    cached = deepcopy(original)
    cached.update(mode="causal", history_mode="cache", kv_source="refresh")
    joint = deepcopy(cached)
    joint["history_mode"] = "joint"
    evaluate.validate_comparison([cached, joint], "history_mode")
    with pytest.raises(ValueError, match="second factor"):
        evaluate.validate_comparison([cached, joint], "adapter")
    joint["kv_source"] = "denoise"
    with pytest.raises(ValueError, match="second factor"):
        evaluate.validate_comparison([cached, joint], "history_mode")
    changed["noise_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="second factor"):
        evaluate.validate_comparison([original, changed], "adapter")


def test_future_noise_probe_preserves_completed_causal_blocks():
    model = _model()
    grid = _grid(_geometry())
    capture = torch.zeros(1, 28, 8)
    noise = torch.ones_like(capture)
    changed = noise.clone()
    changed[:, 12:] = -1
    outputs, record = evaluate.probe_future_noise(
        model, torch.zeros(1, 3, 16), grid, capture, capture, noise, changed,
        change_start_frame=3, mode='causal', mode_settings=CausalSettings(),
        guide_mode='d1', schedule=[0.725, 0], seed=42,
        predict_x0=common.denoised_from_velocity_model(model),
    )
    assert record['earlier_output_bit_identical']
    assert record['earlier_output_max_abs_delta'] == 0
    assert record['later_output_max_abs_delta'] > 0
    assert record['records'][0]['noise_sha256'] != record['records'][1]['noise_sha256']
    assert torch.equal(outputs[0][:, :, :3], outputs[1][:, :, :3])


@pytest.mark.parametrize('invalid', ['nan_later', 'earlier_change', 'identical', 'dtype', 'source_shape', 'inside_block'])
def test_future_noise_invalid_inputs_fail_before_either_sampling(invalid, monkeypatch):
    grid = _grid(_geometry())
    capture = torch.zeros(1, 28, 8)
    noise = torch.ones_like(capture)
    changed = noise.clone()
    changed[:, 12:] = -1
    boundary = 3
    if invalid == 'nan_later':
        changed[:, 12:] = float('nan')
    elif invalid == 'earlier_change':
        changed[:, 0] = 2
    elif invalid == 'identical':
        changed = noise.clone()
    elif invalid == 'dtype':
        changed = changed.double()
    elif invalid == 'source_shape':
        noise, changed = noise[:, :20], changed[:, :20]
    else:
        boundary = 2
    monkeypatch.setattr(evaluate, 'sample_case', lambda *a, **k: pytest.fail('invalid probe executed sampling'))
    with pytest.raises(ValueError):
        evaluate.probe_future_noise(
            None, torch.zeros(1, 3, 16), grid, capture, capture, noise, changed,
            change_start_frame=boundary, mode='causal', mode_settings=CausalSettings(), guide_mode='d1',
            schedule=[0.725, 0], seed=42,
        )


@pytest.mark.parametrize('options', [
    ['--future-noise-start', '3'],
    ['--changed-noise-file', 'changed.pt'],
    ['--changed-noise-file', 'changed.pt', '--future-noise-start', '3'],
    ['--noise-file', 'original.pt', '--changed-noise-file', 'changed.pt', '--future-noise-start', '0'],
])
def test_future_noise_cli_requires_complete_saved_inputs(options):
    with pytest.raises(SystemExit):
        evaluate.parse_args([
            '--mode', 'causal', '--subset', 'unused', '--output', 'unused',
            '--schedule', '0.725', '0', *options,
        ])


@pytest.mark.parametrize('bad_prefix', [False, True])
def test_future_noise_preflight_checks_saved_pair(checked_settings, tmp_path, bad_prefix):
    settings, membership = checked_settings
    source = membership['sources'][0]['relative_dir']
    from scripts.onestep_avatar import dataset
    video = dataset.ClipStore(membership).load(source, require_guide=False)
    original = torch.zeros(1, 7 * video.z_y.shape[2] * video.z_y.shape[3], video.z_y.shape[0], dtype=torch.bfloat16)
    changed = original.clone()
    changed[:, (0 if bad_prefix else 3 * video.z_y.shape[2] * video.z_y.shape[3]):] = 1
    left, right = tmp_path / 'original.pt', tmp_path / 'changed.pt'
    torch.save(original, left)
    torch.save(changed, right)
    args = evaluate.parse_args([
        '--mode', 'causal', '--subset', str(settings.subset), '--output', str(settings.output),
        '--variant', 'dev', '--guide-mode', 'd0', '--source', source,
        '--schedule', '0.725', '0', '--span-latent-frames', '7',
        '--noise-file', str(left), '--changed-noise-file', str(right), '--future-noise-start', '3',
    ])
    if bad_prefix:
        with pytest.raises(ValueError, match='changed earlier noise'):
            evaluate.prepare_evaluation(args)
    else:
        evaluate.prepare_evaluation(args)
        assert torch.equal(args.changed_noise, changed)
    assert not settings.output.exists()


def test_future_noise_publication_preserves_distinct_noise_and_shared_provenance(tmp_path):
    outputs = [torch.zeros(1, 2, 7, 1, 1), torch.ones(1, 2, 7, 1, 1)]
    diagnostic = {'records': [{'noise_sha256': 'a' * 64}, {'noise_sha256': 'b' * 64}], 'earlier_output_bit_identical': False}
    provenance = {'source': 'fixed/view', 'adapter_sha256': 'c' * 64, 'fps': 30}
    completed = evaluate.save_future_noise_probe(outputs, diagnostic, provenance, tmp_path / 'probe')
    assert 'output' not in diagnostic['records'][0]
    for index, record in enumerate(completed['records']):
        assert record['source'] == 'fixed/view' and record['adapter_sha256'] == 'c' * 64
        assert record['noise_sha256'] == ('a' if index == 0 else 'b') * 64
        assert torch.equal(torch.load(record['output']['path'], weights_only=True), outputs[index])
    assert json.loads((tmp_path / 'probe/future_noise.json').read_text()) == completed


@pytest.mark.parametrize('field,value', [('history_mode', 'recompute'), ('kv_source', 'denoise')])
def test_history_comparison_binds_the_same_changed_condition_field(field, value):
    original = {
        **{key: 'a'*64 for key in ('capture_sha256', 'guide_sha256', 'c0_sha256', 'noise_sha256', 'text_sha256')},
        'source': 'actor/view', 'fps': 30, 'frames': 7, 'mode': 'causal', 'mode_settings': {},
        'guide_mode': 'd1', 'schedule': [0.725, 0], 'application_method': 'fused_bf16',
        'adapter': 'same_adapter', 'adapter_sha256': 'b'*64,
        'history_mode': 'cache', 'kv_source': 'refresh',
        'conditions': {'history_mode': 'cache', 'kv_source': 'refresh'},
    }
    changed = deepcopy(original)
    changed[field] = value
    changed['conditions'][field] = value
    evaluate.validate_comparison([original, changed], field)
    changed['conditions']['schedule'] = [0.725, 0.421875, 0]
    with pytest.raises(ValueError, match='second factor'):
        evaluate.validate_comparison([original, changed], field)
    assert original['conditions'] == {'history_mode': 'cache', 'kv_source': 'refresh'}
