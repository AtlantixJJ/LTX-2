"""Product generation uses real small-model paths and has no capture target."""

import pytest
import torch
import json
from types import SimpleNamespace
from safetensors.torch import save_file

from ltx_core.model.transformer.model import X0Model
from scripts.onestep_avatar import infer, dataset
from scripts.onestep_avatar.model import causal
from scripts.onestep_avatar.tests.test_causal_core import _geometry, _grid, _model
from scripts.onestep_avatar.training.config import BidirectionalSettings, CausalSettings
from scripts.onestep_avatar.tests.test_checkpoint_contract import A, B, _contract
from scripts.onestep_avatar.training.checkpoints import CONTRACT_KEY
from scripts.onestep_avatar.precompute import file_fingerprint
from ltx_core.types import SpatioTemporalScaleFactors


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
def test_product_preserves_supplied_image_without_capture_metrics(mode, monkeypatch):
    model = X0Model(_model())
    geometry = _geometry(context_latent_frames=8)
    grid = _grid(geometry)
    generator = torch.Generator().manual_seed(47)
    master = torch.randn(1, 8, 7, 2, 2, generator=generator)
    image = torch.randn(1, 8, 1, 2, 2, generator=generator)
    guide = grid.patchify(master)
    first = grid.patchify(image)
    context = torch.randn(1, 3, 16, generator=generator)
    settings = BidirectionalSettings() if mode == "bidirectional" else CausalSettings()
    if mode == "bidirectional":
        monkeypatch.setattr(
            causal.BlockCache, "allocate", lambda *args, **kwargs: pytest.fail("product allocated cache")
        )
    output, record = infer.generate(
        model, context, grid, guide, first, mode=mode, settings=settings, schedule=[0.725, 0], seed=11
    )
    assert torch.equal(grid.patchify(output)[:, :4], first)
    assert record["capture_reference"] is None
    assert "metrics" not in record
    assert record["call_counts"]["model_calls"] == (1 if mode == "bidirectional" else 6)
    if mode == "causal":
        fixture = (
            dataset.WORKSPACE_ROOT
            / "expr/onestep_avatar/two_mode_restructure_20261005/fixtures/product_causal_legacy.pt"
        )
        if not fixture.exists():
            pytest.skip("saved legacy product fixture is not installed")
        saved = torch.load(fixture, map_location="cpu", weights_only=True)
        assert torch.equal(master, saved["guide"])
        assert torch.equal(image, saved["image"])
        assert torch.equal(context, saved["context"])
        assert torch.equal(output, saved["output"])


def test_product_rejects_capture_history_before_model_calls():
    with pytest.raises(ValueError, match="no capture past"):
        infer.generate(
            None,
            None,
            None,
            None,
            None,
            mode="causal",
            settings=CausalSettings(teacher_forcing=True),
            schedule=[0.725, 0],
            seed=0,
        )


def test_supplied_image_producer_conditions_are_required():
    guide = torch.zeros(1, 8, 7, 2, 2)
    image = guide[:, :, :1]
    record = {
        "objective": "white",
        "fps": 30,
        "box_xyxy": [0, 0, 64, 64],
        "edge": 64,
        "vae_fingerprint": "same-vae",
        "encode_contract_version": 1,
        "pixel_frames": 1,
    }
    infer.check_inputs(guide, image, record, {**record, "input_role": "supplied_image"})
    with pytest.raises(ValueError, match="vae_fingerprint"):
        infer.check_inputs(guide, image, record, {**record, "input_role": "supplied_image", "vae_fingerprint": "other"})
    with pytest.raises(ValueError, match="supplied image"):
        infer.check_inputs(guide, image, record, record)


@pytest.fixture
def product_files(tmp_path, monkeypatch):
    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"vae identity fixture")
    common_record = {
        "schema_version": 2,
        "objective": "white",
        "fps": 30,
        "box_xyxy": [0, 0, 64, 64],
        "edge": 64,
        "encode_contract_version": 1,
        "vae_fingerprint": file_fingerprint(vae),
    }
    guide, image = tmp_path / "guide.pt", tmp_path / "image.pt"
    torch.save({**common_record, "master": torch.zeros(128, 7, 2, 2), "pixel_frames": 49}, guide)
    torch.save(
        {**common_record, "master": torch.zeros(128, 1, 2, 2), "pixel_frames": 1, "input_role": "supplied_image"}, image
    )
    specification = SimpleNamespace(
        scale_factors=SpatioTemporalScaleFactors(8, 32, 32),
        caps=SimpleNamespace(latent_channels=128),
        sigmas=[0.725],
        paths=SimpleNamespace(transformer=lambda: tmp_path / "base.safetensors", video_vae=lambda: vae),
    )
    monkeypatch.setattr(infer.backbone, "resolve", lambda *args: specification)
    monkeypatch.setattr(infer.backbone, "identity", lambda *args, **kwargs: {"base_transformer_sha256": "a" * 64})
    args = [
        "--mode",
        "bidirectional",
        "--guide",
        str(guide),
        "--first-image",
        str(image),
        "--output",
        str(tmp_path / "out"),
        "--variant",
        "dev",
        "--schedule",
        "0.725",
        "0",
    ]
    return args, guide, image


def test_product_dry_run_creates_no_output(product_files, capsys):
    args, _, _ = product_files
    assert infer.main([*args, "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["conditions"]["task"]["guide_mode"] == "d1"
    from pathlib import Path

    assert not Path(args[args.index("--output") + 1]).exists()


def test_product_adapter_wrong_mode_fails_before_session(product_files, tmp_path):
    args, _, _ = product_files
    record = _contract("causal")
    path = tmp_path / "causal.safetensors"
    save_file({A: torch.ones(2, 4), B: torch.zeros(4, 2)}, path, metadata={CONTRACT_KEY: json.dumps(record)})
    with pytest.raises(ValueError, match="incompatible adapter conditions.*mode"):
        infer.main([*args, "--checkpoint", str(path), "--dry-run"])


def test_product_image_condition_mismatch_precedes_base(product_files, monkeypatch):
    args, _, image = product_files
    record = torch.load(image, weights_only=True)
    record["objective"] = "bg"
    torch.save(record, image)
    monkeypatch.setattr(infer.backbone, "resolve", lambda *args: pytest.fail("invalid image reached base resolution"))
    with pytest.raises(ValueError, match="objective"):
        infer.main([*args, "--dry-run"])


def test_review_requires_decoding_before_file_access(product_files):
    args, _, _ = product_files
    with pytest.raises(SystemExit):
        infer.parse_args([*args, "--review"])


def test_product_review_records_decoded_inputs_without_capture(tmp_path, monkeypatch):
    from scripts.onestep_avatar import media, evaluate
    from scripts.onestep_avatar.hashing import sha256

    guide = torch.ones(1, 128, 2, 1, 1)
    image = torch.zeros(1, 128, 1, 1, 1)
    generated = torch.full((9, 3, 16, 16), 0.5)
    decoded = []

    def decode(session, latent, decoder, seed):
        decoded.append((latent, seed))
        return torch.zeros((1 if latent is image else 9, 3, 16, 16))

    monkeypatch.setattr(media, "decode", decode)
    completed = evaluate.save_case(
        guide, {"fps": 30, "mode": "bidirectional", "schedule": [0.725, 0], "input_files": {}}, tmp_path / "raw"
    )
    original_hash = sha256(tmp_path / "raw/result.json")
    args = SimpleNamespace(seed=42, poster_frame=4, output=tmp_path)
    record = infer.render_review(
        None, None, generated, guide, image, completed=completed, args=args, vae_hash="a" * 64
    )
    assert [latent for latent, seed in decoded] == [guide, image]
    assert all(seed == 42 for latent, seed in decoded)
    assert record["layout"] == "inference"
    assert record["source_frames"] == list(range(9))
    assert record["poster_frame"] == 4
    assert record["common_settings"]["capture_reference"] is None
    assert set(record["common_settings"]["decoded_inputs"]) == {"first_image", "guide"}
    assert record["panels"][0]["still"] is True
    assert [panel["role"] for panel in record["panels"]] == ["first_image", "guide", "generated"]
    assert record["common_settings"]["generated_decode_key"] == media.decode_key(
        completed["output"]["sha256"], "a" * 64, list(guide.shape), "native_decode_video", 42,
        media.native_decoder_settings(),
    )
    assert sha256(tmp_path / "raw/result.json") == original_hash


def test_product_causal_preflight_binds_the_recorded_cache_computation(product_files, tmp_path):
    args, _, _ = product_files
    args[args.index('--mode') + 1] = 'causal'
    contract = _contract('causal')
    path = tmp_path / 'causal_adapter.safetensors'
    save_file({A: torch.ones(2, 4), B: torch.zeros(4, 2)}, path,
              metadata={CONTRACT_KEY: json.dumps(contract)})
    _, _, _, _, requested, checked = infer.prepare_product(infer.parse_args(args + ['--checkpoint', str(path)]))
    assert requested['history_mode'] == 'cache' and requested['kv_source'] == 'refresh'
    assert checked['overrides'] == []


@pytest.mark.parametrize('options', [['--history-mode', 'recompute'], ['--kv-source', 'denoise'],
                                     ['--research-override']])
def test_product_exposes_no_diagnostic_history_override(product_files, options):
    args, _, _ = product_files
    args[args.index('--mode') + 1] = 'causal'
    with pytest.raises(SystemExit):
        infer.parse_args(args + options)
