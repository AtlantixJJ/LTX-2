"""CPU checks for the prune-owned whole-clip rollout producer.

A tiny real ``LTXModel`` runs the complete producer on two synthetic corpus views. The
saved files must be the format ``data.whole_clip`` reads: one global epsilon per view,
one latent per sigma and arm, and a manifest that passes the consumer's validation.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.model import causal as causal_core
from scripts.onestep_avatar.model import common
from scripts.prune.core.session import DTYPE
from scripts.prune.data import whole_clip
from scripts.prune.evaluate import whole_clip_baseline as producer

SCALE = SpatioTemporalScaleFactors(time=8, height=32, width=32)
CHANNELS = 8
FRAMES = 5
SIGMAS = (0.725, 0.909375)


@pytest.mark.parametrize("latent_frames", [2, 9, 18])
def test_whole_clip_is_one_block(latent_frames: int) -> None:
    block = producer.whole_clip_block([latent_frames, latent_frames])
    geometry = causal_core.deployed_geometry(SCALE, block_latent_frames=block)
    assert geometry.plan(latent_frames) == [(0, latent_frames)]


def test_whole_clip_rejects_mixed_or_short_views() -> None:
    with pytest.raises(SystemExit):
        producer.whole_clip_block([18, 17])
    with pytest.raises(SystemExit):
        producer.whole_clip_block([1])


def test_option_validation_and_suffixes() -> None:
    with pytest.raises(SystemExit):
        producer.parse_args(["--view", "v", "--output", "o", "--variant", "dev"])
    with pytest.raises(SystemExit):
        producer.parse_args(["--view", "v", "--output", "o", "--steps", "8"])
    with pytest.raises(SystemExit):
        producer.parse_args(["--view", "v", "--output", "o", "--arms", "d0"])
    for retired in ("--whole-clip", "--checkpoint", "--teacher-forcing", "--trajectory-only", "--max-blocks"):
        with pytest.raises(SystemExit):
            producer.parse_args(["--view", "v", "--output", "o", retired])
    args = producer.parse_args(["--view", "v", "--output", "o", "--prompt", ""])
    assert args.prompt == ""
    assert producer.suffix(args) == "one_step"
    args = producer.parse_args(["--view", "v", "--output", "o", "--variant", "dev", "--steps", "8", "--cfg", "3"])
    assert producer.suffix(args) == "dev_n8_cfg3_stg0"


def test_checked_sigmas_requires_the_model_grid() -> None:
    schedule = [1.0, 0.909375, 0.725, 0.421875, 0.0]
    assert producer.checked_sigmas([0.909375, 1.0], schedule) == (0.909375, 1.0)
    for values in ([1.0, 1.0], [0.0], [0.8], []):
        with pytest.raises(SystemExit):
            producer.checked_sigmas(values, schedule)


def test_global_epsilon_is_one_draw_sliced_by_the_plan() -> None:
    geometry = causal_core.CausalGeometry(SCALE, block_latent_frames=2, context_latent_frames=8)
    grid = common.ClipGrid.build(
        9, 64, 64, 30.0, geometry, device=torch.device("cpu"), dtype=torch.float32, latent_channels=CHANNELS
    )
    source = torch.zeros(1, 9 * grid.tokens_per_latent_frame, CHANNELS)
    epsilon, blocks = producer.global_epsilon(source, grid, geometry.plan(9), 42)
    torch.testing.assert_close(epsilon, common.epsilon_block(source, 42), rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(blocks, dim=1), epsilon[:, : sum(b.shape[1] for b in blocks)], rtol=0, atol=0)


@dataclass(frozen=True)
class _Caps:
    latent_channels: int = CHANNELS


def _tiny_model() -> X0Model:
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        num_layers=2,
        cross_attention_dim=8,
        caption_projection=torch.nn.Linear(16, 8),
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return X0Model(model.to(dtype=DTYPE)).eval()


class _Session:
    def __init__(self, vae: Path, *, decoder: object | None = None) -> None:
        self.device = torch.device("cpu")
        self.context = torch.randn(1, 3, 16, generator=torch.Generator().manual_seed(1)).to(dtype=DTYPE)
        self.model = type("Model", (), {})()
        self.model.sigmas = [1.0, 0.909375, 0.725, 0.421875, 0.0]
        self.model.scale_factors = SCALE
        self.model.caps = _Caps()
        self.model.paths = type("Paths", (), {"video_vae": staticmethod(lambda: vae)})()
        self._decoder = decoder

    def stamp(self, *, dtype: str) -> dict:
        return {"model_key": "2.5", "transformer_fingerprint": "tiny", "script": producer.SCRIPT, "dtype": dtype}

    @contextmanager
    def transformer(self, *, loras: tuple = ()):  # noqa: ANN202
        assert loras == ()
        with torch.no_grad():  # as core.session.Session.transformer
            yield _tiny_model()

    @contextmanager
    def decoder(self):  # noqa: ANN202
        yield self._decoder


def _vae(root: Path) -> Path:
    path = root / "vae.safetensors"
    save_file({"weight": torch.zeros(1)}, path)
    return path


def _views(root: Path) -> list[Path]:
    views = []
    for index, actor in enumerate(("0008_01", "0012_09")):
        view = root / actor / "views" / "view00"
        view.mkdir(parents=True)
        generator = torch.Generator().manual_seed(10 + index)
        for name in (dataset.capture_bundle_name("white"), dataset.guide_bundle_name("white")):
            master = torch.randn(CHANNELS, FRAMES, 2, 2, generator=generator).to(dtype=DTYPE)
            torch.save({"schema_version": 2, "master": master, "fps": 30.0}, view / name)
        views.append(view)
    return views


def test_raw_rollout_writes_the_saved_input_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    views = _views(tmp_path / "corpus")
    vae = _vae(tmp_path)
    monkeypatch.setattr(producer, "open_session", lambda *_args, **_kwargs: _Session(vae))
    output = tmp_path / "baseline"
    argv = [arg for view in views for arg in ("--view", str(view))]
    argv += ["--sigmas", *map(str, SIGMAS), "--prompt", "", "--raw-only", "--gpu-id", "0", "--output", str(output)]
    assert producer.main(argv) == 0

    manifest = json.loads((output / "manifest.json").read_text())
    whole_clip.validate_manifest(manifest)
    assert manifest["whole_clip"]
    assert manifest["attention"] == "full_bidirectional"
    assert manifest["history_policy"] == "none_single_block"
    assert manifest["checkpoint"] is None
    assert manifest["geometry"]["block_latent_frames"] == FRAMES - 1
    assert len(manifest["videos"]) == len(views) * len(SIGMAS)
    for view in views:
        stem = f"{view.parent.parent.name}_{view.name}"
        epsilon = torch.load(output / f"{stem}_epsilon.pt", weights_only=True)
        capture, fps = dataset.load_training_master(view / dataset.capture_bundle_name("white"))
        geometry = causal_core.deployed_geometry(SCALE, block_latent_frames=FRAMES - 1)
        grid = producer.clip_grid(capture, fps, geometry, device=torch.device("cpu"), latent_channels=CHANNELS)
        tokens = grid.patchify(capture.unsqueeze(0).to(dtype=DTYPE))
        torch.testing.assert_close(epsilon, common.epsilon_block(tokens, 42), rtol=0, atol=0)
        for sigma in SIGMAS:
            row = whole_clip.records(manifest)[(str(view), sigma)]
            assert row["schedule"] == [sigma, 0.0]
            assert row["blocks"] == [[0, FRAMES]]
            d0 = torch.load(whole_clip.latent_path(output, row), weights_only=True)
            d1 = torch.load(output / f"{stem}_sigma_{sigma:.6f}_one_step_d1.pt", weights_only=True)
            assert d0.shape == d1.shape == (1, CHANNELS, FRAMES, 2, 2)
            assert d0.dtype == DTYPE
            # c0 is the clean capture frame in both arms; later frames differ by source.
            torch.testing.assert_close(d0[0, :, 0], capture[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(d1[0, :, 0], capture[:, 0], rtol=0, atol=0)
            assert not torch.equal(d0[:, :, 1:], d1[:, :, 1:])


def test_decoded_panels_are_capture_d0_d1_at_capture_fps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    views = _views(tmp_path / "corpus")[:1]
    vae = _vae(tmp_path)

    class _Decoder:
        def decode_video(self, latent, _tiling, _generator):  # noqa: ANN001, ANN202
            frames = (latent.shape[2] - 1) * 8 + 1
            yield torch.full((frames, 64, 64, 3), float(latent.float().mean()))

    calls = []
    monkeypatch.setattr(producer, "open_session", lambda *_args, **_kwargs: _Session(vae, decoder=_Decoder()))
    monkeypatch.setattr(producer, "t3_video", lambda *panels, fps: calls.append((panels, fps)))
    output = tmp_path / "decoded"
    producer.main(["--view", str(views[0]), "--sigmas", "0.725", "--prompt", "", "--output", str(output)])
    ((panels, fps),) = calls
    *pixels, path = panels
    assert fps == 30.0
    assert path.name == "0008_01_view00_sigma_0.725000_one_step.mp4"
    assert [tuple(p.shape) for p in pixels] == [((FRAMES - 1) * 8 + 1, 64, 64, 3)] * 3
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["videos"][0]["panels"] == ["ground_truth", "gt_latent_rollout", "rgb_render_latent_rollout"]
    assert manifest["videos"][0]["output"] == str(path)


def test_refuses_a_used_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "used"
    output.mkdir()
    (output / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="fresh and empty"):
        producer.main(["--view", str(tmp_path / "v"), "--output", str(output)])


@pytest.fixture
def saved_raw_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Produce two real tiny sigma results, then forbid model or decoder execution."""
    view = _views(tmp_path / "corpus")[0]
    vae = _vae(tmp_path)
    monkeypatch.setattr(producer, "open_session", lambda *_args, **_kwargs: _Session(vae))
    root = tmp_path / "raw_completion"
    argv = ["--view", str(view), "--sigmas", *map(str, SIGMAS), "--raw-only", "--prompt", "", "--output", str(root)]
    assert producer.main(argv) == 0
    args = producer.parse_args(argv)
    monkeypatch.setattr(producer, "open_session", lambda *_args, **_kwargs: pytest.fail("verification loaded a model"))
    monkeypatch.setattr(producer, "rollout", lambda *_args, **_kwargs: pytest.fail("verification regenerated output"))
    monkeypatch.setattr(producer, "decode_pixels", lambda *_args, **_kwargs: pytest.fail("verification decoded output"))
    return args, root, view


def test_saved_completion_checks_all_rows_and_exposes_only_exact_local_evidence(saved_raw_baseline: tuple) -> None:
    args, root, _ = saved_raw_baseline
    assert producer.verify_completion(args, root)
    assert set(producer.evidence_paths(args, root)) == {path for path in root.iterdir() if path.is_file()}
    assert len(producer.evidence_paths(args, root)) == 6


def test_saved_completion_compares_c0_in_the_actual_bf16_forward_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = _views(tmp_path / "corpus")[0]
    for name in (dataset.capture_bundle_name("white"), dataset.guide_bundle_name("white")):
        path = view / name
        bundle = torch.load(path, weights_only=True)
        bundle["master"] = bundle["master"].float() + 0.001
        torch.save(bundle, path)
    vae = _vae(tmp_path)
    monkeypatch.setattr(producer, "open_session", lambda *_args, **_kwargs: _Session(vae))
    root = tmp_path / "float32_sources"
    argv = ["--view", str(view), "--sigmas", "0.725", "--raw-only", "--arms", "d0", "--output", str(root)]
    assert producer.main(argv) == 0
    args = producer.parse_args(argv)
    capture, _ = dataset.load_training_master(view / dataset.capture_bundle_name("white"))
    output = torch.load(next(root.glob("*_d0.pt")), weights_only=True)
    assert capture.dtype == torch.float32
    assert not torch.equal(output[0, :, 0], capture[:, 0])
    torch.testing.assert_close(output[0, :, 0], capture[:, 0].to(dtype=DTYPE), rtol=0, atol=0)
    monkeypatch.setattr(producer, "open_session", lambda *_args, **_kwargs: pytest.fail("completion loaded a model"))
    assert producer.verify_completion(args, root)


@pytest.mark.parametrize(
    ("target", "defect"),
    [(target, defect) for target in ("latent", "epsilon") for defect in ("dtype", "shape", "nan", "not_tensor")]
    + [("latent", "c0")],
)
def test_rehashed_tensor_still_requires_shape_dtype_finiteness_and_clean_capture(
    saved_raw_baseline: tuple, target: str, defect: str
) -> None:
    args, root, _ = saved_raw_baseline
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifact = manifest["videos"][0]["artifacts"]
    name = artifact["epsilon"] if target == "epsilon" else artifact["latents"][0]["path"]
    path = root / name
    value = torch.load(path, weights_only=True)
    if defect == "dtype":
        value = value.float()
    elif defect == "shape":
        value = value[..., :1]
    elif defect == "nan":
        value.reshape(-1)[0] = float("nan")
    elif defect == "not_tensor":
        value = {"invalid": value}
    else:
        value[:, :, :1] += 1
    torch.save(value, path)
    for row in manifest["videos"]:
        record = row["artifacts"]
        if target == "epsilon":
            record["epsilon_sha256"] = producer.provenance.file_sha256(path)
        else:
            for item in record["latents"]:
                if item["path"] == name:
                    item["sha256"] = producer.provenance.file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    assert not producer.verify_completion(args, root)
    with pytest.raises(ValueError, match=r"tensor|c0"):
        producer.evidence_paths(args, root)


@pytest.mark.parametrize(
    "defect",
    [
        "row_count",
        "row_blocks",
        "row_sigma",
        "row_schedule",
        "shared_epsilon",
        "latent_count",
        "timing_count",
        "forward_calls",
        "seed",
        "fps",
        "capture_hash",
        "epsilon_hash",
        "guidance",
        "prompt",
    ],
)
def test_saved_manifest_refuses_incomplete_or_changed_scientific_coverage(  # noqa: PLR0912 -- each corruption changes one scientific field
    saved_raw_baseline: tuple, defect: str
) -> None:
    args, root, _ = saved_raw_baseline
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    rows = manifest["videos"]
    if defect == "row_count":
        rows.pop()
    elif defect == "row_blocks":
        rows[1]["blocks"] = [[0, FRAMES - 1]]
    elif defect == "row_sigma":
        rows[1]["sigma"] = 0.421875
    elif defect == "row_schedule":
        rows[1]["schedule"] = [rows[1]["sigma"], 0.421875, 0]
    elif defect == "shared_epsilon":
        rows[1]["artifacts"]["epsilon"] = "different_noise.pt"
    elif defect == "guidance":
        manifest["guidance"]["cfg"] = 3
    elif defect == "prompt":
        manifest["text_context"]["prompt"] = "different action"
    else:
        for row in rows:
            record = row["artifacts"]
            if defect == "latent_count":
                record["latents"].pop()
            elif defect == "timing_count":
                record["rollout_timing"].pop()
            elif defect == "forward_calls":
                record["rollout_timing"][0]["forward_passes"] = 2
            elif defect == "seed":
                record["seed"] += 1
            elif defect == "fps":
                record["fps"] = 25
            elif defect == "capture_hash":
                record["capture_sha256"] = "f" * 64
            else:
                record["epsilon_sha256"] = "f" * 64
    path.write_text(json.dumps(manifest))
    assert not producer.verify_completion(args, root)


def test_changed_saved_epsilon_bytes_refuse_before_reusing_outputs(saved_raw_baseline: tuple) -> None:
    args, root, _ = saved_raw_baseline
    record = json.loads((root / "manifest.json").read_text())["videos"][0]["artifacts"]
    path = root / record["epsilon"]
    original = torch.load(path, weights_only=True)
    torch.save(original + 1, path)
    assert not producer.verify_completion(args, root)
    with pytest.raises(ValueError, match="artifact path or hash differs"):
        producer.evidence_paths(args, root)


@pytest.mark.parametrize("payload", [None, [], {"model": []}])
def test_malformed_completion_returns_false_without_loading_models(saved_raw_baseline: tuple, payload: object) -> None:
    args, root, _ = saved_raw_baseline
    (root / "manifest.json").write_text(json.dumps(payload))
    assert not producer.verify_completion(args, root)


def test_direct_dev_is_explicit_and_retains_exact_sigma():
    sigma = 0.9489720463752747
    args = producer.parse_args(
        ["--view", "v", "--output", "o", "--variant", "dev", "--dev-schedule", "direct", "--sigmas", str(sigma)]
    )
    assert producer.dev_schedule(args, sigma) == [sigma, 0.0]
    assert producer.suffix(args) == "dev_direct"
    with pytest.raises(SystemExit):
        producer.parse_args(["--view", "v", "--output", "o", "--dev-schedule", "direct"])
    with pytest.raises(SystemExit):
        producer.parse_args(
            [
                "--view",
                "v",
                "--output",
                "o",
                "--variant",
                "dev",
                "--dev-schedule",
                "direct",
                "--dev-denoising-steps",
                "1",
            ]
        )


def test_capture_only_direct_dev_producer_and_completion(tmp_path, monkeypatch):
    view = _views(tmp_path / "corpus")[0]
    (view / dataset.guide_bundle_name("white")).unlink()
    vae = _vae(tmp_path)
    monkeypatch.setattr(producer, "open_session", lambda *_a, **_k: _Session(vae))
    root = tmp_path / "direct_dev"
    argv = [
        "--view",
        str(view),
        "--output",
        str(root),
        "--variant",
        "dev",
        "--transformer",
        str(vae),
        "--dev-schedule",
        "direct",
        "--sigmas",
        "0.9489720463752747",
        "--raw-only",
        "--arms",
        "d0",
        "--prompt",
        "",
    ]
    assert producer.main(argv) == 0
    manifest = whole_clip.load_manifest(root)
    row = manifest["videos"][0]
    assert row["schedule"] == [0.9489720463752747, 0.0]
    assert row["artifacts"]["guide"] is None
    assert row["artifacts"]["guide_sha256"] is None
    assert row["artifacts"]["rollout_timing"][0]["forward_passes"] == 1
    monkeypatch.setattr(producer, "open_session", lambda *_a, **_k: pytest.fail("verification loaded model"))
    assert producer.verify_completion(producer.parse_args(argv), root)
