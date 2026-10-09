"""Causal physical prefixes keep strict adapter training settings and pre-weight checks."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import evaluate, media, prepare_inputs
from scripts.onestep_avatar.corpus import dataset, precompute, subset
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.test_checkpoint_contract import A, B
from scripts.onestep_avatar.tests.test_subset import old_subset  # noqa: F401 -- full real saved master fixture
from scripts.onestep_avatar.training import checkpoints, config, engine


@pytest.fixture
def causal_coverage(old_subset: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:  # noqa: F811 -- pytest fixture
    """Use real checked D1 masters/contracts; no native model or VAE work is allowed."""
    for source in old_subset["sources"]:
        source["n_latent_frames"] = 18
        view = tmp_path / source["relative_dir"]
        capture_path = view / dataset.capture_bundle_name("white")
        capture = torch.load(capture_path, weights_only=True)
        capture["master"] = torch.arange(144).reshape(2, 18, 2, 2).to(torch.bfloat16)
        torch.save(capture, capture_path)
        render = view / dataset.render_name("white")
        render.write_bytes(b"controlled guide RGB " + source["actor"].encode())
        source["guide_sha256"] = sha256(render)
        guide = {**capture, "master": capture["master"] + 1, "input_fingerprint": sha256(render)}
        torch.save(guide, view / dataset.guide_bundle_name("white"))
        (view / dataset.render_metadata_name("white")).write_text(json.dumps(
            {"compositing_version": dataset.GUIDE_COMPOSITING_VERSION, "objective": "white"}))
    membership, _plan = subset.convert_legacy(old_subset, original_file_sha256="b" * 64, require_guide=True)
    membership_path = tmp_path / "membership.json"
    membership_path.write_text(json.dumps(membership))
    specification = SimpleNamespace(scale_factors=SpatioTemporalScaleFactors(8, 32, 32),
        caps=SimpleNamespace(latent_channels=2, num_layers=2), sigmas=[0.725],
        paths=SimpleNamespace(transformer=lambda: tmp_path / "base.safetensors",
                              video_vae=lambda: tmp_path / "vae.safetensors"))
    monkeypatch.setattr(evaluate.backbone, "resolve", lambda *_args: specification)
    monkeypatch.setattr(evaluate.backbone, "identity", lambda *_args, **_kwargs: {
        "base_transformer_sha256": "a" * 64, "base_transformer_file": "base.safetensors"})
    monkeypatch.setattr(precompute, "file_fingerprint", lambda _path: "saved VAE identity")
    from scripts.prune.core import preflight, session  # noqa: PLC0415 -- pre-weight guard
    monkeypatch.setattr(preflight, "check", lambda *_args, **_kwargs: pytest.fail("preflight opened native work"))
    monkeypatch.setattr(session, "Session", lambda *_args, **_kwargs: pytest.fail("preflight opened model session"))
    paths, contracts = {}, {}
    for label, span in (("e4", None), ("pilot", 7)):
        settings = config.RunSettings("causal", membership_path, tmp_path / label,
            config.CausalSettings(span_latent_frames=span), variant="dev", objective="white",
            lora_rank=2, lora_alpha=2, steps=1 if label == "e4" else 60)
        settings.base_identity = {"base_transformer_file": "base.safetensors",
                                  "base_transformer_sha256": "a" * 64}
        plan = config.build_frame_plan(settings, membership, specification.scale_factors)
        contract = checkpoints.make_contract(settings, membership, plan, settings.steps)
        contract["adapter"]["tensor_shapes"] = {A: [2, 4], B: [4, 2]}
        path = tmp_path / f"{label}.safetensors"
        save_file({A: torch.ones(2, 4), B: torch.zeros(4, 2)}, path,
                  metadata={checkpoints.CONTRACT_KEY: json.dumps(contract)})
        paths[label], contracts[label] = path, contract
    noise = tmp_path / "noise.pt"
    torch.save(torch.zeros(1, 28, 2, dtype=torch.bfloat16), noise)
    arguments = ["--mode", "causal", "--subset", str(membership_path), "--output", str(tmp_path / "unused"),
                 "--source", membership["sources"][0]["relative_dir"], "--variant", "dev", "--guide-mode", "d1",
                 "--schedule", "0.725", "0", "--noise-file", str(noise)]
    return {"arguments": arguments, "paths": paths, "contracts": contracts, "noise": noise,
            "membership": membership, "specification": specification}


@pytest.mark.parametrize(("mode", "options"), [
    ("causal", ["--output-latent-frames", "0"]), ("causal", ["--output-latent-frames", "-1"]),
    ("bidirectional", ["--output-latent-frames", "7"]),
    ("causal", ["--output-latent-frames", "7", "--span-latent-frames", "9"]),
])
def test_output_coverage_invalid_cli_stops_before_inputs(mode: str, options: list[str]) -> None:
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--mode", mode, "--subset", "unused", "--output", "unused",
                             "--schedule", "0.725", "0", *options])


@pytest.mark.parametrize(("label", "options"), [
    ("e4", ["--output-latent-frames", "7"]),
    ("pilot", ["--span-latent-frames", "7"]),
    ("pilot", ["--span-latent-frames", "7", "--output-latent-frames", "7"]),
])
def test_physical_prefix_preserves_real_e4_or_pilot_contract(
    causal_coverage: dict, label: str, options: list[str]
) -> None:
    selected = causal_coverage["contracts"][label]
    assert selected["shape"]["frame_counts"] == ([6, 7] if label == "e4" else [7])
    before = deepcopy(selected)
    args = evaluate.parse_args([*causal_coverage["arguments"], *options,
                                "--checkpoint", str(causal_coverage["paths"][label])])
    _spec, _variants, cases, _membership = evaluate.prepare_evaluation(args)
    video, frames, requested, adapters = cases[0]
    assert video.z_y.shape[1] == 18
    assert frames == 7
    assert requested["mode_settings"]["span_latent_frames"] == (None if label == "e4" else 7)
    assert requested["shape"]["frames"] == 7
    assert requested["history_mode"] == "cache"
    assert requested["kv_source"] == "refresh"
    assert args.saved_noise.shape == (1, 28, 2)
    assert adapters[0]["overrides"] == []
    assert selected == before
    assert not args.output.exists()


@pytest.mark.parametrize(("label", "options"), [
    ("e4", ["--span-latent-frames", "7", "--output-latent-frames", "7"]),
    ("pilot", ["--output-latent-frames", "7"]),
])
def test_output_coverage_does_not_transfer_recorded_settings(
    causal_coverage: dict, label: str, options: list[str]
) -> None:
    args = evaluate.parse_args([*causal_coverage["arguments"], *options,
                                "--checkpoint", str(causal_coverage["paths"][label])])
    with pytest.raises(ValueError, match="incompatible adapter conditions"):
        evaluate.execute_evaluation(args)
    assert not args.output.exists()


@pytest.mark.parametrize("defect", ["noise_shape", "noise_dtype", "noise_nonfinite", "grid", "history", "sigma"])
def test_coverage_only_option_retains_scientific_and_noise_gates(
    causal_coverage: dict, defect: str
) -> None:
    options = ["--output-latent-frames", "7", "--checkpoint", str(causal_coverage["paths"]["e4"])]
    if defect.startswith("noise"):
        noise = torch.load(causal_coverage["noise"], weights_only=True)
        if defect == "noise_shape":
            noise = noise[:, :20]
        elif defect == "noise_dtype":
            noise = noise.float()
        else:
            noise[0, 0, 0] = float("nan")
        torch.save(noise, causal_coverage["noise"])
    elif defect == "grid":
        contract = deepcopy(causal_coverage["contracts"]["e4"])
        contract["shape"]["height"] = 3
        save_file({A: torch.ones(2, 4), B: torch.zeros(4, 2)}, causal_coverage["paths"]["e4"],
                  metadata={checkpoints.CONTRACT_KEY: json.dumps(contract)})
    elif defect == "history":
        options += ["--context-latent-frames", "1"]
    elif defect == "sigma":
        options += ["--schedule", "0.6", "0"]
    args = evaluate.parse_args([*causal_coverage["arguments"], *options])
    with pytest.raises(ValueError, match=r"noise|incompatible adapter conditions"):
        evaluate.execute_evaluation(args)
    assert not args.output.exists()


@pytest.mark.parametrize("frames", [2, 8, 19])
def test_explicit_output_requires_exact_complete_blocks_within_the_master(
    causal_coverage: dict, frames: int
) -> None:
    arguments = causal_coverage["arguments"]
    arguments = arguments[:arguments.index("--noise-file")]
    args = evaluate.parse_args([*arguments, "--output-latent-frames", str(frames)])
    with pytest.raises(ValueError, match=r"complete causal block|does not fit"):
        evaluate.execute_evaluation(args)
    assert not args.output.exists()


@pytest.mark.parametrize(("options", "frames"), [([], 17), (["--span-latent-frames", "7"], 7),
                                           (["--span-latent-frames", "8"], 7)])
def test_omitted_output_coverage_keeps_existing_span_or_master_behavior(
    causal_coverage: dict, options: list[str], frames: int
) -> None:
    arguments = causal_coverage["arguments"]
    arguments = arguments[:arguments.index("--noise-file")]
    args = evaluate.parse_args([*arguments, *options])
    _spec, _variants, cases, _membership = evaluate.prepare_evaluation(args)
    assert cases[0][1] == frames
    assert args.mode_settings.span_latent_frames == (None if not options else int(options[1]))


@pytest.mark.parametrize(("label", "options"), [
    ("e4", ["--output-latent-frames", "7"]),
    ("e4", ["--output-latent-frames=7"]),
    ("pilot", ["--span-latent-frames", "7"]),
    ("pilot", ["--span-latent-frames=7", "--output-latent-frames=7"]),
])
def test_public_preparation_enqueue_roundtrip_keeps_real_adapter_settings(
    causal_coverage: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, options: list[str]
) -> None:
    """Actual public producers retain strict real E4/pilot contracts before weights."""
    from scripts.prune.data import prompt_cache  # noqa: PLC0415 -- controlled CPU text preparation

    case = causal_coverage
    source = case["membership"]["sources"][0]
    vae = Path(case["specification"].paths.video_vae())
    vae.write_bytes(b"controlled VAE")
    frames = tuple(range(49))
    pixels = torch.zeros(49, 3, 64, 64, dtype=torch.uint8)
    panels = [media.Panel(role, title, pixels, frames) for role, title in (
        ("recorded", "Capture RGB"), ("decoded", "VAE-decoded capture"), ("guide", "Guide RGB"))]
    producer = {"source": source["relative_dir"], "objective": "white", "fps": 30,
        "source_frames": list(frames), "capture_encoding_sha256": source["capture_latent_sha256"],
        "guide_rgb_sha256": source["guide_sha256"], "vae_sha256": sha256(vae),
        "panels": [{"role": panel.role, "pixels_sha256": media._pixel_identity(panel.pixels)} for panel in panels]}
    references = tmp_path / "references"
    media.save_training_references(panels, producer, references)
    monkeypatch.setattr(prepare_inputs.preflight, "check", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(prepare_inputs, "torch", SimpleNamespace(
        **{**vars(torch), "device": lambda *_args: torch.device("cpu")}))
    monkeypatch.setattr(prompt_cache, "get_or_build", lambda *_args: torch.ones(1, 3, 4, dtype=torch.bfloat16))
    arguments = case["arguments"].copy()
    index = arguments.index("--output")
    del arguments[index:index + 2]
    args = prepare_inputs.parse_args(["preview", "--references", str(references / "references.json"),
        "--output", str(tmp_path / "prepared"), "--gpu-id", "0", "--evaluation-arguments", *arguments, *options])
    fixed = prepare_inputs.prepare_preview(args)
    canonical = fixed["evaluation_arguments"]
    assert canonical.count("--output-latent-frames") == 1
    assert canonical.count("--span-latent-frames") == (0 if label == "e4" else 1)
    assert not any(token.startswith(("--output-latent-frames=", "--span-latent-frames=")) for token in canonical)
    contract = case["contracts"][label]
    settings = config.RunSettings("causal", Path(fixed["input_files"]["subset"]["path"]), tmp_path / "training",
        config.CausalSettings(span_latent_frames=None if label == "e4" else 7))
    assert engine.read_preview_inputs(args.output / "preview.json", settings) == fixed
    checkpoint = case["paths"][label]
    checkpoint.with_suffix(".complete.json").write_text(json.dumps({"state": "complete", "sha256": sha256(checkpoint)}))
    job = json.loads(engine.enqueue_preview(checkpoint, fixed, settings.output).read_text())
    selected = evaluate.parse_args([*job["fixed_inputs"]["evaluation_arguments"],
        "--checkpoint", job["checkpoint"]["path"], "--output", job["output"]])
    _specification, _variants, cases, _membership = evaluate.prepare_evaluation(selected)
    assert cases[0][1] == 7
    assert cases[0][2]["mode_settings"] == contract["mode_settings"]
    assert cases[0][3][0]["overrides"] == []
    assert selected.saved_noise.shape == (1, 28, 2)
    selected.span_latent_frames = 7 if label == "e4" else None
    selected.mode_settings = config.CausalSettings(span_latent_frames=selected.span_latent_frames)
    with pytest.raises(ValueError, match="incompatible adapter conditions"):
        evaluate.prepare_evaluation(selected)


@pytest.mark.parametrize("mode", ["causal", "bidirectional"])
def test_preview_arguments_pin_paths_and_lengths_once(mode: str, tmp_path: Path) -> None:
    arguments = ["--mode", mode, "--subset=first", "--subset", "second", "--frame-plan=oldplan",
        "--corpus-root", "oldroot", "--noise-file=oldnoise", "--noise-file", "othernoise",
        "--span-latent-frames=7", "--span-latent-frames", "7", "--output-latent-frames=7",
        "--output-latent-frames", "7"]
    args = SimpleNamespace(mode=mode, subset=tmp_path / "membership", frame_plan=tmp_path / "plan",
        corpus_root=tmp_path / "corpus", span_latent_frames=None, output_latent_frames=None)
    noise = tmp_path / "noise"
    result = prepare_inputs.preview_arguments(arguments, args, noise, 7)
    for flag, path in (("--subset", args.subset), ("--frame-plan", args.frame_plan),
                       ("--corpus-root", args.corpus_root), ("--noise-file", noise)):
        assert result.count(flag) == 1
        assert result[result.index(flag) + 1] == str(path.resolve())
    flag = "--output-latent-frames" if mode == "causal" else "--span-latent-frames"
    assert result.count(flag) == 1
    assert result[result.index(flag) + 1] == "7"
    assert ("--span-latent-frames" not in result) if mode == "causal" else ("--output-latent-frames" not in result)
    assert not any("=" in token for token in result)
