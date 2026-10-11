"""Real shared causal calls establish independent E3 control and observer behavior."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.experiments import continuation_identity_check as check
from scripts.onestep_avatar.model import common
from scripts.onestep_avatar.tests.test_causal_core import _context, _geometry, _grid, _model
from scripts.onestep_avatar.training.config import CausalSettings
from scripts.prune.core import session
from scripts.prune.data import prompt_cache


def run_tiny(history: str, sigma: float) -> tuple:
    model = _model(prompt_adaln=True)
    settings = CausalSettings(block_latent_frames=2, context_latent_frames=8)
    context = _context()
    full_grid = _grid(_geometry(8), 17)
    capture = torch.randn(1, 68, 8, generator=torch.Generator().manual_seed(11))
    guide = torch.randn(1, 68, 8, generator=torch.Generator().manual_seed(12))
    noise = torch.randn(1, 68, 8, generator=torch.Generator().manual_seed(13))
    outputs, traces, inputs = {}, {}, {}
    for name, frames in check.ARMS:
        grid = _grid(_geometry(8), frames)
        selected_capture = capture[:, :frames * 4].clone()
        if name == "changed_c0":
            selected_capture[:, :4] += 2
        selected_guide, selected_noise = guide[:, :frames * 4], noise[:, :frames * 4]
        with check.BoundaryObserver(model, grid, history) as observer:
            output, record = evaluate.sample_case(
                model, context, grid, selected_capture, selected_guide, selected_noise,
                mode="causal", mode_settings=settings, guide_mode="d1", schedule=[sigma, 0], seed=42,
                history_mode=history, predict_x0=common.denoised_from_velocity_model(model))
        outputs[name] = output
        traces[name] = observer.calls
        inputs[name] = (grid, selected_capture, selected_guide, selected_noise, context, output, sigma, history)
        assert torch.equal(grid.positions, full_grid.positions[:, :, :frames * 4])
        assert record["call_counts"]["model_calls"] == (frames - 1) // 2 * (2 if history == "cache" else 1)
    return outputs, traces, inputs


@pytest.mark.parametrize("history", ["cache", "recompute"])
@pytest.mark.parametrize("sigma", [1.0, 0.909375])
def test_real_shared_sampler_preserves_repeat_prefix_c0_and_observed_boundary(history: str, sigma: float) -> None:
    outputs, traces, inputs = run_tiny(history, sigma)
    summary = check.summarize(outputs)
    assert summary["state"] == "passed"
    assert summary["exact_controls"] == {"repeat": True, "prefix_5": True, "prefix_13": True}
    assert summary["changed_c0_after_frame_zero"]["difference_rms"] > 0
    for name, _frames in check.ARMS:
        checked = check.verify_boundaries(traces[name], *inputs[name])
        assert checked["clean_c0_input"]
        assert checked["retained_sink_checks"] == (2 if history == "cache" and _frames == 17 else
                                                  1 if history == "cache" and _frames == 13 else 0)
    if history == "cache":
        assert not torch.equal(traces["original"][1]["sink_after"]["k"],
                               traces["changed_c0"][1]["sink_after"]["k"])


@pytest.fixture(scope="module")
def tiny_cached() -> tuple:
    return run_tiny("cache", 0.909375)


@pytest.mark.parametrize("field", ["c0", "timestep", "position", "refresh", "length", "sink"])
def test_observer_rejects_changed_actual_inputs_and_cache_transactions(tiny_cached: tuple, field: str) -> None:
    _outputs, traces, inputs = tiny_cached
    observed = copy.deepcopy(traces["original"])
    if field == "c0":
        observed[0]["latent"][:, :4] += 1
    elif field == "timestep":
        observed[0]["timesteps"][:, :4] = 0.909375
    elif field == "position":
        observed[10]["positions"] += 1
    elif field == "refresh":
        observed[11]["latent"] += 1
    elif field == "length":
        observed[10]["cache_lengths_before"][0] += 1
    else:
        observed[10]["sink_before"]["k"] += 1
    with pytest.raises(ValueError, match="E3"):
        check.verify_boundaries(observed, *inputs["original"])


def test_identity_summary_excludes_frame_zero_and_refuses_bad_controls(tiny_cached: tuple) -> None:
    outputs, _traces, _inputs = tiny_cached
    wrong = {name: value.clone() for name, value in outputs.items()}
    wrong["changed_c0"][:, :, 1:] = wrong["original"][:, :, 1:]
    assert check.summarize(wrong)["state"] == "failed"
    wrong["changed_c0"] = outputs["changed_c0"]
    wrong["prefix_13"][:, :, 12] += 1
    assert check.summarize(wrong)["exact_controls"]["prefix_13"] is False


def test_observer_is_read_only_and_hooks_are_removed_on_failure() -> None:
    model, grid = _model(), _grid(_geometry(8), 17)
    observer = check.BoundaryObserver(model, grid, "cache")
    with pytest.raises(RuntimeError, match="sentinel"), observer:
        raise RuntimeError("sentinel")
    assert not model._forward_hooks  # public registration must be cleaned up on all exits
    assert not model._forward_pre_hooks


def test_native_position_receipt_accepts_exact_device_arithmetic_and_rejects_coordinate_drift() -> None:
    grid = _grid(_geometry(8), 17)
    integer = grid.tools.patchifier.get_patch_grid_bounds(output_shape=grid.tools.target_shape,
                                                         device=torch.device("cpu"))
    native = check.get_pixel_coords(integer, grid.tools.scale_factors,
                                   causal_fix=grid.tools.causal_fix).float()
    native[:, 0] *= torch.tensor(1 / grid.fps, dtype=torch.float32)
    assert not torch.equal(native, grid.positions)
    assert check.validated_positions(native, grid) is native
    assert check.validated_positions(grid.positions, grid) is grid.positions
    bad = native.clone()
    bad[0, 0, 0, 1] = torch.nextafter(bad[0, 0, 0, 1], torch.tensor(float("inf")))
    with pytest.raises(ValueError, match="native positions"):
        check.validated_positions(bad, grid)


@pytest.fixture
def donor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    view = tmp_path / "Part_1/0097_01/views/view00_cam51"
    view.mkdir(parents=True)
    (view.parent.parent / "meta.json").write_text(json.dumps({"actor": {"id": 97}}))
    pixels = tmp_path / "image.png"
    pixels.write_bytes(b"pinned real source pixels")
    image, export_path = tmp_path / "image.pt", tmp_path / "export.json"
    guide_path = view / dataset.guide_bundle_name("white")
    fields = {"schema_version": 2, "objective": "white", "encode_contract_version": 1,
              "vae_fingerprint": "V", "fps": 30, "box_xyxy": [0, 0, 64, 64], "edge": 64}
    guide = {**fields, "source": "Part_1/0097_01/views/view00_cam51",
             "master": torch.ones(8, 17, 2, 2)}
    torch.save(guide, guide_path)
    inputs = {str(pixels.resolve()): check.hashing.sha256(pixels),
              str(guide_path.resolve()): check.hashing.sha256(guide_path)}
    bundle = {**fields, "master": torch.ones(8, 1, 2, 2), "pixel_frames": 1,
              "source": str(pixels.resolve()), "input_fingerprint": inputs[str(pixels.resolve())],
              "input_role": "supplied_image", "preparation": {
                  "input_sha256": inputs, "prepared_pixels_sha256": "prepared"}}
    torch.save(bundle, image)
    export = {"source": guide["source"], "source_frame": 0,
              "capture_preprocessing_pixels_bit_identical": True, "inputs": inputs,
              "output_files": {str(pixels.resolve()): inputs[str(pixels.resolve())]},
              "prepared_pixels_tensor_sha256": "prepared", "box_xyxy": fields["box_xyxy"], "edge": 64}
    export_path.write_text(json.dumps(export))
    args = SimpleNamespace(donor_image=image, donor_image_sha256=check.hashing.sha256(image),
                           donor_export=export_path, donor_export_sha256=check.hashing.sha256(export_path),
                           donor_view=view)
    video = SimpleNamespace(actor="7", z_y=torch.zeros(8, 17, 2, 2), fps=30)
    specification = SimpleNamespace(paths=SimpleNamespace(video_vae=lambda: str(tmp_path / "vae")))
    monkeypatch.setattr(check.precompute, "file_fingerprint", lambda _path: "V")
    return args, video, specification, pixels


def test_donor_requires_another_person_and_unchanged_original_preprocessing(donor: tuple) -> None:
    args, video, specification, pixels = donor
    image, record = check.check_donor(args, video, specification)
    assert image.shape == (8, 1, 2, 2)
    assert record["actor"] == "97"
    video.actor = "97"
    with pytest.raises(ValueError, match="another person"):
        check.check_donor(args, video, specification)
    video.actor = "7"
    pixels.write_bytes(b"changed pixels")
    with pytest.raises(ValueError, match="preprocessing"):
        check.check_donor(args, video, specification)


def test_donor_rejects_a_changed_crop_or_reencoded_bundle_even_when_export_is_unchanged(donor: tuple) -> None:
    args, video, specification, _pixels = donor
    bundle = torch.load(args.donor_image, weights_only=True)
    bundle["box_xyxy"] = [1, 1, 65, 65]
    torch.save(bundle, args.donor_image)
    with pytest.raises(ValueError, match="bytes changed"):
        check.check_donor(args, video, specification)
    args.donor_image_sha256 = check.hashing.sha256(args.donor_image)
    with pytest.raises(ValueError, match="separately encoded"):
        check.check_donor(args, video, specification)


@pytest.fixture
def saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_cached: tuple) -> tuple:
    _outputs, _traces, original_inputs = tiny_cached
    args = SimpleNamespace(output=tmp_path, subset=tmp_path / "subset.json", noise_file=tmp_path / "input_noise.pt",
                           prompt=None, seed=42, sigma=0.909375, history="cache")
    args.subset.write_text("{}")
    torch.save(original_inputs["original"][3], args.noise_file)
    video = SimpleNamespace(source="target", actor="7", hashes={"capture": "C", "guide": "G"},
                            z_y=torch.zeros(8, 17, 2, 2))
    ordinary = SimpleNamespace(mode_settings=CausalSettings(block_latent_frames=2, context_latent_frames=8),
                               saved_noise=original_inputs["original"][3].to(torch.bfloat16))
    prepared = {"video": video, "ordinary": ordinary, "membership": {"sha256": "M"},
                "requested": {"mode": "causal"}, "donor_record": {"actor": "97", "sha256": "D"}}
    producer = {"saved": "software"}
    monkeypatch.setattr(check, "prepare", lambda *_a, **_k: prepared)
    monkeypatch.setattr(check.software, "capture", lambda *_a, **_k: producer)
    monkeypatch.setattr(check.software, "check_current", lambda *_a: None)

    # Convert fixture tensors before creating a fresh trace: native saved checks require bf16.
    model = _model(prompt_adaln=True).to(torch.bfloat16)
    context = _context().to(torch.bfloat16)
    values = {}
    new_traces = {}
    new_outputs = {}
    for name, frames in check.ARMS:
        _old_grid, capture, guide, noise, *_ = original_inputs[name]
        grid = common.ClipGrid.build(frames, 64, 64, 30, _geometry(8), device=torch.device("cpu"),
                                    dtype=torch.bfloat16, latent_channels=8)
        capture, guide, noise = [value.to(torch.bfloat16) for value in (capture, guide, noise)]
        values[name] = (grid, capture, guide, noise)
        with check.BoundaryObserver(model, grid, "cache") as observer:
            output, record = evaluate.sample_case(
                model, context, grid, capture, guide, noise, mode="causal", mode_settings=ordinary.mode_settings,
                guide_mode="d1", schedule=[args.sigma, 0], seed=42,
                predict_x0=common.denoised_from_velocity_model(model))
        new_outputs[name], new_traces[name] = output, observer.calls
        evaluate.save_case(output, {**record, **check.arm_provenance(args, prepared, name, frames, producer)},
                           tmp_path / name)
        torch.save(observer.calls, tmp_path / name / "trace.pt")
    monkeypatch.setattr(check, "input_tokens", lambda _p, frames, changed, _d: values[
        "changed_c0" if changed else "original" if frames == 17 else f"prefix_{frames}"])
    torch.save(context, tmp_path / "text.pt")
    torch.save(ordinary.saved_noise, tmp_path / "noise.pt")
    torch.save(values["original"][0].positions, tmp_path / "positions.pt")
    (tmp_path / "result.json").write_text(json.dumps(check.summarize(new_outputs)))

    def blocked(*_a: object, **_k: object) -> None:
        pytest.fail("saved-only E3 verifier opened a native model or text producer")

    monkeypatch.setattr(session, "Session", blocked)
    monkeypatch.setattr(prompt_cache, "get_or_build", blocked)
    monkeypatch.setattr(check.adapters, "inference_transformer", blocked)
    return args, tmp_path


def test_saved_completion_has_no_native_execution(saved: tuple) -> None:
    args, _root = saved
    assert check.verify_scientific(args)["state"] == "passed"


@pytest.mark.parametrize("field", ["c0_sha256", "source", "noise_file", "donor", "conditions", "output"])
def test_saved_completion_rejects_changed_scientific_claim(saved: tuple, field: str) -> None:
    args, root = saved
    path = root / "changed_c0/result.json"
    record = json.loads(path.read_text())
    record[field] = "changed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="E3"):
        check.verify_scientific(args)


def test_saved_completion_rejects_unpinned_input_and_forged_summary(saved: tuple) -> None:
    args, root = saved
    (root / "result.json").write_text(json.dumps({"state": "passed"}))
    with pytest.raises(ValueError, match="summary"):
        check.verify_scientific(args)
    args.subset.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="fixed inputs"):
        check.verify_scientific(args)


def test_saved_completion_rejects_changed_native_grid(saved: tuple) -> None:
    args, root = saved
    path = root / "positions.pt"
    value = torch.load(path, weights_only=True)
    value[:, 0] += 1
    torch.save(value, path)
    with pytest.raises(ValueError, match="native positions"):
        check.verify_scientific(args)


def test_evidence_paths_skip_only_the_explicit_prior_verifier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(check, "verify_completion", lambda *_a: calls.append("verify"))
    monkeypatch.setattr(check, "read_spec", lambda *_a: SimpleNamespace())
    monkeypatch.setattr(check.spec_protocol, "inspect", lambda *_a: {"artifacts": {"result.json": "hash"}})
    assert check.evidence_paths(tmp_path / "spec.json", tmp_path) == [
        tmp_path / "manifest.json", tmp_path / "result.json"]
    assert calls == ["verify"]
    check.evidence_paths(tmp_path / "spec.json", tmp_path, verified=True)
    assert calls == ["verify"]
