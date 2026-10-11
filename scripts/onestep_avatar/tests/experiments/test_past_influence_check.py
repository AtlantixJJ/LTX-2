"""Real public causal transactions prove independently changed past inputs."""
import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from scripts.onestep_avatar.experiments import conditioning_check
from scripts.onestep_avatar.experiments import continuation_check as continuation
from scripts.onestep_avatar.experiments import past_influence_check as check
from scripts.onestep_avatar.model import causal, common
from scripts.onestep_avatar.tests.test_causal_core import _context, _geometry, _grid, _model
from scripts.prune.core import session


def tiny_sources(root: Path, sigma: float) -> tuple:
    torch.manual_seed(17)
    model = _model(prompt_adaln=True)
    grid = _grid(_geometry(8), 17)
    geometry = causal.CausalGeometry(grid.tools.scale_factors, 2, 8)
    context = _context()
    capture, guide, noise = [torch.randn(1, 68, 8, generator=torch.Generator().manual_seed(seed))
                             for seed in (11, 12, 13)]
    predict = common.denoised_from_velocity_model(model)
    snapshot = root / "capture"
    snapshot.mkdir()
    recorder = continuation.CacheRecorder(predict, grid, geometry, snapshot, sigma=sigma,
        capture=capture, guide=guide, noise=noise, context=context, teacher_forcing=False)
    causal.sample(recorder, context, grid, guide, capture[:, :4], transformer=model,
        geometry=geometry, schedule=[sigma, 0], seed=42, epsilon=noise)
    recorder.finish(len(model.transformer_blocks))
    reference = root / "reference"
    reference.mkdir()
    observed, blocks = continuation.reference_blocks(predict, model, grid, geometry,
        snapshot, recorder.snapshots, context=context, sigma=sigma, output_root=reference)
    return model, predict, grid, geometry, snapshot, recorder.snapshots, reference, observed, blocks


@pytest.mark.parametrize("sigma", [1., .909375])
@pytest.mark.parametrize("history", ["cache", "recompute"])
def test_real_public_transactions_exactly_reconstruct_both_boundaries_then_change_past(
    tmp_path: Path, sigma: float, history: str,
) -> None:
    model, predict, grid, geometry, snapshot, rows, reference, observations, blocks = tiny_sources(tmp_path, sigma)
    for row, block in zip(rows, blocks, strict=True):
        phase = row["phase"]
        values = continuation.load_inputs(snapshot, row)
        original_path = (continuation.checked_path(snapshot, row["cached_prediction"]) if history == "cache" else
                         continuation.checked_path(reference, block["reference_prediction"]))
        original = torch.load(original_path, weights_only=True)
        expected = (check.saved_layers(snapshot, row["layers"], reference=False) if history == "cache" else
                    check.saved_layers(reference, [v for v in observations if v["phase"] == phase], reference=True))
        results, predictions = {}, {}
        for changed in (False, True):
            arm = "changed" if changed else "original"
            folder = tmp_path / history / phase / arm
            result = check.run_arm(predict, model, grid, geometry, values, row, snapshot, folder,
                history, sigma, changed=changed, expected_layers=expected)
            trace = torch.load(folder / "trace.pt", weights_only=True)
            check.verify_trace(trace, grid, geometry, values, tuple(row["span"]), history, sigma,
                               changed, len(model.transformer_blocks))
            assert result["actual_calls"] == (1 if history == "recompute" else 2 if phase == "before" else 6)
            predictions[arm] = torch.load(folder / "prediction.pt", weights_only=True)
            results[arm] = result
        assert torch.equal(predictions["original"], original)
        assert not torch.equal(predictions["original"], predictions["changed"])
        for a, b in zip(results["original"]["layers"], results["changed"]["layers"], strict=True):
            left, right = [torch.load(v["file"]["path"], weights_only=True) for v in (a, b)]
            assert any(not torch.equal(left[name], right[name]) for name in ("k", "v"))
    assert not model._forward_pre_hooks


def test_intervention_preserves_c0_and_current_future_bytes_and_refuses_no_change() -> None:
    grid = _grid(_geometry(8), 17)
    source = torch.ones(1, 68, 8)
    changed = check.changed_history(source, grid, (11, 13))
    assert torch.equal(changed[:, :4], source[:, :4])
    assert torch.equal(changed[:, 44:], source[:, 44:])
    assert torch.equal(changed[:, 4:44], -source[:, 4:44])
    with pytest.raises(ValueError, match="past intervention"):
        check.changed_history(torch.zeros_like(source), grid, (3, 5))


@pytest.fixture(scope="module")
def trace_case(tmp_path_factory: object) -> tuple:
    root = tmp_path_factory.mktemp("past_trace")
    model, predict, grid, geometry, snapshot, rows, _reference, _observations, _blocks = tiny_sources(root, .909375)
    row = rows[1]
    values = continuation.load_inputs(snapshot, row)
    expected = check.saved_layers(snapshot, row["layers"], reference=False)
    folder = root / "changed"
    result = check.run_arm(predict, model, grid, geometry, values, row, snapshot, folder, "cache", .909375,
                          changed=True, expected_layers=expected)
    return torch.load(folder / "trace.pt", weights_only=True), grid, geometry, values, result


@pytest.mark.parametrize("field", ["c0", "past", "current_noise", "position", "timestep", "text", "length", "order"])
def test_native_trace_rejects_changed_fixed_inputs_or_transactions(trace_case: tuple, field: str) -> None:
    original, grid, geometry, values, result = trace_case
    calls = copy.deepcopy(original)
    if field == "c0":
        calls[0]["latent"][:, :4] += 1
    elif field == "past":
        calls[2]["latent"] += 1
    elif field == "current_noise":
        calls[-1]["latent"] += 1
    elif field == "position":
        calls[-1]["positions"] += 1
    elif field == "timestep":
        calls[0]["timesteps"] += 1
    elif field == "text":
        calls[-1]["context_sha256"] = "a" * 64
    elif field == "length":
        calls[-1]["cache_lengths"][0] += 1
    else:
        calls[0], calls[1] = calls[1], calls[0]
    with pytest.raises(ValueError, match="past"):
        check.verify_trace(calls, grid, geometry, values, (11, 13), "cache", .909375, True,
                           len(result["layers"]))


def test_original_all_layer_cache_proof_fails_before_changed_intervention(tmp_path: Path) -> None:
    model, predict, grid, geometry, snapshot, rows, _reference, _observations, _blocks = tiny_sources(tmp_path, 1.)
    row = rows[0]
    expected = check.saved_layers(snapshot, row["layers"], reference=False)
    corrupted = tmp_path / "wrong_layer.pt"
    value = torch.load(expected[-1]["file"]["path"], weights_only=True)
    value["k"] += 1
    torch.save(value, corrupted)
    expected[-1]["file"] = check.file_identity(corrupted)
    with pytest.raises(ValueError, match="original chronological"):
        check.run_arm(predict, model, grid, geometry, continuation.load_inputs(snapshot, row), row,
            snapshot, tmp_path / "invalid", "cache", 1., changed=False, expected_layers=expected)
    assert not model._forward_pre_hooks


@pytest.mark.parametrize("changed", ["sigma", "history", "snapshot", "targets"])
def test_source_requires_matching_generated_capture_and_its_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    capture = {"state": "complete", "phase": "capture", "sigma": 1., "history": "generated",
               "protocol": {}, "fixed_inputs": {}, "software": {},
               "snapshots": [{"phase": name, "span": list(span)} for name, span in continuation.TARGETS]}
    reference = {**capture, "phase": "reference", "reference_blocks": capture["snapshots"]}
    paths = [tmp_path / name / "continuation.json" for name in ("capture", "reference")]
    for path in paths:
        path.parent.mkdir()
        spec = path.parent / "spec.json"
        spec.write_text("{}")
        (path.parent / "manifest.json").write_text(json.dumps({"spec": {"path": str(spec)}}))
    paths[0].write_text(json.dumps(capture))
    reference["source_snapshot"] = check.file_identity(paths[0])
    if changed == "sigma":
        reference["sigma"] = .909375
    elif changed == "history":
        reference["history"] = "capture"
    elif changed == "snapshot":
        reference["source_snapshot"]["sha256"] = "a" * 64
    else:
        reference["reference_blocks"] = reference["reference_blocks"][:1]
    paths[1].write_text(json.dumps(reference))
    monkeypatch.setattr(continuation, "verify_completion", lambda *_args: {})
    args = SimpleNamespace(snapshot=paths[0], reference=paths[1], sigma=1.)
    with pytest.raises(ValueError, match="past control"):
        check.source_records(args)


def test_operation_parser_and_dependencies(tmp_path: Path) -> None:
    args = check.scientific_args(["--snapshot", str(tmp_path / "capture/continuation.json"),
        "--reference", str(tmp_path / "reference/continuation.json"), "--sigma", ".909375",
        "--history", "recompute", "--output", str(tmp_path / "out")])
    assert check.dependency_outputs(args) == [tmp_path / "capture", tmp_path / "reference"]


def saved_case(root: Path, history: str) -> tuple:
    model, predict, grid, geometry, snapshot, rows, reference, observations, blocks = tiny_sources(root, .909375)
    output = root / "saved"
    output.mkdir()
    torch.save(grid.positions, output / "positions.pt")
    args = SimpleNamespace(snapshot=snapshot / "continuation.json", reference=reference / "continuation.json",
                           output=output, sigma=.909375, history=history)
    capture = {"snapshots": rows}
    reference_record = {"layerwise": observations, "reference_blocks": blocks}
    measurements = []
    for row in rows:
        phase = row["phase"]
        values = continuation.load_inputs(snapshot, row)
        expected = check.expected_kv(args, capture, reference_record, phase)
        predictions, arm_layers = {}, {}
        for changed in (False, True):
            arm = "changed" if changed else "original"
            destination = output / phase / arm
            result = check.run_arm(predict, model, grid, geometry, values, row, snapshot, destination,
                history, .909375, changed=changed, expected_layers=expected)
            check.write_json(destination / "result.json", result)
            predictions[arm] = torch.load(destination / "prediction.pt", weights_only=True)
            arm_layers[arm] = result["layers"]
        effects = []
        for a, b in zip(arm_layers["original"], arm_layers["changed"], strict=True):
            left, right = [torch.load(v["file"]["path"], weights_only=True) for v in (a, b)]
            effects.append({"layer": a["layer"], **{name: continuation.compare_chunks(left[name], right[name])
                                                   for name in ("k", "v")}})
        measurements.append({"phase": phase, "span": row["span"],
            "prediction": continuation.compare_chunks(predictions["original"], predictions["changed"]),
            "layerwise": effects, "original_bit_identical": True})
    producer = {"test": "tiny public native producer"}
    pins = {"test": "current source pair"}
    check.write_json(output / "result.json", {"sources": pins, "sigma": args.sigma, "history": history,
                                             "software": producer, "measurements": measurements})
    return args, capture, reference_record, pins, producer, grid


@pytest.fixture(scope="module", params=["cache", "recompute"])
def saved_cached(tmp_path_factory: object, request: pytest.FixtureRequest) -> tuple:
    return saved_case(tmp_path_factory.mktemp("past_saved"), request.param)


@pytest.mark.parametrize("corruption", [None, "prediction", "actual_input", "layer", "measurement", "positions"])
def test_saved_verifier_is_model_free_and_rejects_corruption(  # noqa: PLR0912 -- explicit corruption cases
    saved_cached: tuple, monkeypatch: pytest.MonkeyPatch, corruption: str | None,
) -> None:
    args, capture, reference, pins, producer, grid = saved_cached
    monkeypatch.setattr(check, "source_records", lambda _args: (capture, reference, pins))
    monkeypatch.setattr(check.software, "capture", lambda *_args, **_kwargs: producer)
    monkeypatch.setattr(common.ClipGrid, "build", lambda *_args, **_kwargs: grid)
    monkeypatch.setattr(check.adapters, "inference_transformer", lambda *_args, **_kwargs: pytest.fail("native model"))
    monkeypatch.setattr(session, "Session", lambda *_args, **_kwargs: pytest.fail("native session"))
    modified, prior = None, None
    if corruption == "prediction":
        modified = args.output / "after/original/prediction.pt"
    elif corruption == "actual_input":
        modified = args.output / "after/changed/trace.pt"
    elif corruption == "layer":
        folder = "kv" if args.history == "cache" else "reference_kv/after"
        modified = args.output / f"after/changed/{folder}/layer_0001.pt"
    elif corruption == "positions":
        modified = args.output / "positions.pt"
    elif corruption == "measurement":
        modified = args.output / "result.json"
    if modified is not None:
        prior = modified.read_bytes()
        if corruption == "measurement":
            record = json.loads(modified.read_text())
            record["measurements"][1]["prediction"]["rms_delta"] += 1
            check.write_json(modified, record)
        else:
            value = torch.load(modified, weights_only=True)
            if corruption == "actual_input":
                value[-1]["latent"] += 1
            elif corruption == "layer":
                value["v"] += 1
            else:
                value += 1
            torch.save(value, modified)
    try:
        if corruption is None:
            assert check.verify(args)["state"] == "passed"
        else:
            with pytest.raises(ValueError, match=r"past|E3"):
                check.verify(args)
    finally:
        if modified is not None:
            modified.write_bytes(prior)


def test_frozen_conditioning_dispatcher_selects_only_the_past_operation(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    check.write_json(spec, {"schema_version": 1, "protocol": "conditioning_check", "operation": "past_influence",
        "arguments": ["--snapshot", str(tmp_path / "capture/continuation.json"), "--reference",
                      str(tmp_path / "reference/continuation.json"), "--sigma", "1", "--history", "cache"]})
    args = conditioning_check.read_spec(spec, tmp_path / "out")
    assert args.operation == "past_influence"
    assert conditioning_check.dependency_outputs(spec) == [tmp_path / "capture", tmp_path / "reference"]


@pytest.mark.parametrize("span", [(3, 5), (11, 13)])
def test_presentation_uses_supplied_past_and_exact_current_prediction(span: tuple[int, int]) -> None:
    grid = _grid(_geometry(8), 17)
    history = torch.arange(68 * 8).reshape(1, 68, 8).to(torch.bfloat16)
    prediction = torch.full((1, 8, 8), 7., dtype=torch.bfloat16)
    original = check.presentation_latent(grid, {"clean_history": history}, span, prediction, changed=False)
    changed = check.presentation_latent(grid, {"clean_history": history}, span, prediction, changed=True)
    a, b = grid.patchify(original), grid.patchify(changed)
    lo, _hi = grid.token_span(*span)
    assert original.shape[2] == changed.shape[2] == span[1]
    assert torch.equal(a[:, :lo], history[:, :lo])
    assert torch.equal(a[:, lo:], prediction)
    assert torch.equal(b[:, lo:], prediction)
    assert torch.equal(a[:, :4], b[:, :4])
    assert torch.equal(b[:, 4:lo], -a[:, 4:lo])


@pytest.fixture
def decoded_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    native, capture, _reference, pins, _producer, grid = saved_case(tmp_path, "recompute")
    args = SimpleNamespace(**vars(native), review_of=native.output, gpu_id=0)
    args.output = tmp_path / "decoded"
    member, vae = tmp_path / "member.json", tmp_path / "vae.bin"
    member.write_text("{}")
    vae.write_bytes(b"tiny pinned decoder")
    source = {"shape": [8, 17, 2, 2], "capture_latent_sha256": "c" * 64,
              "guide_sha256": "g" * 64, "capture_encode_record": {"edge": 16}}
    store = SimpleNamespace(root=tmp_path, membership={"sha256": "m" * 64})
    inputs = {"source": source, "store": store, "crop": {"box_xyxy": [0, 0, 16, 16]},
              "video": SimpleNamespace(source="same original view"),
              "specification": SimpleNamespace(paths=SimpleNamespace(video_vae=lambda: str(vae)))}
    prepared = {"capture": {**capture, "fixed_inputs": {"noise": "fixed"}},
        "protocol": {"membership": str(member), "source": "same original view"}, "references": inputs,
        "pins": pins, "software": {"tiny": "decoder only"}, "grid": grid, "seed": 42}
    monkeypatch.setattr(check, "review_preparation", lambda _args: prepared)
    monkeypatch.setattr(check.software, "check_current", lambda _profile: None)
    requested, decoded = [], []
    def rgb(_source: dict, _root: Path, _objective: str, count: int) -> torch.Tensor:
        return torch.full(((count - 1) * 8 + 1, 3, 16, 16), count, dtype=torch.uint8)
    def references(_session: object, _decoder: object, _source: dict, _root: Path,
                   _objective: str, count: int, seed: int, **_kwargs: object) -> tuple:
        requested.append((count, seed))
        value = rgb(source, tmp_path, "white", count)
        mapping = tuple(range(value.shape[0]))
        panels = [check.media.Panel(role, title, value, mapping) for role, title in
                  (("recorded", "Capture RGB"), ("decoded", "Capture VAE"), ("guide", "Guide RGB"))]
        producer = {"source": "same original view", "objective": "white", "fps": 30,
            "source_frames": list(mapping), "capture_encoding_sha256": source["capture_latent_sha256"],
            "guide_rgb_sha256": source["guide_sha256"], "vae_sha256": check.hashing.sha256(vae),
            "decode_seed": seed, "decoder_settings": check.media.native_decoder_settings(),
            "decode_key": check.media.decode_key(source["capture_latent_sha256"], check.hashing.sha256(vae),
                [1, 8, count, 2, 2], "native_decode_video", seed, check.media.native_decoder_settings())}
        producer["panels"] = [{"role": panel.role, "pixels_sha256": check.media._pixel_identity(panel.pixels)}
                              for panel in panels]
        return panels, producer
    def decode(_session: object, latent: torch.Tensor, _decoder: object, seed: int) -> torch.Tensor:
        decoded.append((latent.shape[2], seed))
        return rgb(source, tmp_path, "white", latent.shape[2]).float() / 255
    def render(panels: list, **kwargs: object) -> tuple:
        # Scientific input/coverage/label tests are independent of pixel layout;
        # the shared renderer has its own real full/narrow layout tests.
        count = len(panels[0].source_frames)
        return torch.zeros(count, 3, 32, 32, dtype=torch.uint8), {
            "fps": kwargs["fps"], "poster_frame": kwargs["poster_frame"], "display_size": [32, 32],
            "settings": kwargs["common_settings"], "width": kwargs["viewing_width"],
            "titles": [panel.title for panel in panels], "coverage": [list(panel.source_frames) for panel in panels]}
    def save(pixels: torch.Tensor, record: dict, folder: Path) -> None:
        folder.mkdir(parents=True)
        (folder / "comparison.mp4").write_bytes(b"saved bounded native presentation")
        Image.fromarray(pixels[-1].permute(1, 2, 0).numpy()).save(folder / "poster.png")
        check.write_json(folder / "rendering.json", {**record, "outputs": {
            role: check.file_identity(folder / name) for role, name in
            (("video", "comparison.mp4"), ("poster", "poster.png"))}})
    monkeypatch.setattr(check.media, "open_decoder_session", lambda *_args, **_kwargs:
                        SimpleNamespace(decoder=lambda: nullcontext(object())))
    monkeypatch.setattr(check.media, "prepare_training_references", references)
    monkeypatch.setattr(check.media, "decode", decode)
    monkeypatch.setattr(check.media, "recorded_capture_rgb", rgb)
    monkeypatch.setattr(check.media, "recorded_guide_rgb", rgb)
    monkeypatch.setattr(check.media, "render_panels", render)
    monkeypatch.setattr(check.media, "save_render", save)
    monkeypatch.setattr(check.media, "verify_saved_video", lambda *_args: {"width": 32, "height": 32})
    monkeypatch.setattr(session, "Session", lambda *_args, **_kwargs: pytest.fail("transformer session"))
    assert check.execute_review(args)["state"] == "passed"
    assert requested == [(5, 42), (13, 42)]
    assert decoded == [(5, 42), (5, 42), (13, 42), (13, 42)]
    return args


@pytest.mark.parametrize("corruption", ["latent", "pixels", "mapping", "crop", "poster"])
def test_decoder_saved_verifier_rejects_changed_diagnostic_evidence(decoded_review: tuple, corruption: str) -> None:
    args = decoded_review
    if corruption in ("latent", "pixels"):
        path = args.output / "after" / ("changed_latent.pt" if corruption == "latent" else "changed_rgb.pt")
        value = torch.load(path, weights_only=True)
        value += 1
        torch.save(value, path)
    elif corruption in ("mapping", "crop"):
        path = args.output / "before/references/references.json"
        value = json.loads(path.read_text())
        if corruption == "mapping":
            value["panels"][0]["source_frames"][0] += 1
        else:
            value["producer"]["source_crop_adapter"]["box_xyxy"][0] += 1
        check.write_json(path, value)
    else:
        Image.new("RGB", (32, 32), "white").save(args.output / "after/full/poster.png")
    with pytest.raises(ValueError, match=r"past|E3|reference"):
        check.verify_review(args)


def test_decoder_operation_declares_native_observation_dependency(tmp_path: Path) -> None:
    args = check.scientific_args(["--snapshot", str(tmp_path / "capture/continuation.json"),
        "--reference", str(tmp_path / "reference/continuation.json"), "--sigma", "1", "--history", "cache",
        "--review-of", str(tmp_path / "past"), "--output", str(tmp_path / "decoded")])
    assert check.dependency_outputs(args) == [tmp_path / "capture", tmp_path / "reference", tmp_path / "past"]


@pytest.mark.parametrize("span", [[3, 5], [11, 13]])
def test_public_full_and_narrow_geometry_fits_actual_diagnostic_labels(span: list[int]) -> None:
    pixels = torch.zeros(1, 3, 16, 16)
    refs = [check.media.Panel(role, title, pixels, (0,)) for role, title in
            (("recorded", "Capture RGB"), ("decoded", "VAE-decoded capture"), ("guide", "Guide RGB"))]
    panels = check.review_panels(refs, dict.fromkeys(("original", "changed"), pixels), 1, span)
    layouts = check.review_layouts(panels)
    for name, selected, layout, width in layouts:
        geometry = check.media.layout_geometry(selected, question=check.QUESTION, layout=layout,
                                               panel_size=(320, 320), viewing_width=width)
        assert geometry["font_size"] * width / geometry["display_size"][0] >= 16
        assert [item.source_frames for item in selected if item.pixels is not None] == [(0,)] * 5
        roles = [role for row in geometry["rows"] for role in row]
        assert roles.index("changed") == roles.index("baseline") + 1
        if name == "full":
            assert len(geometry["rows"]) == 2
            assert all(len(row) == 3 for row in geometry["rows"])
