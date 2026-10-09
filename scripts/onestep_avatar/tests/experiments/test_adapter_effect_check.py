"""E2 uses real tiny LTX/PEFT functions and refuses hidden correction/provenance errors."""

import copy
import hashlib
import json
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from peft import get_peft_model_state_dict
from PIL import Image
from safetensors.torch import save_file

from ltx_core.model.transformer.model import X0Model
from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import evaluate, hashing
from scripts.onestep_avatar.corpus import dataset, precompute, subset
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import adapter_effect_check as effect
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters
from scripts.onestep_avatar.tests.test_causal_core import _geometry, _grid, _model
from scripts.onestep_avatar.tests.test_checkpoint_contract import _contract
from scripts.onestep_avatar.training import checkpoints, config, numerics, resources


@pytest.mark.parametrize("defect", [None, "evaluation_digest", "shape", "dtype", "changed_bytes"])
def test_actual_training_context_uses_its_producer_raw_bf16_digest(tmp_path, defect):
    output = tmp_path / "training"
    path = output / "update_states/text.pt"
    path.parent.mkdir(parents=True)
    context = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8).to(torch.bfloat16)
    torch.save(context, path)
    # The trainer stores raw bytes; evaluation separately prefixes shape/dtype.
    digest = hashlib.sha256(context.view(torch.uint8).numpy().tobytes()).hexdigest()
    record = {"path": str(path.resolve()), "sha256": sha256(path), "shape": list(context.shape),
              "dtype": str(context.dtype), "tensor_sha256": digest}
    assert digest != hashing.tensor_sha256(context)
    if defect == "evaluation_digest":
        record["tensor_sha256"] = hashing.tensor_sha256(context)
    elif defect == "shape":
        record["shape"] = [1, 6, 4]
    elif defect == "dtype":
        record["dtype"] = "torch.float32"
    elif defect == "changed_bytes":
        torch.save(context + 1, path)
        record["sha256"] = sha256(path)
    settings = SimpleNamespace(output=output)
    if defect is None:
        assert torch.equal(effect._context(settings, {"update_text": record}), context)
    else:
        with pytest.raises(ValueError, match="context values, shape or dtype"):
            effect._context(settings, {"update_text": record})


def test_repeated_arms_release_cyclic_model_before_the_next_load(tmp_path, monkeypatch):
    models = []

    @contextmanager
    def open_model(*_args, **_kwargs):
        assert all(reference() is None for reference in models)
        model = torch.nn.Linear(2, 2)
        model.__dict__["_cycle"] = model
        models.append(weakref.ref(model))
        yield model

    image = torch.zeros(1, 2, 1, 1, 1)
    monkeypatch.setattr(effect, "reference_transformer", open_model)
    monkeypatch.setattr(effect, "_phase", lambda *_args: nullcontext())
    monkeypatch.setattr(effect, "_check_owners", lambda *_args: None)
    monkeypatch.setattr(numerics, "validate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(effect, "run_sample", lambda *_args, **_kwargs: (image.clone(), {}))
    monkeypatch.setattr(evaluate, "save_case", lambda _output, record, _path: record)
    prepared = {"numerics": {}, "context": torch.zeros(1), "settings": SimpleNamespace(),
                "schedule": [0.725, 0], "seed": 42, "software": {}, "pins": {}}
    case = {name: image for name in ("capture", "guide", "image", "noise")}
    case.update(source="actor/view", coverage={}, fps=30, requested={})
    session = SimpleNamespace(device=torch.device("cpu"))
    grid = SimpleNamespace(patchify=lambda tensor: tensor)
    for _ in range(2):
        effect._arm(prepared, case, session, grid, "reference", "base", tmp_path, [])
        assert models[-1]() is None


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
@pytest.mark.parametrize("steps", [(0, 1), (0, 20, 60)])
def test_actual_loaded_velocity_evaluation_and_product_preserve_saved_correction(  # noqa: PLR0912, PLR0915 -- shared paths
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, steps: tuple[int, ...]
) -> None:
    from ltx_trainer import model_loader  # noqa: PLC0415 -- patch the native loader while executing the real functions

    original = _model().to(dtype=torch.bfloat16)
    original_state = copy.deepcopy(original.state_dict())

    def load(**_kwargs) -> torch.nn.Module:
        base = _model().to(dtype=torch.bfloat16)
        base.load_state_dict(original_state)
        return base

    monkeypatch.setattr(model_loader, "load_transformer", load)
    trained = adapters.attach(original, rank=2, alpha=2, target="attn", init_seed=9)
    for name, parameter in trained.named_parameters():
        if ".lora_B." in name:
            with torch.no_grad():
                parameter.fill_(0.125)
    exported = {name.replace("base_model.model.", "diffusion_model.", 1): value.to(torch.bfloat16).contiguous()
                for name, value in get_peft_model_state_dict(trained).items()}
    selected_settings = config.BidirectionalSettings(7) if mode == "bidirectional" else config.CausalSettings()
    contracts, paths = {}, {}
    states = effect.STATES if steps == (0, 1) else effect.PILOT_STATES
    for state, step in zip(states[1:], steps, strict=True):
        values = {name: value.clone() for name, value in exported.items()}
        if step == 0:
            for name, value in values.items():
                if ".lora_B." in name:
                    value.zero_()
        elif step == 60:
            for name, value in values.items():
                if ".lora_B." in name:
                    value.mul_(2)
        contract = _contract(mode)
        contract["shape"]["channels"] = 8
        contract["adapter"]["step"] = step
        contract["adapter"]["tensor_shapes"] = {name: list(value.shape) for name, value in values.items()}
        path = tmp_path / f"{state}.safetensors"
        save_file(values, path, metadata={checkpoints.CONTRACT_KEY: json.dumps(contract)})
        contracts[state], paths[state] = contract, path
    session = SimpleNamespace(device=torch.device("cpu"), model=SimpleNamespace(
        paths=SimpleNamespace(transformer=lambda: "checked-tiny-base")),
        transformer=lambda **_kwargs: nullcontext(X0Model(load()).eval()))
    settings = config.RunSettings(mode, tmp_path / "unused", tmp_path / "unused_output", selected_settings)
    grid = _grid(_geometry(8))
    generator = torch.Generator().manual_seed(26)
    tensors = [torch.randn(1, 28, 8, generator=generator).to(dtype=torch.bfloat16) for _ in range(3)]
    capture, guide, noise = tensors
    image = torch.full((1, 4, 8), 0.75, dtype=torch.bfloat16)
    assert not torch.equal(capture[:, :4], image)
    assert not torch.equal(guide[:, :4], image)
    context = torch.randn(1, 3, 16, generator=generator).to(dtype=torch.bfloat16)
    outputs, counts = {}, []
    for path in effect.PATHS:
        outputs[path] = {}
        for state in states:
            checkpoint = None if state == "base" else paths[state]
            contract = None if state == "base" else contracts[state]
            if checkpoint is not None:
                requested = {"application_method": adapters.UNMERGED, "global_sigma_dtype": "float32",
                             "mode": mode, "mode_settings": asdict(selected_settings), "model": contract["model"],
                             "task": contract["task"], "shape": {"channels": 8, "height": 2, "width": 2, "frames": 7},
                             "schedule": [0.725, 0.0]}
                if mode == "causal":
                    requested.update(history_mode="cache", kv_source="refresh")
                evaluate.check_adapter(checkpoint, requested, product=True)
            opener = (effect.reference_transformer(session, checkpoint, contract,
                adapter_sha256=None if checkpoint is None else sha256(checkpoint)) if path == "reference" else
                      adapters.inference_transformer(session, checkpoint, contract,
                adapter_sha256=None if checkpoint is None else sha256(checkpoint)))
            with opener as model:
                generated, record = effect.run_sample(
                    model, context, grid, capture, guide, image, noise, path=path, settings=settings,
                    schedule=[0.725, 0], seed=42)
                outputs[path][state] = generated
                counts.append(record["call_counts"])
                assert torch.equal(grid.patchify(generated)[:, :4], image)
    summary = effect.compare_paths(outputs)
    assert summary["passed"]
    assert summary["function_matches"]
    assert summary["learned_effect_demonstrated"]
    compared = [summary] if steps == (0, 1) else list(summary["steps"].values())
    assert all(row["effect"]["relative_l2"] == 0 for step in compared for row in step["paths"].values())
    assert all(row == counts[0] for row in counts)
    assert counts[0]["model_calls"] == (1 if mode == "bidirectional" else 6)


def matched_outputs(effect_size: float = 1e-3) -> dict:
    base = torch.full((1, 8), 100.0, dtype=torch.float64)
    trained = base + effect_size
    return {name: {"base": base.clone(), "zero": base.clone(), "step1": trained.clone()}
            for name in effect.PATHS}


def pilot_outputs() -> dict:
    rows = matched_outputs()
    return {path: {"base": row["base"], "zero": row["zero"],
                   "step20": row["step1"], "step60": row["base"] + 0.01}
            for path, row in rows.items()}


@pytest.mark.parametrize("defect", [None, "boundary20", "boundary60", "swap", "near_zero", "zero", "missing"])
def test_pilot_compares_each_trained_step_against_its_own_reference(defect: str | None) -> None:
    outputs = pilot_outputs()
    if defect in ("boundary20", "boundary60"):
        state = "step20" if defect == "boundary20" else "step60"
        correction = outputs["reference"][state] - outputs["reference"]["base"]
        outputs["product"][state] = outputs["product"]["base"] + correction * 1.051
    elif defect == "swap":
        outputs["evaluation"]["step20"], outputs["evaluation"]["step60"] = (
            outputs["evaluation"]["step60"], outputs["evaluation"]["step20"])
    elif defect == "near_zero":
        for row in outputs.values():
            row["step20"] = row["base"].clone()
    elif defect == "zero":
        outputs["product"]["zero"][0, 0] += 0.1
    elif defect == "missing":
        del outputs["product"]["step20"]
        with pytest.raises(ValueError, match="exact step1 or step20/step60"):
            effect.compare_paths(outputs)
        return
    summary = effect.compare_paths(outputs)
    assert set(summary["steps"]) == {"step20", "step60"}
    assert summary["passed"] == (defect is None)
    if defect == "near_zero":
        assert summary["function_matches"]
        assert not summary["learned_effect_demonstrated"]
        assert summary["steps"]["step20"]["paths"]["reference"]["effect"]["relative_l2"] is None
        assert summary["steps"]["step60"]["passed"]


def test_pilot_fusion_failure_is_diagnostic_for_both_trained_steps() -> None:
    outputs = pilot_outputs()
    outputs["fused"] = copy.deepcopy(outputs["reference"])
    for state in ("step20", "step60"):
        outputs["fused"][state] += 1
    summary = effect.compare_paths(outputs)
    assert summary["passed"]
    assert all(not row["paths"]["fused"]["effect"]["passed"] for row in summary["steps"].values())


@pytest.mark.parametrize("fused", [False, True])
def test_pilot_media_keeps_trained_steps_separate_and_decodes_shared_references_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fused: bool
) -> None:
    outputs = pilot_outputs()
    if fused:
        outputs["fused"] = copy.deepcopy(outputs["reference"])
    prepared = {"states": effect.PILOT_STATES, "seed": 42, "schedule": [0.725, 0], "pins": {}, "software": {},
                "contracts": {state: {"adapter": {"step": step}} for state, step in (("step20", 20), ("step60", 60))}}
    case = {"capture": torch.full((1, 8), 1.), "guide": torch.full((1, 8), 2.),
            "source_frames": [0], "source": "actor/views/view01_cam57", "fps": 30}
    decoded, decoders, rendered = [], [], []

    def decoder() -> object:
        decoders.append(True)
        return nullcontext(object())

    def decode(_session: object, tensor: torch.Tensor, _decoder: object, seed: int) -> torch.Tensor:
        assert seed == 42
        decoded.append(tensor)
        return torch.full((1, 3, 64, 64), float(tensor.mean()))

    def render(panels: list[effect.media.Panel], **options) -> tuple:
        rendered.append((panels, options))
        return torch.empty(0), {"panels": [panel.title for panel in panels], "options": options}

    monkeypatch.setattr(effect, "_phase", lambda *_args: nullcontext())
    monkeypatch.setattr(effect.media, "decode", decode)
    monkeypatch.setattr(effect.media, "render_panels", render)
    monkeypatch.setattr(effect.media, "save_render", lambda _pixels, record, path: {"path": str(path), **record})
    session = SimpleNamespace(device=torch.device("cpu"), decoder=decoder)
    saved = effect._render(prepared, case, outputs, session, tmp_path, [])
    assert len(decoders) == 1
    assert sum(tensor is case["capture"] for tensor in decoded) == 1
    assert sum(tensor is case["guide"] for tensor in decoded) == 1
    assert len(decoded) == 2 + 2 * len(outputs)
    for state, step in (("step20", 20), ("step60", 60)):
        expected = ["Capture VAE", "Guide VAE", f"Reference step {step}",
                    f"Evaluation step {step}", f"Product step {step}"]
        selected = [row for row in saved if f"/media/{state}/" in row["path"]]
        assert len(selected) == (3 if fused else 2)
        for name in ("full", "compact"):
            record = next(row for row in selected if row["path"].endswith("/" + name))
            assert record["panels"] == expected
            assert record["options"]["common_settings"]["checkpoint_step"] == step
        for panels, options in rendered:
            if options["common_settings"]["checkpoint_step"] == step:
                for panel in panels:
                    assert panel.source_frames == (0,)
                    if panel.role in outputs:
                        assert float(panel.pixels.mean()) == pytest.approx(float(outputs[panel.role][state].mean()))


@pytest.mark.parametrize("pilot", [False, True])
@pytest.mark.parametrize("defect", [None, "duplicate", "missing", "step", "lineage", "shape"])
def test_checkpoint_selection_checks_real_steps_matrices_and_common_lineage(
    tmp_path: Path, pilot: bool, defect: str | None
) -> None:
    steps = [0, 20, 60] if pilot else [0, 1]
    selected = []
    for index, step in enumerate(steps):
        contract = _contract()
        contract["adapter"]["step"] = step
        if index == len(steps) - 1:
            if defect == "step":
                contract["adapter"]["step"] = 21
            elif defect == "lineage":
                contract["data"]["membership_sha256"] = "b" * 64
        values = {name: torch.zeros(shape, dtype=torch.bfloat16)
                  for name, shape in contract["adapter"]["tensor_shapes"].items()}
        if defect == "shape" and index == len(steps) - 1:
            values[next(iter(values))] = torch.zeros(1, 1, dtype=torch.bfloat16)
        path = tmp_path / f"step_{step}.safetensors"
        save_file(values, path, metadata={checkpoints.CONTRACT_KEY: json.dumps(contract)})
        selected.append(path)
    if defect == "duplicate":
        selected.append(selected[-1])
    elif defect == "missing":
        selected.pop()
    if defect is not None:
        with pytest.raises(ValueError, match=r"duplicate step|exactly checkpoint steps|lineage differ|tensor shape"):
            effect.checkpoint_selection(selected, pilot=pilot)
    else:
        paths, contracts = effect.checkpoint_selection(list(reversed(selected)), pilot=pilot)
        assert list(paths) == (["zero", "step20", "step60"] if pilot else ["zero", "step1"])
        assert [row["adapter"]["step"] for row in contracts.values()] == steps


@pytest.mark.parametrize("defect", [None, "mode", "model", "history", "precision", "rank", "parent"])
def test_pilot_retains_its_own_lineage_with_the_original_e4_function(defect: str | None) -> None:
    original = _contract()
    selected = copy.deepcopy(original)
    selected["adapter"]["step"] = 60
    selected["data"]["membership_sha256"] = "b" * 64
    selected["data"]["frame_plan_sha256"] = "c" * 64
    if defect == "mode":
        selected["mode"] = "causal"
    elif defect == "model":
        selected["model"]["base_sha256"] = "d" * 64
    elif defect == "history":
        selected["mode_settings"]["start_policy"] = "random"
    elif defect == "precision":
        selected["training"]["global_sigma_dtype"] = "bfloat16"
    elif defect == "rank":
        selected["adapter"]["rank"] = 8
    elif defect == "parent":
        selected["adapter"]["parent"] = {"calibration_transferred": True}
    if defect is None:
        effect.check_calibration_contract(selected, original)
    else:
        with pytest.raises(ValueError, match="original E4 mode or calibrated function"):
            effect.check_calibration_contract(selected, original)


@pytest.mark.parametrize("defect", [None, "span", "frame_counts", "later_start", "random", "cache", "history",
                                  "precision", "shape", "base"])
def test_real_causal_contracts_allow_only_the_prescribed_seven_frame_pilot_restriction(
    tmp_path: Path, defect: str | None
) -> None:
    membership = evaluation_membership(tmp_path, ["actor/views/view01_cam57"])
    membership["splits"] = {"train": ["actor"]}
    membership["sources"][0].update(split="train", n_latent_frames=18, shape=[128, 18, 2, 2])
    membership["sha256"] = subset.membership_hash(membership)
    original_settings = config.RunSettings("causal", tmp_path / "subset.json", tmp_path / "e4",
                                          config.CausalSettings(), variant="dev", objective="white",
                                          steps=1, lora_rank=2, lora_alpha=2)
    original_settings.base_identity = {"base_transformer_file": "base.safetensors",
                                       "base_transformer_sha256": "a" * 64}
    pilot_settings = copy.deepcopy(original_settings)
    pilot_settings.output, pilot_settings.steps = tmp_path / "pilot", 60
    pilot_settings.mode_settings = config.CausalSettings(span_latent_frames=7)
    contracts, paths = {}, {}
    for label, settings in (("original", original_settings), ("pilot", pilot_settings)):
        plan = config.build_frame_plan(settings, membership, SpatioTemporalScaleFactors(8, 32, 32))
        contract = checkpoints.make_contract(settings, membership, plan, settings.steps)
        contract["adapter"]["tensor_shapes"] = _contract("causal")["adapter"]["tensor_shapes"]
        checkpoint = tmp_path / f"{label}.safetensors"
        save_file({name: torch.zeros(shape, dtype=torch.bfloat16)
                   for name, shape in contract["adapter"]["tensor_shapes"].items()}, checkpoint,
                  metadata={checkpoints.CONTRACT_KEY: json.dumps(contract)})
        contracts[label], paths[label] = checkpoints.read_contract(checkpoint), checkpoint
    original, selected = contracts["original"], contracts["pilot"]
    assert original["shape"]["frame_counts"] == [6, 7]
    assert original["mode_settings"]["span_latent_frames"] is None
    assert original["data"]["coverage"][1]["ranges"] == [[7, 9], [9, 11], [11, 13]]
    assert selected["shape"]["frame_counts"] == [7]
    assert selected["data"]["coverage"] == [{"source": "actor/views/view01_cam57", "ranges": [[0, 3], [3, 5], [5, 7]]}]
    if defect == "span":
        selected["mode_settings"]["span_latent_frames"] = 9
    elif defect == "frame_counts":
        selected["shape"]["frame_counts"] = [6]
    elif defect == "later_start":
        selected["data"]["coverage"][0]["ranges"] = [[7, 10], [10, 12], [12, 14]]
    elif defect == "random":
        selected["mode_settings"]["start_policy"] = "random"
    elif defect == "cache":
        selected["mode_settings"]["context_latent_frames"] = 1
    elif defect == "history":
        selected["mode_settings"]["teacher_forcing"] = True
    elif defect == "precision":
        selected["training"]["global_sigma_dtype"] = "bfloat16"
    elif defect == "shape":
        selected["shape"]["height"] = 3
    elif defect == "base":
        selected["model"]["base_sha256"] = "b" * 64
    if defect is not None:
        with pytest.raises(ValueError, match="original E4 mode or calibrated function"):
            effect.check_calibration_contract(selected, original)
        return
    binding = effect.check_calibration_contract(selected, original)
    assert binding["relationship"] == "causal_clip_start_seven_frames"
    assert binding["original"] == {field: original[field] for field in ("shape", "mode_settings", "data")}
    assert binding["selected"] == {field: selected[field] for field in ("shape", "mode_settings", "data")}
    requests = {label: {"application_method": adapters.UNMERGED, "global_sigma_dtype": "float32", "mode": "causal",
                        "mode_settings": contract["mode_settings"], "model": contract["model"], "task": contract["task"],
                        "shape": {"channels": 128, "height": 2, "width": 2, "frames": 7}, "schedule": [0.725, 0],
                        "history_mode": "cache", "kv_source": "refresh"} for label, contract in contracts.items()}
    for label in paths:
        evaluate.check_adapter(paths[label], requests[label], product=True)
        other = "pilot" if label == "original" else "original"
        with pytest.raises(ValueError, match="incompatible adapter conditions"):
            evaluate.check_adapter(paths[label], requests[other], product=True)


def test_small_raw_error_cannot_hide_large_correction_error() -> None:
    outputs = matched_outputs()
    outputs["product"]["step1"] += 1e-4
    summary = effect.compare_paths(outputs)
    assert summary["paths"]["product"]["raw_output"]["passed"]
    assert not summary["paths"]["product"]["effect"]["passed"]
    assert not summary["passed"]


def test_under_five_percent_is_strict_and_not_historical_twenty_percent() -> None:
    reference = torch.ones(8, dtype=torch.float64)
    assert effect.effect_comparison(reference * 1.049, reference)["passed"]
    assert not effect.effect_comparison(reference * 1.05, reference)["passed"]
    assert not effect.effect_comparison(reference * 1.16, reference)["passed"]


def test_near_zero_effect_uses_absolute_rule_without_claiming_learned_effect() -> None:
    zero = torch.zeros(8, dtype=torch.float64)
    assert effect.effect_comparison(torch.full_like(zero, 1e-8), zero)["passed"]
    failed = effect.effect_comparison(torch.full_like(zero, 1.01e-8), zero)
    assert not failed["passed"]
    assert failed["relative_l2"] is None
    assert failed["near_zero"]
    summary = effect.compare_paths(matched_outputs(0))
    assert summary["function_matches"]
    assert not summary["learned_effect_demonstrated"]
    assert not summary["passed"]


def test_fused_failure_does_not_fail_selected_unmerged_paths() -> None:
    outputs = matched_outputs()
    outputs["fused"] = copy.deepcopy(outputs["reference"])
    outputs["fused"]["step1"] += 1.0
    summary = effect.compare_paths(outputs)
    assert summary["passed"]
    assert summary["fused_is_diagnostic_only"]
    assert not summary["paths"]["fused"]["effect"]["passed"]


@pytest.mark.parametrize("path", effect.PATHS)
def test_each_zero_control_must_equal_its_own_base(path: str) -> None:
    outputs = matched_outputs()
    outputs[path]["zero"][0, 0] += 1e-12
    assert not effect.compare_paths(outputs)["passed"]


@pytest.mark.parametrize("defect", ["missing", "shape", "nonfinite", "empty"])
def test_bad_outputs_refuse_comparison(defect: str) -> None:
    outputs = matched_outputs()
    if defect == "missing":
        del outputs["product"]["zero"]
    elif defect == "shape":
        outputs["product"]["step1"] = torch.ones(4)
    elif defect == "nonfinite":
        outputs["product"]["step1"][0, 0] = float("nan")
    else:
        outputs = {name: {state: torch.empty(0) for state in effect.STATES} for name in effect.PATHS}
    with pytest.raises(ValueError, match=r"requires|finite|outputs"):
        effect.compare_paths(outputs)


@pytest.fixture
def prepared_image(tmp_path: Path) -> tuple:
    producer = software.capture("preparation")
    image_path, pixels_path = tmp_path / "image.pt", tmp_path / "prepared_image.png"
    source_path, vae_path, guide_path = (tmp_path / name for name in ("original.png", "vae", "guide"))
    pixels = np.full((64, 64, 3), 200, dtype=np.uint8)
    Image.fromarray(pixels).save(source_path)
    Image.fromarray(pixels).save(pixels_path)
    vae_path.write_bytes(b"checked VAE source")
    guide_path.write_bytes(b"checked guide source")
    identities = {str(path.resolve()): sha256(path) for path in (source_path, vae_path, guide_path)}
    bundle = precompute.master_record(
        torch.zeros(1, 8, 1, 2, 2), source=str(source_path.resolve()), fps=30, pixel_frames=1,
        box_xyxy=(0, 0, 64, 64), edge=64, objective="bg", input_fingerprint=sha256(source_path),
        vae_fingerprint=sha256(vae_path))
    bundle.update(input_role="supplied_image", software=producer,
                  preparation={"input_sha256": identities,
                               "prepared_pixels_sha256": hashing.tensor_sha256(torch.from_numpy(pixels)),
                               "encoder": {"dtype": "bfloat16", "method": "tiled_encode", "tiling": None}})
    torch.save(bundle, image_path)
    record = {"kind": "onestep_avatar.supplied_image_preparation", "schema_version": 1,
              "software": producer, "inputs": identities, "image_latent_shape": list(bundle["master"].shape),
              "outputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                          for name, path in (("bundle", image_path), ("pixels", pixels_path))}}
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps(record))
    return path, image_path, pixels_path, source_path, bundle, record


def test_independent_image_preparation_binds_source_pixels_and_bundle(prepared_image: tuple) -> None:
    path, image, _, _, _, record = prepared_image
    identities = effect.image_preparation_evidence(path, image)
    assert identities[str(image.resolve())] == record["outputs"]["bundle"]["sha256"]
    assert identities[str(path.resolve())] == sha256(path)


@pytest.mark.parametrize("defect", ["bundle", "source", "pixels", "video_c0", "encoder", "shape"])
def test_changed_or_video_derived_c0_refused(prepared_image: tuple, defect: str) -> None:
    path, image, pixels, source, bundle, record = prepared_image
    if defect in ("bundle", "source", "pixels"):
        target = {"bundle": image, "source": source, "pixels": pixels}[defect]
        target.write_bytes(target.read_bytes() + b"changed")
    else:
        if defect == "video_c0":
            bundle["input_role"] = "capture_video_prefix"
        elif defect == "encoder":
            bundle["preparation"]["encoder"]["method"] = "slice_video"
        else:
            record["image_latent_shape"][-1] += 1
        torch.save(bundle, image)
        record["outputs"]["bundle"]["sha256"] = sha256(image)
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=r"different|changed|independent|shape"):
        effect.image_preparation_evidence(path, image)


@pytest.mark.parametrize("defect", ["failed", "reload", "job", "world", "budget"])
def test_failed_or_unrelated_e4_is_rejected_before_source_or_model_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    record = {"state": "passed", "comparison": {"passed": True, "actual_step_one_reload_exact": True},
              "protocol": {"queue_job_sha256": "original", "world_size": 4, "resource_budget": {"budget": 1}}}
    if defect == "failed":
        record["comparison"]["passed"] = False
    elif defect == "reload":
        record["comparison"]["actual_step_one_reload_exact"] = False
    elif defect == "job":
        record["protocol"]["queue_job_sha256"] = "forged"
    elif defect == "world":
        record["protocol"]["world_size"] = 2
    else:
        record["protocol"]["resource_budget"] = {"budget": 2}
    path = tmp_path / "failed_update.json"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(software, "check_current", lambda *_args: pytest.fail("source gate reached after failed E4"))
    with pytest.raises(ValueError, match="passed original E4"):
        effect.validate_update_result(path, {"sha256": "original", "processes": 4}, {}, {"budget": 1})


def test_used_output_refuses_preflight_without_reading_job(tmp_path: Path) -> None:
    args = SimpleNamespace(output=tmp_path)
    with pytest.raises(ValueError, match="fresh output"):
        effect.prepare(args)


def test_changed_input_refuses_publication_without_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "noise.pt"
    torch.save(torch.zeros(1), path)
    prepared = {"software": {}, "budget": None, "pins": {str(path): sha256(path)}}
    monkeypatch.setattr(software, "check_current", lambda _record: None)
    torch.save(torch.ones(1), path)
    with pytest.raises(ValueError, match="inputs changed"):
        effect.check_current(prepared)


@pytest.mark.parametrize("defect", [None, "checkpoint", "missing", "late_input"])
def test_passed_e4_gate_binds_exact_real_files(tmp_path: Path, defect: str | None) -> None:
    paths = {name: tmp_path / f"{name}.safetensors" for name in ("zero", "step1")}
    for path in paths.values():
        save_file({"matrix": torch.zeros(2, 2)}, path)
    noise = tmp_path / "noise.pt"
    torch.save(torch.randn(1, generator=torch.Generator().manual_seed(42)), noise)
    identities = {str(path.resolve()): sha256(path) for path in (*paths.values(), noise)}
    if defect == "checkpoint":
        identities[str(paths["step1"].resolve())] = "0" * 64
    elif defect == "missing":
        del identities[str(paths["zero"].resolve())]
    elif defect == "late_input":
        torch.save(torch.ones(1), noise)
    record = {"state": "passed", "comparison": {"passed": True, "actual_step_one_reload_exact": True},
              "protocol": {"queue_job_sha256": "original", "world_size": 4, "resource_budget": {"budget": 1},
                           "software": software.capture("training", "bidirectional"), "input_files": identities}}
    path = tmp_path / "passed_update.json"
    path.write_text(json.dumps(record))
    if defect is None:
        assert effect.validate_update_result(path, {"sha256": "original", "processes": 4},
                                             paths, {"budget": 1}) == record
    else:
        with pytest.raises(ValueError, match=r"exact|changed"):
            effect.validate_update_result(path, {"sha256": "original", "processes": 4}, paths, {"budget": 1})


@pytest.mark.parametrize("defect", [None, "legacy", "missing", "unsupported", "rank_disagreement"])
def test_original_policy_is_explicit_and_identical_across_native_ranks(defect: str | None) -> None:
    policy = {**numerics.POLICY, "cudnn_allow_tf32": True}
    runtime = {"schema_version": 2, "ranks": [{"numerics": copy.deepcopy(policy)} for _ in range(4)]}
    if defect == "legacy":
        runtime["schema_version"] = 1
    elif defect == "missing":
        runtime["ranks"][0].clear()
    elif defect == "unsupported":
        for rank in runtime["ranks"]:
            rank["numerics"]["allow_tf32"] = True
    elif defect == "rank_disagreement":
        runtime["ranks"][-1]["numerics"]["cudnn_allow_tf32"] = False
    if defect is None:
        assert effect.native_numerics({"runtime": runtime}) == policy
    else:
        with pytest.raises(ValueError, match=r"schema-two|malformed|deterministic|original"):
            effect.native_numerics({"runtime": runtime})


def evaluation_membership(root: Path, sources: list[str], *, objective: str = "white") -> dict:
    record = {"schema_version": 2, "kind": subset.KIND, "objective": objective, "corpus_root": str(root),
              "splits": {"test": ["actor"]}, "excluded": {},
              "sources": [{"relative_dir": source, "actor": "actor", "split": "test", "fps": 30,
                           "shape": [128, 7, 2, 2], "n_latent_frames": 7} for source in sources]}
    record["sha256"] = subset.membership_hash(record)
    return record


@pytest.mark.parametrize("defect", [None, "job", "runtime", "launch", "config", "snapshot", "foreign_path"])
def test_intermediate_pilot_export_requires_its_own_run_marker_and_native_snapshot(  # noqa: PLR0915 -- actual file controls
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str | None
) -> None:
    """Real export/marker/journal files check the gate; synthetic journals prove no native run."""
    membership = evaluation_membership(tmp_path, ["actor/views/view01_cam57"])
    membership["splits"] = {"train": ["actor"]}
    membership["sources"][0]["split"] = "train"
    membership["sha256"] = subset.membership_hash(membership)
    store = dataset.ClipStore(membership)
    settings = config.RunSettings("bidirectional", tmp_path / "subset.json", tmp_path / "pilot",
                                  config.BidirectionalSettings(7), variant="dev", objective="white", steps=60,
                                  lora_rank=2, lora_alpha=2, save_initial=True, save_every=20)
    settings.base_identity = {"base_transformer_file": "base.safetensors", "base_transformer_sha256": "a" * 64}
    plan = config.build_frame_plan(settings, membership, SpatioTemporalScaleFactors(8, 32, 32))
    settings.output.mkdir()
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(json.dumps({"wall_seconds_per_phase": 1800,
                                       "memory_limit_allocated_bytes": 48_000_000_000}))
    budget = resources.read_budget(budget_path)
    saved = {"software": software.capture("training", "bidirectional"), "producer_source_sha256": "b" * 64,
             "queue_launch": {"controlled_launch": True}, "runtime": {"controlled_policy": True}}
    (settings.output / "config.json").write_text(json.dumps(saved))
    (settings.output / "frame_plan.json").write_text(json.dumps(plan))
    job = {"sha256": "original-pilot", "processes": 1}
    selected, contracts = {}, {}
    for state, step in (("zero", 0), ("step20", 20), ("step60", 60)):
        contract = checkpoints.make_contract(settings, membership, plan, step)
        contract["adapter"]["tensor_shapes"] = _contract()["adapter"]["tensor_shapes"]
        checkpoint = settings.output / "checkpoints" / f"lora_weights_step_{step:05d}.safetensors"
        checkpoint.parent.mkdir(exist_ok=True)
        save_file({name: torch.zeros(shape, dtype=torch.bfloat16)
                   for name, shape in contract["adapter"]["tensor_shapes"].items()}, checkpoint,
                  metadata={checkpoints.CONTRACT_KEY: json.dumps(contract)})
        journal = settings.output / "resource_snapshots" / f"step_{step:05d}" / "resources_rank0.jsonl"
        journal.parent.mkdir(parents=True)
        measurements = [{"schema_version": 1, "phase": phase, "rank": 0, "device": "cuda:0", "elapsed_s": 0.1,
                         "peak_allocated_bytes": 1, "peak_reserved_bytes": 1,
                         "budget_sha256": budget["sha256"], "state": "passed", "error": None}
                        for phase in resources.training_phases(settings, step=step)]
        journal.write_text("".join(json.dumps(row) + "\n" for row in measurements))
        marker = {"schema_version": 2, "state": "complete", "step": step, "path": str(checkpoint),
                  "sha256": sha256(checkpoint), "queue_job_sha256": job["sha256"], **saved,
                  "resource_budget": budget, "resource_evidence": {str(journal.resolve()): sha256(journal)},
                  "training_record": {"config_sha256": sha256(settings.output / "config.json"),
                                      "frame_plan_sha256": sha256(settings.output / "frame_plan.json")}}
        if step == 20:
            if defect == "job":
                marker["queue_job_sha256"] = "foreign-run"
            elif defect == "runtime":
                marker["runtime"] = {"changed_policy": True}
            elif defect == "launch":
                marker["queue_launch"] = {"changed_launch": True}
            elif defect == "config":
                marker["training_record"]["config_sha256"] = "c" * 64
            elif defect == "snapshot":
                journal.write_text(journal.read_text().replace('"elapsed_s": 0.1', '"elapsed_s": 0.2'))
            elif defect == "foreign_path":
                replacement = tmp_path / "foreign_step20.safetensors"
                replacement.write_bytes(checkpoint.read_bytes())
                checkpoint = replacement
                marker["path"] = str(checkpoint)
        checkpoint.with_suffix(".complete.json").write_text(json.dumps(marker))
        selected[state], contracts[state] = checkpoint.resolve(), contract
    checked = []
    monkeypatch.setattr(effect.engine, "verify_training_conditions", lambda *_args: checked.append(_args[1]))
    if defect is None:
        evidence = effect.verify_selected_checkpoints(job, settings, store, plan, selected, contracts, saved, budget)
        assert len(evidence) == 3
        assert all(sha256(Path(path)) == digest for path, digest in evidence.items())
    else:
        with pytest.raises(ValueError, match=r"original queue/run binding|resource snapshot differs"):
            effect.verify_selected_checkpoints(job, settings, store, plan, selected, contracts, saved, budget)
    assert checked == [selected["step60"]]


@pytest.mark.parametrize("defect", [None, "objective", "identity"])
def test_separate_test_membership_preserves_original_calibration(tmp_path: Path, defect: str | None) -> None:
    original_membership = evaluation_membership(tmp_path, ["actor/views/view01_cam57"])
    original = dataset.ClipStore(original_membership)
    before = copy.deepcopy(original_membership)
    assert effect.evaluation_store(original, None, None) is original
    evaluation = evaluation_membership(tmp_path, ["heldout/views/view00_cam51", "heldout/views/view01_cam52"])
    if defect == "objective":
        evaluation["objective"] = "bg"
        evaluation["sha256"] = subset.membership_hash(evaluation)
    elif defect == "identity":
        evaluation["sources"][0]["fps"] = 31
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(evaluation))
    if defect is None:
        selected = effect.evaluation_store(original, path, None)
        assert set(selected.sources) == {"heldout/views/view00_cam51", "heldout/views/view01_cam52"}
        assert selected.sources["heldout/views/view00_cam51"]["split"] == "test"
        assert original.membership == before
    else:
        with pytest.raises(ValueError, match=r"objective|SHA-256"):
            effect.evaluation_store(original, path, None)


def test_camera_coverage_distinguishes_two_clips_from_two_camera_views() -> None:
    assert effect.camera_views(["0007_01/views/view01_cam57", "0007_04/views/view01_cam57"]) == ["cam57"]
    assert effect.camera_views(["0097_04/views/view00_cam51", "0097_04/views/view01_cam52"]) == ["cam51", "cam52"]
    assert effect.camera_views(["0007_01/views/view00_cam57", "0007_04/views/view01_cam57"]) == ["cam57"]
    assert effect.camera_views(["unknown/view"]) == []


@pytest.fixture(params=[False, True], ids=["one_update", "pilot"])
def saved_effect(tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Controlled CPU tensors/journals test integrity; they are not native acceptance."""
    directory = tmp_path / "effect"
    directory.mkdir()
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(json.dumps({"wall_seconds_per_phase": 1800, "memory_limit_allocated_bytes": 48_000_000_000}))
    budget = resources.read_budget(budget_path)
    pilot = request.param
    states = effect.PILOT_STATES if pilot else effect.STATES
    steps = [0, 20, 60] if pilot else [0, 1]
    selected = {}
    for state, step in zip(states[1:], steps, strict=True):
        contract = _contract()
        contract["adapter"]["step"] = step
        values = {name: torch.ones(shape, dtype=torch.bfloat16)
                  for name, shape in contract["adapter"]["tensor_shapes"].items()}
        for name, value in values.items():
            if ".lora_B." in name:
                value.fill_(0 if step == 0 else 0.125)
        checkpoint = tmp_path / f"{state}.safetensors"
        save_file(values, checkpoint, metadata={checkpoints.CONTRACT_KEY: json.dumps(contract)})
        selected[state] = {"path": str(checkpoint), "sha256": sha256(checkpoint), "contract": contract}
    noise_path = tmp_path / "noise.pt"
    torch.save(torch.zeros(1, 28, 128, dtype=torch.bfloat16), noise_path)
    source = "actor/views/view01_cam57"
    membership = evaluation_membership(tmp_path, [source])
    membership_path = tmp_path / "evaluation.json"
    membership_path.write_text(json.dumps(membership))
    membership_identity = {"path": str(membership_path), "sha256": sha256(membership_path),
                           "membership_sha256": membership["sha256"]}
    pins = {str(path): sha256(path) for path in (budget_path, noise_path, membership_path,
            *(Path(item["path"]) for item in selected.values()))}
    policy = {**numerics.POLICY, "cudnn_allow_tf32": True}
    producer = software.capture("evaluation", "bidirectional", extra_sources=effect.EXTRA_SOURCES)
    protocol = {"kind": "onestep_avatar.adapter_effect_check", "schema_version": 1,
                "scope": "shared_API_adapter_correction", "software": producer,
                "input_files": pins, "resource_budget": budget, "checkpoints": selected,
                "evaluation_membership": membership_identity,
                "tolerance": effect.TOLERANCE, "schedule": [0.725, 0.0], "seed": 42,
                "runtime": {"world_size": 1, "device": "cuda:0", "numerics": policy}}
    if pilot:
        # This fixture tests the saved array/journal reader. Actual producer lineage has separate controls.
        def check_protocol(_protocol: dict, _prepared: dict, paths: dict, contracts: dict) -> None:
            assert set(paths) == {"zero", "step20", "step60"}
            assert sorted(row["adapter"]["step"] for row in contracts.values()) == steps

        monkeypatch.setattr(effect, "_verify_pilot_protocol", check_protocol)
    (directory / "protocol.json").write_text(json.dumps(protocol))
    conditions = {"application_method": adapters.UNMERGED, "global_sigma_dtype": "float32",
                  "mode": "bidirectional", "mode_settings": asdict(config.BidirectionalSettings(7)),
                  "model": contract["model"], "task": contract["task"],
                  "shape": {"channels": 128, "height": 2, "width": 2, "frames": 7},
                  "schedule": [0.725, 0.0]}
    base = torch.zeros(1, 128, 7, 2, 2, dtype=torch.bfloat16)
    trained = base.clone()
    trained[:, :, 1:] = 0.001
    outputs = {path: {"base": base.clone(), "zero": base.clone(),
                      **{state: trained.clone() * (2 if state == "step60" else 1) for state in states[2:]}}
               for path in effect.PATHS}
    raw, journals = {}, []
    for path in effect.PATHS:
        raw[path] = {}
        for state in states:
            record = {"path": path, "adapter_state": state, "source": source, "source_coverage": [0, 7],
                      "fps": 30, "seed": 42, "software": producer, "input_files": pins, "numerics": policy,
                      "conditions": conditions, "checkpoint": None if state == "base" else selected[state]["path"],
                      "application_method": "base" if state == "base" else adapters.UNMERGED,
                      "call_counts": {"model_calls": 1},
                      **dict.fromkeys(("guide_sha256", "c0_sha256", "noise_sha256", "text_sha256"), "1" * 64)}
            raw[path][state] = evaluate.save_case(outputs[path][state], record, directory / "case_00" / path / state)
            for operation in ("load", "sample"):
                journals.append({"schema_version": 1, "phase": f"{operation}:{source}:{path}:{state}",
                                 "rank": 0, "device": "cuda:0", "elapsed_s": 0.1,
                                 "peak_allocated_bytes": 1, "peak_reserved_bytes": 1,
                                 "budget_sha256": budget["sha256"], "state": "passed", "error": None})
    journal = directory / "case_00/resources_rank0.jsonl"
    journal.write_text("".join(json.dumps(item) + "\n" for item in journals))
    result = {"schema_version": 1, "protocol": protocol, "state": "passed", "e2_complete": False,
              "coverage": {"sources": 1, "views": 1, "camera_labels": ["cam57"],
                           "checkpoint_steps": steps, "decoded": False},
              "cases": [{"source": source, "source_coverage": [0, 7], "comparison": effect.compare_paths(outputs),
                         "raw_records": raw, "renderings": []}], "resource_measurements": journals,
              "output_files": {str(path.relative_to(directory)): sha256(path)
                               for path in directory.rglob("*") if path.is_file()}}
    (directory / "result.json").write_text(json.dumps(result))
    return directory, result, noise_path


def test_saved_reader_recomputes_real_arrays_and_journals_without_restamping(saved_effect: tuple) -> None:
    directory, result, _ = saved_effect
    before = {str(path): sha256(path) for path in directory.rglob("*") if path.is_file()}
    assert effect.verify_saved(directory) == result
    assert {str(path): sha256(path) for path in directory.rglob("*") if path.is_file()} == before
    assert not result["e2_complete"]


def test_saved_near_zero_match_preserves_failed_learned_effect_scope(saved_effect: tuple) -> None:
    directory, result, _noise = saved_effect
    case = result["cases"][0]
    outputs = {}
    for path, rows in case["raw_records"].items():
        base = torch.load(rows["base"]["output"]["path"], weights_only=True)
        outputs[path] = {}
        for state, record in rows.items():
            if state not in ("base", "zero"):
                destination = Path(record["output"]["path"]).parent
                rows[state] = evaluate.save_case(base.clone(), record, destination)
                for artifact in (destination / "generated.pt", destination / "result.json"):
                    result["output_files"][str(artifact.relative_to(directory))] = sha256(artifact)
            outputs[path][state] = base.clone()
    case["comparison"] = effect.compare_paths(outputs)
    result["state"] = "failed"
    (directory / "result.json").write_text(json.dumps(result))
    before = {str(path): sha256(path) for path in directory.rglob("*") if path.is_file()}
    verified = effect.verify_saved(directory)
    assert verified["state"] == "failed"
    assert verified["cases"][0]["comparison"]["function_matches"]
    assert not verified["cases"][0]["comparison"]["learned_effect_demonstrated"]
    assert not verified["e2_complete"]
    assert {str(path): sha256(path) for path in directory.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("defect", ["array", "missing", "redirect", "summary", "input", "calls", "resource",
                                  "complete", "policy", "checkpoint", "dtype", "coverage"])
def test_saved_reader_rejects_changed_evidence_even_if_summary_is_restated(  # noqa: PLR0912 -- independent corruptions
    saved_effect: tuple, defect: str
) -> None:
    directory, result, noise = saved_effect
    trained_state = "step60" if "step60" in result["protocol"]["checkpoints"] else "step1"
    raw = result["cases"][0]["raw_records"]["product"][trained_state]
    artifact = Path(raw["output"]["path"])
    if defect == "array":
        torch.save(torch.ones_like(torch.load(artifact, weights_only=True)), artifact)
    elif defect == "missing":
        artifact.unlink()
    elif defect == "redirect":
        raw["output"]["path"] = str(noise)
    elif defect == "summary":
        comparison = result["cases"][0]["comparison"]
        step = comparison["steps"][trained_state] if "steps" in comparison else comparison
        step["paths"]["product"]["effect"]["relative_l2"] = 0.5
    elif defect == "input":
        torch.save(torch.ones(1), noise)
    elif defect == "calls":
        raw["call_counts"]["model_calls"] = 2
    elif defect == "resource":
        result["resource_measurements"][-1]["peak_allocated_bytes"] = 2
    elif defect == "complete":
        result["e2_complete"] = True
    elif defect == "policy":
        raw["numerics"] = {**raw["numerics"], "cudnn_allow_tf32": False}
    elif defect == "checkpoint":
        raw["checkpoint"] = result["protocol"]["checkpoints"]["zero"]["path"]
    elif defect == "dtype":
        torch.save(torch.load(artifact, weights_only=True).float(), artifact)
        raw["output"]["sha256"] = sha256(artifact)
        artifact.with_name("result.json").write_text(json.dumps(raw))
        for path in (artifact, artifact.with_name("result.json")):
            result["output_files"][str(path.relative_to(directory))] = sha256(path)
    else:
        result["coverage"]["views"] = 2
    (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises((ValueError, FileNotFoundError), match=r"adapter-effect|changed"):
        effect.verify_saved(directory)
