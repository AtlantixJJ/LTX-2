"""Compare saved adapter corrections through shared APIs; see doc/experiments/adapter_effect_check.md."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image
from safetensors.torch import load_file

from scripts.onestep_avatar import evaluate, hashing, infer, media
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.execution import queue, software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters, common
from scripts.onestep_avatar.training import checkpoints, config, engine, resources

if TYPE_CHECKING:
    from scripts.prune.core.session import Session

ENTRY = "scripts/onestep_avatar/experiments/adapter_effect_check.py"
EXTRA_SOURCES = (ENTRY, 'scripts/onestep_avatar/experiments/__init__.py',
                 'scripts/onestep_avatar/infer.py',
                 'scripts/onestep_avatar/training/engine.py',
                 'scripts/onestep_avatar/training/resources.py',
                 'scripts/onestep_avatar/training/numerics.py',
                 'scripts/onestep_avatar/training/runtime.py',
                 )
TOLERANCE = {"relative_effect_l2": 0.05, "near_zero_effect_rms": 1e-8, "absolute_effect_rms": 1e-8}
PATHS = ("reference", "evaluation", "product")
STATES = ("base", "zero", "step1")
PILOT_STATES = ("base", "zero", "step20", "step60")


def checkpoint_selection(paths: list[Path], *, pilot: bool = False) -> tuple[dict, dict]:
    """Select exact same-lineage controls; a step label never replaces actual matrix checks."""
    expected = (0, 20, 60) if pilot else (0, 1)
    selected, contracts = {}, {}
    for source in paths:
        path = source.resolve()
        contract = checkpoints.read_contract(path)
        checkpoints.validate_adapter_tensors(path, contract)
        step = contract["adapter"]["step"]
        if step in selected:
            raise ValueError("adapter-effect checkpoints contain a duplicate step")
        selected[step], contracts[step] = path, contract
    if set(selected) != set(expected):
        raise ValueError(f"adapter-effect comparison requires exactly checkpoint steps {list(expected)}")
    canonical = json.loads(json.dumps(contracts[0]))
    canonical["adapter"]["step"] = 0
    for contract in contracts.values():
        normalized = json.loads(json.dumps(contract))
        normalized["adapter"]["step"] = 0
        if normalized != canonical:
            raise ValueError("adapter-effect checkpoint scientific contracts or lineage differ")
    labels = {step: "zero" if step == 0 else f"step{step}" for step in expected}
    return ({labels[step]: selected[step] for step in expected},
            {labels[step]: contracts[step] for step in expected})


def effect_comparison(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    """Assess complete corrections under a fixed rule, never a raw-output proxy."""
    if (actual.shape != reference.shape or actual.numel() == 0
            or not torch.isfinite(actual).all() or not torch.isfinite(reference).all()):
        raise ValueError("effect comparison requires matching nonempty finite arrays")
    actual, reference = actual.double(), reference.double()
    difference = actual - reference
    reference_rms = float(reference.square().mean().sqrt())
    absolute_rms = float(difference.square().mean().sqrt())
    near_zero = reference_rms < TOLERANCE["near_zero_effect_rms"]
    relative = None if near_zero else absolute_rms / reference_rms
    return {"reference_norm": float(reference.norm()), "actual_norm": float(actual.norm()),
            "reference_rms": reference_rms, "actual_rms": float(actual.square().mean().sqrt()),
            "difference_rms": absolute_rms, "maximum_absolute_difference": float(difference.abs().max()),
            "near_zero": near_zero, "relative_l2": relative,
            "passed": (absolute_rms <= TOLERANCE["absolute_effect_rms"] if near_zero
                       else relative < TOLERANCE["relative_effect_l2"])}


def compare_paths(outputs: dict[str, dict[str, torch.Tensor]]) -> dict:
    """Require each selected trained correction separately; retain fusion separately."""
    if set(outputs) not in (set(PATHS), {*PATHS, "fused"}):
        raise ValueError("adapter-effect comparison requires reference, evaluation and product")
    states = tuple(outputs["reference"])
    if set(states) not in (set(STATES), set(PILOT_STATES)) or any(
        set(rows) != set(states) for rows in outputs.values()
    ):
        raise ValueError("every adapter path requires base, zero and exact step1 or step20/step60 outputs")
    reference = outputs["reference"]["base"]
    for rows in outputs.values():
        if any(
            value.shape != reference.shape or value.numel() == 0 or not torch.isfinite(value).all()
            for value in rows.values()
        ):
            raise ValueError("every adapter path requires matching nonempty finite outputs")
    if set(states) == set(STATES):
        return _compare_step(outputs, "step1")
    comparisons = {state: _compare_step(outputs, state) for state in PILOT_STATES[2:]}
    return {"tolerance": TOLERANCE, "steps": comparisons,
            "function_matches": all(row["function_matches"] for row in comparisons.values()),
            "learned_effect_demonstrated": all(row["learned_effect_demonstrated"] for row in comparisons.values()),
            "passed": all(row["passed"] for row in comparisons.values()),
            "fused_is_diagnostic_only": "fused" in outputs}


def _compare_step(outputs: dict[str, dict[str, torch.Tensor]], state: str) -> dict:
    correction = outputs["reference"][state].double() - outputs["reference"]["base"].double()
    comparisons = {}
    for name, rows in outputs.items():
        effect = rows[state].double() - rows["base"].double()
        comparisons[name] = {
            "zero_equals_base_bitwise": torch.equal(rows["zero"], rows["base"]),
            "effect": effect_comparison(effect, correction),
            "raw_output": effect_comparison(rows[state], outputs["reference"][state]),
        }
    function_matches = all(comparisons[name]["zero_equals_base_bitwise"]
                           and comparisons[name]["effect"]["passed"] for name in PATHS)
    measurable = not comparisons["reference"]["effect"]["near_zero"]
    return {"tolerance": TOLERANCE, "paths": comparisons, "function_matches": function_matches,
            "learned_effect_demonstrated": measurable, "passed": function_matches and measurable,
            "fused_is_diagnostic_only": "fused" in outputs}


def validate_update_result(path: Path, job: dict, checkpoints_by_state: dict, budget: dict) -> dict:
    """Reject failed or unrelated E4 calibration before any transformer loading."""
    record = json.loads(path.read_text())
    protocol, comparison = record.get("protocol", {}), record.get("comparison", {})
    if (record.get("state") != "passed" or comparison.get("passed") is not True
            or comparison.get("actual_step_one_reload_exact") is not True
            or protocol.get("queue_job_sha256") != job["sha256"]
            or protocol.get("world_size") != job["processes"]
            or protocol.get("resource_budget") != budget):
        raise ValueError("adapter-effect calibration requires the passed original E4 update comparison")
    software.check_current(protocol.get("software"))
    identities = protocol.get("input_files")
    if not isinstance(identities, dict) or not identities:
        raise ValueError("E4 update input evidence is missing")
    if any(identities.get(str(checkpoint.resolve())) != sha256(checkpoint)
           for checkpoint in checkpoints_by_state.values()):
        raise ValueError("E4 update did not verify these exact zero/step1 checkpoints")
    if any(sha256(Path(name)) != digest for name, digest in identities.items()):
        raise ValueError("E4 update inputs changed after numerical acceptance")
    return record


def image_preparation_evidence(path: Path, image: Path) -> dict[str, str]:
    """Bind the independent image producer, not merely a claimed bundle role."""
    record = json.loads(path.read_text())
    if record.get("kind") != "onestep_avatar.supplied_image_preparation" or record.get("schema_version") != 1:
        raise ValueError("c0 requires the actual supplied-image preparation result")
    software.validate(record.get("software"))
    if not isinstance(record.get("inputs"), dict) or not {"bundle", "pixels"} <= set(record.get("outputs", {})):
        raise ValueError("supplied-image preparation requires original inputs and bundle/pixels outputs")
    output = record["outputs"]["bundle"]
    if output.get("path") != str(image.resolve()) or output.get("sha256") != sha256(image):
        raise ValueError("supplied-image preparation identifies a different c0 bundle")
    bundle = torch.load(image, map_location="cpu", weights_only=True)
    if (bundle.get("input_role") != "supplied_image" or bundle.get("pixel_frames") != 1
            or bundle.get("software") != record["software"]
            or bundle.get("preparation", {}).get("input_sha256") != record.get("inputs")
            or bundle.get("preparation", {}).get("encoder") != {
                "dtype": "bfloat16", "method": "tiled_encode", "tiling": None}):
        raise ValueError("c0 is not the original independently encoded supplied-image bundle")
    if (record["inputs"].get(bundle.get("source")) != bundle.get("input_fingerprint")
            or record.get("image_latent_shape") != list(bundle["master"].shape)):
        raise ValueError("supplied-image source fingerprint or recorded shape differs")
    import numpy as np  # noqa: PLC0415 -- read prepared pixel evidence without another encode

    with Image.open(record["outputs"]["pixels"]["path"]) as image_pixels:
        if image_pixels.mode != "RGB":
            raise ValueError("prepared supplied-image pixels must be RGB")
        pixels_hash = hashing.tensor_sha256(torch.from_numpy(np.array(image_pixels)))
    if pixels_hash != bundle["preparation"].get("prepared_pixels_sha256"):
        raise ValueError("prepared image pixels differ from the independent encode record")
    identities = {str(path.resolve()): sha256(path), **record["inputs"]}
    for output in record["outputs"].values():
        if not isinstance(output, dict) or set(output) != {"path", "sha256"}:
            raise ValueError("supplied-image output evidence is malformed")
        identities[output["path"]] = output["sha256"]
    if any(sha256(Path(name)) != digest for name, digest in identities.items()):
        raise ValueError("supplied-image preparation inputs or outputs changed")
    return identities


def _path(value: object, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("adapter-effect case paths must be nonempty strings")
    return (root / value).resolve()


def _context(settings: config.RunSettings, saved: dict) -> torch.Tensor:
    path = settings.output / "update_states/text.pt"
    expected = saved.get("update_text", {})
    if expected.get("path") != str(path.resolve()) or expected.get("sha256") != sha256(path):
        raise ValueError("saved training context file differs from its producer")
    context = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(context, torch.Tensor) or context.dtype != torch.bfloat16
            or not torch.isfinite(context).all() or expected.get("shape") != list(context.shape)
            or expected.get("dtype") != str(context.dtype)
            or expected.get("tensor_sha256") != hashlib.sha256(
                context.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()):
        raise ValueError("saved training context values, shape or dtype differ")
    return context


def native_numerics(saved: dict) -> dict:
    """Require explicit actual native policy; historical absence never picks a default."""
    from scripts.onestep_avatar.training import runtime  # noqa: PLC0415 -- one original applied-policy reader

    return runtime.numerical_policy(saved.get("runtime"))


def evaluation_store(original: dataset.ClipStore, path: Path | None, corpus_root: Path | None) -> dataset.ClipStore:
    """Choose separately checked evaluation membership without rewriting calibration."""
    selected = original if path is None else dataset.ClipStore(json.loads(path.read_text()), corpus_root)
    if selected.objective != original.objective:
        raise ValueError("adapter-effect evaluation membership has a different objective")
    return selected


def camera_views(sources: list[str]) -> list[str]:
    """Count recorded camera labels; repeated clips never become distinct views."""
    return sorted({"cam" + suffix for source in sources
                   if (suffix := Path(source).name.rpartition("_cam")[2]).isdigit()})


def verify_selected_checkpoints(job: dict, settings: config.RunSettings, store: dataset.ClipStore,
                                plan: dict, selected: dict, contracts: dict, saved: dict, budget: dict) -> dict:
    """Bind every actual export to its own training run and immutable resource snapshot."""
    final = f"step{settings.steps}"
    engine.verify_training_conditions(job, selected[final], contracts[final])
    identities = {}
    expected_marker = {"queue_job_sha256": job["sha256"], "software": saved["software"],
                       "producer_source_sha256": saved["producer_source_sha256"],
                       "queue_launch": saved["queue_launch"], "runtime": saved["runtime"],
                       "resource_budget": budget,
                       "training_record": {"config_sha256": sha256(settings.output / "config.json"),
                                           "frame_plan_sha256": sha256(settings.output / "frame_plan.json")}}
    for state, path in selected.items():
        step = contracts[state]["adapter"]["step"]
        expected_path = settings.output / "checkpoints" / f"lora_weights_step_{step:05d}.safetensors"
        expected = checkpoints.make_contract(settings, store.membership, plan, step)
        expected["adapter"]["tensor_shapes"] = contracts[state]["adapter"]["tensor_shapes"]
        marker = queue.read_training_marker(path, step)
        if (path.resolve() != expected_path.resolve() or contracts[state] != expected
                or any(marker.get(key) != value for key, value in expected_marker.items())):
            raise ValueError("selected checkpoint conditions or original queue/run binding differ")
        software.check_current(marker["software"])
        records, evidence = resources.read_records(settings.output, job["processes"], step=step)
        resources.validate_records(records, job["processes"], resources.training_phases(settings, step=step), budget)
        if marker.get("resource_evidence") != evidence:
            raise ValueError("selected checkpoint resource snapshot differs from its native marker")
        identities.update(evidence)
    return identities


def check_calibration_contract(selected: dict, original: dict) -> dict:
    """Separate membership/update lineage while retaining the original mode's function."""
    fields = ("mode", "attention", "model", "task", "training", "causal")
    adapter_fields = ("rank", "alpha", "target", "target_modules", "application_method", "tensor_shapes", "parent")
    shape, mode = dict(selected["shape"]), dict(selected["mode_settings"])
    restricted = (selected["mode"] == original["mode"] == "causal"
                  and selected["shape"]["frame_counts"] == [7]
                  and original["shape"]["frame_counts"] in ([7], [6, 7])
                  and selected["mode_settings"]["span_latent_frames"] == 7
                  and original["mode_settings"]["span_latent_frames"] is None
                  and selected["mode_settings"]["start_policy"] == original["mode_settings"]["start_policy"]
                  == "clip_start"
                  and all(selected["mode_settings"].get(field) == value for field, value in (
                      ("block_latent_frames", 2), ("blocks_per_sample", 3), ("context_latent_frames", 8),
                      ("teacher_forcing", False)))
                  and all(row["ranges"] == [[0, 3], [3, 5], [5, 7]] for row in selected["data"]["coverage"]))
    if restricted:
        shape["frame_counts"] = original["shape"]["frame_counts"]
        mode["span_latent_frames"] = None
    if (any(selected.get(field) != original.get(field) for field in fields)
            or shape != original["shape"] or mode != original["mode_settings"]
            or any(selected["adapter"].get(field) != original["adapter"].get(field) for field in adapter_fields)):
        raise ValueError("pilot adapter differs from the original E4 mode or calibrated function")
    return {"relationship": "causal_clip_start_seven_frames" if restricted else "same_function_separate_lineage",
            "original": {field: original[field] for field in ("shape", "mode_settings", "data")},
            "selected": {field: selected[field] for field in ("shape", "mode_settings", "data")}}


def prepare(args: argparse.Namespace) -> dict:  # noqa: PLR0912, PLR0915 -- complete pre-weight evidence gates
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("adapter-effect comparison requires a fresh output directory")
    job = queue.prepare_job(json.loads(args.job.read_text()), args.job.resolve().parent)
    if job["kind"] != "train":
        raise ValueError("adapter-effect calibration must come from a checked training job")
    settings = config.parse_settings(job["arguments"])
    update_job_path = getattr(args, "update_job", None)
    pilot = update_job_path is not None
    if settings.guide_mode != "d1" or settings.steps != (60 if pilot else 1) or settings.init_adapter is not None:
        raise ValueError("bounded E2 requires fresh D1 one-update calibration or the separate 60-update pilot")
    if settings.mode == "causal" and settings.mode_settings.teacher_forcing:
        raise ValueError("product comparison requires generated-history calibration")
    budget = resources.read_budget(settings.resource_budget)
    if budget is None:
        raise ValueError("adapter-effect comparison requires the original native resource budget")
    checkpoint_paths, contracts = checkpoint_selection(args.checkpoints, pilot=pilot)
    update_job = job if not pilot else queue.prepare_job(
        json.loads(update_job_path.read_text()), update_job_path.resolve().parent)
    update_settings = config.parse_settings(update_job["arguments"])
    if (update_job["kind"] != "train" or update_settings.steps != 1 or update_settings.guide_mode != "d1"
            or update_settings.init_adapter is not None or update_settings.mode != settings.mode):
        raise ValueError("pilot requires the original fresh one-update E4 calibration job for its mode")
    update_paths = (checkpoint_paths if not pilot else {
        "zero": update_settings.output / "checkpoints/lora_weights_step_00000.safetensors",
        "step1": update_settings.output / "checkpoints/lora_weights_step_00001.safetensors"})
    update_budget = resources.read_budget(update_settings.resource_budget)
    update = validate_update_result(args.update_check, update_job, update_paths, update_budget)
    store, plan, specification, _used = engine.prepare_run(settings, require_fresh_output=False)
    settings.world_size = job["processes"]
    saved = json.loads((settings.output / "config.json").read_text())
    export_evidence = verify_selected_checkpoints(
        job, settings, store, plan, checkpoint_paths, contracts, saved, budget)
    checkpoints.assert_exported_lora_is_noop(load_file(str(checkpoint_paths["zero"])))
    original_numerics = native_numerics(saved)
    condition_binding = None
    if pilot:
        original_contract = checkpoints.read_contract(update_paths["step1"])
        engine.verify_training_conditions(update_job, update_paths["step1"], original_contract)
        condition_binding = check_calibration_contract(contracts["step60"], original_contract)
        calibration_saved = json.loads((update_settings.output / "config.json").read_text())
        if native_numerics(calibration_saved) != original_numerics:
            raise ValueError("pilot numerical policy differs from the original E4 calibration")
    context = _context(settings, saved)
    inputs = json.loads(args.inputs.read_text())
    if (inputs.get("schema_version") != 1 or inputs.get("kind") != "onestep_avatar.adapter_effect_inputs"
            or not isinstance(inputs.get("cases"), list) or not inputs["cases"]
            or type(inputs.get("seed")) is not int):
        raise ValueError("adapter-effect input specification is incomplete")
    evaluation_path = (args.subset or settings.subset).resolve()
    store = evaluation_store(store, args.subset, settings.corpus_root)
    membership = {"path": str(evaluation_path), "sha256": sha256(evaluation_path),
                  "membership_sha256": store.membership["sha256"]}
    root = args.inputs.resolve().parent
    pins = {str(path.resolve()): sha256(path) for path in (
        args.job, args.inputs, args.update_check, settings.subset, evaluation_path, settings.resource_budget,
        settings.output / "config.json", settings.output / "frame_plan.json",
        settings.output / "update_states/text.pt", Path(specification.paths.transformer()),
        Path(specification.paths.video_vae()), *checkpoint_paths.values(),
        *(path.with_suffix(".complete.json") for path in checkpoint_paths.values()))}
    pins.update(update["protocol"]["input_files"])
    pins.update(export_evidence)
    if pilot:
        pins[str(update_job_path.resolve())] = sha256(update_job_path)
    cases, seen = [], set()
    for case in inputs["cases"]:
        source, frames = case.get("source"), case.get("frames")
        if source not in store.sources or source in seen or type(frames) is not int or frames < 1:
            raise ValueError("adapter-effect cases require unique checked sources and positive frame counts")
        seen.add(source)
        guide_path, image_path, noise_path, preparation_path = (
            _path(case.get(name), root) for name in ("guide", "first_image", "noise", "image_preparation"))
        video = store.load(source, require_guide=True)
        if sha256(guide_path) != video.hashes["guide"]:
            raise ValueError("adapter-effect guide differs from the source's original continuous master")
        pins.update(image_preparation_evidence(preparation_path, image_path))
        product = argparse.Namespace(
            guide=guide_path, first_image=image_path, output=args.output / str(len(cases)),
            checkpoint=None, model=settings.model, variant=settings.variant, mode=settings.mode,
            mode_settings=settings.mode_settings, span_latent_frames=frames,
            schedule=inputs.get("schedule"), decode=args.decode, poster_frame=0)
        specification, guide, image, fps, requested, _checked = infer.prepare_product(product)
        if guide.shape[2] != frames or not torch.equal(guide[0], video.z_g[:, :frames]):
            raise ValueError("adapter-effect prefix coverage differs from the checked continuous guide")
        checked = {name: evaluate.check_adapter(path, requested, product=True)
                   for name, path in checkpoint_paths.items()}
        noise = torch.load(noise_path, map_location="cpu", weights_only=True)
        shape = (1, frames * guide.shape[3] * guide.shape[4], guide.shape[1])
        if (not isinstance(noise, torch.Tensor) or noise.dtype != torch.bfloat16
                or tuple(noise.shape) != shape or not torch.isfinite(noise).all()):
            raise ValueError("adapter-effect noise must be finite native bf16 with the exact prefix token shape")
        pins.update({str(path.resolve()): sha256(path) for path in (guide_path, image_path, noise_path)})
        capture_path = store.root / source / dataset.capture_bundle_name(store.objective)
        pins[str(capture_path.resolve())] = video.hashes["capture"]
        view_path = store.root / source
        pins.update({str(path.resolve()): video.hashes[role] for path, role in (
            (view_path / dataset.render_metadata_name(store.objective), "guide_sidecar"),
            (view_path / dataset.render_name(store.objective), "render"))})
        cases.append({"source": source, "frames": frames, "guide": guide, "image": image,
                      "capture": video.z_y[:, :frames].unsqueeze(0), "noise": noise,
                      "fps": fps, "requested": requested, "checked": checked,
                      "coverage": [0, frames],
                      "source_frames": list(range(common.pixel_frames_for(frames, specification.scale_factors.time)))})
    producer = software.capture("evaluation", settings.mode, decoder=args.decode, extra_sources=EXTRA_SOURCES)
    return {"job": job, "settings": settings, "specification": specification, "cases": cases,
            "checkpoints": checkpoint_paths, "contracts": contracts, "context": context,
            "budget": budget, "pins": pins, "software": producer, "schedule": inputs["schedule"],
            "seed": inputs["seed"], "numerics": original_numerics, "evaluation_membership": membership,
            "states": PILOT_STATES if pilot else STATES, "update_job": update_job,
            "update_result": args.update_check.resolve(),
            "update_job_input": (args.job if not pilot else update_job_path).resolve(),
            "condition_binding": condition_binding}


def check_current(prepared: dict) -> None:
    _check_owners(prepared)
    if any(sha256(Path(path)) != digest for path, digest in prepared["pins"].items()):
        raise ValueError("adapter-effect inputs changed during execution")


def _check_owners(prepared: dict) -> None:
    software.check_current(prepared["software"])
    resources.check_budget(prepared["budget"])


@contextlib.contextmanager
def _phase(prepared: dict, output: Path, device: torch.device, name: str, records: list[dict]) -> Iterator[None]:
    phase = resources.Phase(device, name, 0, prepared["budget"])
    error = None
    try:
        _check_owners(prepared)
        phase.start()
        yield
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        if phase.started is not None:
            record = phase.finish(error)
            records.append(record)
            with (output / "resources_rank0.jsonl").open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            if error is None and record["state"] != "passed":
                raise ValueError(record["error"])


@contextlib.contextmanager
def reference_transformer(
    session: Session, checkpoint: Path | None, contract: dict | None,
    *, adapter_sha256: str | None = None
) -> Iterator[torch.nn.Module]:
    from ltx_trainer.model_loader import load_transformer  # noqa: PLC0415 -- trainer's real velocity loader

    if checkpoint is not None:
        checkpoints.recheck_adapter(checkpoint, contract, adapter_sha256)
    base = load_transformer(checkpoint_path=session.model.paths.transformer(), device=str(session.device),
                            dtype=torch.bfloat16, video_only=True)
    model = base
    try:
        if checkpoint is not None:
            settings = contract["adapter"]
            model = adapters.attach(base, rank=settings["rank"], alpha=settings["alpha"], target=settings["target"])
            adapters.load_weights(model, checkpoint, expected_sha256=adapter_sha256)
        common.base_model(model).set_gradient_checkpointing(False)
        model.requires_grad_(False).eval()
        yield model
    finally:
        del model, base


@torch.no_grad()
def run_sample(  # noqa: PLR0913 -- explicit matched tensors and scientific settings
    model: torch.nn.Module, context: torch.Tensor, grid: common.ClipGrid,
               capture: torch.Tensor, guide: torch.Tensor, image: torch.Tensor, noise: torch.Tensor,
               *, path: str, settings: config.RunSettings, schedule: list[float], seed: int) -> tuple:
    """Dispatch shared functions with independently encoded c0; copy no model algorithm."""
    if path == "product":
        return infer.generate(model, context, grid, guide, image, mode=settings.mode,
                              settings=settings.mode_settings, schedule=schedule, seed=seed, epsilon=noise)
    if path not in ("reference", "evaluation", "fused"):
        raise ValueError("unsupported adapter-effect path")
    prediction = common.denoised_from_velocity_model(model) if path == "reference" else None
    return evaluate.sample_case(
        model, context, grid, common.with_clean_prefix(capture, image), guide, noise,
        mode=settings.mode, mode_settings=settings.mode_settings, guide_mode="d1",
        schedule=schedule, seed=seed, predict_x0=prediction)


def _arm(prepared: dict, case: dict, session: Session, grid: common.ClipGrid, path: str, state: str,
         output: Path, records: list[dict]) -> tuple:
    from scripts.onestep_avatar.training import numerics  # noqa: PLC0415 -- compare actual policy at each arm

    numerics.validate(numerics.capture(), required=True, expected=prepared["numerics"])
    checkpoint = None if state == "base" else prepared["checkpoints"][state]
    contract = None if checkpoint is None else prepared["contracts"][state]
    if checkpoint is not None and sha256(checkpoint) != prepared["pins"][str(checkpoint.resolve())]:
        raise ValueError("adapter-effect checkpoint changed before model loading")
    identity = None if checkpoint is None else prepared["pins"][str(checkpoint.resolve())]
    if checkpoint is not None:
        checkpoints.recheck_adapter(checkpoint, contract, identity)
    opener = (reference_transformer(session, checkpoint, contract, adapter_sha256=identity)
              if path == "reference" else
              adapters.inference_transformer(session, checkpoint, contract, adapter_sha256=identity,
                  method=adapters.FUSED if path == "fused" else adapters.UNMERGED))
    with contextlib.ExitStack() as lifecycle:
        with _phase(prepared, output, session.device, f"load:{case['source']}:{path}:{state}", records):
            transformer = lifecycle.enter_context(opener)
            memory = adapters.parameter_memory(transformer)
        try:
            with _phase(prepared, output, session.device, f"sample:{case['source']}:{path}:{state}", records):
                tensors = {name: grid.patchify(case[name].to(device=session.device, dtype=torch.bfloat16))
                           for name in ("capture", "guide", "image")}
                generated, record = run_sample(
                    transformer, prepared["context"].to(session.device), grid,
                    tensors["capture"], tensors["guide"], tensors["image"], case["noise"].to(session.device),
                    path=path, settings=prepared["settings"], schedule=prepared["schedule"], seed=prepared["seed"])
                if not torch.equal(generated[:, :, :1], case["image"].to(dtype=generated.dtype)):
                    raise ValueError("adapter-effect path changed the independently supplied c0")
                record.update(path=path, adapter_state=state, parameter_storage=memory,
                              application_method=("base" if checkpoint is None else
                                                  adapters.FUSED if path == "fused" else adapters.UNMERGED),
                              source=case["source"], source_coverage=case["coverage"], fps=case["fps"],
                              software=prepared["software"], input_files=prepared["pins"],
                              conditions=case["requested"], checkpoint=None if checkpoint is None else str(checkpoint))
                record["numerics"] = numerics.capture()
                numerics.validate(record["numerics"], required=True, expected=prepared["numerics"])
                _check_owners(prepared)
                if checkpoint is not None and sha256(checkpoint) != prepared["pins"][str(checkpoint.resolve())]:
                    raise ValueError("adapter-effect checkpoint changed during model execution")
                saved = evaluate.save_case(generated, record, output / str(len(records)) / path / state)
        finally:
            del transformer
    gc.collect()
    if session.device.type == "cuda":
        torch.cuda.empty_cache()
    return generated, saved


def _render(prepared: dict, case: dict, outputs: dict, session: Session, output: Path,
            records: list[dict]) -> list[dict]:
    with _phase(prepared, output, session.device, f"decode:{case['source']}", records):
        with session.decoder() as decoder:
            pixels = {name: media.decode(session, case[name], decoder, prepared["seed"])
                      for name in ("capture", "guide")}
            trained = {state: {path: media.decode(session, rows[state], decoder, prepared["seed"])
                               for path, rows in outputs.items()} for state in prepared["states"][2:]}
        mapping = tuple(case["source_frames"])
        result = []
        for state, generated in trained.items():
            selected_pixels = {**pixels, **generated}
            step = prepared["contracts"][state]["adapter"]["step"]
            panels = [media.Panel(name, title, selected_pixels[name], mapping) for name, title in (
                ("capture", "Capture VAE"), ("guide", "Guide VAE"),
                ("reference", f"Reference step {step}"), ("evaluation", f"Evaluation step {step}"),
                ("product", f"Product step {step}"))]
            options = {"question": "Do ordinary paths preserve the saved adapter effect?", "fps": case["fps"],
                       "common_settings": {"source": case["source"], "schedule": prepared["schedule"],
                                           "seed": prepared["seed"], "input_files": prepared["pins"],
                                           "checkpoint_step": step}}
            destination = output / "media" if state == "step1" else output / "media" / state
            for name, layout, viewing_width in (
                ("full", "comparison", 960),
                ("compact", media.compact_layout(panels, question=options["question"], layout="comparison"), 480)):
                rendered, record = media.render_panels(panels, layout=layout, viewing_width=viewing_width, **options)
                record["software"] = prepared["software"]
                result.append(media.save_render(rendered, record, destination / name))
            if "fused" in outputs:
                fused_panels = [panel for panel in panels if panel.role in ("capture", "guide", "reference")]
                fused_panels.append(media.Panel("fused", f"Fused diagnostic step {step}", generated["fused"], mapping))
                question = "Does fusion change the saved adapter effect?"
                layout = media.compact_layout(fused_panels, question=question, layout="comparison")
                rendered, record = media.render_panels(fused_panels, question=question, layout=layout, fps=case["fps"],
                                                     common_settings=options["common_settings"])
                record["software"] = prepared["software"]
                result.append(media.save_render(rendered, record, destination / "fused_diagnostic"))
        return result


def execute(args: argparse.Namespace) -> dict:
    from scripts.onestep_avatar.training import numerics  # noqa: PLC0415 -- original applied policy before CUDA
    from scripts.prune.core import preflight  # noqa: PLC0415 -- native checks after complete scientific preflight
    from scripts.prune.core.session import Session  # noqa: PLC0415

    args.output = args.output.resolve()
    prepared = prepare(args)
    actual_numerics = numerics.apply(expected=prepared["numerics"], environment_required=True)
    check_current(prepared)
    preflight.check(prepared["settings"].model, gpu_id=args.gpu_id,
                    transformer_path=prepared["specification"].paths.transformer())
    device = torch.device(f"cuda:{args.gpu_id}")
    session = Session(prepared["specification"], device, "onestep_avatar.adapter_effect",
                      prepared["context"].to(device))
    check_current(prepared)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"kind": "onestep_avatar.adapter_effect_check", "schema_version": 1,
                "scope": "shared_API_adapter_correction", "job": prepared["job"],
                "job_input": str(args.job.resolve()),
                "calibration": {"job": prepared["update_job"],
                                "job_input": str(prepared["update_job_input"]),
                                "condition_binding": prepared["condition_binding"],
                                "result": {"path": str(prepared["update_result"]),
                                           "sha256": prepared["pins"][str(prepared["update_result"])]}},
                "software": prepared["software"], "input_files": prepared["pins"],
                "evaluation_membership": prepared["evaluation_membership"],
                "resource_budget": prepared["budget"], "tolerance": TOLERANCE,
                "checkpoints": {name: {"path": str(path), "sha256": prepared["pins"][str(path.resolve())],
                                       "contract": prepared["contracts"][name]}
                                for name, path in prepared["checkpoints"].items()},
                "schedule": prepared["schedule"], "seed": prepared["seed"],
                "runtime": {"device": str(device), "world_size": 1, "autocast_enabled": torch.is_autocast_enabled(),
                            "cuda_name": torch.cuda.get_device_name(device),
                            "numerics": actual_numerics,
                            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}}
    dataset.atomic_write(args.output / "protocol.json", lambda path: path.write_text(
        json.dumps(protocol, indent=2) + "\n"))
    records, summaries = [], []
    try:
        for index, case in enumerate(prepared["cases"]):
            case_output = args.output / f"case_{index:02d}"
            case_output.mkdir()
            scales = prepared["specification"].scale_factors
            grid = common.ClipGrid.build(case["frames"], case["guide"].shape[3] * scales.height,
                                        case["guide"].shape[4] * scales.width, case["fps"], prepared["specification"],
                                        device=device, dtype=torch.bfloat16,
                                        latent_channels=case["guide"].shape[1])
            outputs, raw = {}, {}
            for path in (*PATHS, *(("fused",) if args.fused_diagnostic else ())):
                outputs[path], raw[path] = {}, {}
                for state in prepared["states"]:
                    outputs[path][state], raw[path][state] = _arm(
                        prepared, case, session, grid, path, state, case_output, records)
            expected_calls = raw["reference"]["base"]["call_counts"]
            if any(record["call_counts"] != expected_calls for rows in raw.values() for record in rows.values()):
                raise ValueError("adapter-effect paths differ in their actual model-call inventory")
            fields = ("guide_sha256", "c0_sha256", "noise_sha256", "text_sha256")
            expected_inputs = {field: raw["reference"]["base"][field] for field in fields}
            if any({field: record[field] for field in fields} != expected_inputs
                   for rows in raw.values() for record in rows.values()):
                raise ValueError("adapter-effect paths differ in actual consumed input identities")
            renders = _render(prepared, case, outputs, session, case_output, records) if args.decode else []
            summaries.append({"source": case["source"], "source_coverage": case["coverage"],
                              "comparison": compare_paths(outputs), "raw_records": raw, "renderings": renders})
        check_current(prepared)
        resources.validate_records(records, 1, [row["phase"] for row in records], prepared["budget"])
        numerical = all(row["comparison"]["passed"] for row in summaries)
        result = {"schema_version": 1, "protocol": protocol, "cases": summaries,
                  "resource_measurements": records, "state": "passed" if numerical else "failed",
                  "coverage": {"sources": len(summaries),
                               "views": len(camera_views([row["source"] for row in summaries])),
                               "camera_labels": camera_views([row["source"] for row in summaries]),
                               "checkpoint_steps": sorted(contract["adapter"]["step"]
                                                          for contract in prepared["contracts"].values()),
                               "decoded": args.decode},
                  "e2_complete": False, "remaining_acceptance": [
                      *(["trained pilot checkpoints 20 and 60"] if prepared["states"] == STATES else []),
                      *(["at least two matched camera views"]
                        if len(camera_views([row["source"] for row in summaries])) < 2 else []),
                      *(["decoded appearance inspection"] if not args.decode else []),
                      "external bounded supervision and owned-worker absence",
                      "actual product CLI and training preview integration"]}
        result["output_files"] = {str(path.relative_to(args.output)): sha256(path)
                                  for path in sorted(args.output.rglob("*")) if path.is_file()}
        dataset.atomic_write(args.output / "result.json", lambda path: path.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"))
        return result
    except BaseException as failure:
        failure_record = {"state": "failed", "error": f"{type(failure).__name__}: {failure}",
                          "completed_cases": summaries, "resource_measurements": records}
        dataset.atomic_write(args.output / "failure.json", lambda path: path.write_text(
            json.dumps(failure_record, indent=2) + "\n"))
        raise


def _local_artifact(directory: Path, value: str, inventory: dict) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("adapter-effect saved artifact escapes its output directory")
    name = str(path.relative_to(directory))
    if name not in inventory or sha256(path) != inventory[name]:
        raise ValueError("adapter-effect saved artifact is missing or changed")
    return path


def _verify_pilot_protocol(protocol: dict, prepared: dict, selected: dict, contracts: dict) -> None:
    """Recheck distinct pilot and E4 producers; no saved self-description transfers lineage."""
    def read_job(path: str, expected: dict) -> dict:
        source = Path(path).resolve()
        if prepared["pins"].get(str(source)) != sha256(source):
            raise ValueError("adapter-effect saved job input binding differs")
        actual = queue.prepare_job(json.loads(source.read_text()), source.parent)
        if actual != expected:
            raise ValueError("adapter-effect saved job differs from its original input")
        return actual

    job = read_job(protocol["job_input"], protocol["job"])
    calibration = protocol["calibration"]
    update_job = read_job(calibration["job_input"], calibration["job"])
    settings, update_settings = (config.parse_settings(item["arguments"]) for item in (job, update_job))
    if (job["kind"] != "train" or settings.steps != 60 or settings.init_adapter is not None
            or update_job["kind"] != "train" or update_settings.steps != 1
            or update_settings.init_adapter is not None or update_settings.mode != settings.mode):
        raise ValueError("adapter-effect saved pilot or original E4 job selection differs")
    result_path = Path(calibration["result"]["path"]).resolve()
    if calibration["result"]["sha256"] != prepared["pins"].get(str(result_path)):
        raise ValueError("adapter-effect saved E4 result input binding differs")
    update_paths = {"zero": update_settings.output / "checkpoints/lora_weights_step_00000.safetensors",
                    "step1": update_settings.output / "checkpoints/lora_weights_step_00001.safetensors"}
    validate_update_result(
        result_path, update_job, update_paths, resources.read_budget(update_settings.resource_budget))
    original = checkpoints.read_contract(update_paths["step1"])
    engine.verify_training_conditions(update_job, update_paths["step1"], original)
    if calibration.get("condition_binding") != check_calibration_contract(contracts["step60"], original):
        raise ValueError("adapter-effect saved pilot and original E4 condition binding differs")
    store, plan, _specification, _used = engine.prepare_run(settings, require_fresh_output=False)
    settings.world_size = job["processes"]
    saved = json.loads((settings.output / "config.json").read_text())
    evidence = verify_selected_checkpoints(job, settings, store, plan, selected, contracts, saved, prepared["budget"])
    if any(prepared["pins"].get(path) != digest for path, digest in evidence.items()):
        raise ValueError("adapter-effect saved pilot resource input binding differs")
    original_saved = json.loads((update_settings.output / "config.json").read_text())
    if (native_numerics(saved) != protocol["runtime"]["numerics"]
            or native_numerics(original_saved) != native_numerics(saved)):
        raise ValueError("adapter-effect saved pilot and original E4 numerical policies differ")


def verify_saved(directory: Path) -> dict:  # noqa: PLR0912, PLR0915 -- ordered saved-integrity and scientific gates
    """Recompute saved comparisons without model/decoder work or acceptance restamping."""
    from scripts.onestep_avatar.training import numerics  # noqa: PLC0415 -- verify original applied policy

    directory = directory.resolve()
    result = json.loads((directory / "result.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if (result.get("schema_version") != 1 or result.get("protocol") != protocol
            or protocol.get("kind") != "onestep_avatar.adapter_effect_check"
            or protocol.get("scope") != "shared_API_adapter_correction" or protocol.get("tolerance") != TOLERANCE
            or result.get("e2_complete") is not False):
        raise ValueError("adapter-effect saved protocol or acceptance scope differs")
    prepared = {"software": protocol["software"], "budget": protocol["resource_budget"],
                "pins": protocol["input_files"]}
    check_current(prepared)
    evaluation = protocol.get("evaluation_membership", {})
    membership_path = Path(evaluation["path"])
    membership = json.loads(membership_path.read_text())
    store = dataset.ClipStore(membership)
    if (evaluation["sha256"] != prepared["pins"].get(str(membership_path.resolve()))
            or evaluation["membership_sha256"] != store.membership["sha256"]):
        raise ValueError("adapter-effect saved evaluation membership binding differs")
    original_policy = protocol.get("runtime", {}).get("numerics")
    numerics.validate(original_policy, required=True)
    device = torch.device(protocol["runtime"]["device"])
    if protocol["runtime"].get("world_size") != 1 or device.type != "cuda" or device.index is None:
        raise ValueError("adapter-effect saved runtime requires one actual indexed CUDA process")
    selected = protocol.get("checkpoints", {})
    if set(selected) not in ({"zero", "step1"}, {"zero", "step20", "step60"}):
        raise ValueError("adapter-effect saved checkpoint inventory differs")
    pilot = "step60" in selected
    states = PILOT_STATES if pilot else STATES
    paths, contracts = checkpoint_selection([Path(row["path"]) for row in selected.values()], pilot=pilot)
    for state in states[1:]:
        path = Path(selected[state]["path"]).resolve()
        if (selected[state]["sha256"] != prepared["pins"].get(str(path))
                or paths[state] != path or contracts[state] != selected[state]["contract"]):
            raise ValueError("adapter-effect saved checkpoint contract or input binding differs")
    if pilot:
        _verify_pilot_protocol(protocol, prepared, paths, contracts)
    checkpoints.assert_exported_lora_is_noop(load_file(selected["zero"]["path"]))
    inventory = result.get("output_files")
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and path != directory / "result.json"}
    if not isinstance(inventory, dict) or set(inventory) != actual:
        raise ValueError("adapter-effect saved artifact inventory is incomplete")
    for name, digest in inventory.items():
        path = _local_artifact(directory, str(directory / name), inventory)
        if sha256(path) != digest:
            raise ValueError("adapter-effect saved artifact changed")
    cases = result.get("cases")
    if not isinstance(cases, list) or not cases or len({case["source"] for case in cases}) != len(cases):
        raise ValueError("adapter-effect saved view coverage is missing or repeated")
    journals, phases = [], []
    for index, case in enumerate(cases):
        if case["source"] not in store.sources:
            raise ValueError("adapter-effect saved source is outside checked evaluation membership")
        raw, outputs = case["raw_records"], {}
        if set(raw) not in (set(PATHS), {*PATHS, "fused"}):
            raise ValueError("adapter-effect saved path inventory differs")
        expected = raw["reference"]["base"]
        conditions = expected["conditions"]
        frames = conditions["shape"]["frames"]
        if (conditions.get("schedule") != protocol["schedule"] or expected.get("seed") != protocol["seed"]
                or case["source_coverage"] != [0, frames]):
            raise ValueError("adapter-effect saved schedule, seed or source coverage differs")
        for checkpoint in selected.values():
            evaluate.check_adapter(Path(checkpoint["path"]), conditions, product=True)
        fields = ("guide_sha256", "c0_sha256", "noise_sha256", "text_sha256", "call_counts", "numerics",
                  "conditions", "fps", "source_coverage", "seed")
        for path in (*PATHS, *(("fused",) if "fused" in raw else ())):
            if set(raw[path]) != set(states):
                raise ValueError("adapter-effect saved state inventory differs")
            outputs[path] = {}
            for state in states:
                record = raw[path][state]
                if (record.get("state") != "complete" or record.get("adapter_state") != state
                        or record.get("path") != path or record.get("source") != case["source"]
                        or any(record.get(field) != expected.get(field) for field in fields)
                        or record.get("software") != protocol["software"]
                        or record.get("input_files") != protocol["input_files"]):
                    raise ValueError("adapter-effect saved consumed inputs or provenance differ")
                numerics.validate(record["numerics"], required=True, expected=original_policy)
                expected_checkpoint = None if state == "base" else selected[state]["path"]
                expected_method = ("base" if state == "base" else
                                   adapters.FUSED if path == "fused" else adapters.UNMERGED)
                if (record.get("checkpoint") != expected_checkpoint
                        or record.get("application_method") != expected_method):
                    raise ValueError("adapter-effect saved arm used a different checkpoint or application method")
                artifact = _local_artifact(directory, record["output"]["path"], inventory)
                if (record["output"]["sha256"] != inventory[str(artifact.relative_to(directory))]
                        or json.loads(artifact.with_name("result.json").read_text()) != record):
                    raise ValueError("adapter-effect saved raw record differs from its actual artifact")
                generated = torch.load(artifact, map_location="cpu", weights_only=True)
                shape = conditions["shape"]
                expected_shape = [1, shape["channels"], frames, shape["height"], shape["width"]]
                if (not isinstance(generated, torch.Tensor) or generated.ndim != 5
                        or generated.dtype != torch.bfloat16 or list(generated.shape) != record["output"]["shape"]
                        or list(generated.shape) != expected_shape):
                    raise ValueError("adapter-effect saved native raw shape or dtype differs")
                outputs[path][state] = generated
                phases.extend(f"{operation}:{case['source']}:{path}:{state}" for operation in ("load", "sample"))
        if case["comparison"] != compare_paths(outputs):
            raise ValueError("adapter-effect saved numerical comparison differs from its arrays")
        if case["renderings"]:
            phases.append(f"decode:{case['source']}")
        journal = directory / f"case_{index:02d}" / "resources_rank0.jsonl"
        _local_artifact(directory, str(journal), inventory)
        journals.extend(json.loads(line) for line in journal.read_text().splitlines())
    if journals != result["resource_measurements"]:
        raise ValueError("adapter-effect resource summary differs from the actual process journals")
    if any(record.get("device") != str(device) for record in journals):
        raise ValueError("adapter-effect resource devices differ from actual saved runtime")
    resources.validate_records(journals, 1, phases, protocol["resource_budget"])
    expected_state = "passed" if all(case["comparison"]["passed"] for case in cases) else "failed"
    if (result.get("state") != expected_state
            or result.get("coverage") != {"sources": len(cases),
                                          "views": len(camera_views([case["source"] for case in cases])),
                                          "camera_labels": camera_views([case["source"] for case in cases]),
                                          "checkpoint_steps": sorted(contract["adapter"]["step"]
                                                                     for contract in contracts.values()),
                                          "decoded": all(bool(case["renderings"]) for case in cases)}):
        raise ValueError("adapter-effect saved state or coverage differs")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("job", "inputs", "update-check", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--checkpoints", type=Path, nargs="+")
    parser.add_argument("--update-job", type=Path,
                        help="original one-update E4 job when --job selects the 60-step pilot")
    parser.add_argument("--subset", type=Path, help="separate checked evaluation membership; original E4 stays fixed")
    parser.add_argument("--verify", type=Path, help="verify saved evidence without model or decoder work")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--fused-diagnostic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.verify is not None:
        if any(getattr(args, name) is not None for name in (
            "job", "inputs", "update_check", "output", "checkpoints", "subset", "update_job"
        )):
            parser.error("--verify cannot be combined with execution input/output arguments")
        if args.decode or args.fused_diagnostic or args.dry_run:
            parser.error("--verify cannot be combined with execution mode flags")
        record = verify_saved(args.verify)
        preview = {"state": record["state"], "coverage": record["coverage"], "e2_complete": False}
        print(json.dumps(preview, indent=2))  # noqa: T201 -- requested saved-evidence review
        return 0 if record["state"] == "passed" else 2
    if any(getattr(args, name) is None for name in ("job", "inputs", "update_check", "output", "checkpoints")):
        parser.error("execution requires --job, --inputs, --update-check, --output and --checkpoints")
    if args.dry_run:
        prepared = prepare(args)
        preview = {"job": prepared["job"], "sources": [case["source"] for case in prepared["cases"]],
                   "input_files": prepared["pins"], "tolerance": TOLERANCE}
        print(json.dumps(preview, indent=2))  # noqa: T201 -- requested dry-run evidence
        return 0
    return 0 if execute(args)["state"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
