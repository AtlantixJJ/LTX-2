"""Observe independent changed-past controls through public causal operations.

See doc/experiments/past_influence_check.md. The conditioning selector exposes
this operation; its software pins are independent of other operations.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.onestep_avatar import hashing, media
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import continuation_check as continuation
from scripts.onestep_avatar.experiments import continuation_identity_check as identity
from scripts.onestep_avatar.experiments import reference_inputs, spec_protocol
from scripts.onestep_avatar.model import adapters, backbone, causal, common

ENTRY = "scripts/onestep_avatar/experiments/past_influence_check.py"
EXTRAS = ("scripts/onestep_avatar/experiments/__init__.py",
          "scripts/onestep_avatar/experiments/conditioning_check.py", ENTRY,
          "scripts/onestep_avatar/experiments/continuation_check.py",
          "scripts/onestep_avatar/experiments/continuation_identity_check.py", spec_protocol.ENTRY)
FIELDS = ("latent", "sigma", "timesteps", "positions", "keyframes_mask", "attention_mask")
QUESTION = "Does changed past affect the current block?"


def scientific_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("snapshot", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sigma", type=float, choices=(1.0, .909375), required=True)
    parser.add_argument("--history", choices=("cache", "recompute"), required=True)
    parser.add_argument("--review-of", type=Path, help="decode only an already verified native past observation")
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args(argv)


def dependency_outputs(args: argparse.Namespace) -> list[Path]:
    return [args.snapshot.parent.resolve(), args.reference.parent.resolve(),
            *([] if getattr(args, "review_of", None) is None else [args.review_of.resolve()])]


def write_json(path: Path, value: dict) -> None:
    dataset.atomic_write(path, lambda temporary: temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n"))


def file_identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashing.sha256(path)}


def source_records(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """Use the full public saved verifier for both fresh source producers."""
    records, pins = [], {}
    for name, path, phase in (("snapshot", args.snapshot, "capture"), ("reference", args.reference, "reference")):
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        spec = Path(manifest["spec"]["path"])
        continuation.verify_completion(spec, path.parent)
        value = json.loads(path.read_text())
        if (value.get("state") != "complete" or value.get("phase") != phase
                or value.get("history") != "generated" or value.get("sigma") != args.sigma):
            raise ValueError("past control requires complete matching generated-history producers")
        pins[name] = {"record": file_identity(path), "manifest": file_identity(manifest_path),
                      "spec": file_identity(spec)}
        records.append(value)
    capture, reference = records
    if (reference.get("source_snapshot") != file_identity(args.snapshot)
            or any(reference.get(key) != capture.get(key) for key in ("protocol", "fixed_inputs", "software"))):
        raise ValueError("past control reference is not bound to its exact capture")
    if [(row["phase"], row["span"]) for row in capture["snapshots"]] != [
            (name, list(span)) for name, span in continuation.TARGETS]:
        raise ValueError("past control requires both original eviction boundaries")
    if [(row["phase"], row["span"]) for row in reference["reference_blocks"]] != [
            (name, list(span)) for name, span in continuation.TARGETS]:
        raise ValueError("past control reference boundary coverage differs")
    return capture, reference, pins


def prepare(args: argparse.Namespace) -> dict:
    if getattr(args, "review_of", None) is not None:
        prepared = review_preparation(args)
        return {"scope": "decoder-only past diagnostic review", "sources": prepared["pins"],
                "targets": [list(span) for _, span in continuation.TARGETS], "quality_acceptance": None}
    capture, _reference, pins = source_records(args)
    return {"scope": "changed generated-past content only", "sources": pins, "sigma": args.sigma,
            "history": args.history, "targets": [list(span) for _, span in continuation.TARGETS],
            "fixed_inputs": capture["fixed_inputs"], "quality_acceptance": None}


def changed_history(value: torch.Tensor, grid: common.ClipGrid, span: tuple[int, int]) -> torch.Tensor:
    """Negate only completed past frames; preserve c0 and all current/future bytes."""
    result = value.clone()
    lo, hi = grid.token_span(1, span[0])
    result[:, lo:hi] *= -1
    if (torch.equal(result[:, lo:hi], value[:, lo:hi]) or not torch.equal(result[:, :lo], value[:, :lo])
            or not torch.equal(result[:, hi:], value[:, hi:])):
        raise ValueError("past intervention must change only non-c0 completed frames")
    return result


class InputObserver:
    """Read native Modality through the public forward pre-hook; touch no input."""

    def __init__(self, transformer: torch.nn.Module):
        self.transformer, self.calls = transformer, []

    def __enter__(self):
        self.handle = self.transformer.register_forward_pre_hook(self.observe, with_kwargs=True)
        return self

    def __exit__(self, *_exception):
        self.handle.remove()

    def observe(self, _module: object, args: tuple, kwargs: dict) -> None:
        modality = kwargs.get("video", args[0] if args else None)
        if modality is None:
            raise ValueError("past observer did not receive native video Modality")
        row = {name: (getattr(modality, name).detach().cpu().clone()
                      if isinstance(getattr(modality, name), torch.Tensor) else getattr(modality, name))
               for name in FIELDS}
        row.update(context_sha256=hashing.tensor_sha256(modality.context),
                   kv_write=modality.kv_write, kv_start=modality.kv_start,
                   cache_lengths=None if modality.kv_caches is None else [cache.length for cache in modality.kv_caches])
        self.calls.append(row)


def saved_layers(root: Path, rows: list[dict], *, reference: bool) -> list[dict]:
    return [{"layer": row["layer"], "file": file_identity(continuation.checked_path(
        root, row["reference_file"] if reference else row["file"]))} for row in rows]


def observe_cache(cache: causal.BlockCache, destination: Path | None, expected: list[dict]) -> list[dict]:
    """Stream every actual retained layer, checking original byte identity live."""
    if len(cache.caches) != len(expected):
        raise ValueError("past cache layer inventory differs")
    records = []
    for index, (layer, baseline) in enumerate(zip(cache.caches, expected, strict=True)):
        k, v = layer.read(cache.start)
        values = {"k": k.detach().cpu().clone(), "v": v.detach().cpu().clone()}
        if not all(torch.isfinite(value).all() for value in values.values()):
            raise ValueError("past cache contains nonfinite values")
        if destination is None:
            saved = torch.load(baseline["file"]["path"], map_location="cpu", weights_only=True)
            if any(not torch.equal(values[name], saved[name]) for name in ("k", "v")):
                raise ValueError("original chronological cache reconstruction is not bit-identical")
            path = Path(baseline["file"]["path"])
        else:
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / f"layer_{index:04d}.pt"
            dataset.atomic_write(path, lambda temporary, values=values: torch.save(values, temporary))
        records.append({"layer": index, "file": file_identity(path),
                        "observed_tensors": {name: continuation.tensor_record(value) for name, value in values.items()},
                        "scope": "actual retained cache K after normalization/RoPE and native V"})
    return records


@torch.no_grad()
def run_arm(predict: object, transformer: torch.nn.Module, grid: common.ClipGrid,  # noqa: PLR0913 -- explicit native inputs
            geometry: causal.CausalGeometry, values: dict, row: dict, snapshot_root: Path,
            destination: Path, history: str, sigma: float, *, changed: bool,
            expected_layers: list[dict]) -> dict:
    """Replay public refresh transactions or the public clean-history denoise."""
    span = tuple(row["span"])
    clean = changed_history(values["clean_history"], grid, span) if changed else values["clean_history"]
    device = grid.positions.device
    clean, noisy, context = (value.to(device) for value in (clean, values["noisy"], values["text"]))
    destination.mkdir(parents=True, exist_ok=False)
    layers = []
    with InputObserver(transformer) as observer:
        if history == "cache":
            base = common.base_model(transformer)
            cache = causal.BlockCache.allocate(grid, geometry, num_layers=len(base.transformer_blocks),
                inner_dim=base.inner_dim, device=device, dtype=clean.dtype)
            for previous in geometry.plan(grid.latent_frames):
                if previous[1] > span[0]:
                    break
                lo, hi = grid.token_span(*previous)
                causal.refresh_block(predict, grid, cache, clean[:, lo:hi], context, previous)
            layers = observe_cache(cache, destination / "kv" if changed else None, expected_layers)
            prediction = causal.denoise_block(predict, grid, cache, noisy, context, sigma, span)
            del cache
        else:
            # The frozen continuation observer transparently delegates the native
            # attention functions and saves the actual history K/V at every layer.
            with continuation.ReferenceObserver(transformer, snapshot_root, destination / "progress.json") as kv:
                with kv.target(row):
                    prediction = causal.denoise_with_clean_history(
                        predict, grid, geometry, clean, noisy, context, sigma, span)
                for observation, baseline in zip(kv.observations, expected_layers, strict=True):
                    path = continuation.checked_path(destination, observation["reference_file"])
                    actual = torch.load(path, map_location="cpu", weights_only=True)
                    if not changed:
                        saved = torch.load(baseline["file"]["path"], map_location="cpu", weights_only=True)
                        if any(not torch.equal(actual[name], saved[name]) for name in ("k", "v")):
                            raise ValueError("original clean-history reconstruction K/V is not bit-identical")
                    layers.append({"layer": observation["layer"], "file": file_identity(path),
                                   "observed_tensors": {name: continuation.tensor_record(value)
                                                        for name, value in actual.items()},
                                   "scope": observation["scope"]})
    torch.save(observer.calls, destination / "trace.pt")
    torch.save(prediction.cpu(), destination / "prediction.pt")
    return {"layers": layers, "actual_calls": len(observer.calls)}


def verify_trace(calls: list[dict], grid: common.ClipGrid, geometry: causal.CausalGeometry,
                 values: dict, span: tuple[int, int], history: str, sigma: float,
                 changed: bool, layers: int) -> None:
    clean = changed_history(values["clean_history"], grid, span) if changed else values["clean_history"]
    plan = geometry.plan(grid.latent_frames)
    expected = []
    if history == "cache":
        for previous in plan:
            if previous[1] > span[0]:
                break
            lo, hi = grid.token_span(*previous)
            expected.append((clean[:, lo:hi], [(lo, hi)], 0., 0, True,
                             min(previous[0], geometry.context_latent_frames + 1) * grid.tokens_per_latent_frame, None))
        lo, hi = grid.token_span(*span)
        expected.append((values["noisy"], [(lo, hi)], sigma, 0, False,
                         min(span[0], geometry.context_latent_frames + 1) * grid.tokens_per_latent_frame, None))
    else:
        spans = causal.retained_prefix_spans(plan, geometry, span[0])
        slices = [grid.token_span(lo, hi) for lo, hi, _ in spans] + [grid.token_span(*span)]
        sequence = torch.cat([*[clean[:, lo:hi] for lo, hi in slices[:-1]], values["noisy"]], dim=1)
        prefix = sequence.shape[1] - values["noisy"].shape[1]
        ids = causal.block_ids_for([*spans, (*span, plan.index(span))], grid.tokens_per_latent_frame)
        expected.append((sequence, slices, sigma, prefix, False, 0, causal.block_causal_mask(ids)))
    if len(calls) != len(expected):
        raise ValueError("past native call count differs from exact transaction order")
    for call, (latent, slices, level, prefix, write, start, mask) in zip(calls, expected, strict=True):
        timestep = torch.full((1, latent.shape[1], 1), level, dtype=torch.float32)
        timestep[:, :prefix] = 0
        fields = {"latent": latent, "sigma": torch.tensor([level], dtype=torch.float32), "timesteps": timestep,
                  "positions": torch.cat([grid.positions[:, :, lo:hi] for lo, hi in slices], dim=2),
                  "keyframes_mask": torch.cat([grid.keyframes_mask[:, lo:hi] for lo, hi in slices], dim=1)}
        for name, value in fields.items():
            identity.same_tensor(call.get(name), value, "past " + name)
        if (call.get("context_sha256") != hashing.tensor_sha256(values["text"])
                or call.get("kv_write") is not write or call.get("kv_start") != start
                or call.get("cache_lengths") != ([start] * layers if history == "cache" else None)
                or (mask is None) != (call.get("attention_mask") is None)
                or (mask is not None and not torch.equal(call["attention_mask"], mask))):
            raise ValueError("past native text, cache boundary or causal mask differs")


def original_prediction(args: argparse.Namespace, capture: dict, reference: dict, phase: str) -> torch.Tensor:
    if args.history == "cache":
        row = next(row for row in capture["snapshots"] if row["phase"] == phase)
        path = continuation.checked_path(args.snapshot.parent, row["cached_prediction"])
    else:
        row = next(row for row in reference["reference_blocks"] if row["phase"] == phase)
        path = continuation.checked_path(args.reference.parent, row["reference_prediction"])
    return torch.load(path, map_location="cpu", weights_only=True)


def expected_kv(args: argparse.Namespace, capture: dict, reference: dict, phase: str) -> list[dict]:
    if args.history == "cache":
        rows = next(row for row in capture["snapshots"] if row["phase"] == phase)["layers"]
        return saved_layers(args.snapshot.parent, rows, reference=False)
    rows = [row for row in reference["layerwise"] if row["phase"] == phase]
    return saved_layers(args.reference.parent, rows, reference=True)


def verify(args: argparse.Namespace) -> dict:  # noqa: PLR0912 -- ordered saved native input, layer and output checks
    if getattr(args, "review_of", None) is not None:
        return verify_review(args)
    capture, reference, pins = source_records(args)
    producer = software.capture("evaluation", "causal", extra_sources=EXTRAS)
    record = json.loads((args.output / "result.json").read_text())
    if (record.get("sources") != pins or record.get("software") != producer
            or record.get("sigma") != args.sigma or record.get("history") != args.history):
        raise ValueError("past saved scientific bindings differ")
    grid = common.ClipGrid.build(17, 1024, 1024, 30, backbone.resolve("2.5", "dev"),
                                device=torch.device("cpu"), dtype=torch.bfloat16, latent_channels=128)
    positions = identity.validated_positions(torch.load(args.output / "positions.pt", map_location="cpu",
                                                        weights_only=True), grid)
    grid = replace(grid, positions=positions)
    geometry = causal.CausalGeometry(grid.tools.scale_factors, 2, 8)
    measurements = []
    for row in capture["snapshots"]:
        phase, span = row["phase"], tuple(row["span"])
        values = continuation.load_inputs(args.snapshot.parent, row)
        expected = expected_kv(args, capture, reference, phase)
        original = original_prediction(args, capture, reference, phase)
        predictions, layer_values = {}, {}
        for arm in ("original", "changed"):
            folder = args.output / phase / arm
            native = json.loads((folder / "result.json").read_text())
            trace = torch.load(folder / "trace.pt", map_location="cpu", weights_only=True)
            verify_trace(trace, grid, geometry, values, span, args.history, args.sigma, arm == "changed", len(expected))
            if (native.get("actual_calls") != len(trace) or [v["layer"] for v in native["layers"]]
                    != list(range(len(expected)))):
                raise ValueError("past saved layer inventory or actual calls differ")
            predictions[arm] = torch.load(folder / "prediction.pt", map_location="cpu", weights_only=True)
            if (predictions[arm].shape != original.shape or predictions[arm].dtype != original.dtype
                    or not torch.isfinite(predictions[arm]).all()):
                raise ValueError("past saved prediction shape, dtype or finite values differ")
            if arm == "original":
                identity.same_tensor(predictions[arm], original, "past original reconstruction")
            layer_values[arm] = native["layers"]
        layer_effects = []
        for base, a, b in zip(expected, layer_values["original"], layer_values["changed"], strict=True):
            tensors = []
            for item in (a, b):
                path = Path(item["file"]["path"])
                if file_identity(path) != item["file"]:
                    raise ValueError("past observed K/V file bytes changed")
                value = torch.load(path, map_location="cpu", weights_only=True)
                if ({name: continuation.tensor_record(tensor) for name, tensor in value.items()}
                        != item["observed_tensors"] or set(value) != {"k", "v"}):
                    raise ValueError("past observed K/V tensor identities differ")
                tensors.append(value)
            saved = torch.load(base["file"]["path"], map_location="cpu", weights_only=True)
            if any(not torch.equal(tensors[0][name], saved[name]) for name in ("k", "v")):
                raise ValueError("past original all-layer K/V differs from saved producer")
            effect = {name: continuation.compare_chunks(tensors[0][name], tensors[1][name]) for name in ("k", "v")}
            if all(item["bit_identical"] for item in effect.values()):
                raise ValueError("past intervention did not change an observed history layer")
            layer_effects.append({"layer": base["layer"], **effect})
        effect = continuation.compare_chunks(predictions["original"], predictions["changed"])
        if effect["bit_identical"]:
            raise ValueError("past intervention did not affect continuation output")
        measurements.append({"phase": phase, "span": list(span), "prediction": effect,
                             "layerwise": layer_effects, "original_bit_identical": True})
    if record.get("measurements") != measurements:
        raise ValueError("past saved measurements differ from observed outputs")
    return {"state": "passed", "measurements": measurements, "quality_acceptance": None}


def execute(args: argparse.Namespace) -> dict:
    if getattr(args, "review_of", None) is not None:
        return execute_review(args)
    if args.output.exists():
        raise ValueError("past control requires a fresh output")
    capture, reference, pins = source_records(args)
    producer = software.capture("evaluation", "causal", extra_sources=EXTRAS)
    specification = backbone.resolve("2.5", "dev")
    need = (2 * specification.caps.num_layers * 12 * 1024
            * specification.caps.num_heads * specification.caps.head_dim * 2)
    if shutil.disk_usage(args.output.parent).free < need * (2 if args.history == "recompute" else 1) + (256 << 20):
        raise ValueError("insufficient space for exact all-layer past observations")
    from scripts.prune.core import preflight  # noqa: PLC0415 -- native preflight only
    from scripts.prune.core.session import Session  # noqa: PLC0415 -- one native session

    preflight.check("2.5", gpu_id=args.gpu_id, transformer_path=specification.paths.transformer())
    device = torch.device(f"cuda:{args.gpu_id}")
    grid = common.ClipGrid.build(17, 1024, 1024, 30, specification, device=device,
                                dtype=torch.bfloat16, latent_channels=128)
    geometry = causal.CausalGeometry(grid.tools.scale_factors, 2, 8)
    context = continuation.load_inputs(args.snapshot.parent, capture["snapshots"][0])["text"].to(device)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(grid.positions.cpu(), args.output / "positions.pt")
    session = Session(specification, device, "onestep_avatar.past_influence", context)
    measurements = []
    with adapters.inference_transformer(session, None, None, method=adapters.UNMERGED) as transformer:
        predict = common.denoised_from_x0_model(transformer)
        for row in capture["snapshots"]:
            phase = row["phase"]
            values = continuation.load_inputs(args.snapshot.parent, row)
            lo, hi = grid.token_span(*row["span"])
            identity.same_tensor(values["positions"], grid.positions[:, :, lo:hi].cpu(), "past native positions")
            expected = expected_kv(args, capture, reference, phase)
            original = original_prediction(args, capture, reference, phase)
            predictions, layers = {}, {}
            for arm in ("original", "changed"):
                destination = args.output / phase / arm
                observed = run_arm(predict, transformer, grid, geometry, values, row,
                    args.snapshot.parent, destination, args.history, args.sigma,
                    changed=arm == "changed", expected_layers=expected)
                write_json(destination / "result.json", observed)
                predictions[arm] = torch.load(destination / "prediction.pt", map_location="cpu", weights_only=True)
                layers[arm] = observed["layers"]
                if arm == "original":
                    identity.same_tensor(predictions[arm], original, "past original reconstruction")
            effects = []
            for a, b in zip(layers["original"], layers["changed"], strict=True):
                left, right = [torch.load(item["file"]["path"], map_location="cpu", weights_only=True)
                               for item in (a, b)]
                effects.append({"layer": a["layer"], **{name: continuation.compare_chunks(left[name], right[name])
                                                       for name in ("k", "v")}})
            measurements.append({"phase": phase, "span": row["span"],
                "prediction": continuation.compare_chunks(predictions["original"], predictions["changed"]),
                "layerwise": effects, "original_bit_identical": True})
    software.check_current(producer)
    write_json(args.output / "result.json", {"sources": pins, "sigma": args.sigma, "history": args.history,
        "software": producer, "measurements": measurements,
        "scope": "independently changed generated-past input sensitivity; no identity/action quality claim"})
    return verify(args)


def review_preparation(args: argparse.Namespace) -> dict:
    native = argparse.Namespace(**vars(args))
    native.output, native.review_of = args.review_of, None
    verify(native)
    capture, _reference, pins = source_records(args)
    protocol = json.loads(Path(capture["protocol"]["path"]).read_text())
    member = Path(protocol["membership"])
    store = dataset.ClipStore(json.loads(member.read_text()))
    # The native continuation verifier has already checked this original
    # membership. Its scientific data predates canonical publication metadata.
    store.verify(require_guide=True)
    source, crop = reference_inputs.checked_source(store, protocol["source"])
    video = store.load(protocol["source"], require_guide=True)
    specification = backbone.resolve("2.5", "dev")
    from scripts.onestep_avatar.corpus import precompute  # noqa: PLC0415 -- current original VAE fingerprint

    if (precompute.file_fingerprint(Path(specification.paths.video_vae()))
            != source["capture_encode_record"]["vae_fingerprint"]):
        raise ValueError("past review decoder differs from original capture VAE")
    view = store.root / protocol["source"]
    paths = [member, store.root / dataset.CAPTURE_MANIFEST_NAME, Path(specification.paths.video_vae()),
             *(view / name for name in ("rgb.mp4", "bbox.npy", dataset.CAPTURE_MASK_NAME,
               dataset.capture_bundle_name("white"), dataset.guide_bundle_name("white"),
               dataset.render_name("white"), dataset.render_metadata_name("white")))]
    inputs = {"store": store, "source": source, "video": video, "crop": crop, "specification": specification,
              "input_files": {str(path): hashing.sha256(path) for path in paths}}
    pins = {**pins, "past_result": file_identity(args.review_of / "result.json"),
            "past_manifest": file_identity(args.review_of / "manifest.json"),
            "reference_inputs": inputs["input_files"]}
    producer = software.capture("decoding", extra_sources=(*EXTRAS, reference_inputs.ENTRY))
    grid = common.ClipGrid.build(17, 1024, 1024, 30, inputs["specification"],
                                device=torch.device("cpu"), dtype=torch.bfloat16, latent_channels=128)
    return {"capture": capture, "protocol": protocol, "references": inputs, "pins": pins,
            "software": producer, "grid": grid, "seed": protocol["noise"]["seed"]}


def presentation_latent(grid: common.ClipGrid, values: dict, span: tuple[int, int],
                        prediction: torch.Tensor, *, changed: bool) -> torch.Tensor:
    """Keep source frame zero through target end; show the actual supplied past."""
    clean = changed_history(values["clean_history"], grid, span) if changed else values["clean_history"]
    lo, hi = grid.token_span(*span)
    if prediction.shape != clean[:, lo:hi].shape or prediction.dtype != clean.dtype:
        raise ValueError("past presentation prediction shape or dtype differs")
    joined = torch.cat([clean[:, :lo], prediction], dim=1)
    return grid.unpatchify_block(joined, span[1])


def review_panels(references: list[media.Panel], pixels: dict, frames: int, span: list[int]) -> list[media.Panel]:
    mapping = tuple(range(frames))
    return [*references,
            media.Panel("baseline", f"Original current [{span[0]},{span[1]})", pixels["original"], mapping,
                        value="Diagnostic; VAE decoded"),
            media.Panel("changed", f"Changed current [{span[0]},{span[1]})", pixels["changed"], mapping,
                        value="Negated past; VAE decoded")]


def review_layouts(panels: list[media.Panel]) -> tuple:
    gap = media.Panel("control_gap", "Unused", None, (), missing_reason="No second guide")
    narrow = [*panels[:3], gap, *panels[3:]]
    return (("full", panels, "comparison", 960),
            ("narrow", narrow, media.compact_layout(narrow, question=QUESTION, layout="comparison"), 480))


def review_settings(args: argparse.Namespace, prepared: dict, row: dict) -> dict:
    frames = (row["span"][1] - 1) * 8 + 1
    return {"source": prepared["protocol"]["source"], "objective": "white", "seed": prepared["seed"],
            "schedule": [args.sigma, 0], "history": args.history, "geometry": {"B": 2, "D": 8},
            "source_frames": list(range(frames)), "current_rgb_start": (row["span"][0] - 1) * 8 + 1,
            "encoded_current_span": row["span"], "sources": prepared["pins"],
            "fixed_inputs": prepared["capture"]["fixed_inputs"], "quality_acceptance": None,
            "prefix_origin": "saved cached generated prefix; current prediction uses the stated history computation",
            "scope": ("diagnostic current-block sensitivity; negated past, fixed current noise; "
                      "no identity/action quality inference")}


def execute_review(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise ValueError("past diagnostic decoding requires a fresh output")
    prepared = review_preparation(args)
    args.output.mkdir(parents=True, exist_ok=False)
    session = media.open_decoder_session("2.5", args.gpu_id, script="onestep_avatar.past_review")
    records = []
    with session.decoder() as decoder, torch.inference_mode():
        for row in prepared["capture"]["snapshots"]:
            folder = args.output / row["phase"]
            folder.mkdir()
            source = prepared["references"]["source"]
            references, producer = media.prepare_training_references(session, decoder, source,
                prepared["references"]["store"].root, "white", row["span"][1], prepared["seed"], require_guide=True)
            producer.update(membership_file_sha256=hashing.sha256(Path(prepared["protocol"]["membership"])),
                membership_sha256=prepared["references"]["store"].membership["sha256"],
                source_crop_adapter=prepared["references"]["crop"], software=prepared["software"])
            media.save_training_references(references, producer, folder / "references")
            values = continuation.load_inputs(args.snapshot.parent, row)
            pixels, artifacts = {}, {}
            for arm in ("original", "changed"):
                prediction = torch.load(args.review_of / row["phase"] / arm / "prediction.pt",
                                        map_location="cpu", weights_only=True)
                latent = presentation_latent(prepared["grid"], values, tuple(row["span"]), prediction,
                                             changed=arm == "changed")
                latent_path, pixel_path = folder / f"{arm}_latent.pt", folder / f"{arm}_rgb.pt"
                torch.save(latent, latent_path)
                pixels[arm] = media.decode(session, latent, decoder, prepared["seed"]).cpu()
                torch.save(pixels[arm], pixel_path)
                artifacts[arm] = {"latent": file_identity(latent_path), "pixels": file_identity(pixel_path),
                    "pixels_tensor": continuation.tensor_record(pixels[arm]),
                    "decode_key": media.decode_key(hashing.sha256(latent_path), producer["vae_sha256"],
                        list(latent.shape), "native_decode_video", prepared["seed"], media.native_decoder_settings())}
            frames = (row["span"][1] - 1) * 8 + 1
            for name, panels, layout, width in review_layouts(review_panels(references, pixels, frames, row["span"])):
                rendered, rendering = media.render_panels(panels, question=QUESTION, layout=layout, fps=30,
                    viewing_width=width, poster_frame=frames - 1, common_settings=review_settings(args, prepared, row))
                rendering["software"] = prepared["software"]
                media.save_render(rendered, rendering, folder / name)
            records.append({"phase": row["phase"], "span": row["span"], "artifacts": artifacts})
    software.check_current(prepared["software"])
    write_json(args.output / "review.json", {"sources": prepared["pins"], "software": prepared["software"],
                                            "records": records, "quality_acceptance": None})
    return verify_review(args)


def verify_review(args: argparse.Namespace) -> dict:
    prepared = review_preparation(args)
    review = json.loads((args.output / "review.json").read_text())
    if (review.get("sources") != prepared["pins"] or review.get("software") != prepared["software"]
            or [(v["phase"], v["span"]) for v in review["records"]]
            != [(v["phase"], v["span"]) for v in prepared["capture"]["snapshots"]]):
        raise ValueError("past review source, software or boundary inventory differs")
    inputs = prepared["references"]
    for row, record in zip(prepared["capture"]["snapshots"], review["records"], strict=True):
        folder, frames = args.output / row["phase"], (row["span"][1] - 1) * 8 + 1
        references, producer = media.load_training_references(folder / "references/references.json")
        vae = hashing.sha256(Path(inputs["specification"].paths.video_vae()))
        expected = {"source": inputs["video"].source, "objective": "white", "fps": 30,
            "source_frames": list(range(frames)), "software": prepared["software"], "vae_sha256": vae,
            "membership_file_sha256": hashing.sha256(Path(prepared["protocol"]["membership"])),
            "membership_sha256": inputs["store"].membership["sha256"], "source_crop_adapter": inputs["crop"],
            "decode_seed": prepared["seed"], "decoder_settings": media.native_decoder_settings(),
            "capture_encoding_sha256": inputs["source"]["capture_latent_sha256"],
            "guide_rgb_sha256": inputs["source"]["guide_sha256"],
            "decode_key": media.decode_key(inputs["source"]["capture_latent_sha256"], vae,
                [1, inputs["source"]["shape"][0], row["span"][1], *inputs["source"]["shape"][2:]],
                "native_decode_video", prepared["seed"],
                media.native_decoder_settings())}
        if any(producer.get(key) != value for key, value in expected.items()):
            raise ValueError("past review matched reference attribution differs")
        for index, reader in ((0, media.recorded_capture_rgb), (2, media.recorded_guide_rgb)):
            actual = reader(inputs["source"], inputs["store"].root, "white", row["span"][1])
            identity.same_tensor(references[index].pixels, actual, "past review original recorded pixels")
        values, pixels = continuation.load_inputs(args.snapshot.parent, row), {}
        for arm in ("original", "changed"):
            artifact = record["artifacts"][arm]
            latent_path, pixel_path = folder / f"{arm}_latent.pt", folder / f"{arm}_rgb.pt"
            prediction = torch.load(args.review_of / row["phase"] / arm / "prediction.pt",
                                    map_location="cpu", weights_only=True)
            latent = presentation_latent(prepared["grid"], values, tuple(row["span"]), prediction,
                                         changed=arm == "changed")
            identity.same_tensor(torch.load(latent_path, map_location="cpu", weights_only=True), latent,
                                 "past review supplied history and prediction")
            pixels[arm] = torch.load(pixel_path, map_location="cpu", weights_only=True)
            expected_artifact = {"latent": file_identity(latent_path), "pixels": file_identity(pixel_path),
                "pixels_tensor": continuation.tensor_record(pixels[arm]), "decode_key": media.decode_key(
                    hashing.sha256(latent_path), vae, list(latent.shape), "native_decode_video", prepared["seed"],
                    media.native_decoder_settings())}
            edge = inputs["source"]["capture_encode_record"]["edge"]
            if (artifact != expected_artifact or list(pixels[arm].shape) != [frames, 3, edge, edge]
                    or not torch.isfinite(pixels[arm]).all()):
                raise ValueError("past review decoded pixels or identities differ")
        for name, panels, layout, width in review_layouts(review_panels(references, pixels, frames, row["span"])):
            destination = folder / name
            rendered, expected_rendering = media.render_panels(panels, question=QUESTION, layout=layout,
                fps=30, viewing_width=width, poster_frame=frames - 1,
                common_settings=review_settings(args, prepared, row))
            expected_rendering.update(software=prepared["software"], outputs={
                role: file_identity(destination / filename)
                for role, filename in (("video", "comparison.mp4"), ("poster", "poster.png"))})
            if json.loads((destination / "rendering.json").read_text()) != expected_rendering:
                raise ValueError("past review labels, layout, coverage or rendered bytes differ")
            actual = media.verify_saved_video(destination / "comparison.mp4", frames, 30)
            if [actual["width"], actual["height"]] != expected_rendering["display_size"]:
                raise ValueError("past review actual video dimensions differ")
            with Image.open(destination / "poster.png") as poster:
                if not torch.equal(torch.from_numpy(np.array(poster.convert("RGB"))).permute(2, 0, 1), rendered[-1]):
                    raise ValueError("past review rendered poster pixels differ")
    return {"state": "passed", "scope": "matched past diagnostic VAE review", "quality_acceptance": None}
