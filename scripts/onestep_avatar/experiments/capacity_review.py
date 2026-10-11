"""Review saved D0-trained outputs; see doc/experiments/capacity_review.md."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.onestep_avatar import evaluate, hashing, media, metrics
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.execution import queue, software
from scripts.onestep_avatar.experiments import adapter_effect_check, reference_inputs, spec_protocol
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters

STATES = ("base", "zero", "step20", "step60")
EXTRAS = ("scripts/onestep_avatar/experiments/__init__.py",
          "scripts/onestep_avatar/experiments/conditioning_check.py",
          "scripts/onestep_avatar/experiments/capacity_review.py", adapter_effect_check.ENTRY,
          reference_inputs.ENTRY, spec_protocol.ENTRY)
QUESTION = "Does D0 learning improve captured identity and action?"


def scientific_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", required=True, choices=("bidirectional", "causal"))
    parser.add_argument("--evaluation-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args(argv)


def dependency_outputs(args: argparse.Namespace) -> list[Path]:
    return [Path(job["output"]) for job in queue.prepare_jobs(args.evaluation_list)]


def check_outputs(mode: str, outputs: dict[str, torch.Tensor], records: list[dict], capture: torch.Tensor) -> None:
    """Inspect actual four-state outputs rather than state names or raw MSE alone."""
    if set(outputs) != set(STATES) or len(records) != 4:
        raise ValueError("capacity review requires exact base/zero/twenty/sixty outputs")
    evaluate.validate_comparison(records, "adapter")
    if not torch.equal(outputs["base"], outputs["zero"]):
        raise ValueError("capacity review zero output differs from base")
    counts = {"model_calls": 1 if mode == "bidirectional" else 6,
              "denoise_calls": 1 if mode == "bidirectional" else 3,
              "prime_calls": 0, "refresh_calls": 0 if mode == "bidirectional" else 3}
    for state, record in zip(STATES, records, strict=True):
        if (record.get("call_counts") != counts or any(type(value) is not int
                                                      for value in record["call_counts"].values())):
            raise ValueError("capacity review native calls differ")
        if record.get("metrics") != metrics.encoded_metrics(outputs[state], capture):
            raise ValueError("capacity review metric differs from saved output")


def fixed_arguments(arguments: list[str]) -> list[str]:
    """Evaluation arguments without the selected adapter and the output path."""
    kept, index = [], 0
    while index < len(arguments):
        if arguments[index] in ("--checkpoint", "--output"):
            index += 2
            continue
        kept.append(arguments[index])
        index += 1
    return kept


def prepare_case(args: argparse.Namespace, group: list[dict]) -> dict:
    """Check one case: base, zero, twenty and sixty, each saved by its own ordinary process.

    One evaluation process cannot hold four 22B model states on a 48 GB GPU
    (doc/known_gaps.md), so each state is a separate job. The four jobs may differ
    only in the selected adapter and output path.
    """
    if any(job["kind"] != "evaluate" for job in group):
        raise ValueError("capacity review accepts ordinary evaluation jobs only")
    if any(fixed_arguments(job["arguments"]) != fixed_arguments(group[0]["arguments"]) for job in group[1:]):
        raise ValueError("capacity review case changes more than the selected adapter")
    prepared = []
    for job in group:
        ordinary = evaluate.parse_args(job["arguments"])
        specification, variants, selected, membership = evaluate.prepare_evaluation(
            ordinary, require_fresh_output=False)
        if (ordinary.mode != args.mode or ordinary.guide_mode != "d0" or ordinary.variant != "dev"
                or ordinary.model != "2.5" or ordinary.schedule != [.725, 0] or ordinary.cfg != 1
                or ordinary.stg != 0 or ordinary.rescale != 0 or len(selected) != 1
                or ordinary.adapter_application != adapters.UNMERGED or ordinary.research_override
                or ordinary.include_base or len(variants) != 1 or membership["objective"] != "white"):
            raise ValueError("capacity review requires the exact D0 bounded learning comparison")
        subset.verify_prepared_membership(membership)
        prepared.append((ordinary, specification, variants[0], selected[0], membership))
    states = [variant for _ordinary, _specification, variant, _selected, _membership in prepared]
    if states[0] is not None or any(state is None for state in states[1:]):
        raise ValueError("capacity review case requires base first and then three adapters")
    selected_checkpoints, contracts = adapter_effect_check.checkpoint_selection(states[1:], pilot=True)
    if any(contract["task"]["guide_mode"] != "d0" for contract in contracts.values()):
        raise ValueError("capacity review requires separately D0-trained adapters")
    if states[1:] != [selected_checkpoints[state] for state in STATES[1:]]:
        raise ValueError("capacity review checkpoint order differs")
    ordinary, specification, _base, (video, frames, _requested, _checked), membership = prepared[0]
    if frames != 7 or ordinary.saved_noise is None:
        raise ValueError("capacity review requires seven frames and full saved noise")
    paths = []
    for job, (each, *_rest) in zip(group, prepared, strict=True):
        path = each.output / "case_0000/variant_000/result.json"
        evaluate.verify_evaluation_conditions(job["arguments"], [path])
        paths.append(path)
    records = [json.loads(path.read_text()) for path in paths]
    if records[0].get("adapter") is not None or [record.get("contract", {}).get("adapter", {}).get("step")
        for record in records[1:]] != [0, 20, 60]:
        raise ValueError("capacity review actual adapter inventory differs")
    if any(record.get("software") != records[0].get("software") for record in records[1:]):
        raise ValueError("capacity review states come from different producer software")
    outputs = {state: torch.load(path.parent / "generated.pt", map_location="cpu", weights_only=True)
               for state, path in zip(STATES, paths, strict=True)}
    capture = video.z_y[:, :7].unsqueeze(0).to(torch.bfloat16)
    check_outputs(args.mode, outputs, records, capture)
    return {"jobs": group, "ordinary": ordinary, "specification": specification, "video": video,
            "membership": membership, "records": records, "outputs": outputs,
            "input_files": {str(path): sha256(path) for path in (
                args.evaluation_list, ordinary.subset, *paths,
                *(path.parent / "generated.pt" for path in paths))}}


def prepare(args: argparse.Namespace, *, fresh: bool = True) -> dict:
    if fresh and args.output.exists():
        raise ValueError("capacity review requires a fresh output")
    producer = software.capture("evaluation", args.mode, decoder=True, extra_sources=EXTRAS)
    jobs = queue.prepare_jobs(args.evaluation_list)
    if not jobs:
        raise ValueError("capacity review evaluation list is empty")
    if len(jobs) % len(STATES):
        raise ValueError("capacity review requires four one-state evaluation jobs per case")
    cases = [prepare_case(args, jobs[index:index + len(STATES)]) for index in range(0, len(jobs), len(STATES))]
    return {"cases": cases, "software": producer}


def panels_for(references: list[media.Panel], pixels: dict, state: str) -> list[media.Panel]:
    mapping = tuple(range(49))
    return [media.Panel("recorded", "Capture", references[0].pixels, mapping, value="RGB"),
            media.Panel("decoded", "Capture", references[1].pixels, mapping, value="VAE decoded"),
            media.Panel("zero", "D0 zero", pixels["zero"], mapping, value="Output; VAE decoded"),
            media.Panel("baseline", "D0 base", pixels["base"], mapping, value="Output; VAE decoded"),
            media.Panel("changed", f"D0 {state}", pixels[state], mapping, value="Output; VAE decoded")]


def layouts(panels: list[media.Panel]) -> tuple:
    gap = media.Panel("control_gap", "Unused", None, (), missing_reason="No second zero")
    narrow = [*panels[:3], gap, *panels[3:]]
    return (("full", panels, "comparison", 960),
            ("narrow", narrow, media.compact_layout(narrow, question=QUESTION, layout="comparison"), 480))


def settings_for(case: dict, state: str) -> dict:
    return {"source": case["video"].source, "objective": "white", "mode": case["ordinary"].mode,
            "source_arm": "d0", "schedule": [.725, 0], "seed": case["ordinary"].seed,
            "changed_adapter_step": int(state.removeprefix("step")),
            "target_encode_records": case["video"].encode_records, "input_files": case["input_files"],
            "raw_outputs": {state: record["output"] for state, record in zip(STATES, case["records"], strict=True)},
            "noise_sha256": case["records"][0]["noise_sha256"],
            "c0_sha256": case["records"][0]["c0_sha256"],
            "text_sha256": case["records"][0]["text_sha256"],
            "decoder_settings": media.native_decoder_settings(), "quality_acceptance": None}


def render_case(case: dict, session, folder: Path, producer: dict) -> dict:  # noqa: ANN001
    started = time.perf_counter()
    store = dataset.ClipStore(case["membership"])
    source, crop = reference_inputs.checked_source(store, case["video"].source)
    with session.decoder() as decoder, torch.inference_mode():
        references, reference = media.prepare_training_references(
            session, decoder, source, store.root, "white", 7,
            case["ordinary"].seed, require_guide=False)
        pixels = {state: media.decode(session, latent, decoder, case["ordinary"].seed).cpu()
                  for state, latent in case["outputs"].items()}
    reference.update(membership_file_sha256=sha256(case["ordinary"].subset),
                     membership_sha256=case["membership"]["sha256"], source_crop_adapter=crop, software=producer)
    media.save_training_references(references, reference, folder / "references")
    files = {}
    for state, value in pixels.items():
        path = folder / f"{state}_rgb.pt"
        torch.save(value, path)
        files[state] = {"path": str(path), "sha256": sha256(path), "tensor_sha256": hashing.tensor_sha256(value)}
    for state in STATES[2:]:
        for name, panels, layout, width in layouts(panels_for(references, pixels, state)):
            rendered, record = media.render_panels(panels, question=QUESTION, layout=layout,
                fps=case["video"].fps, viewing_width=width, poster_frame=24, common_settings=settings_for(case, state))
            record["software"] = producer
            media.save_render(rendered, record, folder / state / name)
    if session.device.type == "cuda":
        torch.cuda.synchronize(session.device)
    vae_hash = sha256(Path(case["specification"].paths.video_vae()))
    result = {"schema_version": 1, "source": case["video"].source, "seed": case["ordinary"].seed,
              "panels": files, "input_files": case["input_files"], "software": producer,
              "decoded_outputs": {state: media.decode_key(record["output"]["sha256"], vae_hash,
                   list(case["outputs"][state].shape), "native_decode_video", case["ordinary"].seed,
                   media.native_decoder_settings()) for state, record in zip(STATES, case["records"], strict=True)},
              "seconds_decode_and_render": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(session.device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(session.device)}
    (folder / "review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def verify_case(case: dict, folder: Path, producer: dict) -> None:  # noqa: PLR0912 -- ordered saved media gates
    review = json.loads((folder / "review.json").read_text())
    if (review.get("schema_version") != 1 or review.get("source") != case["video"].source
            or review.get("seed") != case["ordinary"].seed or review.get("input_files") != case["input_files"]
            or review.get("software") != producer or set(review.get("panels", {})) != set(STATES)):
        raise ValueError("capacity review provenance or pixel inventory differs")
    references, reference = media.load_training_references(folder / "references/references.json")
    store = dataset.ClipStore(case["membership"])
    source, crop = reference_inputs.checked_source(store, case["video"].source)
    vae_hash = sha256(Path(case["specification"].paths.video_vae()))
    expected_reference = {"source": case["video"].source, "objective": "white", "fps": case["video"].fps,
                          "source_frames": list(range(49)), "capture_encoding_sha256": source["capture_latent_sha256"],
                          "guide_rgb_sha256": source.get("guide_sha256"), "vae_sha256": vae_hash,
                          "decode_seed": case["ordinary"].seed, "decoder_settings": media.native_decoder_settings(),
                          "decode_key": media.decode_key(source["capture_latent_sha256"], vae_hash,
                              list(case["outputs"]["base"].shape), "native_decode_video", case["ordinary"].seed,
                              media.native_decoder_settings()),
                          "membership_file_sha256": sha256(case["ordinary"].subset),
                          "membership_sha256": case["membership"]["sha256"],
                          "source_crop_adapter": crop, "software": producer}
    if any(reference.get(key) != value for key, value in expected_reference.items()):
        raise ValueError("capacity review capture reference differs")
    if not torch.equal(references[0].pixels, media.recorded_capture_rgb(source, store.root, "white", 7)):
        raise ValueError("capacity review recorded crop/objective pixels differ")
    pixels = {}
    edge = case["video"].encode_records["capture"]["edge"]
    if any(panel.pixels is None or list(panel.pixels.shape) != [49, 3, edge, edge]
           or not torch.isfinite(panel.pixels).all() for panel in references[:2]):
        raise ValueError("capacity review matched reference geometry differs")
    for state in STATES:
        path = folder / f"{state}_rgb.pt"
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(value, torch.Tensor) or list(value.shape) != [49, 3, edge, edge]
                or not torch.isfinite(value).all() or review["panels"][state] != {"path": str(path),
                    "sha256": sha256(path), "tensor_sha256": hashing.tensor_sha256(value)}):
            raise ValueError("capacity review saved decoded pixels differ")
        pixels[state] = value
    keys = {state: media.decode_key(record["output"]["sha256"], vae_hash,
                list(case["outputs"][state].shape), "native_decode_video", case["ordinary"].seed,
                media.native_decoder_settings()) for state, record in zip(STATES, case["records"], strict=True)}
    if review.get("decoded_outputs") != keys:
        raise ValueError("capacity review decoder identities differ")
    for state in STATES[2:]:
        for name, panels, layout, width in layouts(panels_for(references, pixels, state)):
            destination = folder / state / name
            rendering = json.loads((destination / "rendering.json").read_text())
            rendered, expected = media.render_panels(panels, question=QUESTION, layout=layout,
                fps=case["video"].fps, viewing_width=width, poster_frame=24, common_settings=settings_for(case, state))
            expected.update(software=producer, outputs={role: {"path": str(destination / filename),
                "sha256": sha256(destination / filename)} for role, filename in
                (("video", "comparison.mp4"), ("poster", "poster.png"))})
            if rendering != expected:
                raise ValueError("capacity review layout, labels, coverage or saved bytes differ")
            actual = media.verify_saved_video(destination / "comparison.mp4", 49, case["video"].fps)
            if [actual["width"], actual["height"]] != rendering["display_size"]:
                raise ValueError("capacity review actual video dimensions differ")
            with Image.open(destination / "poster.png") as poster:
                if not torch.equal(torch.from_numpy(np.array(poster.convert("RGB"))).permute(2, 0, 1), rendered[24]):
                    raise ValueError("capacity review poster differs from saved RGB")
    seconds = review.get("seconds_decode_and_render")
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
            or any(type(review.get(key)) is not int or review[key] <= 0 for key in
                   ("peak_allocated_bytes", "peak_reserved_bytes"))):
        raise ValueError("capacity review cost evidence differs")


def execute(args: argparse.Namespace) -> dict:
    prepared = prepare(args)
    args.output.mkdir(parents=True, exist_ok=False)
    session = media.open_decoder_session("2.5", args.gpu_id, script="onestep_avatar.capacity_review")
    for index, case in enumerate(prepared["cases"]):
        render_case(case, session, args.output / f"case_{index:04d}", prepared["software"])
    software.check_current(prepared["software"])
    verify(args)
    return {"state": "passed", "scope": "separately trained D0 capacity; visual quality requires inspection"}


def verify(args: argparse.Namespace) -> dict:
    prepared = prepare(args, fresh=False)
    for index, case in enumerate(prepared["cases"]):
        verify_case(case, args.output / f"case_{index:04d}", prepared["software"])
    return {"state": "passed", "cases": len(prepared["cases"]), "checkpoint_steps": [0, 20, 60],
            "quality_acceptance": None}
