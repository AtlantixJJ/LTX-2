"""Run matched guide, text and sigma-one controls; see doc/experiments/conditioning_check.md."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.onestep_avatar import evaluate, hashing, media, metrics
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import reference_inputs, spec_protocol
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters, common
from scripts.onestep_avatar.training import checkpoints

SELECTOR = "conditioning_check"
EXTRAS = ("scripts/onestep_avatar/experiments/__init__.py",
          "scripts/onestep_avatar/experiments/conditioning_check.py", reference_inputs.ENTRY, spec_protocol.ENTRY)
ARMS = {"guide": ("matched", "foreign_guide"), "text": ("matched", "foreign_text"),
        "sigma_one": ("d0", "d1"), "capture_oracle": ("d0", "d1")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return spec_protocol.outer_args(argv, __doc__)


def scientific_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--control", required=True, choices=ARMS)
    parser.add_argument("--mode", required=True, choices=("bidirectional", "causal"))
    for name in ("subset", "noise-file", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--donor-source")
    parser.add_argument("--prompt")
    parser.add_argument("--wrong-prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("train", "held_out", "validation", "test"))
    parser.add_argument("--research-override", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args(argv)
    if args.control == "guide" and (not args.donor_source or args.donor_source == args.source):
        parser.error("guide control requires a distinct checked donor source")
    if args.control != "guide" and args.donor_source is not None:
        parser.error("donor source belongs only to the guide control")
    if args.control == "text" and (not args.wrong_prompt or args.wrong_prompt == args.prompt):
        parser.error("text control requires a distinct nonempty wrong prompt")
    if args.control != "text" and args.wrong_prompt is not None:
        parser.error("wrong prompt belongs only to the text control")
    if args.control in ("sigma_one", "capture_oracle") and not args.research_override:
        parser.error("D0/D1 pilot controls require explicit research override")
    return args


def read_spec(spec: Path, output: Path) -> argparse.Namespace:
    return scientific_spec(parse_args(["--spec", str(spec), "--output", str(output)]))


def scientific_spec(outer: argparse.Namespace) -> argparse.Namespace:
    parser = scientific_args
    operation = json.loads(outer.spec.read_text()).get("operation")
    if operation == "prepare_references":
        parser = reference_inputs.scientific_args
    elif operation == "past_influence":
        from scripts.onestep_avatar.experiments import past_influence_check  # noqa: PLC0415 -- separate operation owner

        parser = past_influence_check.scientific_args
    elif operation == "capacity_review":
        from scripts.onestep_avatar.experiments import capacity_review  # noqa: PLC0415 -- explicit scientific operation

        parser = capacity_review.scientific_args
    return spec_protocol.read_spec(outer, SELECTOR, parser,
                                  operations=("run", "capacity_review", "prepare_references", "past_influence"))


def dependency_outputs(spec: Path) -> list[Path]:
    args = read_spec(spec, Path("/tmp/conditioning_check_dependency_output"))
    if args.operation == "past_influence":
        from scripts.onestep_avatar.experiments import past_influence_check  # noqa: PLC0415

        return past_influence_check.dependency_outputs(args)
    if args.operation == "capacity_review":
        from scripts.onestep_avatar.experiments import capacity_review  # noqa: PLC0415

        return capacity_review.dependency_outputs(args)
    return []


def evaluation_arguments(args: argparse.Namespace, guide_mode: str = "d1") -> list[str]:
    sigma = "1.0" if args.control == "sigma_one" else "0.725"
    result = ["--mode", args.mode, "--subset", str(args.subset), "--source", args.source,
              "--model", "2.5", "--variant", "dev", "--guide-mode", guide_mode,
              "--schedule", sigma, "0", "--span-latent-frames", "7", "--seed", str(args.seed),
              "--noise-file", str(args.noise_file), "--output", str(args.output)]
    if args.mode == "causal":
        result += ["--block-latent-frames", "2", "--blocks-per-sample", "3",
                   "--context-latent-frames", "8"]
    if args.checkpoint is not None:
        result += ["--checkpoint", str(args.checkpoint)]
    if args.prompt is not None:
        result += ["--prompt", args.prompt]
    if args.split is not None:
        result += ["--split", args.split]
    if args.research_override:
        result += ["--research-override"]
    return result


def prepare(args: argparse.Namespace, *, fresh: bool = True) -> dict:
    """Check actual branch conditions and distinct donor provenance before native work."""
    ordinary = evaluate.parse_args(evaluation_arguments(args))
    specification, variants, cases, membership = evaluate.prepare_evaluation(ordinary, require_fresh_output=fresh)
    subset.verify_prepared_membership(membership)
    if len(variants) != 1 or len(cases) != 1 or membership["objective"] != "white":
        raise ValueError("conditioning control requires one white target and one model state")
    video, frames, requested, checked = cases[0]
    if frames != 7 or ordinary.saved_noise is None:
        raise ValueError("conditioning control requires seven frames and complete saved noise")
    donor = None
    if args.control == "guide":
        store = dataset.ClipStore(membership)
        if args.donor_source not in store.sources:
            raise ValueError("foreign guide is absent from the checked membership")
        donor = store.load(args.donor_source, require_guide=True)
        if (donor.fps != video.fps or donor.z_g.shape[0] != video.z_g.shape[0]
                or donor.z_g.shape[1] < frames or donor.z_g.shape[2:] != video.z_g.shape[2:]
                or donor.encode_records["guide"]["vae_fingerprint"] !=
                video.encode_records["guide"]["vae_fingerprint"]):
            raise ValueError("foreign guide shape, timebase or VAE differs")
        if torch.equal(video.z_g[:, :frames], donor.z_g[:, :frames]):
            raise ValueError("foreign guide control is a no-op")
    branch_checks = dict.fromkeys(ARMS[args.control], checked[0])
    if args.control in ("sigma_one", "capture_oracle"):
        changed = {**requested, "task": {**requested["task"], "guide_mode": "d0"}}
        branch_checks["d0"] = evaluate.check_adapter(args.checkpoint, changed, override=args.research_override)
    return {"ordinary": ordinary, "specification": specification, "video": video, "donor": donor,
            "membership": membership, "requested": requested, "checks": branch_checks}


def input_tokens(prepared: dict, device: torch.device) -> tuple:
    """Use the ordinary clip grid; the independent donor never replaces the target c0."""
    video, specification = prepared["video"], prepared["specification"]
    grid = common.ClipGrid.build(7, video.z_y.shape[2] * specification.scale_factors.height,
                                video.z_y.shape[3] * specification.scale_factors.width,
                                video.fps, specification, device=device, dtype=torch.bfloat16,
                                latent_channels=specification.caps.latent_channels)
    capture = grid.patchify(video.z_y[:, :7].unsqueeze(0).to(device=device, dtype=torch.bfloat16))
    guide = grid.patchify(video.z_g[:, :7].unsqueeze(0).to(device=device, dtype=torch.bfloat16))
    donor = prepared["donor"]
    foreign = None if donor is None else grid.patchify(
        donor.z_g[:, :7].unsqueeze(0).to(device=device, dtype=torch.bfloat16))
    noise = prepared["ordinary"].saved_noise.to(device=device)
    return grid, capture, guide, foreign, noise


def control_inputs(control: str, context: torch.Tensor, guide: torch.Tensor,
                   *, foreign_guide: torch.Tensor | None = None,
                   foreign_text: torch.Tensor | None = None) -> list[tuple]:
    """Return exactly two arms and reject interventions that change geometry or no values."""
    if control == "guide":
        if (foreign_guide is None or foreign_guide.shape != guide.shape
                or foreign_guide.dtype != guide.dtype or foreign_guide.device != guide.device
                or not torch.isfinite(foreign_guide).all() or torch.equal(guide, foreign_guide)):
            raise ValueError("foreign guide must be finite, same-shaped and different")
        return [("matched", context, guide, "d1"), ("foreign_guide", context, foreign_guide, "d1")]
    if control == "text":
        if (foreign_text is None or foreign_text.shape != context.shape
                or foreign_text.dtype != context.dtype or foreign_text.device != context.device
                or not torch.isfinite(foreign_text).all() or torch.equal(context, foreign_text)):
            raise ValueError("foreign text must be finite, same-shaped and different")
        return [("matched", context, guide, "d1"), ("foreign_text", foreign_text, guide, "d1")]
    if control in ("sigma_one", "capture_oracle"):
        return [("d0", context, guide, "d0"), ("d1", context, guide, "d1")]
    raise ValueError("unknown conditioning control")


def summarize(control: str, outputs: list[torch.Tensor]) -> dict:
    if len(outputs) != 2 or outputs[0].shape != outputs[1].shape or any(
        not torch.isfinite(output).all() for output in outputs
    ):
        raise ValueError("conditioning summary requires two matching finite outputs")
    delta = outputs[1].double() - outputs[0].double()
    identical = torch.equal(*outputs)
    return {"bit_identical": identical, "difference_rms": float(delta.square().mean().sqrt()),
            "difference_max_abs": float(delta.abs().max()),
            "state": "passed" if control != "sigma_one" or identical else "failed",
            "interpretation": "source-weight-zero equality" if control == "sigma_one" else
            "target-provided capture oracle; not D0 learning capacity" if control == "capture_oracle" else
            "insensitive_to_changed_input" if identical else "sensitive_to_changed_input",
            "quality_acceptance": None}


def arm_provenance(args: argparse.Namespace, prepared: dict, name: str, guide_mode: str,
                   prompt: str, producer: dict) -> dict:
    video, donor = prepared["video"], prepared["donor"]
    foreign = name == "foreign_guide"
    requested = {**prepared["requested"],
                 "task": {**prepared["requested"]["task"], "guide_mode": guide_mode}}
    return {"control": args.control, "arm": name, "source": video.source, "fps": video.fps,
            "conditions": requested, "membership_sha256": prepared["membership"]["sha256"],
            "fixed_input_files": {str(path.resolve()): sha256(path) for path in
                                  (args.subset, args.noise_file, *((args.checkpoint,) if
                                                                args.checkpoint is not None else ()))},
            "input_file_hashes": video.hashes, "target_encode_records": video.encode_records,
            "guide_source": donor.source if foreign else video.source,
            "guide_input_file_hashes": donor.hashes if foreign else video.hashes,
            "guide_encode_record": (donor if foreign else video).encode_records["guide"],
            "prompt": prompt, "software": producer, "producer_source_sha256": sha256(Path(__file__)),
            **prepared["checks"][name]}


def review_panels(args: argparse.Namespace, prepared: dict, references: list,
                 pixels: dict[str, torch.Tensor]) -> list[media.Panel]:
    """Label source roles and decoder roles; a donor retains its own recorded crop."""
    mapping = tuple(range(49))
    titles = {"recorded": ("Capture", "RGB"), "decoded": ("Capture", "VAE decoded"),
              "guide": ("Paired guide", "RGB")}
    panels = [media.Panel(panel.role, titles[panel.role][0], panel.pixels, mapping,
                          value=titles[panel.role][1]) for panel in references]
    output_titles = {"matched": "Matched guide" if args.control == "guide" else "Matched text",
                     "foreign_guide": "Foreign guide", "foreign_text": "Wrong text", "d0": "D0", "d1": "D1"}
    panels.extend(media.Panel(role, output_titles[name], pixels[name], mapping, value="Output; VAE decoded")
                  for role, name in zip(("baseline", "changed"), ARMS[args.control], strict=True))
    if args.control == "guide":
        donor = prepared["donor"].source
        person = next(row["actor"] for row in prepared["membership"]["sources"]
                      if row["relative_dir"] == donor)
        panels.append(media.Panel("donor_guide", "Donor guide", pixels["donor_guide"], mapping,
                                  value=f"RGB; person {person}"))
    return panels


def review_variants(panels: list[media.Panel], question: str) -> tuple:
    """Keep the two output conditions beside each other in both reading widths."""
    if len(panels) == 6:
        narrow = [*panels[:3], panels[5], *panels[3:5]]
        layout = media.compact_layout(narrow, question=question, layout="comparison")
    else:
        narrow = panels
        layout = "compact_training"
    return (("full", panels, "comparison", 960), ("narrow", narrow, layout, 480))


def review_settings(args: argparse.Namespace, prepared: dict, records: list[dict]) -> dict:
    """Keep exact experiment and coordinate evidence in the delivered review record."""
    return {"control": args.control, "source": prepared["video"].source, "objective": "white",
            "target_encode_records": prepared["video"].encode_records,
            "donor_encode_records": None if prepared["donor"] is None else prepared["donor"].encode_records,
            "schedule": prepared["ordinary"].schedule, "seed": args.seed,
            "membership_sha256": prepared["membership"]["sha256"],
            "noise_sha256": records[0]["noise_sha256"], "c0_sha256": records[0]["c0_sha256"],
            "adapter_sha256": records[0].get("adapter_sha256"),
            "prompts": {record["arm"]: record["prompt"] for record in records},
            "raw_outputs": {record["arm"]: record["output"] for record in records},
            "decoder_settings": media.native_decoder_settings(), "quality_acceptance": None}


def render_review(args: argparse.Namespace, prepared: dict, session, outputs: list[torch.Tensor],  # noqa: ANN001 -- native session handle
                  records: list[dict], producer: dict) -> dict:
    """Use shared native decoding and rendering after the transformer context is closed."""
    if session.device.type == "cuda":
        torch.cuda.synchronize(session.device)
    started = time.perf_counter()
    store = dataset.ClipStore(prepared["membership"])
    source, crop = reference_inputs.checked_source(store, args.source)
    with session.decoder() as decoder, torch.inference_mode():
        references, reference_producer = media.prepare_training_references(
            session, decoder, source, store.root, "white", 7, args.seed)
        pixels = {name: media.decode(session, output, decoder, args.seed).cpu()
                  for name, output in zip(ARMS[args.control], outputs, strict=True)}
    if args.control == "guide":
        pixels["donor_guide"] = media.recorded_guide_rgb(
            reference_inputs.checked_source(store, args.donor_source)[0], store.root, "white", 7)
    reference_producer.update(membership_file_sha256=sha256(args.subset),
                              membership_sha256=prepared["membership"]["sha256"],
                              source_crop_adapter=crop, software=producer)
    media.save_training_references(references, reference_producer, args.output / "review" / "references")
    panel_files = {}
    for name, value in pixels.items():
        path = args.output / "review" / f"{name}_rgb.pt"
        torch.save(value, path)
        panel_files[name] = {"path": str(path), "sha256": sha256(path),
                             "tensor_sha256": hashing.tensor_sha256(value), "shape": list(value.shape)}
    panels = review_panels(args, prepared, references, pixels)
    questions = {"guide": "Does changing the guide change identity and action?",
                 "text": "Does changing the text change identity and action?",
                 "sigma_one": "Do D0 and D1 agree when source weight is zero?",
                 "capture_oracle": "Does the supplied capture improve identity and action?"}
    settings = review_settings(args, prepared, records)
    renderings = {}
    for name, selected_panels, layout, width in review_variants(panels, questions[args.control]):
        rendered, rendering = media.render_panels(
            selected_panels, question=questions[args.control], layout=layout, fps=prepared["video"].fps,
            viewing_width=width, poster_frame=24, common_settings=settings)
        rendering["software"] = producer
        media.save_render(rendered, rendering, args.output / "review" / name)
        renderings[name] = str(args.output / "review" / name / "rendering.json")
    if session.device.type == "cuda":
        torch.cuda.synchronize(session.device)
    vae = Path(prepared["specification"].paths.video_vae())
    result = {"schema_version": 1, "settings": settings, "panels": panel_files, "renderings": renderings,
              "reference_bundle": str(args.output / "review" / "references" / "references.json"),
              "vae_sha256": sha256(vae), "seconds_decode_and_render": time.perf_counter() - started,
              "decoded_outputs": {record["arm"]: media.decode_key(
                  record["output"]["sha256"], sha256(vae), record["output"]["shape"],
                  "native_decode_video", args.seed, media.native_decoder_settings()) for record in records},
              "memory_boundary": "cumulative runtime peak through decoding; queue peak is not reset",
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(session.device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(session.device)}
    (args.output / "review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def verify_review(args: argparse.Namespace, prepared: dict, outputs: list[torch.Tensor],  # noqa: PLR0912, PLR0915 -- complete saved media gates
                  records: list[dict], producer: dict) -> None:
    """Verify actual matched RGB, native decoder identities and readable saved layouts on CPU."""
    review = json.loads((args.output / "review.json").read_text())
    settings = review_settings(args, prepared, records)
    reference_path = args.output / "review" / "references" / "references.json"
    if (review.get("schema_version") != 1 or review.get("settings") != settings
            or review.get("reference_bundle") != str(reference_path)):
        raise ValueError("conditioning review settings or references differ")
    references, reference = media.load_training_references(reference_path)
    store = dataset.ClipStore(prepared["membership"])
    source, crop = reference_inputs.checked_source(store, args.source)
    vae = Path(prepared["specification"].paths.video_vae())
    vae_hash = sha256(vae)
    expected_reference = {"source": args.source, "objective": "white", "fps": prepared["video"].fps,
                          "capture_encoding_sha256": source["capture_latent_sha256"],
                          "guide_rgb_sha256": source["guide_sha256"], "source_frames": list(range(49)),
                          "vae_sha256": vae_hash, "decode_seed": args.seed,
                          "decoder_settings": media.native_decoder_settings(),
                          "decode_key": media.decode_key(source["capture_latent_sha256"], vae_hash,
                                                         [1, *prepared["video"].z_y[:, :7].shape],
                                                         "native_decode_video", args.seed,
                                                         media.native_decoder_settings()),
                          "membership_file_sha256": sha256(args.subset),
                          "membership_sha256": prepared["membership"]["sha256"],
                          "source_crop_adapter": crop, "software": producer}
    if any(reference.get(key) != value for key, value in expected_reference.items()):
        raise ValueError("conditioning reference source, crop, timebase or decoder differs")
    if (not torch.equal(references[0].pixels, media.recorded_capture_rgb(source, store.root, "white", 7))
            or not torch.equal(references[2].pixels, media.recorded_guide_rgb(source, store.root, "white", 7))):
        raise ValueError("conditioning reference pixels differ from recorded preprocessing")
    names = [*ARMS[args.control], *(("donor_guide",) if args.control == "guide" else ())]
    if set(review.get("panels", {})) != set(names) or review.get("vae_sha256") != vae_hash:
        raise ValueError("conditioning decoded panel inventory or VAE differs")
    keys = {record["arm"]: media.decode_key(record["output"]["sha256"], vae_hash,
                                          list(output.shape), "native_decode_video", args.seed,
                                          media.native_decoder_settings())
            for record, output in zip(records, outputs, strict=True)}
    if review.get("decoded_outputs") != keys:
        raise ValueError("conditioning output decoder keys differ")
    pixels = {}
    edge = prepared["video"].encode_records["capture"]["edge"]
    for name in names:
        path = args.output / "review" / f"{name}_rgb.pt"
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(value, torch.Tensor) or list(value.shape) != [49, 3, edge, edge]
                or not torch.isfinite(value).all() or review["panels"][name] != {
                    "path": str(path), "sha256": sha256(path), "tensor_sha256": hashing.tensor_sha256(value),
                    "shape": list(value.shape)}):
            raise ValueError("conditioning decoded pixel evidence differs")
        pixels[name] = value
    if args.control == "guide" and not torch.equal(pixels["donor_guide"], media.recorded_guide_rgb(
            reference_inputs.checked_source(store, args.donor_source)[0], store.root, "white", 7)):
        raise ValueError("conditioning donor RGB differs from its own recorded preprocessing")
    panels = review_panels(args, prepared, references, pixels)
    questions = {"guide": "Does changing the guide change identity and action?",
                 "text": "Does changing the text change identity and action?",
                 "sigma_one": "Do D0 and D1 agree when source weight is zero?",
                 "capture_oracle": "Does the supplied capture improve identity and action?"}
    expected_renderings = {name: str(args.output / "review" / name / "rendering.json") for name in ("full", "narrow")}
    if review.get("renderings") != expected_renderings:
        raise ValueError("conditioning review rendering inventory differs")
    for name, selected_panels, layout, width in review_variants(panels, questions[args.control]):
        folder = args.output / "review" / name
        rendering = json.loads((folder / "rendering.json").read_text())
        software.check_current(rendering.get("software"))
        rebuilt, expected = media.render_panels(
            selected_panels, question=questions[args.control], layout=layout, fps=prepared["video"].fps,
            viewing_width=width, poster_frame=24, common_settings=settings)
        expected["software"] = producer
        files = {role: {"path": str(folder / filename), "sha256": sha256(folder / filename)}
                 for role, filename in (("video", "comparison.mp4"), ("poster", "poster.png"))}
        if rendering != {**expected, "outputs": files}:
            raise ValueError("conditioning review panel labels, layout, synchronization or bytes differ")
        measured = media.verify_saved_video(folder / "comparison.mp4", 49, prepared["video"].fps)
        if [measured["width"], measured["height"]] != expected["display_size"]:
            raise ValueError("conditioning review actual video dimensions differ")
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415 -- saved poster only

        with Image.open(folder / "poster.png") as poster:
            actual = torch.from_numpy(np.array(poster.convert("RGB"))).permute(2, 0, 1)
        if not torch.equal(actual, rebuilt[24]):
            raise ValueError("conditioning review poster differs from saved panels")
    seconds = review.get("seconds_decode_and_render")
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
            or review.get("memory_boundary") != "cumulative runtime peak through decoding; queue peak is not reset"
            or any(type(review.get(key)) is not int or review[key] <= 0 for key in
                   ("peak_allocated_bytes", "peak_reserved_bytes"))):
        raise ValueError("conditioning decode cost evidence differs")


def execute(args: argparse.Namespace) -> dict:
    prepared = prepare(args)
    producer = software.capture("evaluation", args.mode, decoder=args.decode, extra_sources=EXTRAS)
    from scripts.prune.core.session import DEFAULT_PROMPT, DTYPE, Session  # noqa: PLC0415
    from scripts.prune.data import prompt_cache  # noqa: PLC0415

    device = torch.device(f"cuda:{args.gpu_id}")
    prompt = DEFAULT_PROMPT if args.prompt is None else args.prompt
    context = prompt_cache.get_or_build(prepared["specification"], prompt, DTYPE, device)
    wrong = (prompt_cache.get_or_build(prepared["specification"], args.wrong_prompt, DTYPE, device)
             if args.control == "text" else None)
    grid, capture, guide, foreign, noise = input_tokens(prepared, device)
    branches = control_inputs(args.control, context, guide, foreign_guide=foreign, foreign_text=wrong)
    for checked in prepared["checks"].values():
        if args.checkpoint is not None:
            checkpoints.recheck_adapter(args.checkpoint, checked["contract"], checked["adapter_sha256"])
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(context.cpu(), args.output / "text.pt")
    torch.save(noise.cpu(), args.output / "noise.pt")
    if wrong is not None:
        torch.save(wrong.cpu(), args.output / "wrong_text.pt")
    session = Session(prepared["specification"], device, "onestep_avatar.conditioning_check", context)
    checked = prepared["checks"][ARMS[args.control][-1]]
    outputs, records = [], []
    with adapters.inference_transformer(session, args.checkpoint, checked.get("contract"),
                                       method=adapters.UNMERGED,
                                       adapter_sha256=checked.get("adapter_sha256")) as transformer:
        for name, text, selected_guide, guide_mode in branches:
            software.check_current(producer)
            output, record = evaluate.sample_case(
                transformer, text, grid, capture, selected_guide, noise, mode=args.mode,
                mode_settings=prepared["ordinary"].mode_settings, guide_mode=guide_mode,
                schedule=prepared["ordinary"].schedule, seed=args.seed)
            outputs.append(output)
            selected_prompt = args.wrong_prompt if name == "foreign_text" else prompt
            records.append(evaluate.save_case(output, {**record, **arm_provenance(
                args, prepared, name, guide_mode, selected_prompt, producer)}, args.output / name))
    if args.decode:
        render_review(args, prepared, session, outputs, records, producer)
    summary = summarize(args.control, outputs)
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    software.check_current(producer)
    if args.checkpoint is not None:
        checkpoints.recheck_adapter(args.checkpoint, checked["contract"], checked["adapter_sha256"])
    verify_scientific(args)
    spec_protocol.publish(args, SELECTOR)
    return summary


def verify_scientific(args: argparse.Namespace) -> dict:
    prepared = prepare(args, fresh=False)
    producer = software.capture("evaluation", args.mode, decoder=args.decode, extra_sources=EXTRAS)
    grid, capture, guide, foreign, noise = input_tokens(prepared, torch.device("cpu"))
    context = torch.load(args.output / "text.pt", map_location="cpu", weights_only=True)
    wrong = (torch.load(args.output / "wrong_text.pt", map_location="cpu", weights_only=True)
             if args.control == "text" else None)
    if (not isinstance(context, torch.Tensor) or context.ndim != 3 or context.shape[0] != 1
            or context.dtype != torch.bfloat16 or not torch.isfinite(context).all()):
        raise ValueError("conditioning saved text must be finite native batch-one context")
    saved_noise = torch.load(args.output / "noise.pt", map_location="cpu", weights_only=True)
    if not isinstance(saved_noise, torch.Tensor) or not torch.equal(saved_noise, noise):
        raise ValueError("conditioning saved noise differs")
    branches = control_inputs(args.control, context, guide, foreign_guide=foreign, foreign_text=wrong)
    from scripts.prune.core.session import DEFAULT_PROMPT  # noqa: PLC0415 -- constant only

    prompt = DEFAULT_PROMPT if args.prompt is None else args.prompt
    outputs, records = [], []
    for name, text, selected_guide, guide_mode in branches:
        folder = args.output / name
        record = json.loads((folder / "result.json").read_text())
        software.check_current(record.get("software"))
        selected_prompt = args.wrong_prompt if name == "foreign_text" else prompt
        expected = {**arm_provenance(args, prepared, name, guide_mode, selected_prompt, producer),
                    "schema_version": 2, "state": "complete", "mode": args.mode,
                    "mode_settings": asdict(prepared["ordinary"].mode_settings), "guide_mode": guide_mode,
                    "frames": 7, "schedule": prepared["ordinary"].schedule, "seed": args.seed,
                    "capture_sha256": hashing.tensor_sha256(capture),
                    "guide_sha256": hashing.tensor_sha256(selected_guide),
                    "c0_sha256": hashing.tensor_sha256(capture[:, :grid.tokens_per_latent_frame]),
                    "noise_sha256": hashing.tensor_sha256(noise), "text_sha256": hashing.tensor_sha256(text)}
        if args.mode == "causal":
            expected.update(history_mode="cache", kv_source="refresh")
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("conditioning actual branch evidence differs")
        path = folder / "generated.pt"
        output = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(output, torch.Tensor) or output.dtype != torch.bfloat16
                or list(output.shape) != [1, *prepared["video"].z_y[:, :7].shape]
                or not torch.isfinite(output).all()
                or not torch.equal(output[:, :, :1], prepared["video"].z_y[:, :1].unsqueeze(0).to(torch.bfloat16))):
            raise ValueError("conditioning output shape, dtype, values or clean first image differs")
        if record["output"] != {"path": str(path), "sha256": sha256(path), "shape": list(output.shape)}:
            raise ValueError("conditioning output bytes or record differ")
        expected_counts = {"model_calls": 1 if args.mode == "bidirectional" else 6,
                           "denoise_calls": 1 if args.mode == "bidirectional" else 3,
                           "prime_calls": 0,
                           "refresh_calls": 0 if args.mode == "bidirectional" else 3}
        counts = record.get("call_counts")
        if (counts != expected_counts or any(type(value) is not int for value in counts.values())):
            raise ValueError("conditioning native call counts differ")
        target = grid.unpatchify_block(capture, 7)
        if record.get("metrics") != metrics.encoded_metrics(output, target):
            raise ValueError("conditioning arm metrics differ from saved outputs")
        outputs.append(output)
        records.append(record)
    summary = summarize(args.control, outputs)
    if json.loads((args.output / "result.json").read_text()) != summary:
        raise ValueError("conditioning summary differs from saved outputs")
    if args.decode:
        verify_review(args, prepared, outputs, records, producer)
    return summary


def verify_completion(spec: Path, output: Path) -> dict:
    args = read_spec(spec, output)
    spec_protocol.inspect(args, SELECTOR)
    if args.operation == "prepare_references":
        return reference_inputs.verify(args)
    if args.operation == "past_influence":
        from scripts.onestep_avatar.experiments import past_influence_check  # noqa: PLC0415

        return past_influence_check.verify(args)
    if args.operation == "capacity_review":
        from scripts.onestep_avatar.experiments import capacity_review  # noqa: PLC0415

        return capacity_review.verify(args)
    result = verify_scientific(args)
    if result["state"] != "passed":
        raise ValueError("conditioning equality control failed")
    return result


def evidence_paths(spec: Path, output: Path, *, verified: bool = False) -> list[Path]:
    if not verified:
        verify_completion(spec, output)
    return spec_protocol.evidence(spec_protocol.inspect(read_spec(spec, output), SELECTOR), output)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911 -- separate operation lifecycles
    outer = parse_args(argv)
    args = scientific_spec(outer)
    if args.operation == "past_influence":
        from scripts.onestep_avatar.experiments import past_influence_check  # noqa: PLC0415

        if args.dry_run:
            print(json.dumps(past_influence_check.prepare(args), indent=2))  # noqa: T201
            return 0
        past_influence_check.execute(args)
        spec_protocol.publish(args, SELECTOR)
        return 0
    if args.operation == "prepare_references":
        if args.dry_run:
            prepared = reference_inputs.prepare(args)
            print(json.dumps(prepared["crop"], indent=2))  # noqa: T201
            return 0
        reference_inputs.execute(args)
        spec_protocol.publish(args, SELECTOR)
        return 0
    if args.operation == "capacity_review":
        from scripts.onestep_avatar.experiments import capacity_review  # noqa: PLC0415

        if args.dry_run:
            prepared = capacity_review.prepare(args)
            print(json.dumps({"cases": len(prepared["cases"]), "mode": args.mode}, indent=2))  # noqa: T201
            return 0
        capacity_review.execute(args)
        spec_protocol.publish(args, SELECTOR)
        return 0
    if args.dry_run:
        prepared = prepare(args)
        print(json.dumps({"control": args.control, "arms": ARMS[args.control],  # noqa: T201
                          "target": prepared["video"].source,
                          "donor": None if prepared["donor"] is None else prepared["donor"].source,
                          "conditions": prepared["requested"]}, indent=2))
        return 0
    return 0 if execute(args)["state"] == "passed" else 2


if __name__ == "__main__":
    from scripts.onestep_avatar.execution.runtime import run

    raise SystemExit(run(main))
