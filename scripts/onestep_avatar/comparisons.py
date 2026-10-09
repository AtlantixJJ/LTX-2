"""Render and verify saved comparisons; see doc/comparisons.md."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from scripts.onestep_avatar import evaluate, hashing
from scripts.onestep_avatar import metrics as metrics_ops
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.corpus.dataset import atomic_write
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256


def _saved_panel_path(item: dict, root: Path) -> tuple[Path, bool]:
    """Resolve the shared saved-output/master spelling without loading data."""
    reference = item.get("latent")
    if not isinstance(reference, str) or not reference:
        raise ValueError("saved comparison latent is missing")
    master = reference.startswith(("capture:", "guide:"))
    path = Path(reference.split(":", 1)[1] if master else reference)
    path = (root / path).resolve()
    return path, master



def saved_comparison_inputs_ready(spec_path: Path) -> bool:
    """Wait for saved panel files; invalid empty specifications fail rather than wait."""
    spec = json.loads(spec_path.read_text())
    comparisons = spec.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("saved comparison specification is empty")
    paths = []
    for comparison in comparisons:
        panels = comparison.get("panels")
        if not isinstance(panels, list) or not panels:
            raise ValueError("saved comparison requires panels")
        root = spec_path.resolve().parent
        if "reference_bundle" in comparison:
            reference_path = (root / comparison["reference_bundle"]).resolve()
            paths.append(reference_path)
            if reference_path.is_file():
                reference = json.loads(reference_path.read_text())
                paths.extend(Path(row["path"]) for row in reference.get("panels", []) if row.get("path") is not None)
        for panel in panels:
            if "reference_role" not in panel:
                paths.append(_saved_panel_path(panel, root)[0])
            if "result" in panel:
                paths.append((root / panel["result"]).resolve())
    return all(path.is_file() for path in paths)



def _saved_panel_input(item: dict, root: Path, span: int, fps: float) -> tuple[torch.Tensor, dict]:
    """Resolve an existing output or a checked master without changing recorded bytes."""
    path, master = _saved_panel_path(item, root)
    if not path.is_file():
        raise ValueError("saved comparison latent is missing")
    fingerprint = sha256(path)
    if master:
        latent, source_fps = dataset.load_training_master(path)
        if source_fps != fps:
            raise ValueError("saved comparison bundle fps differs from playback fps")
        if latent.shape[1] < span:
            raise ValueError("saved comparison master has insufficient frame coverage")
        latent = latent[:, :span].unsqueeze(0)
    else:
        latent = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(latent, torch.Tensor) or latent.ndim != 5 or latent.shape[0] != 1
        or not latent.is_floating_point() or min(latent.shape) < 1 or not torch.isfinite(latent).all()
    ):
        raise ValueError("saved comparison requires a finite floating B,C,F,H,W latent with batch one")
    if latent.shape[2] != span:
        raise ValueError("saved comparison encoded geometry differs from requested coverage")
    if sha256(path) != fingerprint:
        raise ValueError("saved comparison latent changed while loading")
    return latent, {"path": str(path), "sha256": fingerprint, "shape": list(latent.shape), "master": master}



def _saved_comparison_inputs(  # noqa: PLR0912 -- all saved RGB/result gates precede decoder work
    comparison: dict, root: Path, model: str, seed: int
) -> list[tuple[torch.Tensor, dict]]:
    """Read latent-only panels or exactly matched, already-prepared RGB references."""
    from scripts.onestep_avatar import media  # noqa: PLC0415 -- checked saved RGB reader
    from scripts.prune.core import model_registry  # noqa: PLC0415 -- geometry only

    span, fps = comparison.get("span", 17), comparison.get("fps", 30)
    panels = comparison["panels"]
    if "reference_bundle" not in comparison:
        if any("reference_role" in panel for panel in panels):
            raise ValueError("saved RGB panels require a reference bundle")
        return [_saved_panel_input(panel, root, span, fps) for panel in panels]
    if "view" in comparison:
        raise ValueError("saved RGB reference comparisons do not use legacy view metrics")
    path = (root / comparison["reference_bundle"]).resolve()
    manifest_hash = sha256(path)
    references, producer = media.load_training_references(path)
    manifest = json.loads(path.read_text())
    if sha256(path) != manifest_hash:
        raise ValueError("saved reference bundle changed while loading")
    software.check_current(producer.get("software"))
    scale = model_registry.resolve(model).scale_factors
    frames = 1 + (span - 1) * scale.time
    if (producer.get("fps") != fps or producer.get("source_frames") != list(range(frames))
            or producer.get("decode_seed") != seed
            or producer.get("decoder_settings") != media.native_decoder_settings()
            or producer.get("vae_sha256") != sha256(Path(model_registry.resolve(model).paths.video_vae()))):
        raise ValueError("saved reference source mapping, timebase or decoder settings differ")
    if (len(panels) < 4 or [item.get("reference_role") for item in panels[:3]] != ["recorded", "decoded", "guide"]
            or any("reference_role" in item for item in panels[3:])):
        raise ValueError("saved reference panels require recorded, decoded, guide before outputs")
    inputs, records = [], []
    for item, panel, row in zip(panels[:3], references, manifest["panels"], strict=True):
        pixels = panel.pixels
        if (item.get("role") != panel.role or "latent" in item or pixels is None
                or pixels.ndim != 4 or pixels.shape[:2] != (frames, 3)
                or not (pixels.is_floating_point() or pixels.dtype == torch.uint8)
                or not torch.isfinite(pixels).all()):
            raise ValueError("saved reference RGB values or panel roles are invalid")
        inputs.append((pixels, {"path": row["path"], "sha256": row["sha256"], "shape": list(pixels.shape),
                               "kind": "rgb_reference", "reference_bundle": {"path": str(path),
                                                                                "sha256": manifest_hash}}))
    for item in panels[3:]:
        latent, evidence = _saved_panel_input(item, root, span, fps)
        if "result" not in item:
            raise ValueError("saved reference output requires its executed result record")
        result_path = (root / item["result"]).resolve()
        result_hash = sha256(result_path)
        record = json.loads(result_path.read_text())
        software.validate(record.get("software"))
        if (record.get("state") != "complete" or record.get("source") != producer.get("source")
                or record.get("fps") != fps or record.get("frames") != span
                or record.get("input_file_hashes", {}).get("capture") != producer.get("capture_encoding_sha256")
                or record.get("input_file_hashes", {}).get("render") != producer.get("guide_rgb_sha256")
                or record.get("membership_sha256") != producer.get("membership_sha256")
                or record.get("conditions", {}).get("task", {}).get("objective") != producer.get("objective")
                or record.get("guide_mode") != "d1"
                or Path(record.get("output", {}).get("path", "")).resolve() != Path(evidence["path"])
                or record.get("output", {}).get("sha256") != evidence["sha256"]
                or record.get("output", {}).get("shape") != list(latent.shape)
                or hashing.tensor_sha256(VideoLatentPatchifier(patch_size=1).patchify(latent[:, :, :1]))
                != record.get("c0_sha256")):
            raise ValueError("saved output result differs from its RGB references or encoding")
        if any(pixels.shape[-2:] != (latent.shape[-2] * scale.height, latent.shape[-1] * scale.width)
               for pixels, _ in inputs[:3]):
            raise ValueError("saved reference RGB dimensions differ from output encoding")
        evidence["result"] = {"path": str(result_path), "sha256": result_hash}
        inputs.append((latent, evidence))
        records.append(record)
    if len(records) > 1:
        evaluate.validate_comparison(records, comparison.get("changed_factor"))
        if any(record.get("software") != records[0].get("software") for record in records[1:]):
            raise ValueError("saved comparison output producers differ")
    return inputs



def _saved_input_bytes_current(inputs: list[tuple[torch.Tensor, dict]]) -> bool:
    """Check every bound pixel, latent, manifest and executed-result file once."""
    files = {}
    for _, record in inputs:
        for evidence in (record, record.get("reference_bundle"), record.get("result")):
            if evidence is not None:
                files[evidence["path"]] = evidence["sha256"]
    return all(sha256(Path(path)) == digest for path, digest in files.items())



def parse_saved_comparison_args(argv: list[str], *, require_gpu: bool = False) -> argparse.Namespace:
    """One parser for direct rendering and queue normalization."""
    parser = argparse.ArgumentParser(description="Render saved comparison latents with the package decoder owner")
    parser.add_argument("--render-saved-comparisons", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=require_gpu, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)



def saved_comparison_identity(spec_path: Path, spec: dict, seed: int) -> dict:
    """Bind a saved render to actual decoder bytes and software, without model loading."""
    from scripts.onestep_avatar import media  # noqa: PLC0415
    from scripts.prune.core import model_registry  # noqa: PLC0415

    model = model_registry.resolve(spec.get("model", "2.5"))
    vae = Path(model.paths.video_vae()).resolve()
    return {
        "spec": str(spec_path.resolve()), "spec_sha256": sha256(spec_path), "seed": seed,
        "decoder": {"model": spec.get("model", "2.5"), "variant": spec.get("variant", "dev"),
                    "vae_path": str(vae), "vae_sha256": sha256(vae), "settings": media.native_decoder_settings()},
        "source_code_sha256": {"evaluate": sha256(Path(__file__)), "media": sha256(Path(media.__file__))},
        "software": software.capture("decoding"),
    }



def _save_comparison_variant(pixels: torch.Tensor, record: dict, output: Path, name: str) -> dict:
    """Save a shared-layout variant and bind its stable report-facing media names."""
    from scripts.onestep_avatar import media  # noqa: PLC0415 -- common RGB media owner

    destination = output / name
    complete = media.save_render(pixels, record, destination)
    named_video, named_poster = output / f"{name}.mp4", output / f"{name}_poster.png"
    (destination / "comparison.mp4").replace(named_video)
    (destination / "poster.png").replace(named_poster)
    for field, path in (("video", named_video), ("poster", named_poster)):
        complete["outputs"][field] = {"path": str(path), "sha256": sha256(path)}
    (destination / "rendering.json").write_text(json.dumps(complete, indent=2) + "\n")
    return {"rendering": complete, "video": named_video.name, "poster": named_poster.name}



def render_saved_comparisons(  # noqa: PLR0912, PLR0915 -- preflight, decode, QA and publication in order
    spec_path: Path, output: Path, *, gpu_id: int, seed: int = 42
) -> dict:
    """Decode and render a saved comparison specification under package ownership.

    The spec contains only saved latent paths and panel metadata. This function
    opens the selected VAE session, never a transformer, and publishes media
    through the shared renderer. Report code receives the manifest afterward.
    """
    from scripts.onestep_avatar import media  # noqa: PLC0415 -- shared preflight/decode/render owner

    spec_hash = sha256(spec_path)
    spec = json.loads(spec_path.read_text())
    if not isinstance(spec.get("comparisons"), list) or not spec["comparisons"]:
        raise ValueError("saved comparison specification is empty")
    if (output / "render_manifest.json").exists():
        raise ValueError("saved comparison output already exists")
    spec_root = spec_path.resolve().parent
    names = set()
    reserved_outputs = set()
    prepared = []
    compact_layouts = []
    for comparison in spec["comparisons"]:
        if not isinstance(comparison, dict) or not isinstance(comparison.get("name"), str) or not comparison["name"]:
            raise ValueError("saved comparison requires a nonempty name")
        if comparison["name"] in names:
            raise ValueError("saved comparison names must be unique")
        names.add(comparison["name"])
        if Path(comparison["name"]).name != comparison["name"] or comparison["name"] in (".", ".."):
            raise ValueError("saved comparison name must be one safe path component")
        for name in (comparison["name"], comparison["name"] + "_compact"):
            for target in (output / name, output / f"{name}.mp4", output / f"{name}_poster.png"):
                if target in reserved_outputs:
                    raise ValueError("saved comparison output names collide")
                reserved_outputs.add(target)
                if target.exists():
                    raise ValueError("saved comparison output already exists")
        span, fps = comparison.get("span", 17), comparison.get("fps", 30)
        if type(span) is not int or span < 1 or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("saved comparison span and fps must be positive")
        panels = comparison.get("panels")
        if not isinstance(panels, list) or not panels:
            raise ValueError("saved comparison requires panels")
        for panel in panels:
            if not isinstance(panel, dict) or not isinstance(panel.get("title"), str):
                raise ValueError("saved comparison panel metadata is invalid")
        inputs = _saved_comparison_inputs(comparison, spec_root, spec.get("model", "2.5"), seed)
        latents = [value for value, record in inputs if record.get("kind") != "rgb_reference"]
        if any(latent.shape != latents[0].shape for latent in latents):
            raise ValueError("saved comparison panels must have identical encoded geometry")
        if "view" in comparison and latents[0].shape[2] < 2:
            raise ValueError("saved comparison metrics require frames after the first image")
        metadata = [media.Panel(item.get("role", f"panel_{index}"), item["title"], None, (),
                                value=item.get("value", "")) for index, item in enumerate(panels)]
        question, layout = comparison.get("question", comparison["name"]), comparison.get("layout", "comparison")
        panel_size = tuple(comparison.get("panel_size", [400, 400]))
        media.layout_geometry(metadata, question=question, layout=layout, panel_size=panel_size,
                              viewing_width=comparison.get("viewing_width", 1280))
        compact_layouts.append(media.compact_layout(metadata, question=question, layout=layout, panel_size=panel_size))
        prepared.append(inputs)

    identity = saved_comparison_identity(spec_path, spec, seed)
    if identity["spec_sha256"] != spec_hash:
        raise ValueError("saved comparison specification changed during preflight")
    if any(not _saved_input_bytes_current(inputs) for inputs in prepared):
        raise ValueError("saved comparison inputs changed during preflight")
    software.check_current(identity["software"])
    session = media.open_decoder_session(
        spec.get("model", "2.5"), gpu_id, script="onestep_avatar.evaluate.saved_comparisons"
    )
    perceptual = None
    if any("view" in comparison for comparison in spec["comparisons"]):
        import lpips  # noqa: PLC0415 -- historical RGB QA only

        perceptual = lpips.LPIPS(net="alex", verbose=False).to(session.device).eval()
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    with session.decoder() as decoder, torch.inference_mode():
        for comparison, inputs, compact_layout in zip(spec["comparisons"], prepared, compact_layouts, strict=True):
            panels = []
            for index, (item, (value, record)) in enumerate(zip(comparison["panels"], inputs, strict=True)):
                pixels = value if record.get("kind") == "rgb_reference" else media.decode(session, value, decoder, seed)
                panels.append(
                    media.Panel(
                        item.get("role", f"panel_{index}"),
                        item["title"],
                        pixels,
                        tuple(range(len(pixels))),
                        value=item.get("value", ""),
                    )
                )
            metrics = []
            if "view" in comparison:
                reference = panels[0].pixels
                view = (spec_root / comparison["view"]).resolve()
                mask = metrics_ops.subject_mask(view / "capture_mask_crop.mp4", len(reference), *reference.shape[-2:])
                for item, panel in zip(comparison["panels"], panels, strict=True):
                    metrics.append({
                        "title": item["title"], "latent": item.get("latent"),
                        "psnr_full": metrics_ops.rgb_metrics(panel.pixels[1:], reference[1:])["psnr"],
                        "psnr_subject": None if mask is None else metrics_ops.subject_rgb_metrics(
                            panel.pixels[1:], reference[1:], mask[1:]
                        )["psnr"],
                        "lpips": metrics_ops.lpips_distance(
                            perceptual, panel.pixels[1:], reference[1:], session.device
                        ),
                    })
            rendered, record = media.render_panels(
                panels,
                question=comparison.get("question", comparison["name"]),
                layout=comparison.get("layout", "comparison"),
                fps=comparison.get("fps", 30),
                common_settings={"spec": str(spec_path.resolve()), "seed": seed},
                poster_frame=min(comparison.get("poster_frame", 96), len(panels[0].pixels) - 1),
                panel_size=tuple(comparison.get("panel_size", [400, 400])),
                viewing_width=comparison.get("viewing_width", 1280),
            )
            record["software"] = identity["software"]
            full = _save_comparison_variant(rendered, record, output, comparison["name"])
            del rendered
            compact_pixels, compact_record = media.render_panels(
                panels, question=comparison.get("question", comparison["name"]), layout=compact_layout,
                fps=comparison.get("fps", 30), panel_size=tuple(comparison.get("panel_size", [400, 400])),
                viewing_width=480, common_settings={"spec": str(spec_path.resolve()), "seed": seed},
                poster_frame=min(comparison.get("poster_frame", 96), len(panels[0].pixels) - 1),
            )
            compact_record["software"] = identity["software"]
            compact = _save_comparison_variant(compact_pixels, compact_record, output, comparison["name"] + "_compact")
            del compact_pixels
            manifest.append({**comparison, **full, "compact": compact, "inputs": [row for _, row in inputs],
                             "frames": len(panels[0].pixels), "metrics": metrics})
    if (saved_comparison_identity(spec_path, spec, seed) != identity
            or any(not _saved_input_bytes_current(inputs) for inputs in prepared)):
        raise ValueError("saved comparison inputs changed before publication")
    result = {"schema_version": 3, **identity, "comparisons": manifest, "results": manifest}
    atomic_write(
        output / "render_manifest.json",
        lambda temporary: temporary.write_text(json.dumps(result, indent=2) + "\n"),
    )
    return result



def _verify_comparison_variant(  # noqa: PLR0912 -- bound layout/media checks
    request: dict, row: dict, output: Path, *, frames: int, fps: float, spec_path: Path, seed: int
) -> bool:
    from scripts.onestep_avatar import media  # noqa: PLC0415 -- single geometry owner

    rendering = row.get("rendering", {})
    software.check_current(rendering.get("software"))
    if (rendering.get("fps") != fps
            or rendering.get("source_frames") != list(range(frames))
            or rendering.get("layout") != request.get("layout", "comparison")
            or rendering.get("question") != request.get("question", request["name"])
            or rendering.get("poster_frame") != min(request.get("poster_frame", 96), frames - 1)
            or rendering.get("panel_size") != request.get("panel_size", [400, 400])
            or rendering.get("viewing_width") != request.get("viewing_width", 1280)
            or rendering.get("common_settings") != {"spec": str(spec_path.resolve()), "seed": seed}):
        raise ValueError("saved comparison completion coverage or settings differ")
    metadata = [media.Panel(item.get("role", f"panel_{index}"), item["title"], None, (),
                            value=item.get("value", "")) for index, item in enumerate(request["panels"])]
    geometry = media.layout_geometry(metadata, question=request.get("question", request["name"]),
                                     layout=request.get("layout", "comparison"),
                                     panel_size=tuple(request.get("panel_size", [400, 400])),
                                     viewing_width=request.get("viewing_width", 1280))
    if (rendering.get("font_size") != geometry["font_size"]
            or rendering.get("display_size") != geometry["display_size"]
            or rendering.get("source_times") != [value / fps for value in range(frames)]):
        raise ValueError("saved comparison completion geometry or readability differs")
    panels = [item for item in rendering.get("panels", []) if item.get("role") != "unused"]
    roles = [panel.get("role", f"panel_{index}") for index, panel in enumerate(request["panels"])]
    if rendering["layout"] in media.COMPARISON_COLUMNS:
        columns = min(media.COMPARISON_COLUMNS[rendering["layout"]], len(roles))
        roles += ["unused"] * (-len(roles) % columns)
        positions = [(index // columns, index % columns, role) for index, role in enumerate(roles)]
    else:
        positions = [(r, c, role) for r, line in enumerate(media.LAYOUTS[rendering["layout"]])
                     for c, role in enumerate(line)]
    if [(item.get("row"), item.get("column"), item.get("role"))
            for item in rendering.get("panels", [])] != positions:
        raise ValueError("saved comparison completion panel positions differ")
    if [item.get("title") for item in panels] != [item["title"] for item in request["panels"]]:
        raise ValueError("saved comparison completion rendered titles differ")
    for index, (panel, requested) in enumerate(zip(panels, request["panels"], strict=True)):
        digest = panel.get("pixels_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("saved comparison completion panel pixel hash is invalid")
        if (panel.get("role") != requested.get("role", f"panel_{index}")
                or panel.get("value") != requested.get("value", "")
                or panel.get("source_frames") != list(range(frames))):
            raise ValueError("saved comparison completion rendered panel settings differ")
    for field, suffix in (("video", ".mp4"), ("poster", "_poster.png")):
        name = row.get(field)
        if name != request["name"] + suffix or Path(name).name != name:
            raise ValueError("saved comparison completion media name differs")
        artifact = output / name
        if not artifact.resolve().is_relative_to(output.resolve()):
            raise ValueError("saved comparison completion media escapes output")
        if not artifact.is_file():
            return False
        recorded = rendering.get("outputs", {}).get(field, {})
        if (Path(recorded.get("path", "")).resolve() != artifact.resolve()
                or artifact.stat().st_size == 0 or sha256(artifact) != recorded.get("sha256")):
            raise ValueError("saved comparison completion media hash or path differs")
    return True



def verify_saved_comparison_completion(  # noqa: PLR0912 -- exact rendering evidence gates together
    spec_path: Path, output: Path, *, seed: int = 42
) -> bool:
    """Verify exact queued render evidence without invoking a decoder or model."""
    from scripts.onestep_avatar import media  # noqa: PLC0415 -- canonical layout declarations
    from scripts.prune.core import model_registry  # noqa: PLC0415

    path = output / "render_manifest.json"
    if not path.is_file():
        return False
    spec, result = json.loads(spec_path.read_text()), json.loads(path.read_text())
    if not saved_comparison_inputs_ready(spec_path):
        return False
    identity = saved_comparison_identity(spec_path, spec, seed)
    if result.get("schema_version") != 3 or any(result.get(key) != value for key, value in identity.items()):
        raise ValueError("saved comparison completion identity differs")
    expected, actual = spec.get("comparisons"), result.get("comparisons")
    if not expected or not isinstance(actual, list) or len(actual) != len(expected) or result.get("results") != actual:
        raise ValueError("saved comparison completion inventory differs")
    scale = model_registry.resolve(spec.get("model", "2.5")).scale_factors.time
    for request, row in zip(expected, actual, strict=True):
        if any(row.get(key) != value for key, value in request.items()):
            raise ValueError("saved comparison completion request fields differ")
        span, fps = request.get("span", 17), request.get("fps", 30)
        saved_inputs = row.get("inputs")
        if not isinstance(saved_inputs, list) or len(saved_inputs) != len(request["panels"]):
            raise ValueError("saved comparison completion input inventory differs")
        try:
            prepared = _saved_comparison_inputs(request, spec_path.resolve().parent, spec.get("model", "2.5"), seed)
        except FileNotFoundError:
            return False
        inputs = [record for _, record in prepared]
        if not _saved_input_bytes_current(prepared):
            raise ValueError("saved comparison completion input hash or path differs")
        del prepared
        if row.get("inputs") != inputs:
            raise ValueError("saved comparison completion input inventory differs")
        frames = 1 + (span - 1) * scale
        if row.get("frames") != frames:
            raise ValueError("saved comparison completion coverage or settings differ")
        if not _verify_comparison_variant(request, row, output, frames=frames, fps=fps, spec_path=spec_path, seed=seed):
            return False
        metadata = [media.Panel(item.get("role", f"panel_{index}"), item["title"], None, (),
                                value=item.get("value", "")) for index, item in enumerate(request["panels"])]
        layout = media.compact_layout(metadata, question=request.get("question", request["name"]),
                                      layout=request.get("layout", "comparison"),
                                      panel_size=tuple(request.get("panel_size", [400, 400])))
        compact_request = {**request, "name": request["name"] + "_compact",
                           "question": request.get("question", request["name"]), "layout": layout, "viewing_width": 480}
        compact = row.get("compact")
        if not isinstance(compact, dict):
            raise ValueError("saved comparison completion requires compact media")
        if not _verify_comparison_variant(compact_request, compact, output, frames=frames, fps=fps,
                                          spec_path=spec_path, seed=seed):
            return False
        def pixel_hashes(record: dict) -> dict:
            return {item["role"]: item.get("pixels_sha256") for item in record["panels"] if item["role"] != "unused"}
        if pixel_hashes(row["rendering"]) != pixel_hashes(compact["rendering"]):
            raise ValueError("saved comparison compact panel pixels differ")
    return True


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    args = parse_saved_comparison_args(arguments, require_gpu=True)
    render_saved_comparisons(args.render_saved_comparisons, args.output, gpu_id=args.gpu_id, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
