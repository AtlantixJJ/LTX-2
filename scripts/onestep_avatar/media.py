"""Decode saved outputs and render synchronized panels; see doc/media.md."""

from __future__ import annotations

import hashlib
import argparse
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.execution import software

if TYPE_CHECKING:
    from scripts.prune.core.session import Session

COMPARISON_COLUMNS = {"comparison": 3, "compact_comparison": 2, "stacked_comparison": 1}

LAYOUTS = {
    "training": (("recorded", "decoded", "guide"), ("baseline", "changed", "unused")),
    "compact_training": (("recorded", "decoded"), ("guide", "unused"), ("baseline", "changed")),
    "inference": (("first_image", "guide", "generated"),),
    "compact_inference": (("first_image", "unused"), ("guide", "generated")),
    "separate_references": (("recorded_baseline", "recorded_changed"), ("baseline", "changed")),
}


def as_fchw(pixels: torch.Tensor) -> torch.Tensor:
    """Normalize the existing decoder layouts without changing float pixel values."""
    if pixels.ndim == 5:
        return pixels.permute(0, 2, 1, 3, 4).flatten(0, 1)
    if pixels.ndim == 4 and pixels.shape[-1] in (1, 3, 4):
        return pixels.permute(0, 3, 1, 2)
    if pixels.ndim != 4:
        raise ValueError(f"expected BCTHW, FCHW or FHWC pixels, got {tuple(pixels.shape)}")
    return pixels


def open_decoder_session(model: str, gpu_id: int, *, script: str) -> Session:
    """Check the selected GPU and create a decoder session with no text work."""
    from scripts.prune.core import preflight  # noqa: PLC0415 -- native device/model checks
    from scripts.prune.core.session import Session  # noqa: PLC0415 -- decoder handle only

    specification = preflight.check(model, gpu_id=gpu_id)
    return Session(specification, torch.device(f"cuda:{gpu_id}"), script, None)


def native_decoder_settings() -> dict:
    """Return the runtime identity of this module's native decoder path."""
    return {"dtype": "bfloat16", "tiling": None, "fresh_generator_per_decode": True, "torch": torch.__version__}


def verify_saved_video(path: Path, frames: int, fps: float) -> dict:
    """Probe actual encoded coverage without opening any model or VAE."""
    import subprocess  # noqa: PLC0415 -- actual saved media acceptance
    from fractions import Fraction  # noqa: PLC0415 -- exact playback rates

    if type(frames) is not int or frames < 1 or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("saved video requires positive frame count and playback rate")
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames,avg_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        streams = json.loads(result.stdout)["streams"]
        if len(streams) != 1:
            raise ValueError("saved video must contain a selected video stream")
        stream = streams[0]
        if int(stream["nb_read_frames"]) != frames or Fraction(stream["avg_frame_rate"]) != Fraction(str(fps)):
            raise ValueError("saved video frame count or playback rate differs")
        if int(stream["width"]) < 1 or int(stream["height"]) < 1:
            raise ValueError("saved video dimensions are invalid")
    except (OSError, subprocess.CalledProcessError, KeyError, TypeError, ZeroDivisionError) as error:
        raise ValueError("cannot verify saved video coverage") from error
    return stream


def verify_saved_png(path: Path, width: int, height: int) -> None:
    """Require a fully readable RGB PNG with the saved video's actual dimensions."""
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB" or image.size != (width, height):
                raise ValueError("saved PNG format or dimensions differ from video")
    except OSError as error:
        raise ValueError("cannot verify saved PNG coverage") from error


def decode(session: Any, latent: torch.Tensor, decoder: Any, seed: int) -> torch.Tensor:  # noqa: ANN401 -- session and decoder are native external handles
    """Use the native decoder with a fresh identical generator for each compared arm."""
    from scripts.prune.evaluate.decode import decode_latent  # noqa: PLC0415 -- RGB rendering has no model imports

    generator = torch.Generator(device=session.device).manual_seed(seed)
    device_context = torch.cuda.device(session.device) if session.device.type == "cuda" else nullcontext()
    with device_context:
        pixels = as_fchw(decode_latent(session, latent.to(session.device), decoder, generator=generator))
    expected = (latent.shape[2] - 1) * 8 + 1
    if len(pixels) != expected:
        raise ValueError(f"decoder returned {len(pixels)} frames; expected {expected}")
    return pixels


def recorded_capture_rgb(source: dict, corpus_root: Path, objective: str, encoded_frames: int) -> torch.Tensor:
    """Replay checked capture pixels using the original producer's recorded crop."""
    from scripts.onestep_avatar import precompute  # noqa: PLC0415 -- CPU producer replay

    if not 1 <= encoded_frames <= source["n_latent_frames"]:
        raise ValueError("recorded RGB range must fit the saved continuous encoding")
    record = source["capture_encode_record"]
    pixel_frames = (encoded_frames - 1) * 8 + 1
    if record.get("source") != source["relative_dir"]:
        raise ValueError("recorded RGB source differs from the encoding")
    if record.get("fps") != source.get("fps") or not math.isfinite(float(source.get("fps", 0))) or source["fps"] <= 0:
        raise ValueError("recorded RGB frame rate differs from the encoding")
    if record.get("pixel_frames", 0) < pixel_frames:
        raise ValueError("recorded RGB range exceeds the saved encoding coverage")
    if record.get("objective") != objective or record.get("box_xyxy") != source["box_xyxy"]:
        raise ValueError("recorded RGB objective/crop differs from the encoding")
    rgb = corpus_root / source["relative_dir"] / "rgb.mp4"
    if sha256(rgb) != source["rgb_sha256"]:
        raise ValueError("recorded RGB source content changed")
    stat = rgb.stat()
    bbox = rgb.with_name("bbox.npy")
    bbox_stat = bbox.stat()
    capture = precompute.CaptureSource(
        source["relative_dir"],
        str(rgb),
        str(bbox),
        f"size={stat.st_size};mtime_ns={stat.st_mtime_ns}",
        f"size={bbox_stat.st_size};mtime_ns={bbox_stat.st_mtime_ns}",
    )
    if precompute.capture_input_fingerprint(capture, objective) != record.get("input_fingerprint"):
        raise ValueError("recorded RGB/matte producer fingerprint changed")
    pixels = precompute.crop_source(capture, pixel_frames, tuple(source["box_xyxy"]), record["edge"], (objective,))[
        objective
    ]
    return torch.from_numpy(pixels).permute(0, 3, 1, 2).contiguous()


def recorded_guide_rgb(source: dict, corpus_root: Path, objective: str, encoded_frames: int) -> torch.Tensor:
    """Read the exact guide video encoded by its checked producer record."""
    import cv2  # noqa: PLC0415 -- video preparation only

    from scripts.onestep_avatar import dataset  # noqa: PLC0415 -- shared artifact names

    record = source["guide_encode_record"]
    if record.get("source") != source["relative_dir"] or record.get("fps") != source["fps"]:
        raise ValueError("guide RGB source/timebase differs from the encoding")
    count = (encoded_frames - 1) * 8 + 1
    if not 1 <= encoded_frames <= source["n_latent_frames"] or record.get("pixel_frames", 0) < count:
        raise ValueError("guide RGB range exceeds the saved encoding coverage")
    if record.get("objective") != objective or record.get("box_xyxy") != source["box_xyxy"]:
        raise ValueError("guide RGB objective/crop differs from the encoding")
    view = corpus_root / source["relative_dir"]
    video, sidecar = view / dataset.render_name(objective), view / dataset.render_metadata_name(objective)
    if sha256(video) != source["guide_sha256"] or record.get("input_fingerprint") != source["guide_sha256"]:
        raise ValueError("guide RGB content differs from the encoded render")
    if sha256(sidecar) != source["guide_sidecar_sha256"]:
        raise ValueError("guide RGB sidecar changed")
    metadata = json.loads(sidecar.read_text())
    if (
        metadata.get("objective") != objective
        or metadata.get("compositing_version") != dataset.GUIDE_COMPOSITING_VERSION
        or metadata.get("out_size") != record["edge"]
        or metadata.get("n_frames", 0) < count
    ):
        raise ValueError("guide RGB sidecar producer conditions differ from the encoding")
    reader = cv2.VideoCapture(str(video))
    frames = []
    try:
        if not reader.isOpened() or not math.isclose(reader.get(cv2.CAP_PROP_FPS), source["fps"], abs_tol=1e-6):
            raise ValueError("guide RGB frame rate differs from the saved timebase")
        for index in range(count):
            valid, pixels = reader.read()
            if not valid:
                raise ValueError(f"guide RGB ended before requested frame {index}")
            if pixels.shape[:2] != (record["edge"], record["edge"]):
                raise ValueError("guide RGB dimensions differ from the recorded encoding")
            frames.append(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
    finally:
        reader.release()
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


def decode_key(
    content_sha256: str,
    vae_sha256: str,
    shape: list[int],
    method: str,
    seed: int,
    settings: dict,
) -> str:
    """Identity includes the actual decoder settings, not an adapter filename."""
    record = {
        "content_sha256": content_sha256,
        "vae_sha256": vae_sha256,
        "shape": shape,
        "method": method,
        "seed": seed,
        "settings": settings,
    }
    for digest in (content_sha256, vae_sha256):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("decode identities require full SHA-256 hashes")
    if not method or not shape or any(not isinstance(x, int) or x <= 0 for x in shape):
        raise ValueError("decode identity requires method and positive dimensions")
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def frame(pixels: torch.Tensor, index: int) -> Image.Image:
    """Quantize a presentation frame; keep scientific measurements in float RGB."""
    value = pixels[index].detach().float().cpu()
    if value.max() > 1.5:
        value = value / 255
    return Image.fromarray(value.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()).convert("RGB")


def _pixel_identity(pixels: torch.Tensor | None) -> str | None:
    if pixels is None:
        return None
    value = pixels.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Panel:
    """RGB pixels and an explicit mapping to original recorded frame numbers."""

    role: str
    title: str
    pixels: torch.Tensor | None
    source_frames: tuple[int, ...] = ()
    value: str = ""
    missing_reason: str = ""
    still: bool = False


def prepare_training_references(
    session: Any,  # noqa: ANN401 -- native session handle
    decoder: Any,  # noqa: ANN401 -- native decoder handle
    source: dict,
    corpus_root: Path,
    objective: str,
    encoded_frames: int,
    seed: int,
    *,
    require_guide: bool = True,
) -> tuple[list[Panel], dict]:
    """Prepare the fixed RGB reference roles without transformer inference."""
    from scripts.onestep_avatar import dataset, precompute  # noqa: PLC0415 -- checked producers

    path = corpus_root / source["relative_dir"] / dataset.capture_bundle_name(objective)
    digest = sha256(path)
    if digest != source["capture_latent_sha256"]:
        raise ValueError("preview capture encoding content changed")
    vae = Path(session.model.paths.video_vae())
    if precompute.file_fingerprint(vae) != source["capture_encode_record"].get("vae_fingerprint"):
        raise ValueError("preview decoder VAE differs from the capture encoding")
    recorded = recorded_capture_rgb(source, corpus_root, objective, encoded_frames)
    guide_hash = source.get("guide_sha256")
    if require_guide and not guide_hash:
        raise ValueError("D1 preview requires a checked guide")
    guide = recorded_guide_rgb(source, corpus_root, objective, encoded_frames) if guide_hash else None
    master, fps = dataset.load_training_master(path)
    if list(master.shape) != source["shape"] or fps != source["fps"]:
        raise ValueError("preview capture encoding shape/timebase changed")
    latent = master[:, :encoded_frames].unsqueeze(0)
    decoded = decode(session, latent, decoder, seed)
    frames = tuple(range((encoded_frames - 1) * 8 + 1))
    panels = [
        Panel("recorded", "Capture RGB", recorded, frames),
        Panel("decoded", "VAE-decoded capture", decoded, frames),
        Panel("guide", "Guide RGB", guide, frames, missing_reason="Guide not used" if guide is None else ""),
    ]
    vae_hash = sha256(vae)
    settings = native_decoder_settings()
    return panels, {
        "source": source["relative_dir"],
        "objective": objective,
        "fps": fps,
        "capture_encoding_sha256": digest,
        "guide_rgb_sha256": guide_hash,
        "source_frames": list(frames),
        "vae_sha256": vae_hash,
        "decode_seed": seed,
        "decode_key": decode_key(digest, vae_hash, list(latent.shape), "native_decode_video", seed, settings),
        "decoder_settings": settings,
        "panels": [{"role": p.role, "pixels_sha256": _pixel_identity(p.pixels)} for p in panels],
    }


def save_training_references(panels: list[Panel], record: dict, destination: Path) -> dict:
    """Publish prepared pixels and their provenance; never overwrite a bundle."""
    from scripts.onestep_avatar.dataset import atomic_write  # noqa: PLC0415 -- common atomic publication

    if "software" in record:
        software.check_current(record["software"])

    if [panel.role for panel in panels] != ["recorded", "decoded", "guide"]:
        raise ValueError("training references require recorded/decoded/guide roles in order")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("reference bundle output is already used")
    expected = {row["role"]: row["pixels_sha256"] for row in record["panels"]}
    for panel in panels:
        if panel.pixels is None:
            if (panel.role != "guide" or panel.missing_reason != "Guide not used"
                    or record.get("guide_rgb_sha256") is not None or expected.get(panel.role) is not None):
                raise ValueError("only an unused D0 guide may lack reference pixels")
        elif _pixel_identity(panel.pixels) != expected.get(panel.role):
            raise ValueError("reference pixels differ from their producer record")
        if (list(panel.source_frames) != record["source_frames"]
                or (panel.pixels is not None and len(panel.pixels) != len(panel.source_frames))):
            raise ValueError("reference frame mapping differs from its producer record")
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for panel in panels:
        path = destination / f"{panel.role}.pt"
        if panel.pixels is not None:
            pixels = panel.pixels.cpu()
            atomic_write(path, lambda temporary, pixels=pixels: torch.save(pixels, temporary))
        rows.append(
            {
                "role": panel.role,
                "title": panel.title,
                "source_frames": list(panel.source_frames),
                "pixels_sha256": expected[panel.role],
                "path": str(path.resolve()) if panel.pixels is not None else None,
                "sha256": sha256(path) if panel.pixels is not None else None,
                "missing_reason": panel.missing_reason,
            }
        )
    manifest = {"schema_version": 2, "kind": "onestep_avatar.training_references", "producer": record, "panels": rows}
    if "software" in record:
        software.check_current(record["software"])
    atomic_write(
        destination / "references.json", lambda temporary: temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    )
    return manifest


def load_training_references(path: Path) -> tuple[list[Panel], dict]:
    """Read pinned prepared RGB; missing pixels are errors, never model work."""
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 2 or manifest.get("kind") != "onestep_avatar.training_references":
        raise ValueError("training references require a version-two bundle")
    if [row["role"] for row in manifest["panels"]] != ["recorded", "decoded", "guide"]:
        raise ValueError("reference bundle has incompatible panel roles")
    expected = {row["role"]: row["pixels_sha256"] for row in manifest["producer"]["panels"]}
    panels = []
    for row in manifest["panels"]:
        if row["source_frames"] != manifest["producer"]["source_frames"]:
            raise ValueError("reference bundle frame mapping changed")
        if row["path"] is None:
            if (row["role"] != "guide" or row.get("missing_reason") != "Guide not used"
                    or manifest["producer"].get("guide_rgb_sha256") is not None
                    or row["sha256"] is not None or row["pixels_sha256"] is not None
                    or expected.get(row["role"]) is not None):
                raise ValueError("only an unused D0 guide may lack reference pixels")
            panels.append(Panel("guide", row["title"], None, tuple(row["source_frames"]),
                                missing_reason="Guide not used"))
            continue
        source = Path(row["path"])
        if sha256(source) != row["sha256"]:
            raise ValueError("reference bundle pixel file changed")
        pixels = torch.load(source, map_location="cpu", weights_only=True)
        if len(pixels) != len(row["source_frames"]):
            raise ValueError("reference bundle pixel coverage differs from its frame mapping")
        if _pixel_identity(pixels) != row["pixels_sha256"] or row["pixels_sha256"] != expected.get(row["role"]):
            raise ValueError("reference bundle pixel identity changed")
        panels.append(Panel(row["role"], row["title"], pixels, tuple(row["source_frames"])))
    return panels, manifest["producer"]


def main(argv: list[str] | None = None) -> int:
    """Prepare checked fixed preview references with a decoder-only session."""
    from scripts.onestep_avatar import dataset, precompute  # noqa: PLC0415 -- checked input owners
    from scripts.onestep_avatar.model import backbone  # noqa: PLC0415 -- model registry only

    parser = argparse.ArgumentParser(description="Prepare fixed training reference pixels without generation")
    parser.add_argument("--prepare-training-references", action="store_true", required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--encoded-frames", type=int, required=True)
    parser.add_argument("--guide-mode", choices=("d0", "d1"), required=True)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--model", default="2.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    producer_software = software.capture("decoding")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("reference bundle output is already used")
    membership_hash = sha256(args.subset)
    store = dataset.ClipStore(json.loads(args.subset.read_text()), args.corpus_root)
    if args.source not in store.sources:
        raise ValueError("reference source is not in the checked video list")
    source = store.sources[args.source]
    if not 1 <= args.encoded_frames <= source["n_latent_frames"]:
        raise ValueError("reference frame count must fit the checked master")
    require_guide = args.guide_mode == "d1"
    if require_guide and not source.get("guide_sha256"):
        raise ValueError("D1 preview requires a checked guide")
    store.load(args.source, require_guide=require_guide or bool(source.get("guide_sha256")))
    specification = backbone.resolve(args.model, "dev")
    vae = Path(specification.paths.video_vae())
    if precompute.file_fingerprint(vae) != source["capture_encode_record"].get("vae_fingerprint"):
        raise ValueError("preview decoder VAE differs from the capture encoding")
    # Bind inputs through publication; never accept references from changed bytes.
    view = store.root / args.source
    paths = [args.subset, vae, view / dataset.capture_bundle_name(store.objective), view / "rgb.mp4",
             view / "bbox.npy"]
    if store.objective == "white":
        paths.append(view / dataset.CAPTURE_MASK_NAME)
    if source.get("guide_sha256"):
        paths += [view / dataset.guide_bundle_name(store.objective), view / dataset.render_name(store.objective),
                  view / dataset.render_metadata_name(store.objective)]
    identities = {path: sha256(path) for path in paths}
    if identities[args.subset] != membership_hash:
        raise ValueError("reference membership changed during preparation")
    software.check_current(producer_software)
    session = open_decoder_session(args.model, args.gpu_id, script="onestep_avatar.training_references")
    with session.decoder() as decoder, torch.inference_mode():
        panels, producer = prepare_training_references(
            session, decoder, source, store.root, store.objective, args.encoded_frames,
            args.seed, require_guide=require_guide,
        )
    if any(sha256(path) != digest for path, digest in identities.items()):
        raise ValueError("reference inputs changed during preparation")
    producer["membership_file_sha256"] = membership_hash
    producer["membership_sha256"] = store.membership["sha256"]
    producer["software"] = producer_software
    save_training_references(panels, producer, args.output)
    return 0


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("DejaVuSans.ttf", size)


def _validate_panels(panels: list[Panel], roles: list[str]) -> tuple[int, ...]:  # noqa: PLR0912 -- ordered input gates
    if len({panel.role for panel in panels}) != len(panels):
        raise ValueError("duplicate panel role")
    if {panel.role for panel in panels} != set(roles) - {"unused"}:
        raise ValueError("panels must match all assigned layout roles")
    mappings = []
    for panel in panels:
        if not panel.title or "\n" in panel.title or "\n" in panel.value:
            raise ValueError("a panel has one role line and at most one value line")
        if panel.pixels is None:
            if not panel.missing_reason:
                raise ValueError("missing panel requires an explicit reason")
            continue
        pixels = panel.pixels
        if pixels.ndim != 4 or pixels.shape[1] != 3 or not torch.isfinite(pixels).all():
            raise ValueError("panels require finite F,3,H,W RGB pixels")
        if panel.still:
            if len(pixels) != 1 or panel.role != "first_image":
                raise ValueError("only the supplied first image can be a still panel")
            continue
        mapping = panel.source_frames
        if len(mapping) != len(pixels) or not mapping:
            raise ValueError("every moving panel requires an explicit source frame mapping")
        if any(not isinstance(x, int) or x < 0 for x in mapping):
            raise ValueError("source frame numbers must be nonnegative integers")
        if any(b <= a for a, b in pairwise(mapping)):
            raise ValueError("source frame mappings must be strictly increasing")
        mappings.append(mapping)
    if not mappings:
        raise ValueError("comparison requires a moving video")
    shared = tuple(x for x in mappings[0] if all(x in other for other in mappings[1:]))
    if not shared:
        raise ValueError("panels have no shared source frames")
    if any(b - a != 1 for a, b in pairwise(shared)):
        raise ValueError("shared source frames must be consecutive at the recorded playback rate")
    return shared


def layout_geometry(  # noqa: PLR0912 -- shared metadata-only placement and measurement
    panels: list[Panel], *, question: str, layout: str,
    panel_size: tuple[int, int] = (320, 320), viewing_width: int = 480,
    font_size: int | None = None, shared_text: str = "",
) -> dict:
    """Plan positions and readable text before any decoding or RGB assembly."""
    if layout not in LAYOUTS and layout not in COMPARISON_COLUMNS:
        raise ValueError(f"unknown layout {layout}")
    if not question or len(question.split()) > 15 or "\n" in question or "\n" in shared_text:
        raise ValueError("question must have at most 15 words on one line")
    width, height = panel_size
    if min(width, height, viewing_width) <= 0:
        raise ValueError("display sizes must be positive")
    if layout in COMPARISON_COLUMNS:
        if not panels or any(panel.role == "unused" for panel in panels):
            raise ValueError("comparison requires panels with nonreserved roles")
        columns = min(COMPARISON_COLUMNS[layout], len(panels))
        assigned = [panel.role for panel in panels]
        assigned += ["unused"] * ((-len(assigned)) % columns)
        rows = tuple(tuple(assigned[index:index + columns]) for index in range(0, len(assigned), columns))
    else:
        rows = LAYOUTS[layout]
    roles = [role for row in rows for role in row]
    if len({panel.role for panel in panels}) != len(panels):
        raise ValueError("duplicate panel role")
    if {panel.role for panel in panels} != set(roles) - {"unused"}:
        raise ValueError("panels must match all assigned layout roles")
    if any(not panel.title or "\n" in panel.title or "\n" in panel.value for panel in panels):
        raise ValueError("a panel has one role line and at most one value line")
    gap = 8
    canvas_width = len(rows[0]) * width + (len(rows[0]) + 1) * gap
    canvas_width += canvas_width % 2
    font_size = font_size or math.ceil(16 * canvas_width / viewing_width)
    if font_size * viewing_width / canvas_width < 16:
        raise ValueError("titles would be smaller than 16 pixels at the viewing width")
    font = _font(font_size)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    title_height = 2 * (font_size + 8) + 8
    top_height = (2 if shared_text else 1) * (font_size + 8) + 8
    for text in (question, shared_text):
        if measure.textbbox((0, 0), text, font=font)[2] > canvas_width - 2 * gap:
            raise ValueError("video title does not fit; use shorter text or a wider viewing size")
    for panel in panels:
        for text in (panel.title, panel.value, panel.missing_reason):
            if measure.textbbox((0, 0), text, font=font)[2] > width - 2 * gap:
                raise ValueError(f"panel title does not fit: {text!r}; use compact layout")
    canvas_height = top_height + len(rows) * (title_height + height + gap) + gap
    canvas_height += canvas_height % 2
    return {"rows": rows, "font_size": font_size, "gap": gap,
            "title_height": title_height, "top_height": top_height,
            "display_size": [canvas_width, canvas_height]}


def compact_layout(panels: list[Panel], *, question: str, layout: str,
                   panel_size: tuple[int, int] = (320, 320)) -> str:
    """Select the first exact-label layout that fits a 480-pixel reading width."""
    if layout in COMPARISON_COLUMNS:
        candidates = ("comparison", "compact_comparison", "stacked_comparison")
    else:
        candidates = ({"training": "compact_training", "inference": "compact_inference"}.get(layout, layout),)
    for candidate in candidates:
        try:
            layout_geometry(panels, question=question, layout=candidate, panel_size=panel_size, viewing_width=480)
        except ValueError:
            continue
        return candidate
    raise ValueError("no readable compact layout fits the exact comparison labels")


def render_panels(
    panels: list[Panel],
    *,
    question: str,
    layout: str,
    fps: float,
    panel_size: tuple[int, int] = (320, 320),
    viewing_width: int = 480,
    font_size: int | None = None,
    shared_text: str = "",
    poster_frame: int = 0,
    common_settings: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """Render checked RGB only. This function has no model or VAE dependency."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    geometry = layout_geometry(panels, question=question, layout=layout, panel_size=panel_size,
                               viewing_width=viewing_width, font_size=font_size, shared_text=shared_text)
    rows = geometry["rows"]
    shared = _validate_panels(panels, [role for row in rows for role in row])
    if not 0 <= poster_frame < len(shared):
        raise ValueError("poster frame is outside shared coverage")
    width, height = panel_size
    canvas_width, canvas_height = geometry["display_size"]
    font_size = geometry["font_size"]
    gap, title_height, top_height = (geometry[key] for key in ("gap", "title_height", "top_height"))
    font = _font(font_size)
    by_role = {panel.role: panel for panel in panels}
    mappings = {p.role: {value: i for i, value in enumerate(p.source_frames)} for p in panels}
    output = []
    for source_frame in shared:
        image = Image.new("RGB", (canvas_width, canvas_height), (32, 32, 32))
        draw = ImageDraw.Draw(image)
        draw.text((gap, 4), question, font=font, fill="white")
        if shared_text:
            draw.text((gap, font_size + 12), shared_text, font=font, fill="white")
        for row_index, row in enumerate(rows):
            for column, role in enumerate(row):
                x = gap + column * (width + gap)
                y = top_height + gap + row_index * (height + title_height + gap)
                panel = by_role.get(role)
                title = "Unused" if panel is None else panel.title
                value = "" if panel is None else panel.value
                draw.text((x + gap, y), title, font=font, fill="white")
                draw.text((x + gap, y + font_size + 8), value, font=font, fill="white")
                if panel is None or panel.pixels is None:
                    reason = "Unused" if panel is None else panel.missing_reason
                    draw.text((x + gap, y + title_height + gap), reason, font=font, fill="white")
                else:
                    index = 0 if panel.still else mappings[role][source_frame]
                    contained = ImageOps.contain(
                        frame(panel.pixels, index), (width, height), method=Image.Resampling.LANCZOS
                    )
                    padded = Image.new("RGB", (width, height), (48, 48, 48))
                    padded.paste(contained, ((width - contained.width) // 2, (height - contained.height) // 2))
                    image.paste(padded, (x, y + title_height))
        output.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
    record = {
        "schema_version": 2,
        "layout_version": 1,
        "layout": layout,
        "question": question,
        "shared_text": shared_text,
        "fps": fps,
        "source_frames": list(shared),
        "source_times": [value / fps for value in shared],
        "poster_frame": poster_frame,
        "panel_size": list(panel_size),
        "display_size": [canvas_width, canvas_height],
        "viewing_width": viewing_width,
        "font_size": font_size,
        "padding": "aspect-preserving contain with neutral padding",
        "common_settings": common_settings or {},
        "panels": [
            {
                "role": role,
                "row": r,
                "column": c,
                "title": by_role[role].title if role in by_role else "Unused",
                "value": by_role[role].value if role in by_role else "",
                "source_frames": list(by_role[role].source_frames) if role in by_role else [],
                "still": by_role[role].still if role in by_role else False,
                "pixels_sha256": _pixel_identity(by_role[role].pixels) if role in by_role else None,
                "missing_reason": by_role[role].missing_reason if role in by_role else "Unused",
            }
            for r, row in enumerate(rows)
            for c, role in enumerate(row)
        ],
    }
    return torch.stack(output), record


def render_from_record(panels: list[Panel], record: dict) -> tuple[torch.Tensor, dict]:
    """Rebuild only from the recorded layout and exactly matching RGB evidence."""
    pixels, rebuilt = render_panels(
        panels,
        question=record["question"],
        layout=record["layout"],
        fps=record["fps"],
        panel_size=tuple(record["panel_size"]),
        viewing_width=record["viewing_width"],
        font_size=record["font_size"],
        shared_text=record["shared_text"],
        poster_frame=record["poster_frame"],
        common_settings=record["common_settings"],
    )
    original = {key: value for key, value in record.items() if key != "outputs"}
    if rebuilt != original:
        raise ValueError("rendering inputs, titles, frame mapping or layout differ from the saved record")
    return pixels, rebuilt


def save_render(pixels: torch.Tensor, record: dict, output: Path) -> dict:
    """Write MP4 and poster first; publish their complete rendering record last."""
    from ltx_trainer.video_utils import save_video  # noqa: PLC0415 -- defer video dependencies until writing

    if "software" in record:
        software.check_current(record["software"])

    if float(record["fps"]) != int(record["fps"]):
        raise ValueError("native MP4 writer requires an integer playback rate")
    output.mkdir(parents=True, exist_ok=True)
    video, poster = output / "comparison.mp4", output / "poster.png"
    save_video(pixels, video, fps=record["fps"], video_format="FCHW")
    frame(pixels, record["poster_frame"]).save(poster)
    complete = {
        **record,
        "outputs": {
            "video": {"path": str(video), "sha256": sha256(video)},
            "poster": {"path": str(poster), "sha256": sha256(poster)},
        },
    }
    destination = output / "rendering.json"
    if "software" in record:
        software.check_current(record["software"])
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(complete, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return complete


if __name__ == "__main__":
    raise SystemExit(main())
