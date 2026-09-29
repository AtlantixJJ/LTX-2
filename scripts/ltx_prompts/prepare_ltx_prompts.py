"""
Video-to-LTX prompts, copied from LTX Trainer's captioning.py and extended here.

The original Qwen3-Omni and Gemini captioners are kept as selectable backends.
The added LiteLLM backend sees sampled video frames, with no audio access.
The CLI at the bottom rewrites each backend's source caption into an LTX prompt.

Original module description:
Audio-visual media captioning using multimodal models.
This module provides captioning capabilities for videos with audio using:
- Qwen3-Omni via a local vLLM server (default)
- Gemini Flash 3.5 (cloud API)
Both produce a single combined English caption per video as a single
continuous paragraph of prose.
The Qwen3-Omni backend runs in a separately-launched vLLM server rather than
in-process, so vLLM's heavy CUDA dependencies stay out of this package. The
captioner talks to it over the OpenAI-compatible HTTP API.
Launch the server once (in an isolated environment) with:
.. code-block:: bash
    uv run python scripts/serve_captioner.py
That helper picks BF16 vs FP8 dynamic quantization based on the GPU's free
memory and forwards everything else to ``vllm serve``. To check the recommended
command without running it, pass ``--print-cmd``.
To use Gemini instead, install ``google-genai`` and either set ``GEMINI_API_KEY``
(Gemini Developer API) or have Google Cloud credentials available (gcloud / an
attached service account), in which case it uses Vertex AI automatically.
"""

import json
import os
import re
import subprocess
import tempfile
import argparse
import base64
import hashlib
import io
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import ClassVar
from scripts.ltx_prompts.backends import (CaptionerType, MediaCaptioningModel, QwenOmniCaptioner,
                          GeminiFlashCaptioner, DEFAULT_VLLM_BASE_URL)
from scripts.ltx_prompts.traces import (_load_jsonl, _trace_slug, _write_trace, _load_ground_truth,
                        render_trace_markdown, build_report)

def create_captioner(captioner_type: CaptionerType, **kwargs) -> "MediaCaptioningModel":
    """Factory function to create a media captioner."""
    match captioner_type:
        case CaptionerType.QWEN_OMNI:
            return QwenOmniCaptioner(**kwargs)
        case CaptionerType.GEMINI_FLASH:
            return GeminiFlashCaptioner(**kwargs)
        case CaptionerType.LITELLM_VISION:
            return LiteLLMVisionCaptioner(**kwargs)
        case _:
            raise ValueError(f"Unsupported captioner type: {captioner_type}")


# The code above is copied from LTX Trainer's captioning.py. The LiteLLM
# backend, call tracing, HTML report, and command below are workspace
# additions; upstream stays untouched.
LTX_GUIDE_URL = "https://ltx.io/blog/prompting-guide-for-ltx-2"
# Frames-only observation prompt. It asks for per-frame facts before any
# motion summary: free-form "describe the changes" answers invented walks and
# turns between frames, while per-frame lines keep the model anchored to what
# each image shows (orientation, hand contents, limb pose).
VISION_INSTRUCTION = """\
The images contain {count} frames sampled in time order from one short video clip; each image is preceded \
by the frame numbers and timestamps it shows, and each frame is labelled with its timestamp. They show the same \
scene from one fixed camera.

Report only what is visible. Use these plain-text sections:

Frames: one line per frame, "t=<seconds>: ..." stating (a) which way the person's body and face point relative \
to the camera (toward the camera, away from the camera with the back visible, or side profile facing frame left \
or frame right), (b) anything held in either hand, naming the object, (c) the position of each arm and where each hand \
is (e.g. at sides, in a pocket, clasped in front, bent in front of chest, at the forehead, above the head, \
extended forward or sideways), (d) the legs (both feet planted, one knee lifted, or mid-stride with the feet \
clearly apart), and (e) the horizontal position of the person's feet in the frame as a number from 0.0 (left \
edge) to 1.0 (right edge).

Motion: the actions implied by the changes between consecutive frames, in order. Use the numbers in (e) to say whether \
and when the person travels across the floor (a change of more than about 0.1) and in which direction, and \
when they stay in one spot. Only claim a turn if the facing direction in (a) actually \
changes. Repeated back-and-forth changes (knees alternately lifted, arms alternately forward) are one ongoing \
action such as marching or swinging; name it as such.

Appearance: sex and approximate age, hair colour and style, hair accessories, each clothing item with colour, \
sleeve and leg length, visible logos or print, shoes with their exact colour, jewellery.

Setting: the room, floor, walls, light sources and other fixtures.

Camera: shot size, vantage (height and side), and whether the view changes between frames.

Rules:
- Left and right mean the person's own left and right. With the back to the camera, their left hand is on the \
image's left; facing the camera, their left hand is on the image's right; in profile, the arm nearer the camera \
is on the side that faces the camera.
- If a held object stops being visible in the hand, look for it on the head, face or body in the later frames \
(worn, tucked away) before deciding what happened. Never say it was put down, dropped or thrown unless you \
can see it somewhere else.
- Name colours as they appear in the crops (pink, mauve, charcoal), not as a default guess. For shoes, look at \
the uppers in the cropped frames and give their colour and any stripes or pattern.
- Do not describe audio, emotions, or intentions."""
REWRITE_INSTRUCTION = """Turn the following frame observations of one video clip into one LTX-2 generation prompt.
Follow the official LTX-2 prompting guide: write one flowing paragraph of 5 to 7 present-tense sentences.
Order: the main action first, then the subject's appearance, the chronological motion in detail, the
setting, and finally the camera and lighting woven into natural sentences. Call the person by a plain
noun phrase ("a young woman", "a man"), not "a female". Describe motion as continuous actions; keep the
facing direction, left/right, and any held object exactly as observed, and say "in place" when the
person does not travel. State the starting pose and anything held at the start, and the final pose,
including where the hands end up. Preserve only details supported by the observations; drop hedges and per-frame
timestamps. The source has no audio, so do not mention sound or silence.
Return only a JSON object with one key, "prompt", whose value is the finished paragraph.

Frame observations:
"""
SEGMENT_REWRITE_INSTRUCTION = """Turn the following observations of one video clip into one LTX-2 generation prompt.
The observations come from separate passes over consecutive segments of the clip, in time order (and,
when present, a coarse overview pass first). Merge them into one continuous account: follow the
segments' motion in order, carry held objects and their fate across segment boundaries, and prefer
the detailed segment passes over the overview when they disagree. For appearance, keep what most
passes agree on.
Follow the official LTX-2 prompting guide: write one flowing paragraph of 5 to 7 present-tense sentences.
Order: the main action first, then the subject's appearance, the chronological motion in detail, the
setting, and finally the camera and lighting woven into natural sentences. Call the person by a plain
noun phrase ("a young woman", "a man"), not "a female". Describe motion as continuous actions; keep the
facing direction, left/right, and any held object exactly as observed, and say "in place" when the
person does not travel. State the starting pose and anything held at the start, and the final pose,
including where the hands end up. Drop hedges, timestamps and segment numbers. The source has no
audio, so do not mention sound or silence.
Return only a JSON object with one key, "prompt", whose value is the finished paragraph.

Observations:
"""
# Bump when frame sampling, cropping, image packing or segment logic changes:
# cached vision observations are reused only under the same version (plus the
# prompt hashes and settings in the vision context).
VISION_STAGE_VERSION = "2026-09-29.1"
MV_SHEET_INSTRUCTION = """You are given observations of ONE short performance recorded simultaneously by several fixed
cameras placed around the person. Each camera's observations were written by a smaller vision model
that saw only that camera's frames, so they contain misreadings, omissions and view-dependent guesses.
Reconcile them into one subject sheet that is true for the whole performance.

The HEADING TABLE below was computed by code from every camera's per-frame facing label and the
camera azimuths (0 = rig front, 90 = rig right, 180 = back, 270 = left). Use it for which way the
person faces and when they turn; do not redo per-frame geometry yourself. Only override it if the
observations clearly describe something it cannot capture. A heading change of about +90 degrees
(e.g. 0 -> 90) is a turn to the person's own RIGHT, and -90 (e.g. 0 -> 270) a turn to their LEFT.
TRAVEL was computed the same way from every camera's horizontal positions; use it for whether, when and
which way the person moves across the floor (travel toward the heading they face is walking forward).

Evidence rules:
- A detail seen clearly by any camera wins over cameras that could not see it (faces, eyewear, age and
  expression from cameras the person faces; hand-held objects from cameras with an unoccluded view).
  A camera that cannot see something is not evidence that it is absent.
- The same arm motion looks different by camera: forward/back swings look like arms "out wide" from the
  side and like arms "in front of the chest" from the front. Describe motion in the person's own body terms.
- Left/right means the person's own left/right. "Image-left/right" hands must be converted with the
  facing in that camera (facing the camera: image-left = their right; back to the camera: their left).
- Clothing and colours: prefer what most cameras with a clear view report.
- Held objects: track what happens to each across the clip; never invent an outcome.
Keep your reasoning brief; the answer is a short JSON object.

Return only a JSON object:
{"person": "sex and approximate age",
 "appearance": "hair, every clothing item with colour, shoes, accessories",
 "objects": "each held object, which hand, and what happens to it (or 'none')",
 "heading": "rig heading at the start and each turn with approximate time and the person's own turn direction",
 "travel": "whether and in which rig direction the person moves across the floor, and when",
 "actions": ["about a-b s: body-relative action", "..."],
 "resolved_conflicts": ["what the cameras disagreed about and how you decided"]}
"""
MV_VIEW_INSTRUCTION = """Write one LTX-2 generation prompt for ONE camera's video of a performance.

You get (1) a subject sheet reconciled from all cameras, authoritative for who the person is, what they
wear and hold, and what they do in body terms; (2) this camera's placement and the person's facing
RELATIVE TO THIS CAMERA over time, already computed by code from the consensus heading (use it as
given); and (3) this camera's own observations, authoritative for what is visible from here.
Forward travel moves the same way the person faces relative to this camera (facing the camera: toward
the camera; back to the camera: away from it; profile facing frame left/right: toward frame left/right).
Elevation above about 15 degrees is a slightly high angle; below that, eye level.

Include every action and object from the sheet, even when partly hidden from this camera (for example,
from behind, "raises the glasses and puts them on"), but do not describe details this camera cannot see
(facial expression from behind). Use the person's own left/right.

Follow the official LTX-2 prompting guide: one flowing paragraph of 5 to 7 present-tense sentences.
Order: the main action first, then the appearance, the chronological motion as this camera sees it
(facing relative to the camera, turns, travel in the frame), the setting, and finally the camera shot,
vantage and lighting. Call the person by a plain noun phrase ("a young woman"). No timestamps, no sound.
Keep your reasoning brief. Return only a JSON object with one key, "prompt".
"""
_FACING_PATTERNS = [
    (re.compile(r"back to (the )?camera|away from (the )?camera|back (is )?visible|rear view(?! .*three)"), 180),
    (re.compile(r"three-quarter (rear|back)\w*[^|;]*?(image|frame)[- ]?left"), 135),
    (re.compile(r"three-quarter (rear|back)\w*[^|;]*?(image|frame)[- ]?right"), 225),
    (re.compile(r"three-quarter[^|;]*?(image|frame)[- ]?left"), 45),
    (re.compile(r"three-quarter[^|;]*?(image|frame)[- ]?right"), 315),
    (re.compile(r"profile[^|;]*?(image|frame)[- ]?left"), 90),
    (re.compile(r"profile[^|;]*?(image|frame)[- ]?right"), 270),
    (re.compile(r"toward(s)? (the )?camera|facing (the )?camera|frontal"), 0),
]
_RELATIVE_NAMES = [(0, "facing the camera"), (45, "in three-quarter view, turned toward frame left"),
                   (90, "in side profile facing frame left"), (135, "in three-quarter rear view, turned toward frame left"),
                   (180, "with the back to the camera"), (225, "in three-quarter rear view, turned toward frame right"),
                   (270, "in side profile facing frame right"), (315, "in three-quarter view, turned toward frame right")]


def _angle_gap(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def _frame_facings(observations: str) -> list[tuple[float, int]]:
    """(time, relative facing angle) for every per-frame line whose facing phrase is recognised."""
    found = []
    for line in observations.splitlines():
        match = re.match(r"\s*t=([\d.]+)s\s*[:|]\s*(.*)", line)
        if not match:
            continue
        facing = re.split(r"\||\(b\)", match.group(2).lower())[0]
        for pattern, angle in _FACING_PATTERNS:
            if pattern.search(facing):
                found.append((round(float(match.group(1)), 2), angle))
                break
    return found


def _heading_consensus(views: dict[str, tuple[float, str]]) -> list[dict]:
    """Per time step, each camera's implied rig heading (azimuth + relative facing) and their circular medoid.

    ``views`` maps a camera name to (azimuth, observations). Code does this arithmetic
    because a reasoning model given the raw labels spent its whole budget on it.
    """
    by_time: dict[float, dict[str, float]] = {}
    for name, (azimuth, observations) in views.items():
        for time_s, relative in _frame_facings(observations):
            by_time.setdefault(time_s, {})[name] = (azimuth + relative) % 360
    rows = []
    for time_s in sorted(by_time):
        votes = by_time[time_s]
        if len(votes) < 2:
            continue
        medoid = min(votes.values(), key=lambda a: sum(_angle_gap(a, b) for b in votes.values()))
        agree = [n for n, a in votes.items() if _angle_gap(a, medoid) <= 45]
        rows.append({"t": time_s, "heading": round(medoid), "agree": len(agree), "cameras": len(votes),
                     "outliers": {n: round(a) for n, a in votes.items() if n not in agree}})
    return rows


def _heading_phases(rows: list[dict]) -> list[tuple[float, float, int]]:
    """Group consecutive consensus rows whose headings stay within 30 degrees of the group's first row:
    (start, end, heading), with the heading the circular medoid of the group so a mid-turn sample at
    a boundary does not set it. Single-sample blips between two agreeing groups are dropped."""
    groups: list[list[dict]] = []
    for row in rows:
        if groups and _angle_gap(row["heading"], groups[-1][0]["heading"]) <= 30:
            groups[-1].append(row)
        elif groups and _angle_gap(row["heading"], groups[-1][-1]["heading"]) <= 30 and len(groups[-1]) == 1:
            groups[-1].append(row)
        else:
            groups.append([row])

    def medoid(group: list[dict]) -> int:
        headings = [r["heading"] for r in group]
        return min(headings, key=lambda a: sum(_angle_gap(a, b) for b in headings))

    groups = [g for i, g in enumerate(groups) if not (len(g) == 1 and 0 < i < len(groups) - 1
                                                      and _angle_gap(medoid(groups[i - 1]), medoid(groups[i + 1])) <= 30)]
    merged: list[list[dict]] = []
    for group in groups:
        if merged and _angle_gap(medoid(group), medoid(merged[-1])) <= 30:
            merged[-1].extend(group)
        else:
            merged.append(list(group))
    return [(g[0]["t"], g[-1]["t"], medoid(g)) for g in merged]


def _frame_positions(observations: str) -> dict[float, float]:
    """Time -> horizontal position (0 = left edge, 1 = right edge) from per-frame lines; duplicates averaged."""
    values: dict[float, list[float]] = {}
    for line in observations.splitlines():
        match = re.match(r"\s*t=([\d.]+)s", line)
        position = re.search(r"position:\s*([\d.]+)|\(e\)\s*([\d.]+)", line)
        if match and position:
            values.setdefault(round(float(match.group(1)), 2), []).append(float(position.group(1) or position.group(2)))
    return {t: sum(v) / len(v) for t, v in values.items()}


def _travel_estimate(views: dict[str, tuple[float, str]]) -> list[dict]:
    """Per time step, the person's net floor displacement since the first frame, in rig terms.

    A camera at azimuth B sees travel toward rig direction B-90 as motion toward frame right, so
    each camera's horizontal displacement is the projection of one 2D travel vector onto that
    direction; least squares over all cameras recovers it. Units are fractions of the (cropped)
    frame width, so only direction and timing are meaningful.
    """
    import math

    series = {name: (azimuth, _frame_positions(obs)) for name, (azimuth, obs) in views.items()}
    times = sorted({t for _, positions in series.values() for t in positions})
    rows = []
    for time_s in times:
        sxx = sxy = syy = bx = by = 0.0
        used = 0
        for azimuth, positions in series.values():
            if time_s not in positions or not positions:
                continue
            start = positions[min(positions)]
            displacement = positions[time_s] - start
            angle = math.radians((azimuth - 90) % 360)
            ux, uy = math.cos(angle), math.sin(angle)
            sxx += ux * ux; sxy += ux * uy; syy += uy * uy; bx += displacement * ux; by += displacement * uy
            used += 1
        det = sxx * syy - sxy * sxy
        if used < 3 or abs(det) < 1e-9:
            continue
        vx = (syy * bx - sxy * by) / det
        vy = (sxx * by - sxy * bx) / det
        rows.append({"t": time_s, "distance": round(math.hypot(vx, vy), 3),
                     "direction": round(math.degrees(math.atan2(vy, vx)) % 360), "cameras": used})
    return rows


def _travel_summary(rows: list[dict], threshold: float = 0.08) -> tuple[float, float, int] | None:
    """(start, end, rig direction) of the main travel, or None when net displacement stays small."""
    if not rows or max(r["distance"] for r in rows) < threshold:
        return None
    final = max(rows, key=lambda r: r["distance"])
    moving = [r["t"] for prev, r in zip(rows, rows[1:]) if r["distance"] - prev["distance"] > 0.02]
    start = moving[0] if moving else rows[0]["t"]
    # The travel ends once 90% of the maximum displacement is reached.
    end = next(r["t"] for r in rows if r["distance"] >= 0.9 * final["distance"])
    return (min(start, end), end, final["direction"])


_TRAVEL_NAMES = [(0, "toward the camera"), (45, "toward the camera and frame left"), (90, "toward frame left"),
                 (135, "away from the camera toward frame left"), (180, "away from the camera"),
                 (225, "away from the camera toward frame right"), (270, "toward frame right"),
                 (315, "toward the camera and frame right")]


def _relative_travel(direction: float, azimuth: float) -> str:
    relative = (direction - azimuth) % 360
    return min(_TRAVEL_NAMES, key=lambda item: _angle_gap(relative, item[0]))[1]


def _relative_facing(heading: float, azimuth: float) -> str:
    relative = (heading - azimuth) % 360
    return min(_RELATIVE_NAMES, key=lambda item: _angle_gap(relative, item[0]))[1]


LOCATE_INSTRUCTION = """\
This image holds {count} video frame(s) side by side, left to right, numbered from 1. Detect the main person in \
each frame, including hands, feet, hair and anything they hold. Return only a JSON list: \
[{{"frame": 1, "box_2d": [ymin, xmin, ymax, xmax]}}, ...] with coordinates normalised to 0-1000 over the whole \
image."""
# Response envelopes whose error text means "try again later", not "bad request".
_TRANSIENT_MARKERS = ("429", "500", "502", "503", "504", "timed out", "timeout", "rate limit",
                      "ResourceExhausted", "Connection", "overloaded")


def _read_litellm_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key in ("LITELLM_BASE_URL", "LITELLM_API_KEY", "LITELLM_MODEL_VISION", "LITELLM_MODEL_TEXT"):
        values[key] = os.environ.get(key) or values.get(key, "")
        if not values[key]:
            raise ValueError(f"{key} is missing from the environment and {path}")
    values["LITELLM_BASE_URL"] = values["LITELLM_BASE_URL"].rstrip("/")
    if not values["LITELLM_BASE_URL"].endswith("/v1"):
        values["LITELLM_BASE_URL"] += "/v1"
    return values


def _jpeg_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


class CallRecord(dict):
    """One chat-completions exchange as sent and received, for the trace/report.

    ``request.messages`` keeps every text part verbatim; each image part is
    replaced by ``{"type": "image", "index": i}`` and the exact JPEG bytes that
    were base64-encoded into the request are kept in ``images[i]``.
    """


def _thinking_body(mode: str) -> dict | None:
    """``extra_body`` for a reasoning switch: "on"/"off" set ``chat_template_kwargs.enable_thinking``
    (honoured by NIM Nemotron-3); "default" sends nothing."""
    return None if mode == "default" else {"chat_template_kwargs": {"enable_thinking": mode == "on"}}


def _chat_traced(client, *, stage: str, model: str, content: list[dict], images: list[bytes],
                 max_tokens: int, retries: int, extra_body: dict | None = None) -> tuple[str, CallRecord]:
    """Send one user message; retry transient failures and empty answers; return text and the trace."""
    shown = []
    image_index = 0
    for part in content:
        if part["type"] == "image_url":
            shown.append({"type": "image", "index": image_index})
            image_index += 1
        else:
            shown.append(part)
    record = CallRecord(stage=stage, model_requested=model, images=images,
                        request={"messages": [{"role": "user", "content": shown}],
                                 "max_tokens": max_tokens, "temperature": 0,
                                 **({"extra_body": extra_body} if extra_body else {})},
                        attempts=[])
    for attempt in range(retries + 1):
        started = time.time()
        entry: dict = {"attempt": attempt + 1}
        try:
            raw = client.chat.completions.with_raw_response.create(
                model=model, messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens, temperature=0, **({"extra_body": extra_body} if extra_body else {}),
            )
            response = raw.parse()
            entry["raw_response"] = response.model_dump(mode="json")
            entry["elapsed_s"] = round(time.time() - started, 2)
            # A LiteLLM group reports its own name in ``model``; the deployment
            # that served the call is only in this header.
            entry["deployment"] = raw.headers.get("x-litellm-model-id")
            entry["model_served"] = entry["deployment"] or response.model
            text = response.choices[0].message.content if response.choices else None
            entry["finish_reason"] = response.choices[0].finish_reason if response.choices else None
            record["attempts"].append(entry)
            if isinstance(text, str) and text.strip():
                record.update(model_served=entry["model_served"], content=text)
                return text, record
            entry["error"] = "empty message content"
        except Exception as exc:  # noqa: BLE001 - recorded, then retried or re-raised
            entry.update(error=f"{type(exc).__name__}: {exc}", elapsed_s=round(time.time() - started, 2))
            record["attempts"].append(entry)
            if not any(marker.lower() in str(exc).lower() for marker in _TRANSIENT_MARKERS):
                raise _TracedError(str(exc), record) from exc
        if attempt < retries:
            time.sleep(min(60, 5 * 2 ** attempt))
    raise _TracedError(f"{stage}: no usable response after {retries + 1} attempts", record)


class _TracedError(RuntimeError):
    def __init__(self, message: str, record: CallRecord):
        super().__init__(message)
        self.record = record


class LiteLLMVisionCaptioner(MediaCaptioningModel):
    """Frame-based captioner compatible with LTX Trainer's captioner interface.

    ``layout="frames"`` sends at most ``max_images`` images, each preceded by a
    text part naming its frames and timestamps; frames beyond that limit share
    an image as a left-to-right strip, each tile keeping ``max_side``.
    ``layout="sheet"`` packs up to four frames per timestamp-labelled contact
    sheet of ``max_side`` pixels (the original behaviour).

    ``subject_crop`` (frames layout only) first asks the same model for the
    person's box in every sampled frame, then crops all frames to the union of
    those boxes. One fixed window keeps travel across the floor visible, and one
    image slot carries a full uncropped frame for the setting and camera. It
    exists because a full-body subject in a wide multi-camera capture is ~15% of
    the frame width: at API resolutions small held objects (eyeglasses) and hand
    poses are unreadable without it.

    ``segments > 1`` splits the clip into consecutive segments of
    ``max_images`` frames and gives every frame its own image slot, one vision
    call per segment. It exists because one call over 12-16 frames forces
    several frames to share an image (fewer pixels each) and makes the model copy
    the previous per-frame line forward. ``segment_mode="sequential"`` runs the
    segment calls independently. ``"hierarchical"`` first sends an overview call
    with the first frame of every segment, uncropped, then passes that overview
    to each segment call as context. The source caption is all observations,
    labelled and in time order; the rewrite merges them.
    """

    def __init__(self, *, base_url: str, api_key: str, model: str, instruction: str | None = None,
                 frames: int = 8, max_side: int = 768, layout: str = "frames", max_images: int = 4,
                 subject_crop: bool = False, crop_margin: float = 0.12, segments: int = 1,
                 segment_mode: str = "sequential", max_tokens: int = 4000, retries: int = 3,
                 timeout_s: float = 240.0):
        from openai import OpenAI

        if segments > 1:
            if layout != "frames" or segment_mode not in {"sequential", "hierarchical"}:
                raise ValueError("segments need layout=frames and segment_mode sequential|hierarchical")
            frames = segments * max_images
        if not 2 <= frames <= 64 or max_side < 128:
            raise ValueError("frames must be 2–64 and max_side must be at least 128")
        if layout not in {"frames", "sheet"}:
            raise ValueError(f"unknown layout: {layout}")
        if subject_crop and (layout != "frames" or max_images < 2):
            raise ValueError("subject_crop needs layout=frames and max_images >= 2")
        self.model = model
        self.instruction = instruction or VISION_INSTRUCTION
        self.frames = frames
        self.max_side = max_side
        self.layout = layout
        self.max_images = max_images
        self.subject_crop = subject_crop
        self.segments = segments
        self.segment_mode = segment_mode
        self.crop_margin = crop_margin
        self.max_tokens = max_tokens
        self.retries = retries
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)
        self.calls: list[CallRecord] = []
        self.subject_box: list[int] | None = None

    def _sample(self, path: Path) -> list[tuple[float, "Image.Image"]]:
        import cv2
        from PIL import Image

        capture = cv2.VideoCapture(str(path))
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if count < 1 or fps <= 0:
                raise ValueError(f"cannot read video frames/fps: {path}")
            indices = sorted({round(i * (count - 1) / (self.frames - 1)) for i in range(self.frames)})
            sampled = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"cannot decode frame {index}: {path}")
                sampled.append((index / fps, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        finally:
            capture.release()
        return sampled

    def _strip(self, group: list[tuple[float, "Image.Image"]]) -> tuple[bytes, list[tuple[int, float]]]:
        """Pack frames left to right, each scaled to ``max_side``; return the JPEG and (x offset, scale) per tile."""
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.load_default(size=max(14, self.max_side // 20))
        tiles = []
        for timestamp, frame in group:
            scale = min(1.0, self.max_side / max(frame.size))
            tiles.append((timestamp, frame.resize((round(frame.width * scale), round(frame.height * scale)),
                                                  Image.Resampling.LANCZOS), scale))
        gap = 8 if len(tiles) > 1 else 0
        strip = Image.new("RGB", (sum(t[1].width for t in tiles) + gap * (len(tiles) - 1),
                                  max(t[1].height for t in tiles)), (255, 255, 255))
        draw = ImageDraw.Draw(strip)
        x, layout = 0, []
        for timestamp, frame, scale in tiles:
            strip.paste(frame, (x, 0))
            draw.text((x + 8, 6), f"t={timestamp:.2f}s", fill="yellow", font=font, stroke_width=2, stroke_fill="black")
            layout.append((x, scale))
            x += frame.width + gap
        return _jpeg_bytes(strip), layout

    def _groups(self, sampled: list, slots: int) -> list[tuple[int, list]]:
        per_image = -(-len(sampled) // slots)
        return [(offset, sampled[offset:offset + per_image]) for offset in range(0, len(sampled), per_image)]

    @staticmethod
    def _label(offset: int, group: list, total: int, what: str = "Frame") -> str:
        if len(group) == 1:
            return f"{what} {offset + 1} of {total}, t={group[0][0]:.2f}s:"
        numbers = ", ".join(str(n) for n in range(offset + 1, offset + len(group) + 1))
        return (f"{what}s {numbers} of {total}, left to right, at "
                + ", ".join(f"t={timestamp:.2f}s" for timestamp, _ in group) + ":")

    def _locate(self, sampled: list) -> list[int] | None:
        """Ask the model for the person's box in every sampled frame; return the padded union in source pixels."""
        width, height = sampled[0][1].size
        boxes = []
        for offset, group in self._groups(sampled, self.max_images):
            jpeg, layout = self._strip(group)
            content = [{"type": "text", "text": LOCATE_INSTRUCTION.format(count=len(group))},
                       {"type": "image_url", "image_url": {"url": _data_url(jpeg)}}]
            text, call = _chat_traced(self._client, stage="locate", model=self.model, content=content,
                                      images=[jpeg], max_tokens=600, retries=self.retries)
            self.calls.append(call)
            from PIL import Image

            strip_width, strip_height = Image.open(io.BytesIO(jpeg)).size
            for item in _json_list(_strip_reasoning(text)):
                try:
                    position = int(item["frame"]) - 1
                    y0, x0, y1, x1 = (float(v) for v in item["box_2d"])
                    tile_x, scale = layout[position]
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
                boxes.append((
                    (x0 / 1000 * strip_width - tile_x) / scale, y0 / 1000 * strip_height / scale,
                    (x1 / 1000 * strip_width - tile_x) / scale, y1 / 1000 * strip_height / scale,
                ))
        if not boxes:
            return None
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
        pad_x, pad_y = (x1 - x0) * self.crop_margin, (y1 - y0) * self.crop_margin
        box = [max(0, round(x0 - pad_x)), max(0, round(y0 - pad_y)),
               min(width, round(x1 + pad_x)), min(height, round(y1 + pad_y))]
        if box[2] - box[0] < width * 0.05 or box[3] - box[1] < height * 0.05:
            return None
        return box

    def _video_content(self, path: Path) -> tuple[list[dict], list[bytes]]:
        from PIL import Image, ImageDraw

        sampled = self._sample(path)
        content: list[dict] = []
        images: list[bytes] = []
        if self.layout == "frames":
            slots = self.max_images
            if self.subject_crop:
                self.subject_box = self._locate(sampled)
            if self.subject_box:
                context, _ = self._strip([sampled[0]])
                content += [{"type": "text", "text": f"Context: the full uncropped frame at t={sampled[0][0]:.2f}s. "
                             "Every following frame is cropped to the same fixed window around the person."},
                            {"type": "image_url", "image_url": {"url": _data_url(context)}}]
                images.append(context)
                slots -= 1
                sampled = [(t, frame.crop(tuple(self.subject_box))) for t, frame in sampled]
            for offset, group in self._groups(sampled, slots):
                jpeg, _ = self._strip(group)
                what = "Cropped frame" if self.subject_box else "Frame"
                content.append({"type": "text", "text": self._label(offset, group, len(sampled), what)})
                content.append({"type": "image_url", "image_url": {"url": _data_url(jpeg)}})
                images.append(jpeg)
            return content, images
        tile = self.max_side // 2
        for offset in range(0, len(sampled), 4):
            sheet = Image.new("RGB", (self.max_side, self.max_side), (24, 24, 24))
            draw = ImageDraw.Draw(sheet)
            for position, (timestamp, frame) in enumerate(sampled[offset:offset + 4]):
                x, y = (position % 2) * tile, (position // 2) * tile
                draw.text((x + 5, y + 4), f"{timestamp:.2f}s", fill="white")
                frame.thumbnail((tile - 8, tile - 30))
                sheet.paste(frame, (x + 4, y + 25))
            jpeg = _jpeg_bytes(sheet)
            content.append({"type": "image_url", "image_url": {"url": _data_url(jpeg)}})
            images.append(jpeg)
        return content, images

    def resolved_instruction(self, frame_count: int) -> str:
        return self.instruction.replace("{count}", str(frame_count))

    def caption(self, path: str | Path, fps: int = 2) -> str:  # fps retained for the copied interface
        from PIL import Image

        path = Path(path)
        self.calls, self.subject_box = [], None
        if self._is_image_file(path):
            with Image.open(path) as image:
                jpeg = _jpeg_bytes(image)
            media = [{"type": "image_url", "image_url": {"url": _data_url(jpeg)}}]
            images = [jpeg]
        elif self._is_video_file(path) and self.segments > 1:
            return self._caption_segmented(path)
        elif self._is_video_file(path):
            media, images = self._video_content(path)
        else:
            raise ValueError(f"unsupported media file: {path}")
        content = [{"type": "text", "text": self.resolved_instruction(self.frames)}, *media]
        try:
            text, call = _chat_traced(self._client, stage="vision", model=self.model, content=content,
                                      images=images, max_tokens=self.max_tokens, retries=self.retries)
        except _TracedError as exc:
            self.calls.append(exc.record)
            raise
        self.calls.append(call)
        return _strip_reasoning(text)


    def _vision_call(self, preamble: str, frames: list, what: str, count: int) -> tuple[str, CallRecord]:
        content: list[dict] = [{"type": "text", "text": preamble + "\n\n" + self.resolved_instruction(count)}]
        images = []
        for position, (timestamp, frame) in enumerate(frames):
            jpeg, _ = self._strip([(timestamp, frame)])
            content.append({"type": "text", "text": f"{what} {position + 1} of {len(frames)}, t={timestamp:.2f}s:"})
            content.append({"type": "image_url", "image_url": {"url": _data_url(jpeg)}})
            images.append(jpeg)
        try:
            return _chat_traced(self._client, stage="vision", model=self.model, content=content, images=images,
                                max_tokens=self.max_tokens, retries=self.retries)
        except _TracedError as exc:
            self.calls.append(exc.record)
            raise

    def _caption_segmented(self, path: Path) -> str:

        sampled = self._sample(path)
        per = self.max_images
        segments = [sampled[i:i + per] for i in range(0, len(sampled), per)]
        if self.subject_crop:
            self.subject_box = self._locate(sampled)
        cropped = ([(t, f.crop(tuple(self.subject_box))) for t, f in sampled] if self.subject_box else sampled)
        crop_note = (" Every frame is cropped to the same fixed window around the person, so a change of position "
                     "inside the crop is real movement across the floor." if self.subject_box else "")
        parts = []
        overview = ""
        if self.segment_mode == "hierarchical":
            firsts = [segment[0] for segment in segments]
            text, call = self._vision_call(
                f"OVERVIEW PASS. These {len(firsts)} full, uncropped frames are the first frames of "
                f"{len(segments)} consecutive segments that together span the whole clip "
                f"(t={sampled[0][0]:.2f}s to {sampled[-1][0]:.2f}s). Describe the clip at this coarse scale; "
                "later passes will look at each segment in detail.", firsts, "Overview frame", len(firsts))
            self.calls.append(call)
            overview = _strip_reasoning(text)
            parts.append(f"Overview (first frame of each segment, uncropped):\n{overview}")

        def run(index: int) -> tuple[str, CallRecord]:
            frames = cropped[index * per:(index + 1) * per]
            preamble = (f"SEGMENT {index + 1} OF {len(segments)} of one clip, covering t={frames[0][0]:.2f}s to "
                        f"{frames[-1][0]:.2f}s.{crop_note}")
            if overview:
                preamble += ("\n\nOverview of the whole clip from an earlier coarse pass, for context only. "
                             "Report what these frames show, and correct the overview where they disagree:\n"
                             + overview)
            return self._vision_call(preamble, frames, "Frame", len(frames))

        with ThreadPoolExecutor(max_workers=len(segments)) as pool:
            results = list(pool.map(run, range(len(segments))))
        for index, (text, call) in enumerate(results):
            self.calls.append(call)
            frames = segments[index]
            parts.append(f"Segment {index + 1} of {len(segments)} (t={frames[0][0]:.2f}s to "
                         f"{frames[-1][0]:.2f}s):\n{_strip_reasoning(text)}")
        return "\n\n".join(parts)


def _json_list(text: str) -> list:
    """Return the first JSON array in ``text`` (models often wrap it in a code fence), or []."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "[":
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, list):
                return value
    return []


def _strip_reasoning(text: str) -> str:
    """Drop a leading ``<think>`` block that some reasoning models inline in ``content``."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def _prompt_from_response(raw: str) -> str:
    decoder = json.JSONDecoder()
    prompts = []
    for index, char in enumerate(raw):
        if char not in "{\"":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and set(obj) == {"prompt"} and isinstance(obj["prompt"], str):
            prompts.append(obj["prompt"])
        elif isinstance(obj, str) and '"prompt"' in obj:
            # Some models return the object JSON-encoded a second time, e.g. "{\\n \\"prompt\\": ...}".
            try:
                prompts.append(_prompt_from_response(obj))
            except ValueError:
                pass
    if not prompts and '\\"prompt\\"' in raw:
        # Some models emit the object with its quotes backslash-escaped ({\\"prompt\\": ...}).
        unescaped = re.sub(r'\\(["\\n])', lambda m: {'"': '"', "\\": "\\", "n": "\n"}[m.group(1)], raw)
        return _prompt_from_response(unescaped)
    if not prompts:
        raise ValueError("text model returned no JSON prompt object")
    prompt = re.sub(r"\s+", " ", prompts[-1]).strip()
    if not prompt:
        raise ValueError("text model returned an empty prompt")
    return prompt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for key in sorted(records):
            handle.write(json.dumps(records[key], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Caption clips with an original LTX backend or LiteLLM frames")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=[choice.value for choice in CaptionerType],
                        default=CaptionerType.LITELLM_VISION.value)
    parser.add_argument("--env-file", type=Path,
                        default=Path(os.environ.get("LTX_ENV_FILE", ".env")))
    parser.add_argument("--vision-model", help="LiteLLM model for frames (default: LITELLM_MODEL_VISION). "
                        "Pin a concrete model: a load-balanced alias can route to non-chat or weak models.")
    parser.add_argument("--text-model", help="LiteLLM model for the rewrite (default: LITELLM_MODEL_TEXT)")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=768,
                        help="Longest side of each frame (layout=frames) or of each contact sheet (layout=sheet)")
    parser.add_argument("--layout", choices=["frames", "sheet"], default="frames")
    parser.add_argument("--max-images", type=int, default=4,
                        help="Images per request for layout=frames (NIM Gemma accepts at most 4)")
    parser.add_argument("--segments", type=int, default=1,
                        help="Split into this many consecutive segments of --max-images frames, one call each "
                             "(overrides --frames)")
    parser.add_argument("--segment-mode", choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--subject-crop", action="store_true",
                        help="Locate the person with the vision model and send crops plus one full context frame")
    parser.add_argument("--vision-max-tokens", type=int, default=4000)
    parser.add_argument("--retries", type=int, default=3, help="Retries per call for transient errors/empty answers")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--instruction", help="Override the selected captioner's source-caption instruction")
    parser.add_argument("--instruction-file", type=Path, help="Read the source-caption instruction from a file")
    parser.add_argument("--rewrite-instruction-file", type=Path, help="Read the rewrite instruction from a file")
    parser.add_argument("--trace-dir", type=Path,
                        help="Save exact images, request text, and raw responses per clip, plus report.md")
    parser.add_argument("--ground-truth", type=Path,
                        help="Optional JSON of reference prompts shown beside each result in the report")
    parser.add_argument("--report-only", action="store_true", help="Rebuild report.md from --trace-dir and exit")
    parser.add_argument("--workers", type=int, default=1, help="Clips processed concurrently")
    parser.add_argument("--observations-from", type=Path, action="append", default=[],
                        help="Earlier output JSONL whose vision observations may be reused when the vision "
                             "context matches (lets text-only stages be iterated without new vision calls)")
    parser.add_argument("--coordinate-views", action="store_true",
                        help="Treat clips in one directory as simultaneous views of one performance: reconcile "
                             "their observations into a subject sheet, then rewrite each view's prompt from it. "
                             "Reads optional camera placement from <dir>/views.json")
    parser.add_argument("--text-max-tokens", type=int, default=16000,
                        help="Token cap for rewrite and coordination calls; reasoning models spend most of it "
                             "on hidden reasoning, and 4000 truncated Nemotron-3-Ultra's JSON")
    parser.add_argument("--text-timeout", type=float, default=900.0,
                        help="Seconds per rewrite/coordination request; a reasoning model reconciling 8 views "
                             "took over 240 s")
    parser.add_argument("--sheet-thinking", choices=["default", "on", "off"], default="on",
                        help="Reasoning for the multi-view subject sheet; with it off Nemotron-3-Ultra got the turn "
                             "direction wrong")
    parser.add_argument("--rewrite-thinking", choices=["default", "on", "off"], default="default",
                        help="Reasoning for single-view and per-view coordinated rewrites")
    parser.add_argument("--coord-model", help="Text model for view coordination (default: --text-model)")
    parser.add_argument("--vllm-url", default=DEFAULT_VLLM_BASE_URL)
    args = parser.parse_args()

    output = args.output.resolve()
    ground_truth = _load_ground_truth(args.ground_truth, output.parent)
    title = f"LTX prompt trace — {output.stem}"
    if args.report_only:
        if not args.trace_dir:
            parser.error("--report-only needs --trace-dir")
        print(f"Report: {build_report(args.trace_dir.resolve(), output, ground_truth, title)}")
        return 0
    if args.instruction and args.instruction_file:
        parser.error("use --instruction or --instruction-file, not both")
    instruction = args.instruction_file.read_text(encoding="utf-8").strip() if args.instruction_file else args.instruction
    rewrite_instruction = (args.rewrite_instruction_file.read_text(encoding="utf-8").rstrip() + "\n"
                           if args.rewrite_instruction_file
                           else SEGMENT_REWRITE_INSTRUCTION if args.segments > 1 else REWRITE_INSTRUCTION)

    config = _read_litellm_env(args.env_file)
    from openai import OpenAI

    text_model = args.text_model or config["LITELLM_MODEL_TEXT"]
    text_client = OpenAI(base_url=config["LITELLM_BASE_URL"], api_key=config["LITELLM_API_KEY"],
                         timeout=args.text_timeout, max_retries=0)
    backend = CaptionerType(args.backend)
    is_litellm = backend == CaptionerType.LITELLM_VISION

    def make_captioner() -> MediaCaptioningModel:
        if is_litellm:
            return create_captioner(
                backend, base_url=config["LITELLM_BASE_URL"], api_key=config["LITELLM_API_KEY"],
                model=args.vision_model or config["LITELLM_MODEL_VISION"], instruction=instruction,
                frames=args.frames, max_side=args.max_side, layout=args.layout, max_images=args.max_images,
                subject_crop=args.subject_crop, segments=args.segments, segment_mode=args.segment_mode,
                max_tokens=args.vision_max_tokens, retries=args.retries,
            )
        if backend == CaptionerType.QWEN_OMNI:
            return create_captioner(backend, base_url=args.vllm_url, instruction=instruction)
        return create_captioner(backend, instruction=instruction)

    captioner = make_captioner()
    if args.input.is_file():
        videos = [args.input]
    else:
        paths = args.input.rglob("*") if args.recursive else args.input.iterdir()
        videos = sorted(p for p in paths if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv", ".avi"})
    if not videos:
        parser.error("no videos found")

    work = output.with_name(output.name + ".work.jsonl")
    previous = {**_load_jsonl(output), **_load_jsonl(work)}
    donors: dict[str, dict] = {}
    for path in args.observations_from:
        for key, record in _load_jsonl(path.resolve()).items():
            # Donor keys are relative to the donor's directory; re-key them to this output.
            absolute = os.path.normpath(path.resolve().parent / key)
            donors[os.path.relpath(absolute, output.parent)] = {**record, "_donor": str(path.resolve())}
    records: dict[str, dict] = {}
    failures = 0
    per_call = captioner.max_images if is_litellm and args.segments > 1 else getattr(captioner, "frames", None)
    vision_instruction = (captioner.resolved_instruction(per_call) if is_litellm
                          else instruction or captioner._resolve_instruction(videos[0]))
    vision_context = {
        "backend": backend.value,
        "caption_model": captioner.model if hasattr(captioner, "model") else GeminiFlashCaptioner.MODEL_ID,
        "caption_endpoint_sha256": hashlib.sha256(
            (config["LITELLM_BASE_URL"] if is_litellm else args.vllm_url).encode()
        ).hexdigest() if backend != CaptionerType.GEMINI_FLASH else None,
        "instruction_sha256": hashlib.sha256(vision_instruction.encode()).hexdigest(),
        "frames": captioner.frames if is_litellm else None,
        "max_side": args.max_side if is_litellm else None,
        "layout": args.layout if is_litellm else None,
        "max_images": args.max_images if is_litellm and args.layout == "frames" else None,
        "subject_crop": args.subject_crop if is_litellm else None,
        "segments": args.segments if is_litellm and args.segments > 1 else None,
        "segment_mode": args.segment_mode if is_litellm and args.segments > 1 else None,
        "locate_sha256": hashlib.sha256(LOCATE_INSTRUCTION.encode()).hexdigest() if is_litellm and args.subject_crop else None,
        "vision_max_tokens": args.vision_max_tokens if is_litellm else None,
        "vision_stage_version": VISION_STAGE_VERSION,
    }
    context = {
        **vision_context,
        "text_model": text_model,
        "text_endpoint_sha256": hashlib.sha256(config["LITELLM_BASE_URL"].encode()).hexdigest(),
        "guide": LTX_GUIDE_URL,
        "rewrite_sha256": hashlib.sha256(rewrite_instruction.encode()).hexdigest(),
        "rewrite_thinking": args.rewrite_thinking,
        "producer_sha256": _sha256(Path(__file__)),
        "producer_components_sha256": {name: _sha256(Path(__file__).with_name(name))
                                       for name in ("backends.py", "media.py", "traces.py")},
    }
    trace_dir = args.trace_dir.resolve() if args.trace_dir else None
    lock = threading.Lock()
    local = threading.local()

    def process(video: Path) -> bool:
        # Match caption_videos.py: preserve a symlinked clip's logical dataset path.
        key = os.path.relpath(video.parent.resolve() / video.name, output.parent)
        source_sha256 = _sha256(video)
        cached = previous.get(key)
        if (not args.overwrite and cached and cached.get("source_sha256") == source_sha256
                and cached.get("context") == context and isinstance(cached.get("caption_single_view",
                                                                               cached.get("caption")), str)):
            with lock:
                records[key] = cached
            print(f"Skipped compatible caption: {key}", flush=True)
            return True
        calls: list[CallRecord] = []
        reusable = [r for r in (cached, donors.get(key)) if r and r.get("source_sha256") == source_sha256
                    and r.get("vision_context") == vision_context and r.get("source_caption")]
        try:
            if reusable and not args.overwrite:
                source_caption = reusable[0]["source_caption"]
                observations_trace = reusable[0].get("observations_trace")
                subject_box = reusable[0].get("subject_box")
                print(f"Reusing vision observations: {key}", flush=True)
            else:
                if not hasattr(local, "captioner"):
                    local.captioner = make_captioner()
                try:
                    source_caption = local.captioner.caption(video)
                finally:
                    calls.extend(getattr(local.captioner, "calls", []))
                observations_trace = str(trace_dir / _trace_slug(key)) if trace_dir else None
                subject_box = getattr(local.captioner, "subject_box", None)
            text, call = _chat_traced(text_client, stage="rewrite", model=text_model,
                                      content=[{"type": "text", "text": rewrite_instruction + source_caption}],
                                      images=[], max_tokens=args.text_max_tokens, retries=args.retries,
                                      extra_body=_thinking_body(args.rewrite_thinking))
            calls.append(call)
            prompt = _prompt_from_response(_strip_reasoning(text))
            record = {
                "media_path": key, "caption": prompt, "caption_single_view": prompt,
                "source_caption": source_caption, "source_sha256": source_sha256, "context": context,
                "vision_context": vision_context, "observations_trace": observations_trace,
                "served_models": {c["stage"]: c.get("model_served") for c in calls},
                "subject_box": subject_box,
            }
            with lock:
                records[key] = record
                _write_jsonl(work, records)
            if trace_dir:
                _write_trace(trace_dir, key, calls, record)
            print(f"Prepared prompt: {key}", flush=True)
            return True
        except Exception as exc:
            if isinstance(exc, _TracedError) and exc.record not in calls:
                calls.append(exc.record)
            if trace_dir:
                _write_trace(trace_dir, key, calls, {"media_path": key, "error": str(exc), "context": context})
            print(f"Failed {key}: {exc}", flush=True)
            return False

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        failures = sum(not ok for ok in pool.map(process, videos))

    if args.coordinate_views and not failures:
        coord_model = args.coord_model or text_model
        failures += _coordinate_views(records, output, trace_dir, text_client, coord_model, args.retries,
                                      max(1, args.workers), args.text_max_tokens, args.sheet_thinking,
                                      args.rewrite_thinking)
        with lock:
            _write_jsonl(work, records)
    if trace_dir:
        print(f"Report: {build_report(trace_dir, output, ground_truth, title)}", flush=True)
    if failures:
        if records:
            _write_jsonl(work, records)
        print(f"Prepared {len(records)}/{len(videos)} prompts; {failures} failure(s); output unchanged.", flush=True)
        return 1
    _write_jsonl(output, records)
    work.unlink(missing_ok=True)
    print(f"Prepared {len(records)}/{len(videos)} prompts in {output}", flush=True)
    return 0


def _coordinate_views(records: dict[str, dict], output: Path, trace_dir: Path | None, client, model: str,
                      retries: int, workers: int, max_tokens: int = 16000, sheet_thinking: str = "default",
                      rewrite_thinking: str = "default") -> int:
    """Reconcile each directory's views into a subject sheet, then rewrite every view's prompt from it.

    Views of one performance are the clips sharing a parent directory; an optional
    ``views.json`` there gives each stem's camera placement. Each record keeps its
    single-view prompt in ``caption_single_view`` and gets the coordinated one in
    ``caption`` plus a ``coordination`` block. Returns the number of failed groups.
    """
    groups: dict[str, list[str]] = {}
    for key in records:
        groups.setdefault(os.path.dirname(key), []).append(key)
    sheet_sha = hashlib.sha256(MV_SHEET_INSTRUCTION.encode()).hexdigest()
    view_sha = hashlib.sha256(MV_VIEW_INSTRUCTION.encode()).hexdigest()
    grouped = [(group, keys) for group, keys in sorted(groups.items()) if len(keys) >= 2]
    if not grouped:
        return 0
    request_slots = threading.BoundedSemaphore(max(1, workers))

    def coordinate_group(item: tuple[str, list[str]]) -> int:
        group, keys = item
        failed = False
        meta_file = output.parent / group / "views.json"
        cameras = json.loads(meta_file.read_text())["views"] if meta_file.is_file() else {}
        calls: list[CallRecord] = []

        def camera_line(key: str) -> str:
            stem = Path(key).stem
            meta = cameras.get(stem, {})
            placement = ", ".join(f"{k}={v}" for k, v in meta.items()) or "placement unknown"
            return f"Camera {stem} ({placement})"

        try:
            azimuths = {k: cameras.get(Path(k).stem, {}).get("azimuth_deg") for k in keys}
            consensus = _heading_consensus({Path(k).stem: (azimuths[k], records[k]["source_caption"])
                                            for k in keys if azimuths[k] is not None})
            phases = _heading_phases(consensus)
            table = "\n".join(
                f"t={row['t']:.2f}s: heading {row['heading']} ({row['agree']}/{row['cameras']} cameras within 45 deg"
                + (f"; outliers {row['outliers']}" if row["outliers"] else "") + ")" for row in consensus)
            phase_text = "; ".join(f"{a:.2f}-{b:.2f}s heading {h}" for a, b, h in phases) or "unknown"
            travel_rows = _travel_estimate({Path(k).stem: (azimuths[k], records[k]["source_caption"])
                                            for k in keys if azimuths[k] is not None})
            travel = _travel_summary(travel_rows)
            travel_text = ("no travel: net displacement stays small; the person stays in one spot" if travel is None
                           else f"travels toward rig direction {travel[2]} between about {travel[0]:.2f}s and "
                                f"{travel[1]:.2f}s, then stays in that spot")
            blocks = [f"=== {camera_line(k)} ===\n{records[k]['source_caption']}" for k in sorted(keys)]
            sheet_prompt = (MV_SHEET_INSTRUCTION
                            + "\nHEADING TABLE (code-computed consensus):\n" + (table or "none")
                            + "\nHeading phases: " + phase_text
                            + "\nTRAVEL (code-computed from every camera's horizontal positions; trust it): "
                            + travel_text + "\n\nCameras and their observations:\n" + "\n\n".join(blocks))
            sheet = None
            # A reasoning model occasionally spends its budget without emitting the JSON; resample twice.
            for _ in range(3):
                with request_slots:
                    text, call = _chat_traced(client, stage="mv_sheet", model=model,
                                              content=[{"type": "text", "text": sheet_prompt}],
                                              images=[], max_tokens=max_tokens, retries=retries,
                                              extra_body=_thinking_body(sheet_thinking))
                calls.append(call)
                sheet_text = _strip_reasoning(text)
                decoder = json.JSONDecoder()
                for index, char in enumerate(sheet_text):
                    if char == "{":
                        try:
                            candidate, _ = decoder.raw_decode(sheet_text[index:])
                        except json.JSONDecodeError:
                            continue
                        if isinstance(candidate, dict):
                            sheet = candidate
                            break
                if sheet is not None:
                    break
            if not isinstance(sheet, dict):
                raise ValueError("coordination model returned no JSON subject sheet in 3 samples")
            sheet_json = json.dumps(sheet, ensure_ascii=False, indent=1)

            def rewrite(key: str) -> tuple[str, str, CallRecord]:
                azimuth = azimuths.get(key)
                facing = ("; ".join(f"{a:.2f}-{b:.2f}s {_relative_facing(h, azimuth)}" for a, b, h in phases)
                          if azimuth is not None and phases else "unknown (use the observations)")
                content = (MV_VIEW_INSTRUCTION + "\nSubject sheet:\n" + sheet_json + "\n\nThis camera: "
                           + camera_line(key) + "\nFacing relative to this camera (code-computed): " + facing
                           + "\nTravel relative to this camera (code-computed): "
                           + ("none; the person stays in one spot" if travel is None or azimuth is None
                              else f"{_relative_travel(travel[2], azimuth)} between about {travel[0]:.1f}s and "
                                   f"{travel[1]:.1f}s, then stays in that spot")
                           + "\n\nThis camera's observations:\n" + records[key]["source_caption"])
                with request_slots:
                    text, call = _chat_traced(client, stage=f"mv_rewrite:{Path(key).stem}", model=model,
                                              content=[{"type": "text", "text": content}], images=[],
                                              max_tokens=max_tokens, retries=retries,
                                              extra_body=_thinking_body(rewrite_thinking))
                return key, _prompt_from_response(_strip_reasoning(text)), call

            with ThreadPoolExecutor(max_workers=min(workers, len(keys))) as pool:
                results = list(pool.map(rewrite, sorted(keys)))
            for key, prompt, call in results:
                calls.append(call)
                records[key]["caption"] = prompt
                records[key]["coordination"] = {
                    "group": group, "views": len(keys), "model": model, "served": call.get("model_served"),
                    "heading_phases": phases, "travel": travel, "sheet_thinking": sheet_thinking, "rewrite_thinking": rewrite_thinking,
                    "sheet": sheet, "sheet_sha256": sheet_sha, "view_sha256": view_sha,
                }
            print(f"Coordinated {len(keys)} views: {group}", flush=True)
        except Exception as exc:
            failed = True
            if isinstance(exc, _TracedError) and exc.record not in calls:
                calls.append(exc.record)
            print(f"Failed coordination {group}: {exc}", flush=True)
        if trace_dir:
            name = "_mv_" + re.sub(r"^(\.\./)+", "", group)
            _write_trace(trace_dir, name, calls, {"media_path": name, "group": group,
                                                            "views": sorted(keys)})
        return int(failed)

    with ThreadPoolExecutor(max_workers=min(workers, len(grouped))) as pool:
        return sum(pool.map(coordinate_group, grouped))


if __name__ == "__main__":
    raise SystemExit(main())
