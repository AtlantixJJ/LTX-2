"""Write and render LTX model-call traces."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

def _load_jsonl(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            records[item["media_path"]] = item
    return records


def _trace_slug(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key.lstrip("./")).strip("_")


def _write_trace(trace_dir: Path, key: str, calls: list[CallRecord], result: dict) -> None:
    """Write one clip's calls: the exact JPEGs sent under ``images/`` and everything else in ``trace.json``."""
    clip_dir = trace_dir / _trace_slug(key)
    if clip_dir.exists():
        for old in clip_dir.glob("*.jpg"):
            old.unlink()
    (clip_dir).mkdir(parents=True, exist_ok=True)
    serialised = []
    for number, call in enumerate(calls):
        call = dict(call)
        files = []
        for index, jpeg in enumerate(call.pop("images", [])):
            name = f"call{number:02d}_{call['stage']}_img{index:02d}.jpg"
            (clip_dir / name).write_bytes(jpeg)
            files.append({"file": name, "sha256": hashlib.sha256(jpeg).hexdigest(), "bytes": len(jpeg)})
        call["images"] = files
        serialised.append(call)
    payload = {"media_path": key, "calls": serialised, "result": result}
    temp = clip_dir / "trace.json.tmp"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    temp.replace(clip_dir / "trace.json")


def _load_ground_truth(path: Path | None, output_dir: Path) -> dict[str, dict]:
    """Map output-relative media paths to ``{"prompt", "key_facts", "forbidden"}``.

    Keys in the file are resolved relative to the ground-truth file itself.
    """
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    mapped = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        media = path.parent / key
        mapped[os.path.relpath(media.parent.resolve() / media.name, output_dir)] = value
    return mapped


def _md_cell(text: str) -> str:
    """Make ``text`` safe for one Markdown table cell."""
    return re.sub(r"\s+", " ", str(text)).replace("|", "\\|").strip()


def _md_fence(text: str, lang: str = "text") -> str:
    """Fence ``text`` verbatim, using a fence longer than any backtick run inside it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text.rstrip()}\n{fence}"


def _md_request(parts: list[dict], images: list[dict], prefix: str) -> list[str]:
    """Render one request's parts in order. A short text label that directly precedes an image
    becomes that image's column header, so consecutive labelled images sit side by side in one
    table row; any other text part is fenced verbatim."""
    lines: list[str] = []
    row: list[tuple[str, dict]] = []

    def flush() -> None:
        if not row:
            return
        lines.append("| " + " | ".join(_md_cell(label) or f"Image {n + 1}" for n, (label, _) in enumerate(row)) + " |")
        lines.append("|" + "---|" * len(row))
        lines.append("| " + " | ".join(f"![{_md_cell(image['file'])}]({prefix}{image['file']})" for _, image in row)
                     + " |")
        lines.append("| " + " | ".join(f"`{image['file']}` · {image['bytes'] // 1024} KiB · sha256 "
                                        f"`{image['sha256'][:12]}`" for _, image in row) + " |")
        lines.append("")
        row.clear()

    pending_label = None
    for part in parts:
        if part["type"] == "image":
            row.append((pending_label or "", images[part["index"]]))
            pending_label = None
            continue
        text = part["text"]
        if pending_label is not None:
            flush()
            lines += [_md_fence(pending_label), ""]
            pending_label = None
        if len(text) <= 200 and "\n" not in text:
            pending_label = text
        else:
            flush()
            lines += [_md_fence(text), ""]
    flush()
    if pending_label is not None:
        lines += [_md_fence(pending_label), ""]
    return lines


def render_trace_markdown(trace_dir: Path, output: Path, ground_truth: dict[str, dict], *, link_prefix: str = "",
                          level: int = 2) -> list[str]:
    """Markdown for every ``trace_dir/*/trace.json``: prompts, exact request parts and images, raw responses.

    Links are relative to ``trace_dir`` joined to ``link_prefix``. Each attempt's full
    response envelope is written next to its trace as ``callNN_attemptM_response.json``
    and linked rather than inlined.
    """
    h = "#" * level
    lines: list[str] = []
    records = {**_load_jsonl(output), **_load_jsonl(output.with_name(output.name + ".work.jsonl"))}
    for trace_file in sorted(trace_dir.glob("*/trace.json")):
        slug = trace_file.parent.name
        trace = json.loads(trace_file.read_text(encoding="utf-8"))
        key = trace["media_path"]
        result = trace.get("result") or {}
        record = records.get(key) or result
        truth = ground_truth.get(key)
        prefix = f"{link_prefix}{slug}/"
        lines += [f"{h} `{key}`", ""]
        context = record.get("context") or result.get("context") or {}
        if context:
            lines += ["Settings: " + ", ".join(f"`{k}={context[k]}`" for k in (
                "caption_model", "text_model", "layout", "frames", "max_side", "subject_crop", "segments",
                "segment_mode") if context.get(k) is not None), ""]
        final = record.get("caption")
        if key.startswith("_mv_"):
            lines += [f"View coordination for `{result.get('group')}` over {len(result.get('views', []))} views. "
                      "The `mv_sheet` call reconciles every view's observations into one subject sheet; each "
                      "`mv_rewrite:<view>` call writes that view's prompt from the sheet and its own observations.",
                      ""]
        elif record.get("coordination") and record.get("caption_single_view"):
            lines += ["| Single-view prompt | Coordinated prompt |" + (" Ground truth |" if truth else ""),
                      "|---|---|" + ("---|" if truth else ""),
                      f"| {_md_cell(record['caption_single_view'])} | {_md_cell(final)} |"
                      + (f" {_md_cell(truth['prompt'])} |" if truth else ""), ""]
            if truth:
                lines += ["Key facts: " + "; ".join(truth.get("key_facts", [])) + ".", ""]
        elif truth:
            lines += ["| Generated LTX prompt | Ground truth |", "|---|---|",
                      f"| {_md_cell(final) if final else '**No prompt (failed run).**'} | {_md_cell(truth['prompt'])} |",
                      ""]
            lines += ["Key facts: " + "; ".join(truth.get("key_facts", [])) + ".", ""]
            if truth.get("forbidden"):
                lines += ["Known wrong readings: " + "; ".join(truth["forbidden"]) + ".", ""]
        else:
            lines += ["**Generated LTX prompt**", "", f"> {final}" if final else "> No prompt (failed run).", ""]
        if result.get("error"):
            lines += [f"**Error:** {result['error']}", ""]
        for number, call in enumerate(trace["calls"]):
            lines += [f"{h}# Call {number}: {call['stage']}", "",
                      f"Requested `{call['model_requested']}`, served by `{call.get('model_served') or '—'}` · "
                      f"`max_tokens={call['request'].get('max_tokens')}`, "
                      f"`temperature={call['request'].get('temperature')}` · {len(call['attempts'])} attempt(s)", "",
                      "**Request message, in order:**", ""]
            lines += _md_request(call["request"]["messages"][0]["content"], call["images"], prefix)
            for attempt in call["attempts"]:
                label = f"Attempt {attempt['attempt']} · {attempt.get('elapsed_s', '?')} s"
                if attempt.get("deployment"):
                    label += f" · deployment `{attempt['deployment']}`"
                if attempt.get("finish_reason"):
                    label += f" · `finish_reason={attempt['finish_reason']}`"
                raw = attempt.get("raw_response")
                if raw:
                    envelope = f"call{number:02d}_attempt{attempt['attempt']}_response.json"
                    (trace_file.parent / envelope).write_text(json.dumps(raw, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
                    message = (raw.get("choices") or [{}])[0].get("message") or {}
                    lines += [f"**{label}** · [full response envelope]({prefix}{envelope})", "",
                              "Raw message content:", "", _md_fence(message.get("content") or ""), ""]
                    reasoning = message.get("reasoning_content") or message.get("reasoning")
                    if reasoning:
                        lines += ["Reasoning returned by the model:", "", _md_fence(str(reasoning)), ""]
                if attempt.get("error"):
                    lines += [f"**{label}** · error: {_md_cell(attempt['error'])}", ""]
    return lines


def build_report(trace_dir: Path, output: Path, ground_truth: dict[str, dict], title: str) -> Path:
    """Render every ``trace_dir/*/trace.json`` into ``trace_dir/report.md``.

    Images and response envelopes are linked relatively, so the report and its
    folder move together.
    """
    lines = [f"# {title}", "",
             f"Output: `{output}`. For each call the report shows the message parts in the order sent, "
             "the exact JPEGs that were base64-encoded into the request, and the raw response.", ""]
    lines += render_trace_markdown(trace_dir, output, ground_truth)
    report = trace_dir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


