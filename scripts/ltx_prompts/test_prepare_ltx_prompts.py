"""Offline behavior checks for the reorganized LTX prompt utility."""

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from scripts.ltx_prompts import backends, traces
from scripts.ltx_prompts import prepare_ltx_prompts as ltx


class TestLtxPackages(unittest.TestCase):
    def test_caption_parsing_and_trace_report(self):
        self.assertEqual(backends._parse_caption_response(
            '<think>reasoning</think>{"combined_caption_english":"A dancer moves."}'),
            "A dancer moves.")
        self.assertEqual(ltx._prompt_from_response('{"prompt":"A dancer moves."}'),
                         "A dancer moves.")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            traces._write_trace(root / "trace", "clip a.mp4", [
                {"stage": "vision", "model_requested": "fixture", "images": [b"jpeg-bytes"],
                 "request": {"messages": [{"content": [{"type": "text", "text": "Describe this frame"}]}]},
                 "attempts": []}], {"media_path": "clip a.mp4", "caption": "A dancer moves."})
            trace = json.loads((root / "trace/clip_a.mp4/trace.json").read_text())
            self.assertEqual(len(trace["calls"][0]["images"]), 1)
            self.assertEqual((root / "trace/clip_a.mp4" /
                              trace["calls"][0]["images"][0]["file"]).read_bytes(), b"jpeg-bytes")
            report = traces.build_report(root / "trace", root / "out.jsonl", {}, "Fixture")
            self.assertIn("Describe this frame", report.read_text())

    def test_resume_requires_compatible_source_and_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "clip.mp4"
            video.write_bytes(b"source one")
            output = root / "prompts.jsonl"
            captioner = mock.Mock()
            captioner.model = "fixture-vision"
            captioner.frames = 2
            captioner.max_images = 2
            captioner.resolved_instruction.return_value = "describe"
            captioner.caption.return_value = "source caption"
            captioner.calls = []
            captioner.subject_box = None
            calls = []

            def chat(_client, **kwargs):
                calls.append(kwargs["stage"])
                return '{"prompt":"final prompt"}', {"stage": kwargs["stage"],
                                                       "model_served": "fixture-text", "images": []}

            args = ["prepare_ltx_prompts.py", str(video), "--output", str(output),
                    "--backend", "litellm_vision", "--frames", "2"]
            config = {"LITELLM_BASE_URL": "http://fixture/v1", "LITELLM_API_KEY": "fixture",
                      "LITELLM_MODEL_VISION": "fixture-vision", "LITELLM_MODEL_TEXT": "fixture-text"}
            with mock.patch.object(ltx, "_read_litellm_env", return_value=config), \
                 mock.patch.object(ltx, "create_captioner", return_value=captioner), \
                 mock.patch.object(ltx, "_chat_traced", side_effect=chat), \
                 mock.patch("openai.OpenAI"), mock.patch.object(sys, "argv", args):
                self.assertEqual(ltx.main(), 0)
                self.assertEqual(ltx.main(), 0)
                self.assertEqual(captioner.caption.call_count, 1)
                self.assertEqual(calls, ["rewrite"])
                video.write_bytes(b"source changed")
                self.assertEqual(ltx.main(), 0)
                self.assertEqual(captioner.caption.call_count, 2)
                self.assertEqual(calls, ["rewrite", "rewrite"])
            record = json.loads(output.read_text().splitlines()[0])
            self.assertEqual(record["caption"], "final prompt")
            self.assertEqual(set(record["context"]["producer_components_sha256"]),
                             {"backends.py", "media.py", "traces.py"})

    def test_multiview_groups_only_siblings(self):
        records = {"take/cam1.mp4": {"source_caption": "front", "caption": "old front",
                                      "caption_single_view": "old front"},
                   "take/cam2.mp4": {"source_caption": "side", "caption": "old side",
                                      "caption_single_view": "old side"},
                   "solo/cam3.mp4": {"source_caption": "back", "caption": "old back"}}
        stages = []

        def chat(_client, **kwargs):
            stage = kwargs["stage"]
            stages.append(stage)
            result = '{"subject":"dancer"}' if stage == "mv_sheet" else \
                     '{"prompt":"coordinated ' + stage + '"}'
            return result, {"stage": stage, "model_served": "fixture", "images": []}

        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.object(ltx, "_chat_traced", side_effect=chat):
            self.assertEqual(ltx._coordinate_views(
                records, Path(temp) / "output.jsonl", None, object(), "fixture", 1, 1), 0)
        self.assertEqual(stages, ["mv_sheet", "mv_rewrite:cam1", "mv_rewrite:cam2"])
        self.assertEqual(records["solo/cam3.mp4"]["caption"], "old back")
        self.assertEqual(records["take/cam1.mp4"]["caption_single_view"], "old front")

    def test_multiview_groups_run_concurrently_with_request_limit(self):
        records = {f"take{group}/cam{view}.mp4": {
            "source_caption": "t=0s: facing the camera", "caption_single_view": "original"
        } for group in range(3) for view in range(2)}
        lock = threading.Lock()
        two_sheets_started = threading.Event()
        active = peak = sheet_starts = 0

        def chat(_client, **kwargs):
            nonlocal active, peak, sheet_starts
            with lock:
                active += 1
                peak = max(peak, active)
                if kwargs["stage"] == "mv_sheet":
                    sheet_starts += 1
                    if sheet_starts >= 2:
                        two_sheets_started.set()
            try:
                if kwargs["stage"] == "mv_sheet":
                    self.assertTrue(two_sheets_started.wait(2), "groups ran sequentially")
                time.sleep(0.01)
                answer = '{"person":"dancer"}' if kwargs["stage"] == "mv_sheet" else \
                         '{"prompt":"coordinated dancer"}'
                return answer, {"stage": kwargs["stage"], "model_served": "fixture", "images": []}
            finally:
                with lock:
                    active -= 1

        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.object(ltx, "_chat_traced", side_effect=chat):
            self.assertEqual(ltx._coordinate_views(
                records, Path(temp) / "output.jsonl", None, object(), "fixture", 1, 2), 0)
        self.assertTrue(two_sheets_started.is_set())
        self.assertEqual(peak, 2)
        self.assertTrue(all(row["caption"] == "coordinated dancer" for row in records.values()))
