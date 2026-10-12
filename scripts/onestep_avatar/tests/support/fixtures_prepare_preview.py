"""Shared saved-input builders and fixtures; never a collected test module."""

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.prune.data.prompt_cache as prompts
from scripts.onestep_avatar import evaluate, media, prepare_inputs
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.hashing import sha256


@pytest.fixture
def preview_case(checked_settings: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple:
    settings, membership = checked_settings
    spec = evaluate.backbone.resolve("2.5", "dev")
    Path(spec.paths.video_vae()).write_bytes(b"controlled VAE")
    frames = tuple(range(49))
    pixels = torch.zeros(49, 3, 16, 16, dtype=torch.uint8)
    source = membership["sources"][0]
    calls = []
    monkeypatch.setattr(prepare_inputs.preflight, "check", lambda *_a, **_k: calls.append("preflight"))

    def text(_spec: object, prompt: str, *_a) -> torch.Tensor:
        calls.append(prompt)
        return torch.ones(1, 3, 4, dtype=torch.bfloat16) * (2 if prompt == "negative" else 1)

    monkeypatch.setattr(prompts, "get_or_build", text)
    monkeypatch.setattr(
        prepare_inputs, "torch", SimpleNamespace(**{**vars(torch), "device": lambda *_a: torch.device("cpu")})
    )

    def setup(mode: str = "bidirectional", task: str = "d0", extra: tuple[str, ...] = ()) -> Namespace:
        if task == "d1":
            for row in membership["sources"]:
                view = Path(membership["corpus_root"]) / row["relative_dir"]
                render = view / dataset.render_name("white")
                render.write_bytes(b"controlled guide" + row["relative_dir"].encode())
                sidecar = view / dataset.render_metadata_name("white")
                sidecar.write_text(
                    json.dumps({"objective": "white", "compositing_version": dataset.GUIDE_COMPOSITING_VERSION})
                )
                record = torch.load(view / dataset.capture_bundle_name("white"), weights_only=True)
                record["input_fingerprint"] = sha256(render)
                guide = view / dataset.guide_bundle_name("white")
                torch.save(record, guide)
                row.update(
                    guide_sha256=sha256(render),
                    guide_latent_sha256=sha256(guide),
                    guide_sidecar_sha256=sha256(sidecar),
                    guide_encode_record={k: v for k, v in record.items() if k not in ("schema_version", "master")},
                )
            membership["sha256"] = subset.membership_hash(membership)
            settings.subset.write_text(json.dumps(membership))
        panels = [
            media.Panel("recorded", "Capture RGB", pixels, frames),
            media.Panel("decoded", "VAE-decoded capture", pixels, frames),
            media.Panel(
                "guide",
                "Guide RGB",
                pixels if task == "d1" else None,
                frames,
                missing_reason="" if task == "d1" else "Guide not used",
            ),
        ]
        producer = {
            "source": source["relative_dir"],
            "objective": "white",
            "fps": 30,
            "source_frames": list(frames),
            "capture_encoding_sha256": source["capture_latent_sha256"],
            "guide_rgb_sha256": source.get("guide_sha256"),
            "vae_sha256": sha256(Path(spec.paths.video_vae())),
            "panels": [{"role": p.role, "pixels_sha256": media._pixel_identity(p.pixels)} for p in panels],
        }
        references = tmp_path / (mode + "_" + task + "_refs")
        media.save_training_references(panels, producer, references)
        command = [
            "preview",
            "--references",
            str(references / "references.json"),
            "--output",
            str(tmp_path / (mode + "_" + task + "_fixed")),
            "--gpu-id",
            "3",
            "--evaluation-arguments",
            "--mode",
            mode,
            "--subset",
            str(settings.subset),
            "--source",
            source["relative_dir"],
            "--guide-mode",
            task,
            "--variant",
            "dev",
            "--span-latent-frames",
            "7",
            "--schedule",
            "0.725",
            "0",
            *extra,
        ]
        return prepare_inputs.parse_args(command)

    return setup, settings, calls


build_preview_case = preview_case.__wrapped__
