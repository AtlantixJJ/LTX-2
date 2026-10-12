# ruff: noqa: PLC0415
# Keep fixture dependencies local: ordinary imports must work with experiments blocked.
"""Shared corpus fixtures preserve their exact capture, guide and membership inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.onestep_avatar.hashing import sha256


def build_membership(tmp_path: Path) -> dict:
    """Author current capture records directly, independently of membership discovery."""
    from scripts.onestep_avatar.corpus import dataset, subset

    sources, windows = [], []
    for actor, group in (("1", "train"), ("2", "held_out")):
        name = f"Part_1/{actor}/views/view00"
        view = tmp_path / name
        view.mkdir(parents=True)
        (view / "rgb.mp4").write_bytes(b"saved capture bytes" + actor.encode())
        metadata = view.parent.parent / "meta.json"
        metadata.write_text(
            json.dumps({"actor": {"id": actor}, "_pipeline": {"state": "done"}, "fps": 30.0, "n_frames": 49})
        )
        encoding = {
            "schema_version": 2,
            "fps": 30.0,
            "objective": "white",
            "source": name,
            "vae_fingerprint": "saved VAE identity",
            "box_xyxy": [0, 0, 64, 64],
            "edge": 64,
            "encode_contract_version": 1,
            "pixel_frames": 49,
            "input_fingerprint": "saved pixel inputs",
        }
        capture = view / dataset.capture_bundle_name("white")
        torch.save({**encoding, "master": torch.arange(56).reshape(2, 7, 2, 2).float()}, capture)
        sources.append(
            {
                "relative_dir": name,
                "actor": actor,
                "split": group,
                "fps": 30.0,
                "n_latent_frames": 7,
                "shape": [2, 7, 2, 2],
                "rgb_sha256": sha256(view / "rgb.mp4"),
                "capture_latent_sha256": sha256(capture),
                "capture_encode_record": encoding,
                "clip_metadata_sha256": sha256(metadata),
            }
        )
        windows.append(
            {
                "bundle": f"{name}/{dataset.capture_bundle_name('white')}",
                "index": 0,
                "start": 0,
                "end": 49,
                "box_xyxy": [0, 0, 64, 64],
            }
        )
    manifest = tmp_path / dataset.CAPTURE_MANIFEST_NAME
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "one_step_raw_capture_target_latents",
                "pad_factor": 1.2,
                "edge": 64,
                "resolution": 64,
                "windows": windows,
            }
        )
    )
    membership = {
        "schema_version": 2,
        "kind": subset.KIND,
        "objective": "white",
        "corpus_root": str(tmp_path),
        "capture_manifest": {"sha256": sha256(manifest)},
        "sources": sources,
        "splits": {"train": ["1"], "held_out": ["2"]},
        "excluded": {},
        "content_pin_scope": "raw_and_latents",
        "requires_guide_latent": False,
    }
    membership["sha256"] = subset.membership_hash(membership)
    subset.validate_membership(membership)
    return membership


@pytest.fixture
def two_videos(current_membership: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    import json
    from types import SimpleNamespace

    import pytest

    from ltx_core.types import SpatioTemporalScaleFactors
    from scripts.onestep_avatar.corpus import precompute
    from scripts.onestep_avatar.training import engine

    membership = current_membership
    _write_guides(membership)
    path = tmp_path / "membership.json"
    path.write_text(json.dumps(membership))
    specification = SimpleNamespace(
        scale_factors=SpatioTemporalScaleFactors(8, 32, 32),
        caps=SimpleNamespace(latent_channels=2),
        sigmas=[0.725],
        paths=SimpleNamespace(transformer=lambda: tmp_path / "base.safetensors", video_vae=lambda: tmp_path / "vae"),
    )
    monkeypatch.setattr(engine.backbone, "resolve", lambda *_args: specification)
    monkeypatch.setattr(precompute, "file_fingerprint", lambda _path: "saved VAE identity")
    monkeypatch.setattr(
        engine.backbone,
        "identity",
        lambda *_args, **_kwargs: {
            "base_transformer_sha256": "a" * 64,
            "base_transformer_file": "base.safetensors",
            "base_variant": "dev",
            "model_key": "2.5",
        },
    )
    monkeypatch.setattr(engine, "build_transformer", lambda *_args: pytest.fail("preflight loaded a transformer"))
    monkeypatch.setattr(engine, "Accelerator", lambda *_args, **_kwargs: pytest.fail("preflight reached Accelerator"))
    return path, membership


def _membership() -> dict:

    from scripts.onestep_avatar.corpus import subset

    membership = {
        "schema_version": 2,
        "kind": subset.KIND,
        "objective": "white",
        "corpus_root": "/unused",
        "splits": {"train": ["actor"]},
        "sources": [
            {
                "relative_dir": "actor/view",
                "actor": "actor",
                "split": "train",
                "n_latent_frames": 17,
                "shape": [128, 17, 2, 2],
                "fps": 30,
            }
        ],
        "excluded": {},
    }
    membership["sha256"] = subset.membership_hash(membership)
    return membership


@pytest.fixture
def reference_cli_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    import json
    from contextlib import nullcontext
    from types import SimpleNamespace

    import pytest
    import torch

    from scripts.onestep_avatar import media
    from scripts.onestep_avatar.hashing import sha256
    from scripts.onestep_avatar.model import backbone

    """Real checked-master/producer flow; only pixel readers and VAE are controlled."""

    from scripts.onestep_avatar.corpus import dataset, precompute, subset

    view = tmp_path / "actor/view"
    view.mkdir(parents=True)
    for name in ("rgb.mp4", "bbox.npy", dataset.CAPTURE_MASK_NAME):
        (view / name).write_bytes(name.encode())
    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"controlled VAE")
    encoding = {
        "source": "actor/view",
        "objective": "white",
        "fps": 30,
        "vae_fingerprint": precompute.file_fingerprint(vae),
        "edge": 32,
        "box_xyxy": [0, 0, 32, 32],
        "pixel_frames": 9,
        "encode_contract_version": 1,
    }
    capture = view / dataset.capture_bundle_name("white")
    torch.save({"schema_version": 2, "master": torch.zeros(128, 2, 1, 1), **encoding}, capture)
    source = {
        "relative_dir": "actor/view",
        "actor": "actor",
        "split": "train",
        "shape": [128, 2, 1, 1],
        "fps": 30,
        "n_latent_frames": 2,
        "capture_encode_record": encoding,
        "capture_latent_sha256": sha256(capture),
        "rgb_sha256": sha256(view / "rgb.mp4"),
        "guide_sha256": None,
    }
    membership = {
        "schema_version": 2,
        "kind": subset.KIND,
        "objective": "white",
        "corpus_root": str(tmp_path),
        "sources": [source],
        "splits": {"train": ["actor"]},
    }
    crop_manifest = tmp_path / dataset.CAPTURE_MANIFEST_NAME
    crop_manifest.write_text(
        json.dumps(
            {
                "pad_factor": 1.2,
                "edge": 32,
                "windows": [{"bundle": "actor/view/ltx_vae_latent_white.pt", "box_xyxy": [0, 0, 32, 32]}],
            }
        )
    )
    membership["capture_manifest"] = {"sha256": sha256(crop_manifest)}
    membership["sha256"] = subset.membership_hash(membership)
    path = tmp_path / "membership.json"
    path.write_text(json.dumps(membership))
    model = SimpleNamespace(paths=SimpleNamespace(video_vae=lambda: vae))
    session = SimpleNamespace(model=model, decoder=lambda: nullcontext(None))
    pixels = torch.zeros(9, 3, 32, 32)
    monkeypatch.setattr(backbone, "resolve", lambda *_a: model)
    monkeypatch.setattr(media, "recorded_capture_rgb", lambda *_a: pixels)
    monkeypatch.setattr(media, "recorded_guide_rgb", lambda *_a: pytest.fail("capture-only D0 read guide RGB"))
    monkeypatch.setattr(media, "decode", lambda *_a: pixels)
    monkeypatch.setattr(media, "open_decoder_session", lambda *_a, **_kw: session)
    output = tmp_path / "references"
    arguments = [
        "--prepare-training-references",
        "--subset",
        str(path),
        "--source",
        "actor/view",
        "--encoded-frames",
        "2",
        "--guide-mode",
        "d0",
        "--gpu-id",
        "4",
        "--output",
        str(output),
    ]
    return arguments, source, session, view, path, output


@pytest.fixture
def causal_coverage(current_membership: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:  # noqa: PLR0915 -- complete saved-input fixture
    import json
    from types import SimpleNamespace

    import pytest
    import torch
    from safetensors.torch import save_file

    from ltx_core.types import SpatioTemporalScaleFactors
    from scripts.onestep_avatar import evaluate
    from scripts.onestep_avatar.corpus import dataset, precompute, subset
    from scripts.onestep_avatar.hashing import sha256
    from scripts.onestep_avatar.tests.support.contracts import A, B
    from scripts.onestep_avatar.training import checkpoints, config

    """Use real checked D1 masters/contracts; no native model or VAE work is allowed."""
    membership = current_membership
    for source in membership["sources"]:
        source["n_latent_frames"] = 18
        source["shape"] = [2, 18, 2, 2]
        view = tmp_path / source["relative_dir"]
        capture_path = view / dataset.capture_bundle_name("white")
        capture = torch.load(capture_path, weights_only=True)
        capture["master"] = torch.arange(144).reshape(2, 18, 2, 2).to(torch.bfloat16)
        capture["pixel_frames"] = 137
        torch.save(capture, capture_path)
        source["capture_latent_sha256"] = sha256(capture_path)
        source["capture_encode_record"] = {key: value for key, value in capture.items() if key != "master"}
        render = view / dataset.render_name("white")
        render.write_bytes(b"controlled guide RGB " + source["actor"].encode())
        source["guide_sha256"] = sha256(render)
        guide = {**capture, "master": capture["master"] + 1, "input_fingerprint": sha256(render)}
        guide_path = view / dataset.guide_bundle_name("white")
        torch.save(guide, guide_path)
        sidecar = view / dataset.render_metadata_name("white")
        sidecar.write_text(json.dumps({"compositing_version": dataset.GUIDE_COMPOSITING_VERSION, "objective": "white"}))
        source.update(
            guide_latent_sha256=sha256(guide_path),
            guide_sidecar_sha256=sha256(sidecar),
            guide_encode_record={key: value for key, value in guide.items() if key != "master"},
        )
    membership["requires_guide_latent"] = True
    membership["sha256"] = subset.membership_hash(membership)
    membership_path = tmp_path / "membership.json"
    membership_path.write_text(json.dumps(membership))
    specification = SimpleNamespace(
        scale_factors=SpatioTemporalScaleFactors(8, 32, 32),
        caps=SimpleNamespace(latent_channels=2, num_layers=2),
        sigmas=[0.725],
        paths=SimpleNamespace(
            transformer=lambda: tmp_path / "base.safetensors", video_vae=lambda: tmp_path / "vae.safetensors"
        ),
    )
    monkeypatch.setattr(evaluate.backbone, "resolve", lambda *_args: specification)
    monkeypatch.setattr(
        evaluate.backbone,
        "identity",
        lambda *_args, **_kwargs: {"base_transformer_sha256": "a" * 64, "base_transformer_file": "base.safetensors"},
    )
    monkeypatch.setattr(precompute, "file_fingerprint", lambda _path: "saved VAE identity")
    from scripts.prune.core import preflight, session

    monkeypatch.setattr(preflight, "check", lambda *_args, **_kwargs: pytest.fail("preflight opened native work"))
    monkeypatch.setattr(session, "Session", lambda *_args, **_kwargs: pytest.fail("preflight opened model session"))
    paths, contracts = {}, {}
    for label, span in (("e4", None), ("pilot", 7)):
        settings = config.RunSettings(
            "causal",
            membership_path,
            tmp_path / label,
            config.CausalSettings(span_latent_frames=span),
            variant="dev",
            objective="white",
            lora_rank=2,
            lora_alpha=2,
            steps=1 if label == "e4" else 60,
        )
        settings.base_identity = {"base_transformer_file": "base.safetensors", "base_transformer_sha256": "a" * 64}
        plan = config.build_frame_plan(settings, membership, specification.scale_factors)
        contract = checkpoints.make_contract(settings, membership, plan, settings.steps)
        contract["adapter"]["tensor_shapes"] = {A: [2, 4], B: [4, 2]}
        path = tmp_path / f"{label}.safetensors"
        save_file(
            {A: torch.ones(2, 4), B: torch.zeros(4, 2)}, path, metadata={checkpoints.CONTRACT_KEY: json.dumps(contract)}
        )
        paths[label], contracts[label] = path, contract
    noise = tmp_path / "noise.pt"
    torch.save(torch.zeros(1, 28, 2, dtype=torch.bfloat16), noise)
    arguments = [
        "--mode",
        "causal",
        "--subset",
        str(membership_path),
        "--output",
        str(tmp_path / "unused"),
        "--source",
        membership["sources"][0]["relative_dir"],
        "--variant",
        "dev",
        "--guide-mode",
        "d1",
        "--schedule",
        "0.725",
        "0",
        "--noise-file",
        str(noise),
    ]
    return {
        "arguments": arguments,
        "paths": paths,
        "contracts": contracts,
        "noise": noise,
        "membership": membership,
        "specification": specification,
    }


def evaluation_membership(root: Path, sources: list[str], *, objective: str = "white") -> dict:

    from scripts.onestep_avatar.corpus import subset

    record = {
        "schema_version": 2,
        "kind": subset.KIND,
        "objective": objective,
        "corpus_root": str(root),
        "splits": {"test": ["actor"]},
        "excluded": {},
        "sources": [
            {
                "relative_dir": source,
                "actor": "actor",
                "split": "test",
                "fps": 30,
                "shape": [128, 7, 2, 2],
                "n_latent_frames": 7,
            }
            for source in sources
        ],
    }
    record["sha256"] = subset.membership_hash(record)
    return record


@pytest.fixture
def original_artifacts(tmp_path: Path) -> tuple:
    import copy
    import json
    from types import SimpleNamespace

    import torch

    from ltx_core.types import SpatioTemporalScaleFactors
    from scripts.onestep_avatar.corpus import dataset, subset
    from scripts.onestep_avatar.execution import software
    from scripts.onestep_avatar.experiments import training_slice_check as slice_check
    from scripts.onestep_avatar.hashing import sha256
    from scripts.onestep_avatar.model import adapters
    from scripts.onestep_avatar.tests.support.model import _model
    from scripts.onestep_avatar.tests.support.training import CPUAccelerator
    from scripts.onestep_avatar.training import checkpoints, config, resources

    root = tmp_path / "original"
    settings = config.parse_settings(
        [
            "--mode",
            "bidirectional",
            "--subset",
            str(tmp_path / "membership.json"),
            "--output",
            str(root),
            "--guide-mode",
            "d0",
            "--objective",
            "white",
            "--variant",
            "dev",
            "--span-latent-frames",
            "7",
            "--lora-rank",
            "2",
            "--chains-per-rank",
            "2",
            "--steps",
            "1",
            "--save-initial",
            "--save-update-state",
        ]
    )
    settings.world_size = 4
    settings.base_identity = {
        "model_key": "2.5",
        "base_variant": "dev",
        "base_transformer_file": "base.safetensors",
        "base_transformer_fingerprint": "b" * 16,
        "base_transformer_sha256": "b" * 64,
    }
    membership = {
        "schema_version": 2,
        "kind": subset.KIND,
        "objective": "white",
        "splits": {"train": ["actor"]},
        "sources": [
            {
                "relative_dir": "actor/view",
                "actor": "actor",
                "split": "train",
                "shape": [8, 7, 2, 2],
                "n_latent_frames": 7,
            }
        ],
    }
    membership["sha256"] = subset.membership_hash(membership)
    plan = config.select_frame_plan(settings, membership, SpatioTemporalScaleFactors(8, 32, 32))
    corpus = tmp_path / "corpus"
    capture = corpus / "actor/view" / dataset.capture_bundle_name("white")
    capture.parent.mkdir(parents=True)
    torch.save({"master": torch.ones(8, 7, 2, 2)}, capture)
    store = SimpleNamespace(root=corpus, membership=membership)
    accelerator_config = tmp_path / "fsdp.yaml"
    accelerator_config.write_text("mixed_precision: bf16\n")
    original = {"sha256": "d" * 64, "processes": 4, "accelerate_config": str(accelerator_config)}
    profile = software.capture("training", "bidirectional")
    budget_path = tmp_path / "budget.json"
    _write_json(
        budget_path,
        {
            "wall_seconds_per_phase": 1800,
            "memory_limit_allocated_bytes": 48000000000,
            "tolerance": slice_check.training_update_check.TOLERANCE,
        },
    )
    budget = resources.read_budget(budget_path)
    saved = {
        "base_identity": copy.deepcopy(settings.base_identity),
        "membership_sha256": membership["sha256"],
        "frame_plan_sha256": plan["sha256"],
        "software": profile,
        "producer_source_sha256": profile["sources"]["scripts/onestep_avatar/training/engine.py"],
        "queue_launch": {"original": "already validated by prepare"},
        "runtime": {"original": "four ranks"},
    }
    _write_json(root / "config.json", saved)
    _write_json(root / "frame_plan.json", plan)
    model = adapters.attach(_model().bfloat16(), rank=2, alpha=settings.lora_alpha, target="attn", init_seed=11)
    for step in (0, 1):
        if step == 1:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if ".lora_B." in name:
                        parameter.fill_(0.0001)
        metadata = {checkpoints.CONTRACT_KEY: json.dumps(checkpoints.make_contract(settings, membership, plan, step))}
        path = checkpoints.save_lora(
            model, CPUAccelerator(), root / "checkpoints", step, metadata, verify_noop=step == 0
        )
        marker = {
            "schema_version": 2,
            "step": step,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "state": "complete",
            "queue_job_sha256": original["sha256"],
            **{key: saved[key] for key in ("producer_source_sha256", "software", "queue_launch", "runtime")},
            "resource_budget": budget,
            "resource_evidence": {},
            "consumer_trace_evidence": {},
            "training_record": {
                "config_sha256": sha256(root / "config.json"),
                "frame_plan_sha256": sha256(root / "frame_plan.json"),
            },
        }
        _write_json(path.with_suffix(".complete.json"), marker)
    visits = slice_check.training_update_check.first_update_visits(settings, plan["samples"], 4)
    for rank in range(4):
        selected = [visit for visit in visits if visit["rank"] == rank]
        row = {
            "rank": rank,
            "step": 1,
            "sigma0": selected[0]["sigma"],
            "samples": [
                {
                    "source": visit["sample"]["source"],
                    "ranges": visit["sample"]["ranges"],
                    "noise_seed": visit["noise_seed"],
                }
                for visit in selected
            ],
            "call_counts": {"prime": 0, "denoise": 2, "backward": 2, "refresh": 0},
        }
        (root / f"metrics_rank{rank}.jsonl").write_text(json.dumps(row) + "\n")
    return original, settings, saved, store, plan, budget


def _write_guides(membership: dict) -> None:
    from scripts.onestep_avatar.corpus import dataset, subset

    """Give each saved view a current guide encode, render and compositing sidecar."""
    root = Path(membership["corpus_root"])
    for source in membership["sources"]:
        view = root / source["relative_dir"]
        render = view / dataset.render_name("white")
        render.write_bytes(b"render" + source["actor"].encode())
        sidecar = view / dataset.render_metadata_name("white")
        sidecar.write_text(json.dumps({"compositing_version": dataset.GUIDE_COMPOSITING_VERSION, "objective": "white"}))
        capture = torch.load(view / dataset.capture_bundle_name("white"), weights_only=True)
        guide = {**capture, "master": capture["master"] + 1, "input_fingerprint": sha256(render)}
        guide_path = view / dataset.guide_bundle_name("white")
        torch.save(guide, guide_path)
        source.update(
            guide_sha256=sha256(render),
            guide_latent_sha256=sha256(guide_path),
            guide_sidecar_sha256=sha256(sidecar),
            guide_encode_record={key: value for key, value in guide.items() if key != "master"},
        )
    membership["requires_guide_latent"] = True
    membership["sha256"] = subset.membership_hash(membership)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def build_runtime_membership(tmp_path: Path, seed: int) -> tuple[dict, Path]:
    """Keep identical runtime/archive corpus bytes, varying only the saved seed."""
    from scripts.onestep_avatar.corpus import dataset, subset

    view = tmp_path / "corpus" / "actor" / "view"
    view.mkdir(parents=True)
    path = view / dataset.capture_bundle_name("white")
    generator = torch.Generator().manual_seed(seed)
    capture = torch.randn(8, 7, 2, 2, generator=generator)
    bundle = {
        "schema_version": 2,
        "master": capture,
        "fps": 30.0,
        "objective": "white",
        "source": "actor/view",
        "vae_fingerprint": "runtime VAE identity",
    }
    torch.save(bundle, path)
    membership = {
        "schema_version": 2,
        "kind": subset.KIND,
        "corpus_root": str(tmp_path / "corpus"),
        "objective": "white",
        "splits": {"train": ["actor"]},
        "sources": [
            {
                "relative_dir": "actor/view",
                "actor": "actor",
                "split": "train",
                "n_latent_frames": 7,
                "shape": [8, 7, 2, 2],
                "fps": 30.0,
                "capture_latent_sha256": sha256(path),
                "capture_encode_record": {
                    k: bundle[k] for k in ("schema_version", "fps", "objective", "source", "vae_fingerprint")
                },
            }
        ],
    }
    membership["sha256"] = subset.membership_hash(membership)
    list_path = tmp_path / "membership.json"
    list_path.write_text(json.dumps(membership))
    return membership, list_path
