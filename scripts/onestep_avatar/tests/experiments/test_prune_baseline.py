"""CPU controls for queued pruning completion; native quality remains a separate gate."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from scripts.onestep_avatar import LTX_ROOT
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.execution import completion as queue_completion
from scripts.onestep_avatar.execution import jobs as queue_jobs
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import prune_baseline as study
from scripts.onestep_avatar.experiments import spec_protocol
from scripts.onestep_avatar.hashing import sha256
from scripts.prune.core import provenance
from scripts.prune.data import prompt_cache
from scripts.prune.evaluate import whole_clip_baseline as producer
from scripts.prune.tests.test_whole_clip_baseline import SCALE, _Caps, _Session, _vae, _views


def write_spec(path: Path, arguments: list[str]) -> Path:
    path.write_text(json.dumps({"schema_version": 1, "protocol": study.SELECTOR, "arguments": arguments}))
    return path


def outer(spec: Path, output: Path) -> list[str]:
    return ["--spec", str(spec), "--output", str(output)]


class _BoundSession(_Session):
    def __init__(self, vae: Path, transformer: Path) -> None:
        super().__init__(vae)
        self._transformer = transformer
        self._vae = vae
        self.model.caps = _BoundCaps()

    def stamp(self, *, dtype: str) -> dict:
        return {
            "model_key": "2.5",
            "model_version": [2, 5],
            "transformer_path": str(self._transformer.resolve()),
            "transformer_fingerprint": provenance.checkpoint_fingerprint(self._transformer),
            "video_vae_path": str(self._vae.resolve()),
            "scale_factors": list(SCALE),
            "script": producer.SCRIPT,
            "dtype": dtype,
        }


@dataclass(frozen=True)
class _BoundCaps(_Caps):
    num_layers: int = 2


@pytest.fixture
def saved_pruning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Run the real tiny numerical producer once, then prohibit any model execution."""
    view = _views(tmp_path / "corpus")[0]
    vae = _vae(tmp_path)
    transformer, text = tmp_path / "transformer.safetensors", tmp_path / "text.safetensors"
    for path in (transformer, text):
        save_file({"weight": torch.zeros(1)}, path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fixed tiny baseline text")
    components = SimpleNamespace(
        key="2.5",
        version=(2, 5),
        scale_factors=SCALE,
        caps=_BoundCaps(),
        sigmas=[1.0, 0.909375, 0.725, 0.421875, 0.0],
        paths=SimpleNamespace(transformer=lambda: transformer, text_encoder=lambda: text, video_vae=lambda: vae),
    )
    monkeypatch.setattr(study.backbone, "resolve", lambda *_args: components)
    session = _BoundSession(vae, transformer)
    cache = tmp_path / "fixed_context.pt"
    torch.save(session.context, cache)
    monkeypatch.setattr(prompt_cache, "cache_path", lambda *_args: cache)
    monkeypatch.setattr(producer, "open_session", lambda *_args, **_kwargs: session)
    spec = write_spec(
        tmp_path / "scientific.json",
        ["--view", str(view), "--sigmas", "0.725", "--raw-only", "--arms", "d0", "d1", "--prompt-file", str(prompt)],
    )
    output = tmp_path / "queued"
    assert study.main(outer(spec, output)) == 0
    args = study.read_spec(spec, output)
    monkeypatch.setattr(producer, "open_session", lambda *_args, **_kwargs: pytest.fail("completion opened a model"))
    monkeypatch.setattr(producer, "rollout", lambda *_args, **_kwargs: pytest.fail("completion regenerated output"))
    return (
        args,
        spec,
        output,
        {
            "capture": view / dataset.capture_bundle_name("white"),
            "guide": view / dataset.guide_bundle_name("white"),
            "transformer": transformer,
            "text": text,
            "vae": vae,
            "prompt": prompt,
            "cache": cache,
        },
    )


def job_for(spec: Path, output: Path) -> dict:
    return queue_jobs.prepare_job(
        {
            "id": "pruning",
            "kind": "experiment",
            "experiment": study.SELECTOR,
            "spec": str(spec),
            "spec_sha256": sha256(spec),
            "arguments": outer(spec, output),
            "output": str(output),
            "completion": {"manifest": str(output / "manifest.json")},
        },
        spec.parent,
    )


def test_completion_uses_saved_raw_outputs_and_pins_all_current_inputs(saved_pruning: tuple) -> None:
    args, spec, root, inputs = saved_pruning
    assert study.verify_completion(spec, root)
    record = json.loads((root / "acceptance.json").read_text())
    assert record["inputs"] == {str(path.resolve()): sha256(path) for path in [spec, *inputs.values()]}
    assert record["settings"] == spec_protocol.settings(args)
    assert record["software"] == software.capture("evaluation", "causal", extra_sources=study.EXTRA_SOURCES)
    expected = {root / "manifest.json", *(root / name for name in spec_protocol.inventory(root))}
    assert set(study.evidence_paths(spec, root)) == expected
    assert root / "baseline/manifest.json" in expected
    assert len([path for path in expected if path.suffix == ".pt"]) == 3


def test_public_queue_route_binds_parser_launch_and_saved_completion(saved_pruning: tuple) -> None:
    _, spec, root, _ = saved_pruning
    job = job_for(spec, root)
    assert queue_jobs.EXPERIMENTS[study.SELECTOR] == "scripts.onestep_avatar.experiments.prune_baseline"
    command, environment = queue_jobs.job_command(job, (5,))
    assert command[2] == queue_jobs.EXPERIMENTS[study.SELECTOR]
    assert command[-2:] == ["--gpu-id", "0"]
    assert environment["CUDA_VISIBLE_DEVICES"] == "5"
    assert queue_completion.verify_completion(job)
    assert study.verify_completion(spec, root)


@pytest.mark.parametrize("asset", ["capture", "guide", "transformer", "text", "vae", "prompt", "cache", "spec"])
def test_changed_scientific_input_refuses_saved_completion(saved_pruning: tuple, asset: str) -> None:
    _, spec, root, inputs = saved_pruning
    path = spec if asset == "spec" else inputs[asset]
    path.write_bytes(path.read_bytes() + b"changed after production")
    with pytest.raises((ValueError, OSError), match=r"differ|changed|JSON|Extra data"):
        study.verify_completion(spec, root)


def test_changed_current_source_refuses_even_with_original_saved_artifacts(
    saved_pruning: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, spec, root, _ = saved_pruning
    owner = LTX_ROOT / "scripts/prune/evaluate/whole_clip_baseline.py"
    original = software.sha256
    monkeypatch.setattr(
        software, "sha256", lambda path: "f" * 64 if Path(path).resolve() == owner.resolve() else original(path)
    )
    with pytest.raises(ValueError, match="software differ"):
        study.verify_completion(spec, root)


@pytest.mark.parametrize("change", ["missing", "changed", "extra", "symlink"])
def test_inventory_refuses_missing_changed_extra_or_escaped_results(saved_pruning: tuple, change: str) -> None:
    _, spec, root, _ = saved_pruning
    artifact = next((root / "baseline").glob("*_d0.pt"))
    if change == "missing":
        artifact.unlink()
    elif change == "changed":
        artifact.write_bytes(artifact.read_bytes() + b"altered")
    elif change == "extra":
        (root / "extra.bin").write_bytes(b"unrecorded")
    else:
        (root / "escaped.pt").symlink_to(artifact)
    with pytest.raises(ValueError, match=r"artifacts differ|artifact escapes"):
        study.verify_completion(spec, root)


@pytest.mark.parametrize(
    "field",
    [
        "model_key",
        "model_version",
        "transformer_path",
        "transformer_fingerprint",
        "video_vae_path",
        "video_vae_fingerprint",
        "scale_factors",
        "capabilities",
        "context_hash",
        "context_shape",
    ],
)
def test_rehashed_result_still_requires_registered_model_coordinates_and_exact_context(
    saved_pruning: tuple, field: str
) -> None:
    _, spec, root, _ = saved_pruning
    path = root / "baseline/manifest.json"
    manifest = json.loads(path.read_text())
    if field == "model_version":
        manifest["model"][field] = [2, 3]
    elif field == "model_key":
        manifest["model"][field] = "2.3"
    elif field.endswith("_path"):
        manifest["model"][field] = str(root / "different.safetensors")
    elif field.endswith("_fingerprint"):
        manifest["model"][field] = "incorrect component identity"
    elif field == "scale_factors":
        manifest["model"][field][1] = 16
        manifest["geometry"][field][1] = 16
    elif field == "capabilities":
        manifest[field]["num_layers"] = 3
    elif field == "context_hash":
        manifest["text_context"]["sha256"] = "f" * 64
    else:
        manifest["text_context"]["shape"][1] += 1
    path.write_text(json.dumps(manifest))
    outer_path = root / "manifest.json"
    outer_manifest = json.loads(outer_path.read_text())
    outer_manifest["artifacts"] = spec_protocol.inventory(root)
    outer_path.write_text(json.dumps(outer_manifest))
    with pytest.raises(ValueError, match=r"scientific completion|component identity|text context"):
        study.verify_completion(spec, root)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--view", "/one", "--raw-only", "--arms", "d1"],
        ["--view", "/one", "--raw-only", "--seeds", "1", "2"],
        ["--view", "/one", "--raw-only", "--objective", "bg"],
        ["--view", "/one", "--raw-only", "--variant", "dev", "--steps", "8"],
        ["--view", "/one", "--raw-only", "--model", "2.3"],
        ["--view", "/one", "--raw-only", "--cfg", "3"],
        ["--view", "/one", "--raw-only", "--stg", "1"],
        ["--view", "/one", "--raw-only", "--negative-prompt", "changed"],
        ["--view", "/one", "--raw-only", "--context-latent-frames", "4"],
        ["--view", "/one", "--raw-only", "--sigmas", "0.725", "0.725"],
        ["--view", "/one", "--raw-only", "--sigmas", "nan"],
        ["--view", "/one", "--raw-only", "--sigmas", "0"],
        ["--view", "/one"],
        ["--view", "relative", "--raw-only"],
        ["--view", "/one", "--raw-only", "--output", "/overridden"],
    ],
)
def test_restricted_spec_refuses_before_model_input_reads_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    spec = write_spec(tmp_path / "spec.json", arguments)
    output = tmp_path / "output"
    monkeypatch.setattr(study.backbone, "resolve", lambda *_args: pytest.fail("invalid spec resolved model inputs"))
    monkeypatch.setattr(producer, "open_session", lambda *_args, **_kwargs: pytest.fail("invalid spec loaded model"))
    with pytest.raises(ValueError, match=r"requires|paths|override"):
        study.read_spec(spec, output)
    assert not output.exists()


def test_unknown_selector_refuses_without_importing_or_launching_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, output = tmp_path / "missing.json", tmp_path / "output"
    raw = {
        "id": "invalid",
        "kind": "experiment",
        "experiment": "prune_baseline_unreviewed",
        "spec": str(spec),
        "spec_sha256": "a" * 64,
        "arguments": outer(spec, output),
        "output": str(output),
        "completion": {"manifest": str(output / "manifest.json")},
    }
    monkeypatch.setattr(
        queue_jobs.importlib, "import_module", lambda *_args: pytest.fail("unknown selector imported owner")
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("unknown selector launched child"))
    with pytest.raises(ValueError, match="selector is unknown"):
        queue_jobs.prepare_job(raw, tmp_path)
    assert not output.exists()
