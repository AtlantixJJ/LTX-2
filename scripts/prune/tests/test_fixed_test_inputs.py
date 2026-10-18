"""Frozen full-source test identity rejects monitor caches and changed evidence."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.prune.data import fixed_test_inputs as owner
from scripts.prune.data.checkpoint_pairs import TEST_ACTORS
from scripts.prune.learn import checkpoints


def fixture(tmp_path, monkeypatch):
    sources = []
    for actor in TEST_ACTORS:
        for view_id in range(2):
            view = tmp_path / "corpus" / actor / "views" / str(view_id)
            view.mkdir(parents=True)
            for name in ("capture.pt", "rgb.mp4", "mask.mp4"):
                (view / name).write_bytes(f"fixture {actor} {view_id} {name}".encode())
            (view.parent.parent / "meta.json").write_text("{}")
            sources.append(
                dict(
                    actor=actor,
                    relative_dir=f"{actor}/views/{view_id}",
                    resolved_dir=str(view),
                    capture=str(view / "capture.pt"),
                    capture_latent_sha256=owner.pin(view / "capture.pt")["sha256"],
                    capture_encode_record=dict(box_xyxy=[0, 0, 1024, 1024], objective="white"),
                    n_latent_frames=28,
                    shape=[128, 28, 32, 32],
                    fps=30.0,
                )
            )
    context = torch.zeros(1, 2, 4)
    sigmas = [owner.SIGMAS[0], 0.94, 0.93, 0.92, 0.91, 0.88, 0.83, owner.SIGMAS[1], 0.66, owner.SIGMAS[2]]
    study = dict(
        identity="fixture",
        root=str(tmp_path / "study"),
        splits=dict(test=sources, test_actors=TEST_ACTORS, learning_actors=["training"]),
        config=dict(progressive_policy=dict(method="clip01_ste3_reset_v2"), sigmas=sigmas, numerics=dict(tf32=False)),
        text_context=dict(tensor_sha256=checkpoints.tensor_hash(context)),
        components={"transformer": "fixture"},
        software={"torch": "fixture"},
        code=dict(sha256="fixture"),
    )

    def input_case(study, current, source, seed, sigma):
        directory = tmp_path / "inputs" / source["relative_dir"]
        directory.mkdir(parents=True, exist_ok=True)
        epsilon = torch.ones(1, 28, 2)
        capture = torch.zeros_like(epsilon)
        checkpoints.publish_tensor(directory / "epsilon.pt", epsilon)
        native = dict(videos=[dict(artifacts=dict(epsilon="epsilon.pt"))], geometry=dict(latent_frames=28))
        checkpoints.publish_json(directory / "manifest.json", native)
        return SimpleNamespace(tokens_per_latent_frame=1), None, capture[:, :1], capture, epsilon, directory, native

    monkeypatch.setattr(owner, "input_case", input_case)
    return study, SimpleNamespace(context=context)


def test_freeze_and_replay_exact_input_manifest(tmp_path, monkeypatch):
    study, current = fixture(tmp_path, monkeypatch)
    pin = owner.freeze(study, current)
    manifest = json.loads(open(pin["path"]).read())
    assert len(manifest["cases"]) == 36 and len(manifest["inputs"]) == 12
    assert owner.validate_manifest(manifest, study, expected_digest=pin["content_sha256"]) == manifest
    assert owner.freeze(study, current) == pin
    # Actual saved epsilon bytes, rather than seed labels, remain authoritative.
    noise = manifest["inputs"][0]["epsilon"]["path"]
    with open(noise, "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="epsilon bytes changed"):
        owner.validate_manifest(manifest, study, expected_digest=pin["content_sha256"])


@pytest.mark.parametrize("change", ["actor", "noise", "GT", "schedule", "context", "missing", "four_monitors"])
def test_changed_case_or_development_cache_rejected(tmp_path, monkeypatch, change):
    study, current = fixture(tmp_path, monkeypatch)
    pin = owner.freeze(study, current)
    manifest = json.loads(open(pin["path"]).read())
    changed = copy.deepcopy(manifest)
    if change == "actor":
        changed["actors"][0] = "35"
    elif change == "noise":
        changed["cases"][0]["noise_seed"] = 2043
    elif change == "GT":
        changed["inputs"][0]["capture"]["sha256"] = "changed"
    elif change == "schedule":
        changed["schedules"][0] = [0.9, 0.0]
    elif change == "context":
        changed["context_tensor_sha256"] = "changed"
    elif change == "missing":
        changed["cases"].pop()
    else:
        changed = dict(actors=["35", "25", "84", "96"], rows=changed["cases"][:12])
    with pytest.raises(ValueError, match="fixed-test identity"):
        owner.validate_manifest(changed, study, expected_digest=pin["content_sha256"], verify_files=False)


def test_f18_and_training_actor_cannot_enter_test(tmp_path, monkeypatch):
    study, current = fixture(tmp_path, monkeypatch)
    study["splits"]["test"][0]["n_latent_frames"] = 18
    with pytest.raises(ValueError, match="full F28"):
        owner.freeze(study, current)
    study["splits"]["test"][0]["n_latent_frames"] = 28
    study["splits"]["learning_actors"].append(TEST_ACTORS[0])
    with pytest.raises(ValueError, match="no training actor"):
        owner.freeze(study, current)
