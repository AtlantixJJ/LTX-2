import numpy as np
import pytest
import torch

from scripts.prune.data import resume_learned_heads as resume
from scripts.prune.learn import checkpoints
from scripts.prune.learn.adapters import ProjectionAdapter
from scripts.prune.learn.gates import HeadGates
from scripts.prune.learn.heads import train_identity


def fixture(tmp_path, monkeypatch):
    old_root = tmp_path / "old"
    plan = old_root / "plan.md"
    plan.parent.mkdir()
    plan.write_text("Original loss")
    draw_path = old_root / "draws.json"
    draw_path.write_text("[]")
    code = {"sha256": "code", "files": {"unchanged.py": "same"}}
    monkeypatch.setattr(resume, "code_identity", lambda: code)
    study = {
        "format": "learned_head_study_v1",
        "root": str(old_root),
        "config": {"lambda_max": 0.001, "soft_updates": 700},
        "code": code,
        "software": {},
        "splits": {"learning": []},
        "text_context": {"tensor_sha256": "context"},
        "components": {},
        "plan": {"path": str(plan), "sha256": resume.file_sha256(plan)},
        "draws": {"42:ordinary": {"path": str(draw_path), "sha256": resume.file_sha256(draw_path)}},
    }
    study["identity"] = checkpoints.digest(study)
    study_path = old_root / "study.json"
    checkpoints.publish_json(study_path, study)
    torch.manual_seed(7)
    np.random.seed(7)
    adapter = ProjectionAdapter(torch.nn.Linear(4, 4), rank=2)
    gates = HeadGates(1, 4, [[2, 4]])
    ao = torch.optim.AdamW([adapter.A, adapter.B], lr=0.001)
    go = torch.optim.AdamW([gates.alpha], lr=0.01)
    (adapter(torch.randn(1, 4)).square().mean() + gates.alpha.sigmoid().mean()).backward()
    ao.step()
    go.step()
    identity = train_identity(study, "learned_joint", 42, 2, study["draws"]["42:ordinary"])
    checkpoint = old_root / "training/learned_joint/replicate_42/checkpoints/update_0001.pt"
    checkpoints.save(
        checkpoint,
        identity=identity,
        update=1,
        phase="soft",
        gates=gates,
        adapters={"0.attn1.v": adapter},
        adapter_optimizer=ao,
        gate_optimizer=go,
    )
    inputs = old_root / "inputs"
    inputs.mkdir()
    epsilon = inputs / "epsilon.pt"
    epsilon.write_bytes(b"saved epsilon")
    manifest = inputs / "manifest.json"
    checkpoints.publish_json(manifest, {"videos": [{"artifacts": {"epsilon": "epsilon.pt"}}]})
    draw = {
        "input_manifest": str(manifest),
        "input_manifest_sha256": resume.file_sha256(manifest),
        "epsilon_sha256": resume.file_sha256(epsilon),
    }
    checkpoints.publish_json(
        checkpoint.parent.parent / "updates/0001.json", {"study": study["identity"], "update": 1, "draws": [draw, draw]}
    )
    checkpoints.publish_json(
        old_root / "checks/feasibility.json", {"study": study["identity"], "method": "four_rank_fsdp", "passed": True}
    )
    checkpoints.publish_json(old_root / "provenance/environment.json", {})
    return study_path, checkpoint, plan, epsilon


def test_loss_amendment_preserves_parameters_moments_rng_and_draw_cursor(tmp_path, monkeypatch):
    previous, checkpoint, plan, _ = fixture(tmp_path, monkeypatch)
    source = checkpoints.load(checkpoint)
    study, record = resume.prepare(previous, checkpoint, plan, tmp_path / "new")
    state = checkpoints.load(record["checkpoint"])
    assert state["identity"] != source["identity"]
    assert resume.unchanged_state_hash(source) == resume.unchanged_state_hash(state)
    assert state["update"] == 1 and state["draw_cursor"] == 2
    assert study["config"]["lambda_max"] == 1 and study["config"]["sparsity_penalty"] == "relu_logits"
    assert record["next_update"] == 2 and record["next_lambda"] == 2 / 300
    assert len(state["lineage"]["prefix_receipts"]) == 1
    assert checkpoints.load(checkpoint)["identity"] == source["identity"]
    assert int(state["gate_optimizer"]["state"][0]["step"]) == 1


def test_amendment_rejects_changed_noise_or_unrelated_code(tmp_path, monkeypatch):
    previous, checkpoint, plan, epsilon = fixture(tmp_path, monkeypatch)
    epsilon.write_bytes(b"different epsilon")
    with pytest.raises(ValueError, match="epsilon changed"):
        resume.prepare(previous, checkpoint, plan, tmp_path / "new")
    monkeypatch.setattr(resume, "code_identity", lambda: {"files": {"unchanged.py": "changed"}})
    with pytest.raises(ValueError, match="unrelated production"):
        resume.prepare(previous, checkpoint, plan, tmp_path / "new")
