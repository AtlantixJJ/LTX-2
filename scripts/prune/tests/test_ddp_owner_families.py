"""Each DDP family module exposes the complete hook set the shared owner calls."""

import pytest

from scripts.prune.learn import ffn, joint

HOOKS = (
    "state",
    "label",
    "script",
    "native_fixture",
    "native_prepare",
    "native_first_checks",
    "native_freeze",
    "check_start",
    "after_update",
    "receipt_fields",
    "reviews",
    "checkpoint_due",
    "after_checkpoint",
    "monitor_due",
    "final_update",
)
STATE = ("JointStep", "optimizers", "update", "state", "save", "read", "restore")


@pytest.mark.parametrize("family", [ffn, joint])
def test_family_hooks_and_state_owner(family):
    assert [name for name in HOOKS if not hasattr(family, name)] == []
    assert [name for name in STATE if not hasattr(family.state, name)] == []


def test_ffn_and_v3_boundaries():
    cfg = dict(selection_updates=1000, head_soft_updates=700, monitor_interval=50, checkpoint_interval=50)

    class Ctx:
        pass

    ctx = Ctx()
    ctx.cfg = cfg
    assert ffn.checkpoint_due(ctx, 700, 1300) and ffn.checkpoint_due(ctx, 1300, 1300)
    assert not ffn.checkpoint_due(ctx, 701, 1300)
    assert ffn.monitor_due(ctx, 1000, 1300) and not ffn.monitor_due(ctx, 1001, 1300)
    assert not joint.monitor_due(ctx, 100, 1700) and joint.monitor_due(ctx, 150, 1700)
    with pytest.raises(ValueError, match="pilot"):
        ffn.check_start({}, None, cfg, pilot=True, max_updates=10)


def test_memory_releases_cache_before_reading_free_memory(monkeypatch):
    from types import SimpleNamespace

    import torch

    from scripts.prune.learn import ddp_owner

    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("synchronize"))
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: calls.append("mem_get_info") or (3 << 30, 48 << 30)
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2)
    runtime = SimpleNamespace(rank=0, device="cuda:0", gather=lambda row: [row])
    assert ddp_owner.memory(runtime, 2 << 30)[0]["free"] == 3 << 30
    assert calls == ["empty_cache", "synchronize", "mem_get_info"]



def test_pending_joint_reviews_block_legacy_and_descriptive_names(tmp_path):
    import json

    import pytest

    from scripts.prune.learn import joint

    folder = tmp_path / "reviews"
    folder.mkdir()
    for name in ("G1_0100.json", "G2_1500.json", "allocation_review_0100.json", "alignment_review_1500.json"):
        path = folder / name
        path.write_text(json.dumps(dict(triggers=["monitor_above_three_percent"])))
        with pytest.raises(ValueError, match="review gate remains pending"):
            joint.check_start({}, tmp_path, {}, pilot=True, max_updates=50)
        path.unlink()
    joint.check_start({}, tmp_path, {}, pilot=True, max_updates=50)


def test_new_joint_allocation_review_uses_descriptive_name(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import torch

    from scripts.prune.learn import ddp_owner, joint

    (tmp_path / "monitor").mkdir()
    (tmp_path / "monitor/update_0000.json").write_text(json.dumps(dict(capture_mse=1.0)))
    (tmp_path / "training/events").mkdir(parents=True)
    for number in (50, 100):
        torch.save(dict(families=dict(channels=dict(global_cap_bound=False))), tmp_path / f"training/events/{number:04d}.pt")
    monkeypatch.setattr(ddp_owner, "monitor", lambda *a, **k: dict(capture_mse=1.0))
    step = SimpleNamespace(event=None, policy=SimpleNamespace(channels=SimpleNamespace(kept=torch.ones(2, 128))))
    ctx = SimpleNamespace(step=step, root=tmp_path, cfg=dict(ffn_width=128), study=dict(identity="fixture"), current=None,
                          runtime=SimpleNamespace(primary=True, gather=lambda value: [value]))
    assert not joint.reviews(ctx, 100)
    receipt = json.loads((tmp_path / "reviews/allocation_review_0100.json").read_text())
    assert receipt["gate"] == "allocation_review" and not receipt["triggers"]
