"""Fresh-state replacement and exact-step comparison guards."""

import copy

import pytest
import torch

from scripts.prune.data.checkpoint_pairs import checkpoint_record
from scripts.prune.data.clipped_reset_study import initialize, pilot_config
from scripts.prune.data.resume_learned_heads import unchanged_state_hash
from scripts.prune.learn import checkpoints
from scripts.prune.learn.progressive import ProgressivePolicy


def source_zero():
    adapters = {
        f"{b}.attn1.{p}.{k}": torch.randn(2, 2) if k == "A" else torch.zeros(2, 2)
        for b in range(48)
        for p in ("v", "out")
        for k in ("A", "B")
    }
    return dict(
        update=0,
        draw_cursor=0,
        phase="soft",
        draws_per_update=4,
        alpha=torch.full((48, 32), 3.0),
        mask=None,
        adapters=adapters,
        adapter_optimizer=dict(state={}, param_groups=[]),
        gate_optimizer=dict(state={}, param_groups=[]),
        rng=checkpoints.rng_state(),
        identity=dict(study="fixture"),
        lineage={},
    )


def test_initialization_changes_only_explicit_scores_and_gate_policy():
    source = source_zero()
    before = unchanged_state_hash(source)
    cfg = dict(gate_lr=0.01, betas=[0.9, 0.999], adam_epsilon=1e-8)
    from scripts.prune.learn.clipped_reset import DEFAULTS

    cfg["progressive_policy"] = dict(DEFAULTS)
    result = initialize(source, cfg)
    assert before == unchanged_state_hash(source)
    assert torch.equal(result["alpha"], torch.ones(48, 32))
    assert not result["gate_optimizer"]["state"]
    assert len(result["adapters"]) == 192
    assert result["rng"]["python"] == source["rng"]["python"]
    for key in ("alpha", "gate_optimizer", "progressive"):
        source.pop(key, None)
        result.pop(key, None)
    assert unchanged_state_hash(source) == unchanged_state_hash(result)


@pytest.mark.parametrize(
    "alteration", ["update", "draw_cursor", "B", "adapter_optimizer", "gate_optimizer", "alpha", "mask"]
)
def test_initialization_rejects_trained_or_masked_source(alteration):
    source = source_zero()
    if alteration == "B":
        source["adapters"]["0.attn1.v.B"].fill_(1)
    elif alteration in ("adapter_optimizer", "gate_optimizer"):
        source[alteration]["state"] = {0: dict(step=torch.tensor(1))}
    elif alteration == "alpha":
        source["alpha"].fill_(1)
    elif alteration == "mask":
        source["mask"] = torch.ones(48, 32)
    else:
        source[alteration] = 1
    with pytest.raises(ValueError, match="untrained|B matrices"):
        initialize(source, {})


@pytest.mark.parametrize("step", [0, 500, 550, 800])
def test_exact_step_pair_record_and_cursor_rejection(tmp_path, monkeypatch, step):
    alpha = torch.full((48, 32), 3.0)
    policy = ProgressivePolicy(alpha)
    # A constant trajectory keeps all original heads alive and produces valid
    # undefined-regret receipts without changing the historical sigmoid gates.
    for update in range(1, step + 1):
        policy.record(update, alpha)
    study = dict(identity="previous", draws={"42:ordinary": dict(sha256="draws")})
    state = dict(
        identity=dict(study="previous", arm="learned_progressive", seed=42, draws="draws"),
        update=step,
        draw_cursor=step * 4,
        draws_per_update=4,
        phase="soft",
        alpha=alpha,
        progressive=policy.state_dict(),
        adapters={},
        adapter_optimizer=dict(state={i: dict(step=torch.tensor(step)) for i in range(192)} if step else {}),
    )
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"pinned checkpoint fixture")
    path.with_suffix(".json").write_text("{}")
    monkeypatch.setattr(checkpoints, "load", lambda _: copy.deepcopy(state))
    original_policy = copy.deepcopy(state["progressive"])
    result = checkpoint_record(study, path, step, list(range(3200)))
    assert result["update"] == step and result["draw_cursor"] == step * 4
    assert result["removed"] == 0 and result["retained"] == 1536
    assert result["alpha_sha256"] == checkpoints.tensor_hash(alpha)
    assert state["progressive"] == original_policy
    state["draw_cursor"] += 4
    with pytest.raises(ValueError, match="global step"):
        checkpoint_record(study, path, step, list(range(3200)))
    state["draw_cursor"] -= 4
    state["update"] += 50
    with pytest.raises(ValueError, match="global step"):
        checkpoint_record(study, path, step, list(range(3200)))


def test_pilot_config_retains_numerics_and_freezes_budget():
    from scripts.prune.learn import protocol

    original = dict(
        soft_updates=700,
        binary_updates=300,
        compact_updates=500,
        distributed_training="ddp_full_replicas",
        uniform_f18=True,
        global_unique_clips=4,
        gpu_ids=[0, 1, 2, 3],
        accumulation=1,
        rank_draw_assignment=[0, 1, 2, 3],
        gate_lr=0.01,
        lambda_warmup=100,
        numerics=dict(tf32=False),
        rank=128,
    )
    config = pilot_config(dict(config=original))
    assert protocol.phase_budget(config) == (500, 500, 800)
    assert config["wall_time_hours"] == 72
    for key in ("rank", "numerics", "gate_lr", "lambda_warmup", "rank_draw_assignment"):
        assert config[key] == original[key]
    assert original["soft_updates"] == 700
