"""Global ranking, configurable percentage removal, protected Adam and exact replay."""

import copy
import math

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from scripts.prune.learn import checkpoints, losses, protocol
from scripts.prune.learn.gates import HeadGates
from scripts.prune.learn.progressive import DEFAULTS, ProgressivePolicy, selection, statistics, validate_config


def config(interval=2, fraction=0.01, threshold=0.1):
    return dict(hardening_interval=interval, removal_fraction=fraction, regret_threshold=threshold)


@pytest.mark.parametrize(
    "bad",
    [
        dict(hardening_interval=0),
        dict(hardening_interval=True),
        dict(removal_fraction=0),
        dict(removal_fraction=1.01),
        dict(removal_fraction=float("nan")),
        dict(regret_threshold=-0.1),
        dict(regret_threshold=1.1),
        dict(regret_threshold=True),
        dict(packet=4),
    ],
)
def test_invalid_config(bad):
    with pytest.raises(ValueError):
        validate_config({**DEFAULTS, **bad})


def test_statistics_and_constant_regret():
    travel, decrease, regret = statistics(
        [torch.tensor([3.0, 3.0, 3.0, 3.0]), torch.tensor([2.0, 4.0, 4.0, 3.0]), torch.tensor([1.0, 3.0, 5.0, 3.0])]
    )
    assert travel.tolist() == [2, 2, 2, 0]
    assert decrease.tolist() == [2, 0, -2, 0]
    assert regret[:3].tolist() == [0, 1, 0]
    assert math.isnan(regret[3])


def test_ranking_threshold_equality_and_shortfall():
    free = torch.ones((2, 3), dtype=torch.bool)
    travel = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=torch.float64)
    decrease = torch.tensor([[0.5, 0.9, 0.9], [1.0, -1.0, 0.0]], dtype=torch.float64)
    regret = torch.tensor([[0.5, 0.1, 0.1], [0.0, 0.0, float("nan")]], dtype=torch.float64)
    choice = selection(free, travel, decrease, regret, config(fraction=0.5, threshold=0.1))
    assert choice["selected_ids"] == [[1, 0], [0, 1], [0, 2]]
    assert [c["rank"] for c in choice["candidates"]] == [1, 2, 3]
    assert all(c["regret"] <= 0.1 for c in choice["candidates"])
    fewer = selection(free, travel, decrease, regret, config(fraction=1, threshold=0))
    assert fewer["eligible_count"] == fewer["selected_count"] == 1
    assert fewer["shortfall"] == 5


@pytest.mark.parametrize("interval,fraction,expected", [(1, 0.01, 16), (3, 0.001, 2), (7, 1 / 1536, 1), (2, 0.1, 154)])
def test_configurable_intervals_and_arbitrary_counts(interval, fraction, expected):
    alpha = torch.full((48, 32), 3.0)
    policy = ProgressivePolicy(alpha, config(interval, fraction))
    for t in range(1, interval + 1):
        event = policy.record(t, alpha - t * 0.001)
        if t < interval:
            assert event is None
    assert event["selected_count"] == expected
    assert event["selected_ids"] == [[i // 32, i % 32] for i in range(expected)]
    assert policy.counts() == dict(free=1536 - expected, removed=expected)
    assert ProgressivePolicy.from_state(policy.state_dict(), policy.config).state_dict() == policy.state_dict()


def test_original_denominator_no_caps_and_irreversibility():
    alpha = torch.full((48, 32), 3.0)
    policy = ProgressivePolicy(alpha, config(interval=1))
    removed = []
    for t in range(1, 11):
        event = policy.record(t, alpha - t * 0.01)
        assert event["requested_count"] == event["selected_count"] == 16
        assert not set(map(tuple, event["selected_ids"])) & set(map(tuple, removed))
        removed += event["selected_ids"]
    assert policy.counts() == dict(free=1376, removed=160)
    assert int((~policy.free[0]).sum()) == 32


def test_constant_increasing_small_decrease_and_no_quota():
    alpha = torch.full((1, 4), 3.0)
    policy = ProgressivePolicy(alpha, config(fraction=1))
    policy.record(1, torch.tensor([[3.0, 4.0, 2.9999, 4.0]]))
    event = policy.record(2, torch.tensor([[3.0, 5.0, 2.9998, 2.9]]))
    assert event["selected_ids"] == [[0, 2]]
    assert event["shortfall"] == 3
    assert event["regret"][0][0] is None


def test_mixed_gradients_and_adam_protection():
    gates = HeadGates()
    gates.progressive = ProgressivePolicy(gates.alpha, config(interval=1, fraction=3 / 1536))
    opt = torch.optim.AdamW([gates.alpha], lr=0.01, weight_decay=0)
    gates.alpha.sum().backward()
    opt.step()
    opt.zero_grad()
    gates.progressive.record(1, gates.alpha, opt)
    fixed = ~gates.progressive.free
    assert fixed.sum() == 3
    before = gates.alpha.detach().clone()
    actual = gates.microbatch()
    assert not actual[fixed].any()
    value = losses.sparsity_value(actual, logits=gates.alpha, kind="relu_logits", free_mask=~fixed)
    assert value == (torch.relu(gates.alpha + 3) * (~fixed)).sum() / 1536
    x = torch.ones_like(gates.alpha, requires_grad=True)
    result = checkpoint(lambda v: (v * gates.microbatch()).square().sum(), x, use_reentrant=False)
    (result + value).backward()
    assert not gates.alpha.grad[fixed].any()
    opt.step()
    gates.progressive.protect(gates.alpha, opt)
    assert torch.equal(before[fixed], gates.alpha.detach()[fixed])
    for v in opt.state[gates.alpha].values():
        if v.shape == gates.alpha.shape:
            assert not v[fixed].any()


def test_resume_across_event_and_changed_hyperparameters():
    gates = HeadGates()
    gates.progressive = ProgressivePolicy(gates.alpha, config(interval=3))
    with torch.no_grad():
        gates.alpha.sub_(0.1)
    gates.progressive.record(1, gates.alpha)
    optimizer = torch.optim.AdamW([gates.alpha], lr=0.01)

    def step():
        optimizer.zero_grad()
        loss = (gates.microbatch() * torch.randn_like(gates.alpha)).sum()
        loss.backward()
        optimizer.step()
        gates.progressive.protect(gates.alpha, optimizer)
        gates.progressive.record(gates.progressive.update + 1, gates.alpha, optimizer)
        return gates.alpha.detach().clone(), copy.deepcopy(optimizer.state_dict()), gates.progressive.state_dict()

    step()
    saved = copy.deepcopy(
        (gates.alpha.detach(), optimizer.state_dict(), gates.progressive.state_dict(), checkpoints.rng_state())
    )
    expected = [step(), step()]
    gates.alpha.data.copy_(saved[0])
    optimizer.load_state_dict(saved[1])
    gates.progressive = ProgressivePolicy.from_state(saved[2], config(interval=3))
    checkpoints.restore_rng(saved[3])
    for alpha, opt, state in expected:
        a, o, s = step()
        assert torch.equal(a, alpha) and s == state
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(o["state"][0][k], opt["state"][0][k])
    with pytest.raises(ValueError, match="hyperparameters changed"):
        ProgressivePolicy.from_state(saved[2], config(interval=4))
    changed = copy.deepcopy(expected[0][2])
    changed["history"][0]["selected_ids"] = []
    with pytest.raises(ValueError, match="selection differs"):
        ProgressivePolicy.from_state(changed)


def test_checkpoint_io_and_monitor_rng(tmp_path, monkeypatch):
    from scripts.prune.learn import heads
    from scripts.prune.learn.adapters import ProjectionAdapter

    gates = HeadGates()
    gates.progressive = ProgressivePolicy(gates.alpha, config(interval=1))
    gates.alpha.data.sub_(0.01)
    event = gates.progressive.record(1, gates.alpha)
    adapter = ProjectionAdapter(torch.nn.Linear(4, 4), 2)
    ao = torch.optim.AdamW([adapter.A, adapter.B], lr=0.001)
    go = torch.optim.AdamW([gates.alpha], lr=0.01)
    checkpoints.publish_json(tmp_path / "event.json", event)
    kwargs = dict(
        identity={"study": "tiny"},
        update=1,
        phase="soft",
        gates=gates,
        adapters={"0.attn1.v": adapter},
        adapter_optimizer=ao,
        gate_optimizer=go,
        draws_per_update=4,
    )
    checkpoints.save(tmp_path / "state.pt", **kwargs)
    checkpoints.save(tmp_path / "state.pt", **kwargs)
    assert checkpoints.load(tmp_path / "state.pt")["progressive"] == gates.progressive.state_dict()
    rng = checkpoints.rng_state()

    def fail(*args):
        torch.rand(5)
        gates.phase = "compact"
        raise RuntimeError("monitor failure")

    monkeypatch.setattr(heads, "_monitor", fail)
    with pytest.raises(RuntimeError, match="monitor failure"):
        heads.monitor(None, None, None, gates, "learned_progressive", 42, 1, None)
    assert torch.equal(torch.get_rng_state(), rng["torch"]) and gates.phase == "soft"


def test_progressive_budget_and_config_validation():
    cfg = dict(soft_updates=500, binary_updates=0, compact_updates=0, progressive_policy=dict(DEFAULTS))
    assert protocol.phase_budget(cfg) == (500, 500, 500)
    with pytest.raises(ValueError):
        protocol.validate({**cfg, "progressive_policy": dict(DEFAULTS, removal_fraction=0)})
