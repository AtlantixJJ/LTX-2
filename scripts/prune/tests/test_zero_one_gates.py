"""Migration, independent zero/one selection, fixed coordinates and continuation budgets."""

import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from scripts.prune.learn import checkpoints, losses, protocol
from scripts.prune.learn.gates import HeadGates
from scripts.prune.learn.progressive import ProgressivePolicy, new_policy, validate_config
from scripts.prune.learn.wandb_logger import training_metrics
from scripts.prune.learn.zero_one import ZeroOnePolicy, select_one


def config(**kw):
    return dict(
        hardening_interval=2,
        removal_fraction=0.125,
        regret_threshold=0.1,
        fixed_one_fraction=0.125,
        fixed_one_regret_threshold=0.5,
        **kw,
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(fixed_one_fraction=0),
        dict(fixed_one_fraction=True),
        dict(fixed_one_fraction=float("nan")),
        dict(fixed_one_regret_threshold=0.1),
        dict(fixed_one_regret_threshold=1.1),
        dict(fixed_one_regret_threshold=True),
    ],
)
def test_reject_invalid_high_policy(changes):
    with pytest.raises(ValueError):
        validate_config({**config(), **changes})


def test_high_gate_first_then_regret_then_original_ids_and_equality():
    free = torch.ones(2, 4, dtype=torch.bool)
    alpha = torch.tensor([[1.0, 2.0, 2.0, 3.0], [2.0, 2.0, 2.0, 2.0]])
    regret = torch.tensor([[0.9, 0.5, 0.7, 0.49], [0.7, 0.7, 0.7, float("nan")]], dtype=torch.float64)
    travel = torch.ones_like(regret)
    travel[1, 2] = 0
    actual = select_one(free, alpha, travel, regret, {**config(), "fixed_one_fraction": 0.5})
    assert actual["one_selected_ids"] == [[0, 2], [1, 0], [1, 1], [0, 1]]
    assert actual["one_eligible_count"] == 5 and actual["one_shortfall"] == 0
    free.zero_()
    none = select_one(free, alpha, travel, regret, config())
    assert none["one_selected_count"] == 0 and none["one_shortfall"] == 1


def mixed():
    gates = HeadGates(2, 4)
    gates.progressive = new_policy(gates.alpha, config())
    a = gates.alpha.detach().clone()
    a[0, 0] = 2.0
    a[0, 1] = 5.0
    gates.alpha.data.copy_(a)
    gates.progressive.record(1, gates.alpha)
    a[0, 0] = 1.0
    a[0, 1] = 3.0
    gates.alpha.data.copy_(a)
    event = gates.progressive.record(2, gates.alpha)
    return gates, event


def test_both_commits_actual_gates_and_fixed_gradients_adam():
    gates, event = mixed()
    policy = gates.progressive
    assert event["selected_ids"] == [[0, 0]] and event["one_selected_ids"] == [[0, 1]]
    assert policy.counts() == dict(free=6, removed=1, fixed_one=1)
    actual = gates.microbatch()
    assert actual[0, 0] == 0 and actual[0, 1] == 1
    opt = torch.optim.AdamW([gates.alpha], lr=0.01, weight_decay=0.01)
    # Seed nonzero moments to check clearing independently of the mask's natural zero derivative.
    opt.state[gates.alpha].update(
        step=torch.tensor(2.0), exp_avg=torch.ones_like(gates.alpha), exp_avg_sq=torch.ones_like(gates.alpha)
    )
    before = gates.alpha.detach().clone()
    policy.protect(gates.alpha, opt)
    x = torch.ones_like(gates.alpha, requires_grad=True)
    value = checkpoint(lambda y: (y * gates.microbatch()).sum(), x, use_reentrant=False)
    sparse = losses.sparsity_value(actual, logits=gates.alpha, kind="relu_logits", free_mask=policy.free)
    assert sparse == (torch.relu(gates.alpha + 3) * policy.free).sum() / 8
    (value + sparse).backward()
    assert not gates.alpha.grad[~policy.free].any()
    opt.step()
    policy.protect(gates.alpha, opt)
    assert torch.equal(before[~policy.free], gates.alpha.detach()[~policy.free])
    assert all(not opt.state[gates.alpha][k][~policy.free].any() for k in ("exp_avg", "exp_avg_sq"))
    snapshot = gates.snapshot()
    assert snapshot["population"] == "free_heads_only"
    assert snapshot["quantiles"][0] == float(gates.alpha.detach().sigmoid()[policy.free].min())


def test_migration_preserves_parent_and_starts_next_interval():
    alpha = torch.full((2, 4), 3.0)
    old = ProgressivePolicy(alpha, dict(hardening_interval=2, removal_fraction=0.125, regret_threshold=0.1))
    old.record(1, alpha - 0.1)
    alpha = alpha - 0.2
    old.record(2, alpha)
    base = copy.deepcopy(old.state_dict())
    new = ZeroOnePolicy(alpha, config(), base)
    assert new.update == 2 and new.start_update == 2 and len(new.window) == 1
    assert torch.equal(new.compose(alpha), old.compose(alpha))
    assert new.counts() == dict(free=7, removed=1, fixed_one=0)
    assert new.record(3, alpha) is None
    event = new.record(4, alpha)
    assert event["selected_count"] == event["one_selected_count"] == 0
    assert event["shortfall"] == event["one_shortfall"] == 1
    assert base == old.state_dict()
    assert ProgressivePolicy.from_state(new.state_dict(), config()).state_dict() == new.state_dict()


def test_serialization_tampering_and_hyperparameter_change():
    gates, _ = mixed()
    state = gates.progressive.state_dict()
    assert ProgressivePolicy.from_state(state).state_dict() == state
    for key in ("one", "free"):
        bad = copy.deepcopy(state)
        bad[key][0][2] = not bad[key][0][2]
        with pytest.raises(ValueError):
            ProgressivePolicy.from_state(bad)
    bad = copy.deepcopy(state)
    bad["history"][0]["one_selected_ids"] = []
    with pytest.raises(ValueError, match="selection"):
        ProgressivePolicy.from_state(bad)
    with pytest.raises(ValueError, match="hyperparameters"):
        ProgressivePolicy.from_state(state, {**config(), "fixed_one_regret_threshold": 0.6})


def test_exact_optimizer_rng_resume_with_both_fixed_states():
    gates, _ = mixed()
    opt = torch.optim.AdamW([gates.alpha], lr=0.01)

    def step():
        opt.zero_grad()
        (gates.microbatch() * torch.randn_like(gates.alpha)).sum().backward()
        opt.step()
        gates.progressive.protect(gates.alpha, opt)
        gates.progressive.record(gates.progressive.update + 1, gates.alpha, opt)
        return gates.alpha.detach().clone(), copy.deepcopy(opt.state_dict()), gates.progressive.state_dict()

    step()
    saved = (
        gates.alpha.detach().clone(),
        copy.deepcopy(opt.state_dict()),
        gates.progressive.state_dict(),
        checkpoints.rng_state(),
    )
    expected = [step(), step()]
    gates.alpha.data.copy_(saved[0])
    opt.load_state_dict(saved[1])
    gates.progressive = ProgressivePolicy.from_state(saved[2])
    checkpoints.restore_rng(saved[3])
    for a, o, s in expected:
        actual, opt_state, state = step()
        assert torch.equal(a, actual) and s == state
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(o["state"][0][k], opt_state["state"][0][k])


def test_free_population_exhaustion_omits_distribution_and_continues_graph():
    gates = HeadGates(1, 2)
    cfg = {**config(), "removal_fraction": 1.0}
    gates.progressive = ZeroOnePolicy(gates.alpha, cfg)
    gates.alpha.data.sub_(0.1)
    gates.progressive.record(1, gates.alpha)
    gates.alpha.data.sub_(0.1)
    gates.progressive.record(2, gates.alpha)
    assert gates.progressive.counts() == dict(free=0, removed=2, fixed_one=0)
    assert gates.snapshot()["mean"] is None and gates.snapshot()["quantiles"] is None
    (
        gates.microbatch().sum()
        + losses.sparsity_value(gates.current, logits=gates.alpha, kind="relu_logits", free_mask=gates.progressive.free)
    ).backward()
    assert not gates.alpha.grad.any()
    gates.progressive.record(3, gates.alpha)
    event = gates.progressive.record(4, gates.alpha)
    assert event["eligible_count"] == event["one_eligible_count"] == 0
    assert ProgressivePolicy.from_state(gates.progressive.state_dict()).state_dict() == gates.progressive.state_dict()


def test_global_budget_extends_only_zero_one_and_preserves_draw_cursor():
    cfg = dict(
        soft_updates=4000, binary_updates=0, compact_updates=0, progressive_policy=config(), global_unique_clips=4
    )
    assert protocol.phase_budget(cfg) == (4000, 4000, 4000)
    draws = list(range(16000))
    assert protocol.update_draws(draws, 501, cfg) == list(range(2000, 2004))
    assert protocol.update_draws(draws, 4000, cfg) == list(range(15996, 16000))
    with pytest.raises(ValueError):
        protocol.phase_budget({**cfg, "binary_updates": 1})


def test_wandb_empty_free_population_and_both_branch_counts():
    row = dict(
        update=551,
        phase="soft",
        draws=[dict(official=0.1, penalty=0.0)],
        seconds=1.0,
        sparsity_lambda=0.01,
        legal_mask_changes=0,
        gates=dict(mean=None, quantiles=None, le_0_1=None, lt_0_5=None, ge_0_9=None),
        progressive=dict(
            counts=dict(free=0, removed=6, fixed_one=2),
            selected_count=0,
            one_selected_count=0,
            eligible_count=0,
            requested_count=1,
            shortfall=1,
            one_eligible_count=0,
            one_requested_count=1,
            one_shortfall=1,
        ),
    )
    for key in (
        "adapter_gradient_norm",
        "logit_gradient_norm",
        "velocity_logit_gradient_norm",
        "sparsity_logit_gradient_norm",
    ):
        row[key] = 0.0
    cfg = dict(
        global_unique_clips=1,
        adapter_lr=0.0001,
        gate_lr=0.01,
        warmup_updates=50,
        soft_updates=4000,
        binary_updates=0,
        compact_updates=0,
        progressive_policy=config(),
    )
    values = training_metrics(row, cfg)
    assert values["gates/fixed_one"] == 2 and values["gates/removed"] == 6
    assert values["gates/one_shortfall"] == 1 and "gates/min" not in values and "gates/mean" not in values
