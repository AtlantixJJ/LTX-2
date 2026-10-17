"""Eligible budgets, inherited masks, and exact 300-update fixed-gate tail."""

import copy

import pytest
import torch

from scripts.prune.learn.eligible import EligiblePolicy
from scripts.prune.learn.progressive import ProgressivePolicy, selection
from scripts.prune.learn.zero_one import ZeroOnePolicy, select_one

CFG = dict(
    hardening_interval=2,
    removal_fraction=0.02,
    regret_threshold=0.1,
    fixed_one_fraction=0.02,
    fixed_one_regret_threshold=0.5,
    selection_population="eligible",
    minimum_count=16,
    finalize_updates=300,
)


def test_eligible_budgets():
    free = torch.ones(1, 2000, dtype=torch.bool)
    travel = torch.ones(1, 2000, dtype=torch.float64)
    decrease, regret = travel.clone(), torch.zeros_like(travel)
    regret[:, 1200:] = 0.6
    zero = selection(free, travel, decrease, regret, CFG)
    one = select_one(free, torch.zeros_like(travel).float(), travel, regret, CFG)
    assert (zero["eligible_count"], zero["requested_count"]) == (1200, 24)
    assert (one["one_eligible_count"], one["one_requested_count"]) == (800, 16)
    free[:, 3:] = False
    zero = selection(free, travel, decrease, regret, CFG)
    assert zero["requested_count"] == 16 and zero["selected_count"] == 3


def test_force_one_exact_300_resume_and_tamper():
    alpha = torch.zeros(1, 20)
    policy = EligiblePolicy(alpha, CFG)
    policy.record(1, alpha + 1)
    event = policy.record(2, alpha + 2)
    assert event["eligible_count"] == event["one_eligible_count"] == 0
    assert event["forced_one_selected_count"] == 20
    assert policy.finalization_start == 2 and policy.final_update == 302
    assert policy.counts() == dict(free=0, removed=0, fixed_one=20)
    value = alpha + 2
    for update in range(3, 303):
        assert policy.record(update, value) is None
        if update in (3, 52, 301):
            policy = ProgressivePolicy.from_state(policy.state_dict(), CFG)
    assert policy.update == 302 and len(policy.new_history) == 1
    with pytest.raises(ValueError):
        policy.record(303, value)
    broken = copy.deepcopy(policy.state_dict())
    broken["final_update"] += 1
    with pytest.raises(ValueError):
        ProgressivePolicy.from_state(broken)
    broken = copy.deepcopy(policy.state_dict())
    broken["history"][0]["forced_one_selected_ids"] = []
    with pytest.raises(ValueError):
        ProgressivePolicy.from_state(broken)


def test_zero_exhaustion_finalizes():
    alpha = torch.zeros(1, 3)
    policy = EligiblePolicy(alpha, CFG)
    policy.record(1, alpha - 1)
    event = policy.record(2, alpha - 2)
    assert event["selected_count"] == 3 and event["terminal_reason"] == "no_free_heads"
    assert policy.final_update == 302 and not policy.one.any()


def test_partial_migration_preserves_both_masks():
    cfg = dict(CFG, hardening_interval=3, minimum_count=1)
    legacy = {k: v for k, v in cfg.items() if k not in ("selection_population", "minimum_count", "finalize_updates")}
    legacy.update(removal_fraction=0.1, fixed_one_fraction=0.1)
    parent = ZeroOnePolicy(torch.zeros(1, 3), legacy)
    parent.record(1, torch.tensor([[-1.0, 1.0, 1.0]]))
    parent.record(2, torch.tensor([[-2.0, -1.0, 2.0]]))
    parent.record(3, torch.tensor([[-3.0, 1.0, 3.0]]))
    value = torch.tensor([[-3.0, 1.0, 4.0]])
    parent.record(4, value)
    policy = EligiblePolicy(value, cfg, parent.state_dict())
    assert policy.counts() == parent.counts()
    assert torch.equal(policy.compose(value), parent.compose(value))
    assert len(policy.window) == 2 and policy.start_update == 4
    policy.record(5, torch.tensor([[-3.0, 1.0, 5.0]]))
    event = policy.record(6, torch.tensor([[-3.0, 1.0, 6.0]]))
    assert event["forced_one_selected_count"] == 1 and event["counts"]["fixed_one"] == 2
    assert ProgressivePolicy.from_state(policy.state_dict()).state_dict() == policy.state_dict()


def test_gate_only_includes_constant_and_low_regret_with_id_ties():
    from scripts.prune.learn.eligible import select_gate_only

    cfg = dict(
        CFG, fixed_one_selection="gate_only", fixed_one_regret_threshold=None, minimum_count=1, fixed_one_fraction=0.4
    )
    free = torch.ones(1, 5, dtype=torch.bool)
    choice = select_gate_only(free, torch.tensor([[1.0, 4.0, 4.0, 3.0, 2.0]]), cfg)
    assert choice["one_eligible_count"] == 5
    assert choice["one_selected_ids"] == [[0, 1], [0, 2]]
    assert all("regret" not in c and "travel" not in c for c in choice["one_candidates"])


def test_gate_only_zero_priority_and_remaining_population_budget():
    cfg = dict(
        CFG,
        hardening_interval=1,
        minimum_count=1,
        removal_fraction=1.0,
        fixed_one_fraction=0.4,
        fixed_one_selection="gate_only",
        fixed_one_regret_threshold=None,
    )
    alpha = torch.zeros(1, 6)
    policy = EligiblePolicy(alpha, cfg)
    # Head 0 is the largest final gate and zero-eligible. Zero has priority.
    value = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]])
    policy = EligiblePolicy(torch.tensor([[10.0, 8.0, 7.0, 6.0, 5.0, 4.0]]), cfg)
    event = policy.record(1, value)
    assert event["selected_ids"] == [[0, 0]]
    assert event["one_eligible_count"] == 5 and event["one_requested_count"] == 2
    assert event["one_selected_ids"] == [[0, 1], [0, 2]]
    assert event["counts"] == dict(removed=1, fixed_one=2, free=3)
    assert ProgressivePolicy.from_state(policy.state_dict(), cfg).state_dict() == policy.state_dict()


def test_gate_only_terminal_forces_remaining_and_resumes_exact_tail():
    cfg = dict(
        CFG,
        hardening_interval=1,
        minimum_count=1,
        fixed_one_fraction=0.1,
        fixed_one_selection="gate_only",
        fixed_one_regret_threshold=None,
    )
    alpha = torch.arange(20, dtype=torch.float32).reshape(1, 20) / 20
    policy = EligiblePolicy(alpha, cfg)
    event = policy.record(1, alpha)
    assert event["eligible_count"] == 0 and event["one_eligible_count"] == 20
    assert event["one_selected_count"] == 2 and event["forced_one_selected_count"] == 18
    assert event["new_fixed_one_count"] == 20 and policy.final_update == 301
    for update in range(2, 302):
        assert policy.record(update, alpha) is None
        if update == 77:
            policy = ProgressivePolicy.from_state(policy.state_dict(), cfg)
    assert policy.update == 301 and not policy.free.any()


def test_gate_only_migration_keeps_old_events_and_partial_window():
    alpha = torch.zeros(1, 30)
    parent = EligiblePolicy(alpha, CFG)
    parent.record(1, alpha + 1)
    cfg = dict(CFG, fixed_one_selection="gate_only", fixed_one_regret_threshold=None)
    migrated = EligiblePolicy(alpha + 1, cfg, parent.state_dict())
    assert len(migrated.window) == 2 and migrated.start_update == 1
    assert torch.equal(parent.compose(alpha + 1), migrated.compose(alpha + 1))
    event = migrated.record(2, alpha + 2)
    assert event["one_eligible_count"] == 30 and event["new_fixed_one_count"] == 30
    assert ProgressivePolicy.from_state(migrated.state_dict()).state_dict() == migrated.state_dict()
