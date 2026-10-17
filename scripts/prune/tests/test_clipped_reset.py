"""Clipped score gradients, event resets, immutable history and exact resume."""

import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from scripts.prune.learn import checkpoints, losses
from scripts.prune.learn.clipped_reset import DEFAULTS, ClippedResetPolicy, clipped_gate
from scripts.prune.learn.gates import HeadGates
from scripts.prune.learn.progressive import ProgressivePolicy, new_policy, statistics


def config(**changes):
    return dict(DEFAULTS, hardening_interval=2, selection_updates=4, minimum_count=1, **changes)


def test_forward_surrogate_endpoints_and_unprojected_scores():
    alpha = torch.tensor([[-4., -3., -1., 0., .5, 1., 2., 3., 4.]], requires_grad=True)
    original = alpha.detach().clone()
    value = clipped_gate(alpha, torch.ones_like(alpha, dtype=torch.bool))
    assert value.tolist() == [[0, 0, 0, 0, .5, 1, 1, 1, 1]]
    value.sum().backward()
    assert alpha.grad.tolist() == [[0, 1, 1, 1, 1, 1, 1, 1, 0]]
    assert torch.equal(alpha, original)
    alpha.grad = None
    checkpoint(lambda a: clipped_gate(a, torch.zeros_like(a, dtype=torch.bool)), alpha, use_reentrant=False).sum().backward()
    assert not alpha.grad.any()


def test_initial_execution_and_separate_statistics():
    gates = HeadGates(1, 4, activation="clip01_v1")
    gates.progressive = new_policy(gates.alpha, config())
    assert torch.equal(gates.microbatch(), torch.ones(1, 4))
    assert torch.equal(HeadGates(1, 4).alpha, torch.full((1, 4), 3.))
    gates.alpha.data.copy_(torch.tensor([[-1., .5, 3., 0.]]))
    snapshot = gates.snapshot()
    assert snapshot["counts"] == dict(alive=4, removed=0)
    assert snapshot["alive_clipped_zero"] == 2 and snapshot["alive_interior"] == snapshot["alive_clipped_one"] == 1
    assert snapshot["raw_score_mean"] != snapshot["executed_gate_mean"]


def test_reset_jump_is_outside_both_windows():
    alpha = torch.ones(1, 3)
    policy = ClippedResetPolicy(alpha, config())
    policy.record(1, torch.tensor([[.8, .8, 1.2]]))
    alpha.copy_(torch.tensor([[.6, .6, 1.4]]))
    event = policy.record(2, alpha)
    assert event["selected_ids"] == [[0, 0]]
    assert event["regret"] == [[0., 0., 0.]]
    assert alpha.tolist() == [[0., 1., 1.]]
    assert len(policy.window) == 1 and torch.equal(policy.window[0], alpha)
    assert event["pre_event_scores"][0][1] == pytest.approx(.6)
    assert event["post_reset_scores"] == [[0., 1., 1.]]
    _, _, regret = statistics([torch.tensor(1.), torch.tensor(.8), torch.tensor(.6), torch.tensor(1.)])
    assert regret == 1
    assert ProgressivePolicy.from_state(policy.state_dict(), config()).state_dict() == policy.state_dict()
    # The survivor is still eligible at the next event.
    policy.record(3, torch.tensor([[0., .8, 1.2]]))
    event = policy.record(4, torch.tensor([[0., .6, 1.4]]))
    assert event["selected_ids"] == [[0, 1]] and event["terminal_reason"] == "pilot_cap"


def test_eligible_denominator_floor_and_arbitrary_counts():
    alpha = torch.ones(48, 32)
    policy = ClippedResetPolicy(alpha)
    for update in range(1, 51):
        alpha.sub_(.001)
        event = policy.record(update, alpha)
    assert event["eligible_count"] == 1536
    assert event["selected_count"] == 31
    assert event["selected_ids"] == [[i // 32, i % 32] for i in range(31)]
    assert policy.counts() == dict(alive=1505, removed=31)


def test_no_eligibility_exact_tail_and_tamper_rejection():
    alpha = torch.ones(1, 3)
    policy = ClippedResetPolicy(alpha, config())
    policy.record(1, alpha)
    event = policy.record(2, alpha)
    assert event["terminal_reason"] == "no_eligible_heads"
    assert event["regret"] == [[None, None, None]]
    assert (policy.finalization_start, policy.final_update) == (2, 302)
    assert policy.counts() == dict(alive=3, removed=0)
    for update in range(3, 303):
        assert policy.record(update, alpha) is None
        if update in (3, 51, 301):
            policy = ProgressivePolicy.from_state(policy.state_dict(), config())
    with pytest.raises(ValueError):
        policy.record(303, alpha)
    for field, value in (("final_update", 303), ("event_stage", "pre_reset"), ("free", [[False, True, True]])):
        state = policy.state_dict()
        state[field] = value
        with pytest.raises(ValueError):
            ProgressivePolicy.from_state(state)
    for key in DEFAULTS:
        altered = dict(config())
        value = altered[key]
        altered[key] = value + 1 if type(value) in (int, float) else "altered"
        with pytest.raises(ValueError):
            ProgressivePolicy.from_state(policy.state_dict(), altered)


def test_all_removed_and_protection():
    cfg = dict(config(), minimum_count=16)
    alpha = torch.ones(1, 3)
    policy = ClippedResetPolicy(alpha, cfg)
    policy.record(1, alpha - .2)
    alpha.sub_(.4)
    event = policy.record(2, alpha)
    assert event["terminal_reason"] == "all_heads_removed" and not alpha.any()
    assert event["selected_count"] == 3 and event["shortfall"] == 13
    alpha.add_(1)
    policy.protect(alpha)
    assert not alpha.any()
    clipped_gate(alpha.requires_grad_(), policy.free).sum().backward()
    assert not alpha.grad.any()


def test_raw_sparsity_and_fixed_denominator():
    alpha = torch.tensor([[-4., -3., -1., 1., 4.]], requires_grad=True)
    alive = torch.tensor([[True, True, True, False, True]])
    value = losses.sparsity_value(clipped_gate(alpha, alive), logits=alpha, kind="relu_logits", free_mask=alive)
    coefficient = losses.sparsity_coefficient(50, "soft", lambda_max=.01, lambda_warmup=100, lambda_start_multiplier=.1)
    (coefficient * value).backward()
    actual = losses.sparsity_gradient(alpha, update=50, phase="soft", kind="relu_logits", free_mask=alive,
                                     lambda_max=.01, lambda_warmup=100, lambda_start_multiplier=.1)
    assert torch.equal(alpha.grad, actual)
    assert actual[0, 4] == pytest.approx(coefficient / 5)
    assert not losses.sparsity_gradient(alpha, update=500, phase="recovery").any()


def test_adam_reset_adapter_continuity_and_exact_resume():
    torch.manual_seed(42)
    alpha = torch.nn.Parameter(torch.ones(1, 4))
    adapter = torch.nn.Parameter(torch.ones(1, 4))
    go = torch.optim.AdamW([alpha], lr=.05, weight_decay=0)
    ao = torch.optim.AdamW([adapter], lr=.01, weight_decay=0)
    policy = ClippedResetPolicy(alpha, config())

    def step():
        go.zero_grad()
        ao.zero_grad()
        update = policy.update + 1
        (clipped_gate(alpha, policy.free) * adapter * (torch.rand_like(alpha) + 1)).sum().backward()
        policy.mask_gradient(alpha)
        ao.step()
        if not policy.recovering:
            go.step()
        policy.protect(alpha, go)
        event = policy.record(update, alpha, go)
        assert ao.state[adapter]["step"] == update
        if event:
            assert not go.state and not alpha.grad
        return copy.deepcopy((alpha.detach(), adapter.detach(), go.state_dict(), ao.state_dict(), policy.state_dict()))

    step()
    saved = copy.deepcopy((alpha.detach(), adapter.detach(), go.state_dict(), ao.state_dict(), policy.state_dict(), checkpoints.rng_state()))
    expected = [step() for _ in range(5)]  # Cross two events and enter recovery.
    alpha.data.copy_(saved[0])
    adapter.data.copy_(saved[1])
    go.load_state_dict(saved[2])
    ao.load_state_dict(saved[3])
    policy = ProgressivePolicy.from_state(saved[4], config())
    checkpoints.restore_rng(saved[5])
    for reference in expected:
        actual = step()
        assert checkpoints.tensor_hash(actual[0]) == checkpoints.tensor_hash(reference[0])
        assert torch.equal(actual[1], reference[1]) and actual[4] == reference[4]
        for index in (2, 3):
            for key, state in actual[index]["state"].items():
                for name, value in state.items():
                    assert torch.equal(value, reference[index]["state"][key][name])


def test_shared_optimizer_rejected_before_mutation():
    alpha = torch.nn.Parameter(torch.ones(1, 3))
    adapter = torch.nn.Parameter(torch.ones(1, 3))
    policy = ClippedResetPolicy(alpha, config())
    opt = torch.optim.AdamW([alpha, adapter])
    before = policy.state_dict()
    with pytest.raises(ValueError, match="gate-only Adam"):
        policy.record(1, alpha, opt)
    assert before == policy.state_dict()
