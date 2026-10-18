"""Clipped score gradients, event resets, immutable history and exact resume."""

import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from scripts.prune.learn import checkpoints, losses
from scripts.prune.learn.clipped_reset import DEFAULTS, ClippedResetPolicy, clipped_gate, drop_statistics
from scripts.prune.learn.gates import HeadGates
from scripts.prune.learn.progressive import ProgressivePolicy, new_policy


def config(**changes):
    return dict(DEFAULTS, hardening_interval=2, late_updates=1, selection_updates=4, **changes)


def run_window(policy, *snapshots):
    """Record one window of explicit post-optimizer scores; return the event."""
    alpha = None
    for values in snapshots:
        alpha = torch.tensor(values, dtype=torch.float32)
        event = policy.record(policy.update + 1, alpha)
    return event, alpha


def test_forward_surrogate_endpoints_and_unprojected_scores():
    alpha = torch.tensor([[-4.0, -3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    original = alpha.detach().clone()
    value = clipped_gate(alpha, torch.ones_like(alpha, dtype=torch.bool))
    assert value.tolist() == [[0, 0, 0, 0, 0.5, 1, 1, 1, 1]]
    value.sum().backward()
    assert alpha.grad.tolist() == [[0, 1, 1, 1, 1, 1, 1, 1, 0]]
    assert torch.equal(alpha, original)
    alpha.grad = None
    checkpoint(
        lambda a: clipped_gate(a, torch.zeros_like(a, dtype=torch.bool)), alpha, use_reentrant=False
    ).sum().backward()
    assert not alpha.grad.any()


def test_initial_execution_and_separate_statistics():
    gates = HeadGates(1, 4, activation="clip01_v1")
    gates.progressive = new_policy(gates.alpha, config())
    assert torch.equal(gates.microbatch(), torch.ones(1, 4))
    assert torch.equal(HeadGates(1, 4).alpha, torch.full((1, 4), 3.0))
    gates.alpha.data.copy_(torch.tensor([[-1.0, 0.5, 3.0, 0.0]]))
    snapshot = gates.snapshot()
    assert snapshot["counts"] == dict(alive=4, removed=0)
    assert snapshot["alive_clipped_zero"] == 2 and snapshot["alive_interior"] == snapshot["alive_clipped_one"] == 1
    assert snapshot["raw_score_mean"] != snapshot["executed_gate_mean"]


def test_reset_jump_is_outside_both_windows_and_capped_flag_persists():
    policy = ClippedResetPolicy(torch.ones(1, 3), config(cap_fraction=0.5))
    event, alpha = run_window(policy, [[0.8, 0.8, 1.2]], [[0.6, 0.6, 1.4]])
    # Both heads are strong; the cap keeps one, and ID breaks the drop tie.
    assert event["selected_ids"] == [[0, 0]] and event["cap"] == 1 and event["cap_bound"]
    assert (event["eligible_count"], event["shortfall"], event["strong_count"]) == (2, 1, 2)
    assert event["path_regret"] == [[0.0, 0.0, 0.0]] and event["giveback"] == [[0.0, 0.0, None]]
    assert alpha.tolist() == [[0.0, 1.0, 1.0]] and event["post_reset_scores"] == [[0.0, 1.0, 1.0]]
    assert len(policy.window) == 1 and torch.equal(policy.window[0], alpha)
    assert event["outgoing_persistence"] == [[False, True, False]]
    # Including the reset jump would give path regret one and give-back one.
    reset = [torch.tensor(v) for v in (1.0, 0.8, 0.6, 1.0)]
    drop, giveback, _, _, path_regret = drop_statistics(reset, 1)
    assert path_regret == 1 and giveback == pytest.approx(1) and drop == 0
    assert ProgressivePolicy.from_state(policy.state_dict(), config(cap_fraction=0.5)).state_dict() == (
        policy.state_dict()
    )
    # The capped survivor needs only the persistent drop in the next window.
    event, _ = run_window(policy, [[0.0, 0.9, 1.2]], [[0.0, 0.8, 1.4]])
    assert event["selected_ids"] == [[0, 1]] and event["candidates"][0]["tier"] == "persistent"
    assert event["terminal_reason"] == "pilot_cap"


def test_giveback_late_change_and_threshold_equality():
    cfg = dict(config(cap_fraction=1.0, giveback_limit=0.125), hardening_interval=3, selection_updates=6)
    policy = ClippedResetPolicy(torch.ones(1, 4), cfg)
    # Head 0 gives back 25%; head 1 stops falling; heads 2 and 3 sit exactly on a limit.
    event, _ = run_window(
        policy, [[0.6, 0.8, 0.5, 0.9]], [[0.8, 0.7, 0.625, 0.875]], [[0.7, 0.7, 0.5625, 0.75]]
    )
    assert event["giveback"][0][0] == pytest.approx(0.25) and event["late_change"][0][1] == 0
    assert event["giveback"][0][2] == 0.125 and event["drop"][0][3] == 0.25
    assert event["selected_ids"] == [[0, 2], [0, 3]]
    assert [c["tier"] for c in event["candidates"]] == ["strong", "strong"]


def test_persistence_needs_the_preceding_window():
    cfg = dict(config(cap_fraction=1.0), selection_updates=8)
    policy = ClippedResetPolicy(torch.ones(1, 2), cfg)
    event, _ = run_window(policy, [[0.9, 1.0]], [[0.8, 1.0]])
    assert event["selected_count"] == 0 and event["outgoing_persistence"] == [[True, False]]
    event, _ = run_window(policy, [[0.95, 1.0]], [[0.9, 1.0]])  # a 0.1 drop clears the flag
    assert event["selected_count"] == 0 and event["outgoing_persistence"] == [[False, False]]
    event, _ = run_window(policy, [[0.9, 1.0]], [[0.8, 1.0]])
    assert event["selected_count"] == 0 and event["empty_events"] == 0
    event, _ = run_window(policy, [[0.9, 1.0]], [[0.8, 1.0]])
    assert event["selected_ids"] == [[0, 0]] and event["persistent_count"] == 1
    assert ProgressivePolicy.from_state(policy.state_dict(), cfg).state_dict() == policy.state_dict()


def test_lowest_first_ranking_alive_cap_and_no_minimum():
    alpha = torch.ones(48, 32)
    policy = ClippedResetPolicy(alpha)
    rate = torch.full((48, 32), 0.006)
    rate[47, 31] = 0.009  # the lowest final score ranks first
    for update in range(1, 51):
        alpha.sub_(rate)
        event = policy.record(update, alpha)
    assert event["eligible_count"] == 1536 and event["cap"] == 61 and event["selected_count"] == 61
    assert event["selected_ids"] == [[47, 31]] + [[i // 32, i % 32] for i in range(60)]
    assert policy.counts() == dict(alive=1475, removed=61)
    # A shallow window has no candidates; nothing is forced out.
    for update in range(51, 101):
        alpha.sub_(0.001 * policy.free)
        event = policy.record(update, alpha)
    assert event["selected_count"] == 0 and event["terminal_reason"] is None


def test_three_empty_events_exact_tail_and_tamper_rejection():
    cfg = dict(config(cap_fraction=1.0), selection_updates=10)
    alpha = torch.ones(1, 3)
    policy = ClippedResetPolicy(alpha, cfg)
    for update in range(1, 5):
        policy.record(update, alpha)
    event, _ = run_window(policy, [[1.0, 0.8, 1.0]], [[1.0, 0.7, 1.0]])
    assert event["selected_count"] == 1 and event["empty_events"] == 0
    for update in range(7, 11):
        event = policy.record(update, torch.tensor([[1.0, 0.0, 1.0]]))
    assert event["empty_events"] == 2 and event["terminal_reason"] == "pilot_cap"
    policy = ClippedResetPolicy(torch.ones(1, 3), cfg)
    for update in range(1, 7):
        event = policy.record(update, torch.ones(1, 3))
    assert event["terminal_reason"] == "evidence_exhausted" and event["giveback"] == [[None, None, None]]
    assert (policy.finalization_start, policy.final_update) == (6, 306)
    for update in range(7, 307):
        assert policy.record(update, torch.ones(1, 3)) is None
        if update in (7, 51, 305):
            policy = ProgressivePolicy.from_state(policy.state_dict(), cfg)
    with pytest.raises(ValueError):
        policy.record(307, torch.ones(1, 3))
    for field, value in (("final_update", 307), ("event_stage", "pre_reset"), ("empty_events", 0)):
        state = policy.state_dict()
        state[field] = value
        with pytest.raises(ValueError):
            ProgressivePolicy.from_state(state)
    for key in DEFAULTS:
        altered = dict(cfg)
        value = altered[key]
        altered[key] = value + 1 if type(value) in (int, float) else "altered"
        with pytest.raises(ValueError):
            ProgressivePolicy.from_state(policy.state_dict(), altered)


def test_all_removed_and_protection():
    alpha = torch.ones(1, 3)
    policy = ClippedResetPolicy(alpha, config(cap_fraction=1.0))
    policy.record(1, alpha - 0.2)
    alpha.sub_(0.4)
    event = policy.record(2, alpha)
    assert event["terminal_reason"] == "all_heads_removed" and not alpha.any()
    assert event["selected_count"] == 3 and event["shortfall"] == 0
    alpha.add_(1)
    policy.protect(alpha)
    assert not alpha.any()
    clipped_gate(alpha.requires_grad_(), policy.free).sum().backward()
    assert not alpha.grad.any()


def test_raw_sparsity_and_fixed_denominator():
    alpha = torch.tensor([[-4.0, -3.0, -1.0, 1.0, 4.0]], requires_grad=True)
    alive = torch.tensor([[True, True, True, False, True]])
    value = losses.sparsity_value(clipped_gate(alpha, alive), logits=alpha, kind="relu_logits", free_mask=alive)
    coefficient = losses.sparsity_coefficient(
        50, "soft", lambda_max=0.01, lambda_warmup=100, lambda_start_multiplier=0.1
    )
    (coefficient * value).backward()
    actual = losses.sparsity_gradient(
        alpha,
        update=50,
        phase="soft",
        kind="relu_logits",
        free_mask=alive,
        lambda_max=0.01,
        lambda_warmup=100,
        lambda_start_multiplier=0.1,
    )
    assert torch.equal(alpha.grad, actual)
    assert actual[0, 4] == pytest.approx(coefficient / 5)
    assert not losses.sparsity_gradient(alpha, update=500, phase="recovery").any()


def test_adam_reset_adapter_continuity_and_exact_resume():
    torch.manual_seed(42)
    alpha = torch.nn.Parameter(torch.ones(1, 4))
    adapter = torch.nn.Parameter(torch.ones(1, 4))
    go = torch.optim.AdamW([alpha], lr=0.05, weight_decay=0)
    ao = torch.optim.AdamW([adapter], lr=0.01, weight_decay=0)
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
    saved = copy.deepcopy(
        (
            alpha.detach(),
            adapter.detach(),
            go.state_dict(),
            ao.state_dict(),
            policy.state_dict(),
            checkpoints.rng_state(),
        )
    )
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


def test_global_budget_separates_selection_and_recovery():
    from scripts.prune.learn import protocol

    cfg = dict(soft_updates=500, binary_updates=0, compact_updates=0, gate_lr=0.01, progressive_policy=dict(DEFAULTS))
    assert protocol.phase_budget(cfg) == (500, 500, 800)
    with pytest.raises(ValueError, match="budget differs"):
        protocol.phase_budget(dict(cfg, soft_updates=800))
    with pytest.raises(ValueError, match="full-speed drop"):
        protocol.phase_budget(dict(cfg, gate_lr=0.02))


def test_pre_event_transaction_and_canonical_checkpoint(tmp_path):
    from scripts.prune.learn.adapters import ProjectionAdapter

    gates = HeadGates(1, 3, activation="clip01_v1")
    gates.progressive = new_policy(gates.alpha, config())
    adapter = ProjectionAdapter(torch.nn.Linear(4, 4), 2)
    ao = torch.optim.AdamW([adapter.A, adapter.B], lr=0.001)
    go = torch.optim.AdamW([gates.alpha], lr=0.01, weight_decay=0)
    identity = {"study": "tiny_clip_reset"}
    attached = {"0.attn1.v": adapter}
    for update in (1, 2):
        go.zero_grad()
        gates.microbatch().sum().backward()
        go.step()
        if update == 1:
            gates.progressive.record(update, gates.alpha, go)
            continue
        transaction = tmp_path / "transactions/pre_event_0002.pt"
        kwargs = dict(
            identity=identity, update=2, gates=gates, adapters=attached, adapter_optimizer=ao, gate_optimizer=go
        )
        checkpoints.save_pre_event(transaction, **kwargs)
        # Recover an interruption after atomic tensor publication, before sidecar.
        transaction.with_suffix(".json").unlink()
        checkpoints.save_pre_event(transaction, **kwargs)
        assert transaction.with_suffix(".json").exists()
        saved = torch.load(transaction, map_location="cpu", weights_only=False)
        assert saved["stage"] == "post_optimizer_pre_event"
        assert saved["gate_optimizer"]["state"][0]["step"] == 2
        pre_scores = gates.alpha.detach().clone()
        event = gates.progressive.record(update, gates.alpha, go)
    assert torch.equal(saved["alpha"], pre_scores)
    assert saved["alpha"].tolist() == event["pre_event_scores"]
    path = tmp_path / "checkpoints/update_0002.pt"
    checkpoints.save(path, phase="soft", draws_per_update=4, **kwargs)
    canonical = checkpoints.load(path, identity)
    assert canonical["alpha"].tolist() == event["post_reset_scores"]
    assert canonical["draw_cursor"] == 8 and canonical["gate_optimizer"]["state"] == {}
    assert canonical["progressive"]["event_stage"] == "post_reset"


def test_stop_rule_no_strong_or_no_candidates():
    cfg = dict(config(cap_fraction=1.0), selection_updates=20, stop_rule="no_strong_or_no_candidates")
    policy = ClippedResetPolicy(torch.ones(1, 3), cfg)
    event, _ = run_window(policy, [[0.7, 0.9, 1.0]], [[0.6, 0.8, 1.0]])  # strong + persistent flag
    assert event["strong_count"] == 1 and event["terminal_reason"] is None
    event, _ = run_window(policy, [[0.0, 0.9, 1.0]], [[0.0, 0.8, 1.0]])  # persistent only: commit, then stop
    assert event["selected_ids"] == [[0, 1]] and event["strong_count"] == 0
    assert event["terminal_reason"] == "no_strong_or_no_candidates" and policy.final_update == 304
    assert ProgressivePolicy.from_state(policy.state_dict(), cfg).state_dict() == policy.state_dict()
    empty = ClippedResetPolicy(torch.ones(1, 3), cfg)
    event, _ = run_window(empty, [[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]])
    assert event["terminal_reason"] == "no_strong_or_no_candidates"
    with pytest.raises(ValueError, match="stop_rule"):
        ClippedResetPolicy(torch.ones(1, 3), dict(cfg, stop_rule="never"))
