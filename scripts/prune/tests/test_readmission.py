"""Independent V3 event, boundary, optimizer and checkpoint controls."""

import copy

import pytest
import torch

from ltx_core.model.transformer.model import X0Model
from scripts.prune.core.records import unchanged_state_hash
from scripts.prune.learn import checkpoints, joint_training
from scripts.prune.learn.readmission import Family, JointPolicy, budget_projection
from scripts.prune.tests.test_ffn_learning import Velocity, case, config


def window(f, endpoints):
    start = f.kept.float()
    end = torch.as_tensor(endpoints, dtype=torch.float32).expand_as(start)
    for n in range(1, 51):
        f.observe(start + (end - start) * n / 50)


def test_remove_return_second_removal_is_permanent_and_excludes_penalty():
    f = Family((1, 100), strong=0.25, persistent=0.15, minimum=1)
    endpoint = torch.ones(1, 100)
    endpoint[0, :4] = 0.6
    window(f, endpoint)
    assert f.event(50)["selected"].tolist() == [0, 1, 2, 3]
    endpoint = f.kept.float()
    endpoint[0, :4] = 0.4
    window(f, endpoint)
    assert f.event(100)["readmitted"].tolist() == [0, 1, 2]  # cap floor(.04*96)
    assert f.returned[0, :3].all()
    endpoint = f.kept.float()
    endpoint[0, :3] = 0.8  # persistent cannot remove returned units
    for t in (150, 200):
        window(f, endpoint)
        assert f.event(t)["selected"].numel() == 0
    endpoint[0, :3] = 0.6
    window(f, endpoint)
    f.event(250)
    assert f.permanent[0, :3].all()
    alpha = torch.nn.Parameter(f.kept.float())
    (f.compose(alpha).sum() + f.penalty(alpha)).backward()
    assert torch.equal(alpha.grad[0, :3], torch.zeros(3))
    assert alpha.grad[0, 3] > 0  # returnable removed gate still trains


def test_block_caps_minimum_ties_and_alignment_order():
    f = Family((2, 128), strong=0.35, persistent=0.25, minimum=4, block_cap=0.25)
    window(f, 0.5)
    e = f.event(50)
    assert len(e["selected"]) == 10
    assert e["selected"].tolist() == list(range(10))
    f.kept[0] = False
    f.kept[0, :4] = True
    f.reset_summary(f.kept.float())
    window(f, 0.5)
    assert all(i >= 128 for i in f.event(100)["selected"].tolist())
    f.permanent[0, 4] = True
    f.last_rise[0, 4] = 100
    restored = f.align(8)
    assert (0, 4) not in restored  # returnable before permanent regardless of rise
    assert all(n % 8 == 0 for n in f.kept.sum(1).tolist())


def test_persistence_giveback_late_and_pre_event_counts():
    f = Family((1, 100), strong=0.25, persistent=0.15, minimum=1)
    window(f, 0.8)
    assert f.event(50)["selected"].numel() == 0
    window(f, 0.8)
    assert f.event(100)["selected"].numel() == 4
    # Removed IDs cannot be proposed for removal and return at one event.
    window(f, 0.4)
    e = f.event(150)
    assert not set(e["selected"].tolist()) & set(e["readmitted"].tolist())
    assert e["cap"] == 3


def test_quiet_stop_and_persistent_only_activity():
    p = JointPolicy((1, 100), (1, 128), alignment=4)
    h, c = torch.ones(1, 100), torch.ones(1, 128)
    for n in range(1, 201):
        if n > 100:
            h.fill_(1 - 0.2 * ((n - 1) % 50 + 1) / 50)
        e = p.record(n, h, c, [])
    assert e["families"]["heads"]["eligible_removals"] == 100
    assert p.quiet == 0 and not p.recovering  # persistent-only removal prevents stop
    p = JointPolicy((1, 100), (1, 128), alignment=4)
    for n in range(1, 201):
        p.record(n, torch.ones_like(h), c, [])
    assert p.selection_end == 200 and p.final_update == 400


def test_return_only_hard_limit_and_resume_midwindow():
    p = JointPolicy((1, 100), (1, 128), alignment=4)
    p.update = 1400
    h, c = torch.ones(1, 100), torch.ones(1, 128)
    for n in range(1401, 1501):
        h.fill_(1 - 0.5 * ((n - 1) % 50 + 1) / 50)
        e = p.record(n, h, c, [])
        if n == 1421:
            p = JointPolicy.from_state(p.state_dict())
    assert e["return_only"] and p.selection_end == 1500
    assert p.heads.kept.all() and p.stop_reason == "hard_limit"


def test_budget_guard_prefix_and_stable_ties():
    h, c = torch.zeros(2, 2), torch.zeros(2, 8)
    hm, cm, receipt = budget_projection(h, c, 15, cost=4, minimum=2)
    assert receipt == dict(spent=12, slack=3)  # next head costs 4; stop instead of filling channels
    assert hm.tolist() == [[1.0, 0.0], [1.0, 0.0]]
    assert cm.tolist() == [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2
    with pytest.raises(ValueError):
        budget_projection(h, c, 11, cost=4, minimum=2)


def make_step():
    cfg = config()
    cfg.update(
        arm="joint_clip_reset_v3",
        rho=1 / 704,
        policy=dict(interval=2, late=1, earliest=4, limit=6, recovery=2, alignment=4),
    )
    torch.manual_seed(42)
    return joint_training.JointStep(X0Model(Velocity()), cfg, blocks=1, heads=4)


@pytest.mark.parametrize("saved_update", [1, 2])
def test_native_tiny_graph_exact_resume_and_fp32_trace(tmp_path, saved_update):
    torch.set_num_threads(1)
    step = make_step()
    ao, go = joint_training.optimizers(step)
    for number in range(1, saved_update + 1):
        joint_training.update(step, step, case(number=number), number, ao, go)
    payload = joint_training.state(
        step,
        ao,
        go,
        study="tiny",
        update=saved_update,
        world_size=1,
        rng=[checkpoints.rng_state()],
        input_chain="initial",
    )
    path = tmp_path / "state.pt"
    joint_training.save(path, payload)
    for number in range(saved_update + 1, 5):
        joint_training.update(step, step, case(number=number), number, ao, go)
    expected = joint_training.state(
        step, ao, go, study="tiny", update=4, world_size=1, rng=[checkpoints.rng_state()], input_chain="initial"
    )
    joint_training.restore(step, ao, go, joint_training.read(path, world_size=1))
    for number in range(saved_update + 1, 5):
        joint_training.update(step, step, case(number=number), number, ao, go)
    repeated = joint_training.state(
        step, ao, go, study="tiny", update=4, world_size=1, rng=[checkpoints.rng_state()], input_chain="initial"
    )
    assert unchanged_state_hash(expected) == unchanged_state_hash(repeated)
    for opt in go.items:
        assert not opt.state  # event cleared moments AND steps
    gradients = dict(channels=torch.full((1, 16), 1e-9, dtype=torch.float32))
    torch.save(gradients, tmp_path / "grad.pt")
    assert torch.equal(torch.load(tmp_path / "grad.pt", weights_only=True)["channels"], gradients["channels"])
    step.close()


def v3_ddp_worker(rank, store):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + store, rank=rank, world_size=4)
    step, serial = make_step(), make_step()
    wrapped = DistributedDataParallel(step, broadcast_buffers=False)
    for target in (step, serial):
        for gates in (target.heads, target.ffn):
            gates.family.kept[0, :2] = False
            gates.family.permanent[0, 1] = True
        target.policy.reset(target.heads.alpha, target.ffn.alpha, [])
    c = case(rank)
    wrapped(c[1], c[4], c[3], 1, 1)[0].backward()
    for other in range(4):
        c = case(other)
        (serial(c[1], c[4], c[3], 1, 1)[0] / 4).backward()
    for actual, expected in zip(step.parameters(), serial.parameters(), strict=True):
        if actual.requires_grad:
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-7, rtol=2e-5)
    for g in (step.heads, step.ffn):
        assert g.alpha.grad[0, 0] != 0
        assert g.alpha.grad[0, 1] == 0
    step.close()
    serial.close()
    dist.destroy_process_group()


def test_v3_ddp_averages_once_and_permanence_masks_total_loss(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(v3_ddp_worker, args=(str(tmp_path / "store"),), nprocs=4, start_method="fork", join=True)


@pytest.mark.parametrize("arm", ["joint_clip_reset_v3", "heads_clip_reset_v3", "budget_joint_v3"])
def test_recipe_loss_and_optimizer_separation(arm):
    from scripts.prune.learn import adapters

    step = make_step()
    # Rebuild with the actual arm so adapter inventory is independent of the
    # configuration mutation used in earlier numerical fixtures.
    cfg = copy.deepcopy(step.config)
    step.close()
    cfg["arm"] = arm
    if arm == "budget_joint_v3":
        cfg["selection_updates"] = 200
        cfg["budget"] = 100
    torch.manual_seed(42)
    step = joint_training.JointStep(X0Model(Velocity()), cfg, blocks=1, heads=4)
    ao, go = joint_training.optimizers(step)
    assert len(go.items) == 2
    assert go.items[1].param_groups[0]["eps"] == 1e-12
    assert go.items[0].param_groups[0]["eps"] == (1e-12 if arm == "budget_joint_v3" else 1e-8)
    if arm == "heads_clip_reset_v3":
        assert all(".ff." not in n for n in adapters.state_dict(step.attached))
        assert step.ffn.family.config["enabled"] is False
    c = case()
    step.configure(1)
    loss, official, penalty = step(c[1], c[4], c[3], 1, 1)
    loss.backward()
    assert float(penalty) == 0 if arm == "budget_joint_v3" else float(penalty) > 0
    if arm == "heads_clip_reset_v3":
        assert step.ffn.alpha.grad.eq(0).all()
    step.close()


def test_reject_checkpoint_score_summary_divergence(tmp_path):
    step = make_step()
    ao, go = joint_training.optimizers(step)
    joint_training.update(step, step, case(), 1, ao, go)
    state = joint_training.state(
        step, ao, go, study="tiny", update=1, world_size=1, rng=[checkpoints.rng_state()], input_chain="initial"
    )
    state["ffn_alpha"][0, 0] += 0.1
    with pytest.raises(ValueError, match="window summary"):
        joint_training.restore(step, ao, go, state)
    step.close()


def test_c_exact_resume_with_projected_masks(tmp_path):
    cfg = make_step().config
    cfg = copy.deepcopy(cfg)
    cfg.update(arm="budget_joint_v3", selection_updates=200, budget=708, budget_ramp_start=0, budget_ramp_end=1)
    torch.manual_seed(42)
    step = joint_training.JointStep(X0Model(Velocity()), cfg, blocks=1, heads=4)
    ao, go = joint_training.optimizers(step)
    joint_training.update(step, step, case(number=1), 1, ao, go)
    payload = joint_training.state(
        step, ao, go, study="tiny", update=1, world_size=1, rng=[checkpoints.rng_state()], input_chain="initial"
    )
    path = tmp_path / "c.pt"
    joint_training.save(path, payload)
    joint_training.update(step, step, case(number=2), 2, ao, go)
    expected = joint_training.state(
        step, ao, go, study="tiny", update=2, world_size=1, rng=[checkpoints.rng_state()], input_chain="initial"
    )
    joint_training.restore(step, ao, go, joint_training.read(path))
    joint_training.update(step, step, case(number=2), 2, ao, go)
    actual = joint_training.state(
        step, ao, go, study="tiny", update=2, world_size=1, rng=[checkpoints.rng_state()], input_chain="initial"
    )
    assert unchanged_state_hash(expected) == unchanged_state_hash(actual)
    assert step.projection["spent"] <= 708
    step.close()


def test_clipped_gate_forward_surrogate_endpoints():
    from torch.utils.checkpoint import checkpoint

    from scripts.prune.learn.readmission import clipped_gate

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
