import pytest
import torch

from scripts.prune.learn.losses import (
    phase_loss,
    sparsity_coefficient,
    sparsity_gradient,
    velocity_mse,
    velocity_target,
)


def test_official_target_mask_sign_and_regions():
    y = torch.arange(24.0).reshape(1, 8, 3)
    e = -y
    v = velocity_target(e, y)
    v[:, :2] = 999
    assert velocity_mse(v, e, y, 2)["official"] == 0
    v[:, 2:] += 2
    w = torch.zeros(1, 8)
    w[:, 3:5] = 0.5
    terms = velocity_mse(v, e, y, 2, {"person": w, "left_hand": w, "right_hand": w * 0})
    assert terms["official"] == 4 and terms["person"] == 4 and terms["hands"] == 4
    assert terms["coverage"]["right_hand"] == 0
    with pytest.raises(ValueError, match="empty person"):
        velocity_mse(v, e, y, 2, {"person": w * 0})


def test_penalty_accumulation_and_disabling():
    alpha = torch.full((48, 32), 3.0, requires_grad=True)
    for _ in range(2):
        loss, _ = phase_loss({"official": torch.tensor(0.0), "coverage": {}}, alpha.sigmoid(), update=300, phase="soft")
        loss.backward()
    p = alpha.detach().sigmoid()
    torch.testing.assert_close(alpha.grad, 0.001 * p * (1 - p) / 1536)
    assert abs(float(alpha.grad[0, 0]) - 2.941186e-8) < 1e-12
    for phase in ("binary", "compact"):
        _, penalty = phase_loss({"official": torch.tensor(1.0), "coverage": {}}, p, update=800, phase=phase)
        assert penalty == 0


def test_matches_official_fp32_masked_routine():
    from ltx_trainer.training_strategies.flexible import FlexibleStrategy

    e, y, v = [torch.randn(1, 10, 4) for _ in range(3)]
    mask = torch.ones(1, 10, dtype=torch.bool)
    mask[:, :2] = False
    stock = FlexibleStrategy._compute_modality_loss(v, e - y, mask).mean()
    torch.testing.assert_close(velocity_mse(v, e, y, 2)["official"], stock)


@pytest.mark.parametrize("update", [1, 150, 300, 547])
def test_raw_logit_l1_floor_warmup_and_two_clip_accumulation(update):
    alpha = torch.tensor([-4.0, -3.0, -2.0, 0.0, 3.0], requires_grad=True)
    for _ in range(2):
        loss, penalty = phase_loss(
            {"official": torch.tensor(0.0), "coverage": {}},
            alpha.sigmoid(),
            update=update,
            phase="soft",
            lambda_max=1.0,
            logits=alpha,
            sparsity_kind="relu_logits",
        )
        loss.backward()
    coefficient = min(update / 300, 1.0)
    torch.testing.assert_close(penalty, torch.tensor(2.0 * coefficient))
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]) * coefficient / 5
    torch.testing.assert_close(alpha.grad, expected)
    torch.testing.assert_close(
        alpha.grad, sparsity_gradient(alpha, update=update, phase="soft", lambda_max=1, kind="relu_logits")
    )


@pytest.mark.parametrize("phase", ["binary", "compact"])
def test_raw_logit_penalty_is_disabled_after_soft_selection(phase):
    alpha = torch.full((2, 3), 3.0, requires_grad=True)
    _, penalty = phase_loss(
        {"official": torch.tensor(1.0), "coverage": {}},
        alpha.sigmoid(),
        update=800,
        phase=phase,
        lambda_max=1,
        logits=alpha,
        sparsity_kind="relu_logits",
    )
    assert penalty == 0
    assert torch.count_nonzero(sparsity_gradient(alpha, update=800, phase=phase, lambda_max=1, kind="relu_logits")) == 0


@pytest.mark.parametrize("update,expected", [(1, 0.0109), (50, 0.055), (100, 0.1), (200, 0.1)])
def test_nonzero_start_ramp_matches_accumulated_autograd(update, expected):
    schedule = {"lambda_max": 0.1, "lambda_warmup": 100, "lambda_start_multiplier": 0.1}
    assert sparsity_coefficient(update, "soft", **schedule) == pytest.approx(expected)
    alpha = torch.tensor([-4.0, -3.0, -2.0, 0.0, 3.0], requires_grad=True)
    for _ in range(2):
        loss, penalty = phase_loss(
            {"official": torch.tensor(0.0), "coverage": {}}, alpha.sigmoid(),
            update=update, phase="soft", logits=alpha, sparsity_kind="relu_logits", **schedule,
        )
        loss.backward()
    assert float(penalty.detach()) == pytest.approx(2 * expected)
    torch.testing.assert_close(
        alpha.grad, sparsity_gradient(alpha, update=update, phase="soft", kind="relu_logits", **schedule)
    )
    torch.testing.assert_close(alpha.grad[:2], torch.zeros(2))
    for phase in ("binary", "compact", "fixed"):
        assert sparsity_coefficient(update, phase, **schedule) == 0


@pytest.mark.parametrize("bad", [
    {"lambda_max": float("nan")}, {"lambda_max": float("inf")},
    {"lambda_warmup": 0}, {"lambda_warmup": -1},
    {"lambda_start_multiplier": -0.1}, {"lambda_start_multiplier": 1.1},
])
def test_invalid_sparsity_ramps_fail(bad):
    with pytest.raises(ValueError, match="ramp"):
        sparsity_coefficient(1, "soft", **bad)
