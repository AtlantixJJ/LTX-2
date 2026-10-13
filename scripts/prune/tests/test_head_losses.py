import pytest
import torch

from scripts.prune.learn.losses import phase_loss, velocity_mse, velocity_target


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
