import torch

from scripts.prune.learn.adapters import ProjectionAdapter, transfer, transfer_optimizer


def test_zero_fusion_and_frozen_weights():
    base = torch.nn.Linear(8, 8).to(torch.bfloat16)
    before = base.weight.clone()
    adapter = ProjectionAdapter(base, rank=2, generator=torch.Generator().manual_seed(99))
    x = torch.randn(1, 4, 8).to(torch.bfloat16)
    assert torch.equal(adapter(x), base(x))
    adapter(x).float().square().mean().backward()
    assert base.weight.grad is None and torch.equal(base.weight, before)
    assert adapter.B.grad is not None and adapter.B.grad.norm() > 0
    adapter.B.data.fill_(0.1)
    torch.testing.assert_close(adapter.merged_weight(), (before.float() + adapter.B @ adapter.A).to(torch.bfloat16))


def test_original_ids_and_moments_slice():
    state = {
        "0.attn1.v.A": torch.randn(2, 8),
        "0.attn1.v.B": torch.arange(16.0).reshape(8, 2),
        "0.attn1.out.A": torch.arange(16.0).reshape(2, 8),
        "0.attn1.out.B": torch.randn(8, 2),
    }
    mask = torch.tensor([[1, 0, 1, 0]])
    compact = transfer(state, mask, head_dim=2)
    indices = [0, 1, 4, 5]
    assert torch.equal(compact["0.attn1.v.B"], state["0.attn1.v.B"][indices])
    assert torch.equal(compact["0.attn1.out.A"], state["0.attn1.out.A"][:, indices])
    opt = {
        "param_groups": [{"params": [0, 1, 2, 3]}],
        "state": {
            i: {"step": torch.tensor(7.0), "exp_avg": v.clone(), "exp_avg_sq": v.square()}
            for i, v in enumerate(state.values())
        },
    }
    result = transfer_optimizer(opt, list(state), mask, head_dim=2)
    for i, name in enumerate(state):
        assert result["state"][i]["step"] == 7 and torch.equal(result["state"][i]["exp_avg"], compact[name])
