import itertools

import pytest
import torch

from scripts.prune.learn.gates import HeadGates, nearest_removed, select_legal


def test_init_and_fresh_graphs():
    g = HeadGates(2, 4, [[2, 4]] * 2)
    assert torch.equal(g.alpha, torch.full((2, 4), 3.0))
    g.begin_update("soft")
    first = g.microbatch().detach().clone()
    for _ in range(2):
        value = g.microbatch()
        assert torch.equal(value.detach(), first)
        (value.mean() / 2).backward()
    torch.testing.assert_close(g.alpha.grad, first * (1 - first) / 8)


def test_dp_against_bruteforce_and_infeasible():
    scores = torch.tensor([[1.0, 4.0, 3.0, 2.0], [6.0, 1.0, 2.0, 5.0], [0.0, 5.0, 3.0, 1.0]])
    allowed = [[1, 2, 4]] * 3
    for kept in (3, 4, 5, 6, 7, 8, 9, 10, 12):
        mask = select_legal(scores, kept, allowed)
        candidates = []
        for vector in itertools.product(*allowed):
            if sum(vector) == kept:
                candidates.append(
                    sum(sum(sorted(row.tolist(), reverse=True)[:count]) for row, count in zip(scores, vector))
                )
        assert float((mask * scores).sum()) == max(candidates)
        assert mask.sum() == kept
    with pytest.raises(ValueError, match="infeasible"):
        select_legal(scores, 11, allowed)
    assert torch.equal(
        select_legal(torch.ones_like(scores), 6, allowed), select_legal(torch.ones_like(scores), 6, allowed)
    )
    assert select_legal(torch.ones_like(scores), 6, allowed).sum(1).tolist() == [4.0, 1.0, 1.0]
    assert nearest_removed(2, [[2, 4]] * 2, heads=4) == 2
    assert nearest_removed(1, [[2, 4]] * 2, heads=4) == 0


def test_binary_forward_and_ste():
    g = HeadGates(2, 4, [[2, 4]] * 2)
    g.begin_update("binary", 6)
    m = g.microbatch()
    assert set(m.detach().flatten().tolist()) == {0.0, 1.0}
    assert m.sum() == 6
    m.sum().backward()
    p = g.alpha.detach().sigmoid()
    torch.testing.assert_close(g.alpha.grad, p * (1 - p))
