import pytest

from scripts.prune.learn import protocol


def test_frozen_budgets_give_global_boundaries():
    assert protocol.phase_budget({"soft_updates": 700, "binary_updates": 300, "compact_updates": 500}) == (
        700,
        1000,
        1500,
    )
    # Extended soft selection without binary refinement: selection ends where soft training ends.
    assert protocol.phase_budget({"soft_updates": 2000, "binary_updates": 0, "compact_updates": 500}) == (
        2000,
        2000,
        2500,
    )


@pytest.mark.parametrize(
    "budget",
    [
        {"soft_updates": 0, "binary_updates": 300, "compact_updates": 500},
        {"soft_updates": 700, "binary_updates": -1, "compact_updates": 500},
        {"soft_updates": 700, "binary_updates": 300, "compact_updates": 0},
        {"soft_updates": 700.0, "binary_updates": 300, "compact_updates": 500},
    ],
)
def test_invalid_budgets_fail(budget):
    with pytest.raises(ValueError, match="phase budgets"):
        protocol.phase_budget(budget)


def test_missing_budget_fails():
    with pytest.raises(KeyError):
        protocol.phase_budget({"soft_updates": 700, "binary_updates": 300})


def test_recovery_end_follows_the_frozen_budget():
    default = {"soft_updates": 700, "binary_updates": 300, "compact_updates": 500}
    extended = {"soft_updates": 2000, "binary_updates": 0, "compact_updates": 500}
    assert protocol.recovery_end(default, "learned_joint") == 1500
    assert protocol.recovery_end(default, "learned_sequential") == 2500
    assert protocol.recovery_end(extended, "learned_joint") == 2500
