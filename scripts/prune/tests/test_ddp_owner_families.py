"""Each DDP family module exposes the complete hook set the shared owner calls."""

import pytest

from scripts.prune.learn import ffn, joint

HOOKS = (
    "state",
    "label",
    "script",
    "native_fixture",
    "native_prepare",
    "native_first_checks",
    "native_freeze",
    "check_start",
    "after_update",
    "receipt_fields",
    "reviews",
    "checkpoint_due",
    "after_checkpoint",
    "monitor_due",
    "final_update",
)
STATE = ("JointStep", "optimizers", "update", "state", "save", "read", "restore")


@pytest.mark.parametrize("family", [ffn, joint])
def test_family_hooks_and_state_owner(family):
    assert [name for name in HOOKS if not hasattr(family, name)] == []
    assert [name for name in STATE if not hasattr(family.state, name)] == []


def test_ffn_and_v3_boundaries():
    cfg = dict(selection_updates=1000, head_soft_updates=700, monitor_interval=50, checkpoint_interval=50)

    class Ctx:
        pass

    ctx = Ctx()
    ctx.cfg = cfg
    assert ffn.checkpoint_due(ctx, 700, 1300) and ffn.checkpoint_due(ctx, 1300, 1300)
    assert not ffn.checkpoint_due(ctx, 701, 1300)
    assert ffn.monitor_due(ctx, 1000, 1300) and not ffn.monitor_due(ctx, 1001, 1300)
    assert not joint.monitor_due(ctx, 100, 1700) and joint.monitor_due(ctx, 150, 1700)
    with pytest.raises(ValueError, match="pilot"):
        ffn.check_start({}, None, cfg, pilot=True, max_updates=10)
