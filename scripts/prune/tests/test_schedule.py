from __future__ import annotations

import pytest

from scripts.prune.core import refine_task


def test_k2_is_the_deployed_two_forward_schedule(model):
    assert refine_task.schedule_for(model.sigmas, "k2") == [0.725, 0.421875, 0.0]


def test_k_step_tails_nest(model):
    sigmas = model.sigmas
    assert refine_task.schedule_for(sigmas, "k1") == [0.421875, 0.0]
    assert refine_task.schedule_for(sigmas, "k8") == sigmas
    for shorter, longer in (("k1", "k2"), ("k2", "k3"), ("k3", "k4")):
        assert refine_task.schedule_for(sigmas, shorter) == refine_task.schedule_for(sigmas, longer)[1:]


def test_unknown_k_step_raises(model):
    with pytest.raises(ValueError, match="Unknown k_step"):
        refine_task.schedule_for(model.sigmas, "k5")
