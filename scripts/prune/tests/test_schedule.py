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


def test_one_step_is_not_a_k_step_tail(model):
    """The whole reason ``ONE_STEP`` needs its own constructor (plan 2026-09-10 §7.3).

    The distilled grid's tail from 0.725 is ``k2`` -- two forwards. One forward means jumping
    straight to 0, which no slice of the grid expresses, so repurposing ``schedule_for`` would
    have silently produced the k2 baseline under a new name.
    """
    one_step = refine_task.one_step_schedule(model.sigmas)
    assert one_step == [0.725, 0.0]
    assert len(one_step) - 1 == 1
    assert one_step != refine_task.schedule_for(model.sigmas, "k2")
    assert one_step not in [refine_task.schedule_for(model.sigmas, k) for k in ("k1", "k2", "k3", "k4", "k8")]


def test_one_step_refuses_a_sigma_off_the_distilled_grid(model):
    """§2.3(2): the distilled model is a map defined at nine sigmas, not on a continuum."""
    with pytest.raises(ValueError, match="not on the distilled sigma grid"):
        refine_task.one_step_schedule(model.sigmas, sigma0=0.8)


def test_one_step_accepts_the_other_two_candidate_sigmas(model):
    """§4.2 names exactly three: k1's, k2's and k3's start. C3 may revisit 0.909375."""
    for sigma0 in (0.421875, 0.725, 0.909375):
        assert refine_task.one_step_schedule(model.sigmas, sigma0) == [sigma0, 0.0]


def test_one_step_conditions_refuse_a_multi_step_schedule(model):
    """§9 risk 13: extra steps are extrapolation for a fixed-sigma adapter, and fail silently."""
    metadata = {"onestep_avatar_sigma0": "0.725"}
    with pytest.raises(ValueError, match="Extra steps are extrapolation"):
        refine_task.assert_one_step_conditions(metadata, 0.725, refine_task.schedule_for(model.sigmas, "k2"))


def test_one_step_conditions_refuse_a_sigma_the_checkpoint_was_not_trained_at(model):
    metadata = {"onestep_avatar_sigma0": "0.725"}
    with pytest.raises(ValueError, match="trained at sigma0"):
        refine_task.assert_one_step_conditions(
            metadata, 0.909375, refine_task.one_step_schedule(model.sigmas, 0.909375)
        )


def test_one_step_conditions_pass_on_matching_conditions(model):
    metadata = {"onestep_avatar_sigma0": "0.725"}
    refine_task.assert_one_step_conditions(metadata, 0.725, refine_task.one_step_schedule(model.sigmas))


def test_one_step_conditions_are_permissive_about_an_unlabelled_checkpoint(model):
    """A checkpoint from before the metadata existed is not rejected -- only a CONTRADICTION is.
    The step-count check still applies, since it needs no metadata at all."""
    refine_task.assert_one_step_conditions({}, 0.725, refine_task.one_step_schedule(model.sigmas))
