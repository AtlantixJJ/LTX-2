from __future__ import annotations

import pytest

from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
from scripts.onestep_avatar import sampling


@pytest.fixture
def model_sigmas():
    return list(DISTILLED_SIGMA_VALUES)


def test_one_step_refuses_a_sigma_off_the_distilled_grid(model_sigmas):
    """§2.3(2): the distilled model is a map defined at nine sigmas, not on a continuum."""
    with pytest.raises(ValueError, match="not on the distilled sigma grid"):
        sampling.one_step_schedule(model_sigmas, sigma0=0.8)


def test_one_step_accepts_the_other_two_candidate_sigmas(model_sigmas):
    """§4.2 names exactly three: k1's, k2's and k3's start. C3 may revisit 0.909375."""
    for sigma0 in (0.421875, 0.725, 0.909375):
        assert sampling.one_step_schedule(model_sigmas, sigma0) == [sigma0, 0.0]


def test_one_step_conditions_refuse_a_multi_step_schedule(model_sigmas):
    """§9 risk 13: extra steps are extrapolation for a fixed-sigma adapter, and fail silently."""
    metadata = {"onestep_avatar_sigma0": "0.725"}
    with pytest.raises(ValueError, match="Extra steps are extrapolation"):
        sampling.assert_one_step_conditions(metadata, 0.725, [0.725, 0.421875, 0.0])


def test_one_step_conditions_refuse_a_sigma_the_checkpoint_was_not_trained_at(model_sigmas):
    metadata = {"onestep_avatar_sigma0": "0.725"}
    with pytest.raises(ValueError, match="trained at sigma0"):
        sampling.assert_one_step_conditions(metadata, 0.909375, sampling.one_step_schedule(model_sigmas, 0.909375))


def test_one_step_conditions_pass_on_matching_conditions(model_sigmas):
    metadata = {"onestep_avatar_sigma0": "0.725"}
    sampling.assert_one_step_conditions(metadata, 0.725, sampling.one_step_schedule(model_sigmas))


def test_one_step_conditions_are_permissive_about_an_unlabelled_checkpoint(model_sigmas):
    """A checkpoint from before the metadata existed is not rejected -- only a CONTRADICTION is.
    The step-count check still applies, since it needs no metadata at all."""
    sampling.assert_one_step_conditions({}, 0.725, sampling.one_step_schedule(model_sigmas))
