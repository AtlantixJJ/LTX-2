from __future__ import annotations

import pytest
import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
from scripts.onestep_avatar.model import sampling as model_sampling


@pytest.fixture
def model_sigmas():
    return list(DISTILLED_SIGMA_VALUES)


def test_one_step_refuses_a_sigma_off_the_distilled_grid(model_sigmas):
    """§2.3(2): the distilled model is a map defined at nine sigmas, not on a continuum."""
    with pytest.raises(ValueError, match="not on the distilled sigma grid"):
        model_sampling.one_step_schedule(model_sigmas, sigma0=0.8)


def test_one_step_accepts_the_other_two_candidate_sigmas(model_sigmas):
    """§4.2 names exactly three: k1's, k2's and k3's start. C3 may revisit 0.909375."""
    for sigma0 in (0.421875, 0.725, 0.909375):
        assert model_sampling.one_step_schedule(model_sigmas, sigma0) == [sigma0, 0.0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_positive_euler_steps_match_native_operation_order(dtype):
    generator = torch.Generator().manual_seed(91)
    initial = torch.randn(1, 1024, 128, generator=generator).to(dtype)
    prediction = torch.randn(initial.shape, generator=generator).to(dtype)
    levels = torch.tensor([1.0, 0.725, 0.421875], dtype=torch.float32)
    actual = initial.clone()
    reference = initial.clone()
    for index in range(2):
        actual = model_sampling.euler_to(actual, prediction, float(levels[index]), float(levels[index + 1]))
        reference = EulerDiffusionStep().step(reference, prediction, levels, index)
        assert torch.equal(actual, reference)
    if dtype == torch.bfloat16:
        interpolation = prediction + float(levels[1]) * (initial - prediction)
        first_native = EulerDiffusionStep().step(initial, prediction, levels, 0)
        assert not torch.equal(interpolation, first_native)


def test_direct_endpoint_is_exact_and_stock_bf16_rounding_is_distinct():
    generator = torch.Generator().manual_seed(92)
    initial = torch.randn(1, 1024, 128, generator=generator).bfloat16()
    prediction = torch.randn(initial.shape, generator=generator).bfloat16()
    actual = model_sampling.euler_to(initial, prediction, 0.725, 0.0)
    stock = EulerDiffusionStep().step(initial, prediction, torch.tensor([0.725, 0.0]), 0)
    assert torch.equal(actual, prediction)
    assert not torch.equal(stock, prediction)
