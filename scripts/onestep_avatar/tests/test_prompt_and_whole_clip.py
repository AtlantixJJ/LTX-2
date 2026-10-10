"""CPU tests for prompt selection, dev schedules and guidance passes.

Prompt selection must default to ``REFINE_PROMPT`` so every saved run reproduces, and the
hash recorded in a manifest must be the hash of the text actually encoded. The whole-clip
producer's one-block plan and option checks live in
``scripts/prune/tests/test_whole_clip_baseline.py``.
"""

from __future__ import annotations

import argparse
import itertools

import pytest
import torch

from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.modality import Modality
from scripts.onestep_avatar.model import causal as causal_core
from scripts.onestep_avatar.model import common
from scripts.onestep_avatar.model import sampling as model_sampling
from scripts.prune.core.session import DEFAULT_PROMPT, add_prompt_args, resolve_prompt

SCALE = causal_core.SpatioTemporalScaleFactors(8, 32, 32)


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_prompt_args(parser)
    return parser.parse_args(argv)


def test_default_prompt_is_refine_prompt() -> None:
    assert resolve_prompt(_parse([])) == DEFAULT_PROMPT


def test_prompt_text_and_file(tmp_path) -> None:  # noqa: ANN001
    assert resolve_prompt(_parse(["--prompt", ""])) == ""
    path = tmp_path / "p.txt"
    path.write_text("a person on plain white\n", encoding="utf-8")
    assert resolve_prompt(_parse(["--prompt-file", str(path)])) == "a person on plain white"


def test_prompt_flags_are_exclusive(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(SystemExit):
        _parse(["--prompt", "x", "--prompt-file", str(tmp_path / "p.txt")])


# --- dev model: rescaled schedule and guidance passes -------------------------------------

SHAPE = (1, 128, 18, 32, 32)


@pytest.mark.parametrize("sigma", [0.421875, 0.725, 0.909375, 0.975, 1.0])
@pytest.mark.parametrize("steps", [1, 4, 8, 16, 30])
def test_rescaled_schedule_shape(sigma: float, steps: int) -> None:
    schedule = model_sampling.rescaled_schedule(sigma, steps)
    assert len(schedule) == steps + 1
    assert schedule[0] == sigma
    assert schedule[-1] == 0.0
    assert all(b < a for a, b in itertools.pairwise(schedule))


def test_rescaled_schedule_at_one_is_stock() -> None:

    stock = LTX2Scheduler().execute(steps=30).tolist()
    assert model_sampling.rescaled_schedule(1.0, 30) == pytest.approx(tuple(stock))


def test_rescaled_schedule_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="sigma_start"):
        model_sampling.rescaled_schedule(0.0, 8)
    with pytest.raises(ValueError, match="steps"):
        model_sampling.rescaled_schedule(0.5, 0)


class _FakeX0:
    """Returns a distinct constant per pass so the combination can be checked by hand."""

    num_blocks = 48

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, video, audio, perturbations):  # noqa: ANN001

        del audio
        if perturbations is not None:
            self.calls.append("ptb")
            value = 3.0
        elif video.context is not None and float(video.context.sum()) < 0:
            self.calls.append("uncond")
            value = 2.0
        else:
            self.calls.append("cond")
            value = 5.0
        return torch.full_like(video.latent, value), None


def _modality(context):  # noqa: ANN001, ANN202

    return Modality(
        latent=torch.zeros(1, 4, 2),
        sigma=torch.ones(1),
        timesteps=torch.ones(1, 4),
        positions=torch.zeros(1, 3, 4, 2),
        context=context,
    )


def test_guided_denoise_combines_passes() -> None:

    model = _FakeX0()
    neg = -torch.ones(1, 2, 8)
    guider = MultiModalGuider(MultiModalGuiderParams(cfg_scale=3.0, stg_scale=1.0, stg_blocks=[28]), neg)
    out = common.guided_denoised_from_x0_model(model, guider, neg)(_modality(torch.ones(1, 2, 8)))
    assert model.calls == ["cond", "uncond", "ptb"]
    # Guided value: 5 + (3 - 1) * (5 - 2) + 1 * (5 - 3) = 13.
    assert torch.allclose(out, torch.full_like(out, 13.0))


def test_unguided_is_one_conditional_pass() -> None:

    model = _FakeX0()
    guider = MultiModalGuider(MultiModalGuiderParams())
    out = common.guided_denoised_from_x0_model(model, guider)(_modality(torch.ones(1, 2, 8)))
    assert model.calls == ["cond"]
    assert torch.allclose(out, torch.full_like(out, 5.0))


@pytest.mark.parametrize("steps", [1, 2, 8, 30])
def test_truncated_schedule(steps: int) -> None:

    assert model_sampling.truncated_schedule(1.0, steps) == pytest.approx(
        tuple(LTX2Scheduler().execute(steps=steps).tolist()) if steps > 1 else (1.0, 0.0)
    )
    counts = []
    for sigma in (0.421875, 0.725, 0.909375, 0.975, 1.0):
        sched = model_sampling.truncated_schedule(sigma, steps)
        assert sched[0] == sigma
        assert sched[-1] == 0.0
        assert all(b < a for a, b in itertools.pairwise(sched))
        counts.append(len(sched) - 1)
    assert counts == sorted(counts)  # lower start noise never takes more steps
