"""CPU tests for causal-mode training through ``model.causal.train_sample``.

These checks moved from the retired mode-less engine tests (refactor group G11). That
engine's ``train_chain`` only built a token grid from two masters and called
``train_sample``; these tests now call the model owner directly with the same inputs.
Each one pins which tensor the loop hands to which call:

(i)   ``K = 1`` starting at block 0 reproduces a plain non-AR step exactly.
(ii)  ``z_g == z_y`` reduces the objective to the ordinary flow-matching target ``eps - z_y``.
(iii) at fixed sigma_0, velocity MSE times sigma_0**2 equals x0 MSE.
(iv)  block ``i+1``'s cached context comes from block ``i``'s own output, detached -- and
      teacher forcing is exactly the one-tensor swap to the ground truth.

The transformer is a stub on purpose: what is under test is the plumbing. The cache/mask
machinery is tested against a real ``LTXModel`` in ``test_causal_core.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ltx_core.types import SpatioTemporalScaleFactors
from ltx_core.utils import to_velocity
from scripts.onestep_avatar.model import causal as causal_core
from scripts.onestep_avatar.model import common
from scripts.onestep_avatar.model.causal import BlockCache, CausalGeometry

SCALE = SpatioTemporalScaleFactors(time=8, height=32, width=32)
GEOMETRY = CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=2)
CHANNELS = 8
EDGE = 64  # -> a 2x2 latent grid: 4 tokens per latent frame
LATENT_FRAMES = 7  # -> blocks [(0,3), (3,5), (5,7)]
FPS = 30.0
SIGMA0 = 0.725
DEVICE = torch.device("cpu")
DTYPE = torch.bfloat16


class StubTransformer(torch.nn.Module):
    """A trainable stand-in with the real ``(video, audio, perturbations) -> (v, a)`` shape.

    Linear in the latent so an expected gradient can be written down in closed form. It
    ignores the K/V cache entirely; ``test_causal_core`` covers that against a real model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, video, audio, perturbations) -> tuple[torch.Tensor, None]:  # noqa: ANN001, ARG002
        return self.scale * video.latent, None


class CapturingStubTransformer(StubTransformer):
    """Stub that retains modalities so the c0 input contract can be asserted directly."""

    def __init__(self) -> None:
        super().__init__()
        self.modalities = []

    def forward(self, video, audio, perturbations) -> tuple[torch.Tensor, None]:  # noqa: ANN001
        self.modalities.append(video)
        return super().forward(video, audio, perturbations)


def _clip(*, same: bool = False, seed: int = 0) -> SimpleNamespace:
    """Capture ``z_y`` and guide ``z_g`` masters of one clip."""
    generator = torch.Generator().manual_seed(seed)
    shape = (CHANNELS, LATENT_FRAMES, EDGE // SCALE.height, EDGE // SCALE.width)
    z_g = torch.randn(shape, generator=generator)
    z_y = z_g.clone() if same else torch.randn(shape, generator=generator)
    return SimpleNamespace(z_g=z_g, z_y=z_y)


def _grid(geometry: CausalGeometry = GEOMETRY) -> common.ClipGrid:
    return common.ClipGrid.build(
        LATENT_FRAMES, EDGE, EDGE, FPS, geometry, device=DEVICE, dtype=DTYPE, latent_channels=CHANNELS
    )


def _tokens(master: torch.Tensor, grid: common.ClipGrid) -> torch.Tensor:
    return grid.patchify(master.unsqueeze(0).to(device=DEVICE, dtype=DTYPE))


def _cache(grid: common.ClipGrid, geometry: CausalGeometry = GEOMETRY, **kwargs) -> BlockCache:
    return BlockCache.allocate(grid, geometry, num_layers=1, inner_dim=4, device=DEVICE, dtype=DTYPE, **kwargs)


def _run(clip: SimpleNamespace, model: torch.nn.Module, *, blocks: list[int] | None = None,
         geometry: CausalGeometry = GEOMETRY, cache: BlockCache | None = None, **kwargs) -> dict:
    grid = _grid(geometry)
    return causal_core.train_sample(
        model, torch.zeros(1, 1, 8), grid, _tokens(clip.z_y, grid), _tokens(clip.z_g, grid), geometry,
        [0] if blocks is None else blocks, lambda loss: loss.backward(),
        sigma=SIGMA0, seed=0, cache=_cache(grid, geometry) if cache is None else cache, **kwargs,
    )


def _spy_on(monkeypatch: pytest.MonkeyPatch, name: str, capture: list) -> None:
    """Record the tokens handed to ``causal_core.<name>`` as ``train_sample`` calls it."""
    real = getattr(causal_core, name)

    def spy(denoise_fn, grid, cache, tokens, *args, **kwargs):  # noqa: ANN001, ANN202
        capture.append(tokens)
        return real(denoise_fn, grid, cache, tokens, *args, **kwargs)

    monkeypatch.setattr(causal_core, name, spy)


def test_guided_init_is_the_linear_interpolant_of_the_guide(monkeypatch: pytest.MonkeyPatch) -> None:
    """The noisy state is ``(1 - sigma_0) z_g + sigma_0 eps`` -- built from the GUIDE."""
    clip = _clip()
    noisy: list[torch.Tensor] = []
    _spy_on(monkeypatch, "denoise_block", noisy)
    _run(clip, StubTransformer())

    grid = _grid()
    lo, hi = grid.token_span(*GEOMETRY.plan(LATENT_FRAMES)[0])
    z_g_tokens = _tokens(clip.z_g, grid)[:, lo:hi].float()
    eps = (noisy[0].float() - (1 - SIGMA0) * z_g_tokens) / SIGMA0
    assert eps.std().item() > 0.5  # a real N(0, 1) draw, not a copy of z_g
    assert torch.allclose(noisy[0].float(), (1 - SIGMA0) * z_g_tokens + SIGMA0 * eps, atol=8e-3)


def test_full_frame_mse_averages_every_token_and_channel() -> None:
    """The training objective has no subject, alpha, or disagreement weighting."""
    assert common.full_frame_mse(torch.zeros(1, 16, 4), torch.ones(1, 16, 4)).item() == 1.0


def test_identical_guide_and_capture_reduce_to_flow_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``z_g == z_y`` the optimal velocity is exactly ``eps - z_y``."""
    clip = _clip(same=True)
    noisy: list[torch.Tensor] = []
    _spy_on(monkeypatch, "denoise_block", noisy)
    _run(clip, StubTransformer())

    grid = _grid()
    lo, hi = grid.token_span(*GEOMETRY.plan(LATENT_FRAMES)[0])
    z_y_tokens = _tokens(clip.z_y, grid)[:, lo:hi].float()
    eps = (noisy[0].float() - (1 - SIGMA0) * z_y_tokens) / SIGMA0
    v_star = (noisy[0].float() - z_y_tokens) / SIGMA0
    assert torch.allclose(v_star, eps - z_y_tokens, atol=3e-2)


def test_velocity_mse_equals_x0_mse_over_sigma_squared(monkeypatch: pytest.MonkeyPatch) -> None:
    """At fixed sigma_0 the velocity loss already IS an x0 loss, scaled by sigma_0**2."""
    clip = _clip()
    noisy: list[torch.Tensor] = []
    _spy_on(monkeypatch, "denoise_block", noisy)
    model = StubTransformer()
    _run(clip, model)

    grid = _grid()
    span = GEOMETRY.plan(LATENT_FRAMES)[0]
    lo, hi = grid.token_span(*span)
    target = _tokens(clip.z_y, grid)[:, lo:hi]
    z0 = causal_core.denoise_block(
        common.denoised_from_velocity_model(model), grid, _cache(grid), noisy[0], torch.zeros(1, 1, 8), SIGMA0, span
    )
    v_hat = to_velocity(noisy[0], SIGMA0, z0)
    v_star = to_velocity(noisy[0], SIGMA0, target)
    x0_mse = (z0.float() - target.float()).pow(2).mean()
    v_mse = (v_hat.float() - v_star.float()).pow(2).mean()
    assert torch.allclose(v_mse * SIGMA0**2, x0_mse, rtol=1e-3, atol=1e-6)


def test_cache_refresh_uses_the_models_own_output_and_carries_no_grad(monkeypatch: pytest.MonkeyPatch) -> None:
    """What enters the cache must be the MODEL's own output, detached -- not the GT."""
    clip = _clip()
    denoised: list[torch.Tensor] = []
    refreshed: list[torch.Tensor] = []
    _spy_on(monkeypatch, "denoise_block", denoised)
    _spy_on(monkeypatch, "refresh_block", refreshed)
    _run(clip, StubTransformer(), blocks=[0, 1])

    # Two blocks, one refresh: the sample's last refresh is skipped (nothing reads it).
    assert len(refreshed) == 1
    assert not any(tokens.requires_grad for tokens in refreshed)
    grid = _grid()
    lo, hi = grid.token_span(*GEOMETRY.plan(LATENT_FRAMES)[0])
    gt = _tokens(clip.z_y, grid)[:, lo:hi]
    # ...and it is NOT the ground truth, which is what teacher forcing would have put there.
    assert not torch.allclose(refreshed[0].float(), gt.float(), atol=1e-2)


@pytest.mark.parametrize("blocks", [[0], [0, 1, 2], [1, 2]])
def test_the_last_block_of_a_sample_is_not_refreshed(monkeypatch: pytest.MonkeyPatch, blocks: list[int]) -> None:
    """K blocks -> K denoise forwards and K - 1 refresh forwards; K=1 runs no refresh."""
    denoised: list[torch.Tensor] = []
    refreshed: list[torch.Tensor] = []
    _spy_on(monkeypatch, "denoise_block", denoised)
    _spy_on(monkeypatch, "refresh_block", refreshed)
    result = _run(_clip(), StubTransformer(), blocks=blocks)
    assert len(denoised) == len(blocks)
    assert len(refreshed) == len(blocks) - 1
    assert result["denoise_calls"] == len(blocks)
    assert result["refresh_calls"] == len(blocks) - 1


def test_a_clip_start_single_block_sample_runs_on_a_one_frame_cache() -> None:
    """A K=1 clip-start sample writes nothing to the cache, so a one-frame placeholder is
    accepted; a sample that would write (K > 1) must still refuse it."""
    grid = _grid()
    totals = _run(_clip(), StubTransformer(), blocks=[0], cache=_cache(grid, capacity_latent_frames=1))
    assert len(totals["per_block"]) == 1
    with pytest.raises(ValueError, match="cache"):
        _run(_clip(), StubTransformer(), blocks=[0, 1], cache=_cache(grid, capacity_latent_frames=1))


def test_teacher_forcing_refreshes_with_the_gt_instead_of_self_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Teacher forcing is one tensor: the refresh receives the capture target."""
    clip = _clip()
    refreshed: list[torch.Tensor] = []
    _spy_on(monkeypatch, "refresh_block", refreshed)
    _run(clip, StubTransformer(), blocks=[0, 1], teacher_forcing=True)

    grid = _grid()
    target = _tokens(clip.z_y, grid)
    assert refreshed
    for span, tokens in zip(GEOMETRY.plan(LATENT_FRAMES), refreshed, strict=False):
        lo, hi = grid.token_span(*span)
        assert torch.equal(tokens, target[:, lo:hi])


def test_d0_teacher_forcing_conditions_block_zero_on_clean_capture_frame() -> None:
    """D0's source is the capture, but c0 still bypasses noising and AdaLN time."""
    clip = _clip(seed=9)
    model = CapturingStubTransformer()
    _run(clip, model, guide_mode="d0", teacher_forcing=True)

    grid = _grid()
    target = _tokens(clip.z_y, grid)
    denoise = next(
        modality
        for modality in model.modalities
        if not modality.kv_write and torch.count_nonzero(modality.timesteps) > 0
    )
    c0_tokens = grid.tokens_per_latent_frame
    torch.testing.assert_close(denoise.latent[:, :c0_tokens], target[:, :c0_tokens], rtol=0, atol=0)
    assert torch.count_nonzero(denoise.timesteps[:, :c0_tokens]) == 0
    assert torch.all(denoise.timesteps[:, c0_tokens:] == SIGMA0)


def test_a_mid_clip_sample_primes_the_cache_from_the_ground_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sample that starts at block 0 primes nothing; a mid-clip sample primes from the capture."""
    primed: list[tuple[torch.Tensor, int]] = []
    real = causal_core.prime_cache

    def spy(denoise_fn, grid, cache, tokens, geometry, context, *, upto_latent_frame):  # noqa: ANN001, ANN202
        primed.append((tokens, upto_latent_frame))
        return real(denoise_fn, grid, cache, tokens, geometry, context, upto_latent_frame=upto_latent_frame)

    monkeypatch.setattr(causal_core, "prime_cache", spy)
    _run(_clip(), StubTransformer(), blocks=[0])
    assert primed[-1][1] == 0  # block 0 starts at latent frame 0: nothing to prime

    clip = _clip()
    _run(clip, StubTransformer(), blocks=[2])
    tokens, upto = primed[-1]
    assert upto == GEOMETRY.plan(LATENT_FRAMES)[2][0]
    assert torch.equal(tokens, _tokens(clip.z_y, _grid()))


def test_k1_sample_matches_a_single_non_ar_step() -> None:
    """``K = 1`` from block 0 is the teacher-forced control and must be a plain step."""
    clip = _clip(seed=3)
    model_sample = StubTransformer()
    totals = _run(clip, model_sample)
    sample_grad = model_sample.scale.grad.clone()

    model_plain = StubTransformer()
    grid = _grid()
    span = GEOMETRY.plan(LATENT_FRAMES)[0]
    lo, hi = grid.token_span(*span)
    guide = _tokens(clip.z_g, grid)
    target = _tokens(clip.z_y, grid)
    c0 = target[:, : grid.tokens_per_latent_frame]
    noisy = common.with_clean_prefix(common.noise_block(guide[:, lo:hi], SIGMA0, 0), c0)
    z0 = causal_core.denoise_block(
        common.denoised_from_velocity_model(model_plain), grid, _cache(grid), noisy,
        torch.zeros(1, 1, 8), SIGMA0, span, clean_prefix_tokens=c0.shape[1],
    )
    loss = common.full_frame_mse(common.with_clean_prefix(z0, c0), target[:, lo:hi])
    loss.backward()

    assert torch.allclose(sample_grad, model_plain.scale.grad, rtol=1e-4, atol=1e-8)
    assert abs(totals["mse"] - float(loss.detach())) < 1e-5
    assert [entry["block_index"] for entry in totals["per_block"]] == [0]


def test_train_sample_records_one_per_block_entry_in_sample_order() -> None:
    """One per-block entry per block actually run, in order; the mean is their average."""
    totals = _run(_clip(), StubTransformer(), blocks=[0, 1, 2])
    assert [entry["block_index"] for entry in totals["per_block"]] == [0, 1, 2]
    assert abs(totals["mse"] - sum(entry["mse"] for entry in totals["per_block"]) / 3) < 1e-6


def test_d0_noises_the_capture_not_the_guide(monkeypatch: pytest.MonkeyPatch) -> None:
    """D0 noises the loss target ``z_y``; it is a capacity control, not a deployable arm."""
    clip = _clip()  # same=False: z_g and z_y are genuinely different draws
    noisy: list[torch.Tensor] = []
    _spy_on(monkeypatch, "denoise_block", noisy)
    _run(clip, StubTransformer(), guide_mode="d0")

    grid = _grid()
    lo, hi = grid.token_span(*GEOMETRY.plan(LATENT_FRAMES)[0])
    z_g_tokens = _tokens(clip.z_g, grid)[:, lo:hi].float()
    z_y_tokens = _tokens(clip.z_y, grid)[:, lo:hi].float()
    eps = (noisy[0].float() - (1 - SIGMA0) * z_y_tokens) / SIGMA0
    assert eps.std().item() > 0.5
    assert torch.allclose(noisy[0].float(), (1 - SIGMA0) * z_y_tokens + SIGMA0 * eps, atol=8e-3)
    assert not torch.allclose(noisy[0].float(), (1 - SIGMA0) * z_g_tokens + SIGMA0 * eps, atol=0.1)


def test_a_cache_too_small_for_this_clip_is_refused_before_any_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """An undersized cache fails before the step's forwards, so every FSDP rank fails alike."""
    deep = CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=16)
    undersized = _cache(_grid(deep), deep, capacity_latent_frames=3)
    monkeypatch.setattr(causal_core, "prime_cache", lambda *_args, **_kwargs: pytest.fail("forward before refusal"))
    with pytest.raises(ValueError, match="sized from the subset's LONGEST clip"):
        _run(_clip(), StubTransformer(), blocks=[0, 1], geometry=deep, cache=undersized)


def test_source_selection_is_the_one_tensor_the_arm_changes() -> None:
    """D0 noises the capture and D1 the guide; a missing or mismatched guide never falls back."""
    capture = torch.zeros(1, 4, 2)
    guide = torch.ones(1, 4, 2)
    assert common.source_for(capture, guide, "d0") is capture
    assert common.source_for(capture, guide, "d1") is guide
    with pytest.raises(ValueError, match="D1 requires guide data"):
        common.source_for(capture, None, "d1")
    with pytest.raises(ValueError, match="guide shape"):
        common.source_for(capture, torch.ones(1, 3, 2), "d1")
    with pytest.raises(ValueError, match="guide_mode"):
        common.source_for(capture, guide, "d2")


def test_explicit_block_epsilon_reuses_noise_across_sigma_mixtures() -> None:
    clean = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    epsilon = common.epsilon_block(clean, 42)
    assert torch.equal(common.mix_block_noise(clean, epsilon, 1.0), epsilon)
    assert torch.allclose(common.mix_block_noise(clean, epsilon, 0.25), torch.lerp(clean, epsilon, 0.25))


def test_rollout_slices_blocks_out_of_one_master_encode() -> None:
    """Deployment slices the clip's ONE continuous encode, exactly as training does."""
    clip = _clip()
    grid = _grid()
    guide = _tokens(clip.z_g, grid)
    tokens, forwards = causal_core.rollout(
        common.denoised_from_velocity_model(StubTransformer()), grid, GEOMETRY, _cache(grid), guide,
        torch.zeros(1, 1, 8), SIGMA0, first_frame_condition=guide[:, : grid.tokens_per_latent_frame],
    )
    assert forwards == 2 * len(GEOMETRY.plan(LATENT_FRAMES))  # one denoise plus one cache refresh per block
    assert tokens.shape == guide.shape
