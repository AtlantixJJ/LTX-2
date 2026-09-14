"""SS7.1's four CPU tests for the AR training loop -- small dims, no GPU, no checkpoint.

Each one pins a property the plan names, and each is the cheap version of a bug that would
otherwise look like a quality problem in C1 (SS9 risk 12):

(i)   ``K = 1`` with the GT carryover reproduces a plain non-AR step exactly.
(ii)  ``z_g == z_y`` reduces the objective to the ordinary flow-matching target ``eps - z_y``.
(iii) at fixed sigma_0, velocity MSE times sigma_0**2 equals x0 MSE -- SS3's second identity.
(iv)  window ``i+1``'s carryover slot IS window ``i``'s output at ``CARRYOVER_LATENT_IDX``,
      and carries no gradient.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from ltx_core.types import SpatioTemporalScaleFactors
from ltx_core.utils import to_velocity
from ltx_pipelines.utils.types import DenoisedLatentResult
from scripts.onestep_avatar import onestep_core, train
from scripts.prune.core import refine_core

SCALE = SpatioTemporalScaleFactors(time=8, height=32, width=32)
GEOMETRY = refine_core.WindowGeometry(window_frames=25, overlap_frames=9, scale_factors=SCALE)
CHANNELS = 8
EDGE = 64  # -> a 2x2 latent grid: 4 latent frames x 4 positions = 16 tokens
LATENT_FRAMES = GEOMETRY.latent_frames
LATENT_EDGE = EDGE // SCALE.height
SIGMA0 = 0.725
DEVICE = torch.device("cpu")


class StubTransformer(torch.nn.Module):
    """A trainable stand-in with the real ``(video, audio, perturbations) -> (v, a)`` shape.

    Deliberately linear in the latent so an expected gradient can be written down in closed
    form where a test needs one; the point of these tests is the loop's plumbing, not the
    backbone.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, video, audio, perturbations) -> tuple[torch.Tensor, None]:  # noqa: ANN001, ARG002
        return self.scale * video.latent, None


def _window(index: int = 0, *, same: bool = False, seed: int = 0) -> train.Window:
    generator = torch.Generator().manual_seed(seed + index)
    shape = (CHANNELS, LATENT_FRAMES, LATENT_EDGE, LATENT_EDGE)
    z_g = torch.randn(shape, generator=generator)
    z_y = z_g.clone() if same else torch.randn(shape, generator=generator)
    return train.Window(z_g=z_g, z_y=z_y, fps=30.0, index=index, source="stub/view00", loss_mask=None, z0_base=None)


def _forward(window: train.Window, carry: torch.Tensor | None, model: StubTransformer, *, seed: int = 7) -> tuple:
    context = torch.zeros(1, 1, 8)
    return train.one_window_forward(
        model, context, window, carry, GEOMETRY,
        sigma0=SIGMA0, seed=seed, device=DEVICE, latent_channels=CHANNELS,
    )


def test_guided_init_is_the_linear_interpolant_of_the_guide() -> None:
    """SS3: the noisy state is ``(1 - sigma_0) z_g + sigma_0 eps`` -- built from the GUIDE.

    The whole trainer gap (SS1.1 "cannot build the noisy state from a different latent than the
    loss target") is this one line, so it gets its own test rather than being implied.
    """
    window = _window()
    model = StubTransformer()
    _, target_tokens, weights, state, tools = _forward(window, None, model)

    z_g_tokens = tools.patchifier.patchify(window.z_g.unsqueeze(0))
    z_y_tokens = tools.patchifier.patchify(window.z_y.unsqueeze(0))
    # eps recovered from the state; it must be noise, i.e. uncorrelated with either latent.
    eps = (state.latent.float() - (1 - SIGMA0) * z_g_tokens.float()) / SIGMA0

    # bf16 round-trip: the loop casts to the model dtype, so this is the bf16 noise floor.
    assert torch.allclose(target_tokens.float(), z_y_tokens.float(), atol=8e-3)
    assert eps.std().item() > 0.5  # a real N(0, 1) draw, not a copy of z_g
    # With no carryover every token is denoised, so the interpolant holds everywhere.
    assert (weights > 0).all()
    recovered = (1 - SIGMA0) * z_g_tokens.float() + SIGMA0 * eps
    assert torch.allclose(state.latent.float(), recovered, atol=8e-3)


def test_carryover_tokens_are_excluded_from_the_loss() -> None:
    """The frozen carryover and its slot are conditioning, not prediction (SS4.4).

    ``make_window_state`` zeroes ``denoise_mask`` there, and the loop weights the loss by it --
    without that the model would be scored on reproducing a value it was handed, which both
    dilutes the gradient and rewards copying the carryover into the output.
    """
    window = _window()
    model = StubTransformer()
    carry = torch.randn(1, CHANNELS, GEOMETRY.context_latent_frames, LATENT_EDGE, LATENT_EDGE)
    _, _, weights, _, _ = _forward(window, carry, model)

    free = weights.squeeze(-1) > 0
    assert free.any(), "a carryover window must have free tokens"
    assert (~free).any(), "a carryover window must have frozen tokens"
    tokens_per_frame = free.shape[1] // GEOMETRY.latent_frames
    idx = refine_core.CARRYOVER_LATENT_IDX
    frozen = ~free[0]
    assert frozen[idx * tokens_per_frame : (idx + GEOMETRY.context_latent_frames) * tokens_per_frame].all()
    assert free[0, : idx * tokens_per_frame].all()  # the index-0 causal keyframe stays fresh


def test_identical_guide_and_capture_reduce_to_flow_matching() -> None:
    """SS3 identity 1: with ``z_g == z_y`` the optimal velocity is exactly ``eps - z_y``.

    This is what makes SS7.1 a strict generalisation of the shipped trainer rather than a
    separate objective -- if it ever stops holding, the guided init has drifted from the
    linear interpolant the rest of the stack assumes.
    """
    window = _window(same=True)
    model = StubTransformer()
    _, target_tokens, _, state, tools = _forward(window, None, model)

    z_y_tokens = tools.patchifier.patchify(window.z_y.unsqueeze(0)).float()
    eps = (state.latent.float() - (1 - SIGMA0) * z_y_tokens) / SIGMA0
    v_star = (state.latent.float() - target_tokens.float()) / SIGMA0
    assert torch.allclose(v_star, eps - z_y_tokens, atol=3e-2)


def test_velocity_mse_equals_x0_mse_over_sigma_squared() -> None:
    """SS3 identity 2: at fixed sigma_0 the existing velocity loss already IS an x0 loss.

    So the regression term needs no loss-function change -- only a different target -- and
    predicting z0 in the loop (which is what makes the anchor and the capture directly
    comparable) costs nothing in objective terms.
    """
    window = _window()
    model = StubTransformer()
    z0_tokens, target_tokens, _, state, _ = _forward(window, None, model)

    v_hat = to_velocity(state.latent, SIGMA0, z0_tokens)
    v_star = to_velocity(state.latent, SIGMA0, target_tokens)
    x0_mse = (z0_tokens.float() - target_tokens.float()).pow(2).mean()
    v_mse = (v_hat.float() - v_star.float()).pow(2).mean()
    assert torch.allclose(v_mse * SIGMA0**2, x0_mse, rtol=1e-3, atol=1e-6)


def test_carryover_is_the_previous_window_output_and_carries_no_grad() -> None:
    """SS4.4: the carryover must be the MODEL's own output, detached -- not the GT latent.

    Feeding the GT is the teacher forcing the plan removes; detaching is what keeps peak
    activation memory at one window rather than ``K`` (SS8.1).
    """
    model = StubTransformer()
    window0, window1 = _window(0), _window(1)

    z0_tokens, _, _, state, tools = _forward(window0, None, model)
    z0_latent = refine_core.finalize(train.replace(state, latent=z0_tokens), tools)
    carry = refine_core.carry_from(z0_latent, GEOMETRY).detach()
    assert not carry.requires_grad
    assert carry.shape[2] == GEOMETRY.context_latent_frames

    _, _, _, next_state, next_tools = _forward(window1, carry, model)
    clean = next_tools.unpatchify(train.replace(next_state, latent=next_state.clean_latent)).latent
    idx = refine_core.CARRYOVER_LATENT_IDX
    assert torch.allclose(clean[:, :, idx : idx + carry.shape[2]].float(), carry.float(), atol=1e-2)
    # ...and it is NOT the ground truth, which is what teacher forcing would have put there.
    assert not torch.allclose(
        clean[:, :, idx].float(), window1.z_y.unsqueeze(0)[:, :, idx].float(), atol=1e-2
    )


def test_k1_chain_matches_a_single_non_ar_step() -> None:
    """SS7.1 test (i): ``K = 1`` is the teacher-forced control, and must be a plain step.

    A2 uses exactly this as its control arm, so the difference in rollout slope it reports is
    only the value of AR training if ``K = 1`` really is the non-AR baseline.
    """
    window = _window(seed=3)
    chain = train.Chain(
        source="stub/view00", split="train", actor="stub", seed_is_clip_start=True, windows=[window]
    )

    model_chain = StubTransformer()
    accelerator = _StubAccelerator()
    totals = train.train_chain(
        model_chain, torch.zeros(1, 1, 8), chain, GEOMETRY, accelerator,
        sigma0=SIGMA0, seed=0, anchor_weight=0.0, latent_channels=CHANNELS,
    )
    chain_grad = model_chain.scale.grad.clone()

    model_plain = StubTransformer()
    z0_tokens, target_tokens, weights, _, _ = _forward(
        window, None, model_plain, seed=0 + window.index
    )
    loss = train.masked_mse(z0_tokens, target_tokens, weights)
    loss.backward()

    assert torch.allclose(chain_grad, model_plain.scale.grad, rtol=1e-4, atol=1e-8)
    assert abs(totals["mse"] - float(loss.detach())) < 1e-5
    assert totals["anchor"] == 0.0
    assert [w["window_index"] for w in totals["per_window"]] == [window.index]
    assert abs(totals["per_window"][0]["mse"] - float(loss.detach())) < 1e-5


def test_train_chain_records_one_per_window_entry_in_chain_order() -> None:
    """SS7.4(a): "today it writes one row per chain... that effect is unobservable" -- the fix
    is a per-window breakdown alongside the existing chain-mean, in chain order, one entry per
    window actually run (not per corpus window)."""
    window0, window1, window2 = _window(0), _window(1), _window(2)
    chain = train.Chain(
        source="stub/view00", split="train", actor="stub", seed_is_clip_start=True,
        windows=[window0, window1, window2],
    )
    model = StubTransformer()
    totals = train.train_chain(
        model, torch.zeros(1, 1, 8), chain, GEOMETRY, _StubAccelerator(),
        sigma0=SIGMA0, seed=0, anchor_weight=0.0, latent_channels=CHANNELS,
    )

    assert [w["window_index"] for w in totals["per_window"]] == [0, 1, 2]
    # The chain-mean is exactly the average of the per-window entries it was built from --
    # otherwise the two views of the same chain would disagree with each other.
    assert abs(totals["mse"] - sum(w["mse"] for w in totals["per_window"]) / 3) < 1e-6


class _StubAccelerator:
    """The two ``Accelerator`` members ``train_chain`` uses, without a distributed context."""

    device = DEVICE

    @staticmethod
    def backward(loss: torch.Tensor) -> None:
        loss.backward()


def test_masked_mse_is_scale_free_in_the_masked_area() -> None:
    """A tight crop and a wide one must contribute comparably (SS4.3 row 1).

    Averaging over all tokens instead would make a subject occupying a tenth of the frame
    contribute a tenth of the gradient, which is a silent reweighting by crop tightness.
    """
    pred = torch.zeros(1, 16, 4)
    target = torch.ones(1, 16, 4)
    narrow = torch.zeros(1, 16, 1)
    narrow[:, :2] = 1.0
    wide = torch.ones(1, 16, 1)
    assert torch.allclose(train.masked_mse(pred, target, narrow), train.masked_mse(pred, target, wide))


def test_combine_masks_covers_the_four_disagreement_readings() -> None:
    record = {
        "render_alpha": torch.tensor([[[1.0, 0.0]]]),
        "capture_mask": torch.tensor([[[0.0, 1.0]]]),
    }
    assert train._combine_masks(record, "render").tolist() == [[[1.0, 0.0]]]
    assert train._combine_masks(record, "capture").tolist() == [[[0.0, 1.0]]]
    assert train._combine_masks(record, "union").tolist() == [[[1.0, 1.0]]]
    assert train._combine_masks(record, "intersection").tolist() == [[[0.0, 0.0]]]


def test_chain_seed_carryover_is_the_gt_at_the_destination_slot() -> None:
    """The GT seed is ``z_y[w][:, :, 1]``, never ``carry_from(z_y[w])``.

    ``carry_from`` takes a window's LAST latent frame, which is correct when the value comes
    from the previous window's *output*: at the deployed 16-frame stride, window ``w-1``'s last
    latent frame and window ``w``'s latent frame 1 are the same master frame. Applied to window
    ``w``'s own latents it picks a frame two latent steps into the future instead -- a seed no
    rollout ever produces, and one that would look like a quality problem rather than a bug.
    """
    window = _window(5)
    chain = train.Chain(
        source="stub/view00", split="train", actor="stub", seed_is_clip_start=False, windows=[window]
    )
    model = StubTransformer()
    seen: list[torch.Tensor | None] = []
    real_forward = train.one_window_forward

    def spy(*args, **kwargs):  # noqa: ANN202
        seen.append(args[3])
        return real_forward(*args, **kwargs)

    train.one_window_forward = spy
    try:
        train.train_chain(
            model, torch.zeros(1, 1, 8), chain, GEOMETRY, _StubAccelerator(),
            sigma0=SIGMA0, seed=0, anchor_weight=0.0, latent_channels=CHANNELS,
        )
    finally:
        train.one_window_forward = real_forward

    idx, n = refine_core.CARRYOVER_LATENT_IDX, GEOMETRY.context_latent_frames
    expected = window.z_y.unsqueeze(0)[:, :, idx : idx + n]
    assert seen[0] is not None
    assert torch.allclose(seen[0].float(), expected.float(), atol=8e-3)  # bf16 cast in the loop
    # ...and specifically NOT the window's last latent frame, which is what carry_from gives.
    assert not torch.allclose(seen[0].float(), refine_core.carry_from(window.z_y.unsqueeze(0), GEOMETRY).float())


def test_d0_noises_the_capture_not_the_guide() -> None:
    """SS4.1 D0: a training-only sanity check, not a deployable arm.

    It sets ``l_init = z_y`` instead of ``z_g`` -- the noisy branch is built from the LOSS
    TARGET, so optimal denoising is exactly SS3 identity 1's ordinary flow-matching target,
    decoupled from the render entirely. This measures the architecture's capacity ceiling at
    sigma_0 (compare against the measured `r`, SS0.3), not a render-correction model -- there
    is no `z_y` at inference, so `onestep_core.guide_conditionings` refuses this mode.
    """
    window = _window()  # same=False: z_g and z_y are genuinely different draws
    model = StubTransformer()
    z0_tokens, target_tokens, weights, state, tools = train.one_window_forward(
        model, torch.zeros(1, 1, 8), window, None, GEOMETRY,
        sigma0=SIGMA0, seed=7, device=DEVICE, latent_channels=CHANNELS, guide_mode="d0",
    )

    z_g_tokens = tools.patchifier.patchify(window.z_g.unsqueeze(0)).float()
    z_y_tokens = tools.patchifier.patchify(window.z_y.unsqueeze(0)).float()
    eps = (state.latent.float() - (1 - SIGMA0) * z_y_tokens) / SIGMA0

    # The noisy state is built from z_y, NOT z_g -- the one thing d0 changes.
    assert eps.std().item() > 0.5  # a real N(0, 1) draw, not a copy of either latent
    recovered_from_capture = (1 - SIGMA0) * z_y_tokens + SIGMA0 * eps
    assert torch.allclose(state.latent.float(), recovered_from_capture, atol=8e-3)
    not_from_guide = (1 - SIGMA0) * z_g_tokens + SIGMA0 * eps
    assert not torch.allclose(state.latent.float(), not_from_guide, atol=0.1)

    # No D2-style reference tokens: d0 is plain flow-matching, the sequence does not double.
    assert state.latent.shape[1] == target_tokens.shape[1]
    assert z0_tokens.shape[1] == target_tokens.shape[1] == weights.shape[1]


def test_d0_rejects_the_anchor_term() -> None:
    """base_denoised/ is Phi(lerp(z_g, eps, sigma_0)) -- off-input for a z_y-noised run."""
    with pytest.raises(SystemExit, match="off-input"):
        train.main(
            [
                "--subset", "/nonexistent.json", "--precomputed", "/nonexistent",
                "--output", "/nonexistent", "--guide-mode", "d0", "--anchor-weight", "0.1",
            ]
        )


def test_multilevel_sigma_schedule_assigns_one_fixed_level_per_rank() -> None:
    """Each rank trains one level for the whole run; extra ranks rotate back through the list."""
    levels = (0.909375, 0.725, 0.421875)
    args = argparse.Namespace(sigma0=0.725, sigma_levels=list(levels))
    assert train.training_sigmas(args) == levels
    # 4 ranks, 3 levels: rank 3 rotates back to rank 0's level rather than needing a 4th value.
    assert [train.sigma_for_rank(levels, rank) for rank in range(4)] == [
        0.909375, 0.725, 0.421875, 0.909375,
    ]


def test_sigma_zero_is_refused() -> None:
    """sigma=0.0 adds no noise, so its loss and gradient are identically zero -- never trained."""
    args = argparse.Namespace(sigma0=0.725, sigma_levels=[0.909375, 0.725, 0.421875, 0.0])
    with pytest.raises(SystemExit, match="sigma=0.0"):
        train.training_sigmas(args)


def test_d2_appends_a_clean_pixel_aligned_copy_of_the_guide() -> None:
    """§4.1's D2: the guide a SECOND time, clean, at timestep 0, on the target's own positions.

    Four properties, and each one is the reason D2 is supposed to beat D1:

    * the sequence doubles (that is the ~2.3x attention cost being paid);
    * the appended tokens hold the guide *undegraded* -- the init's copy is buried under
      sigma_0 noise and therefore carries a weaker constraint, which is the whole argument;
    * they share the target's RoPE positions at scale factor 1, so the copy is pixel-aligned
      rather than merely present;
    * they are excluded from the loss, so the model is not scored on reproducing its own input.
    """
    window = _window()
    model = StubTransformer()
    z0_d1, target_d1, weights_d1, state_d1, tools = _forward(window, None, model)
    z0_d2, target_d2, weights_d2, state_d2, _ = train.one_window_forward(
        model, torch.zeros(1, 1, 8), window, None, GEOMETRY,
        sigma0=SIGMA0, seed=7, device=DEVICE, latent_channels=CHANNELS, guide_mode="d2",
    )

    n = target_d1.shape[1]
    assert state_d1.latent.shape[1] == n
    assert state_d2.latent.shape[1] == 2 * n, "D2 must double the sequence"

    # The appended half is the guide, clean -- not noised, and not the capture.
    z_g_tokens = tools.patchifier.patchify(window.z_g.unsqueeze(0)).float()
    assert torch.allclose(state_d2.latent[:, n:].float(), z_g_tokens, atol=8e-3)
    assert not torch.allclose(state_d2.latent[:, :n].float(), z_g_tokens, atol=0.1)

    # Timestep 0 there, via denoise_mask 0 -- and therefore out of the loss.
    assert (state_d2.denoise_mask[:, n:] == 0).all()
    assert torch.allclose(state_d2.positions[:, :, n:].float(), state_d2.positions[:, :, :n].float())

    # The loss still sees exactly the target's tokens, under both arms.
    for z0, target, weights in ((z0_d1, target_d1, weights_d1), (z0_d2, target_d2, weights_d2)):
        assert z0.shape[1] == n
        assert weights.shape[1] == n
        assert target.shape[1] == n


# --- onestep_core: the deployment counterpart of the training loop -----------------------


def test_rollout_slices_one_master_encode_and_never_re_keys() -> None:
    """§4.4's load-bearing rule: the rollout must NOT inherit K_STEP's per-window re-encode.

    `precompute.py` builds every training init by encoding a source once and slicing, so past
    a clip's first window slot 0 holds a regular multi-frame block. `refine_core`'s own k2
    windowing instead re-encodes each window from pixels, giving every one a fresh causal
    keyframe there. A rollout that did that would feed the model an input it was never trained
    on -- and it would read as a quality number, not an error.
    """
    model = StubTransformer()
    sigmas = torch.tensor([SIGMA0, 0.0])
    # 4 windows' worth of master: window i starts at latent frame 2i (16-frame stride / 8).
    master = torch.randn(1, CHANNELS, 2 * 3 + LATENT_FRAMES, LATENT_EDGE, LATENT_EDGE)

    seen: list[torch.Tensor] = []
    real = onestep_core.one_step_window

    def spy(*args, **kwargs):  # noqa: ANN202 -- forwards whatever rollout passes
        seen.append(args[2])
        return real(*args, **kwargs)

    onestep_core.one_step_window = spy
    try:
        result = onestep_core.rollout(
            model, _StubDenoiser(), master, GEOMETRY, sigmas, 30.0,
            device=DEVICE, latent_channels=CHANNELS,
        )
    finally:
        onestep_core.one_step_window = real

    assert result.windows == len(seen) >= 2
    assert result.forwards == result.windows, "one step means exactly one forward per window"
    for i, z_g in enumerate(seen):
        # Sliced verbatim out of the single master encode -- not re-derived, not re-keyed.
        assert torch.equal(z_g, master[:, :, 2 * i : 2 * i + LATENT_FRAMES])
    # Window 1's slot 0 is window 0's slot 2: a regular block, shared with the previous window.
    assert torch.equal(seen[1][:, :, 0], seen[0][:, :, 2])


def test_rollout_carries_the_models_own_output_forward() -> None:
    """Deployment's carryover is the previous window's output, matching §4.4's training loop."""
    model = StubTransformer()
    sigmas = torch.tensor([SIGMA0, 0.0])
    master = torch.randn(1, CHANNELS, 2 * 2 + LATENT_FRAMES, LATENT_EDGE, LATENT_EDGE)

    carries: list[torch.Tensor | None] = []
    real = onestep_core.one_step_window

    def spy(*args, **kwargs):  # noqa: ANN202
        carries.append(args[3])
        return real(*args, **kwargs)

    onestep_core.one_step_window = spy
    try:
        result = onestep_core.rollout(
            model, _StubDenoiser(), master, GEOMETRY, sigmas, 30.0,
            device=DEVICE, latent_channels=CHANNELS,
        )
    finally:
        onestep_core.one_step_window = real

    assert carries[0] is None, "a clip's first window has no predecessor at deployment either"
    for i in range(1, len(carries)):
        assert torch.equal(carries[i], refine_core.carry_from(result.latents[i - 1], GEOMETRY))


def test_one_step_window_refuses_a_multi_step_schedule() -> None:
    tools = refine_core.tools_for_window(GEOMETRY, EDGE, EDGE, 30.0, latent_channels=CHANNELS)
    with pytest.raises(ValueError, match="2-point schedule"):
        onestep_core.one_step_window(
            StubTransformer(), _StubDenoiser(),
            torch.randn(1, CHANNELS, LATENT_FRAMES, LATENT_EDGE, LATENT_EDGE), None,
            torch.tensor([0.725, 0.421875, 0.0]), tools, 0, DEVICE,
        )


def test_guide_conditionings_match_the_training_arms() -> None:
    """The arm must agree between training and deployment: a checkpoint trained with clean
    reference tokens and run without them works from half its input, and nothing raises."""
    z_g = torch.randn(1, CHANNELS, LATENT_FRAMES, LATENT_EDGE, LATENT_EDGE)
    assert onestep_core.guide_conditionings(z_g, "d1") == ()
    assert len(onestep_core.guide_conditionings(z_g, "d2")) == 1
    with pytest.raises(ValueError, match="unknown guide mode"):
        onestep_core.guide_conditionings(z_g, "d3")
    # d0 is training-only: there is no z_y at inference to noise, so it has no deployment
    # counterpart and must be refused here exactly like any other unknown mode.
    with pytest.raises(ValueError, match="unknown guide mode"):
        onestep_core.guide_conditionings(z_g, "d0")


class _StubDenoiser:
    """`SimpleDenoiser`'s shape, without a context tensor or the real modality plumbing."""

    def __call__(self, transformer, video_state, audio_state, sigmas, step_index):  # noqa: ANN001, ARG002
        denoised, _ = transformer(video=_Mod(video_state), audio=None, perturbations=None)
        return DenoisedLatentResult.result_or_none(denoised=denoised), None


class _Mod:
    """The one field `StubTransformer` reads off a Modality."""

    def __init__(self, state) -> None:  # noqa: ANN001
        self.latent = state.latent
