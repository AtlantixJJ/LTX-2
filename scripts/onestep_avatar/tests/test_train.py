"""SS7.1's CPU tests for the causal AR training loop -- small dims, no GPU, no checkpoint.

Each one pins a property the plan names, and each is the cheap version of a bug that would
otherwise look like a quality problem in C1 (SS9 risk 12):

(i)   ``K = 1`` starting at block 0 reproduces a plain non-AR step exactly.
(ii)  ``z_g == z_y`` reduces the objective to the ordinary flow-matching target ``eps - z_y``.
(iii) at fixed sigma_0, velocity MSE times sigma_0**2 equals x0 MSE -- SS3's second identity.
(iv)  block ``i+1``'s cached context comes from block ``i``'s own output, detached -- and
      ``--teacher-forcing`` is exactly the one-tensor swap to the ground truth.

The cache/mask machinery those rest on is tested separately, against a real ``LTXModel``, in
``test_causal_core.py``. Here the transformer is a stub on purpose: what is under test is which
tensor the loop hands to which call, and a stub makes that visible.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest
import torch

from ltx_core.types import SpatioTemporalScaleFactors
from ltx_core.utils import to_velocity
from scripts.onestep_avatar import causal_core, onestep_core, train
from scripts.onestep_avatar.causal_core import BlockCache, CausalGeometry
from scripts.prune.core import model_registry

SCALE = SpatioTemporalScaleFactors(time=8, height=32, width=32)
GEOMETRY = CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=2)
CHANNELS = 8
EDGE = 64  # -> a 2x2 latent grid: 4 tokens per latent frame
LATENT_FRAMES = 7  # -> blocks [(0,3), (3,5), (5,7)]
FPS = 30.0
SIGMA0 = 0.725
DEVICE = torch.device("cpu")


class X0Stub(torch.nn.Module):
    """A trainable stand-in shaped like the ``X0Model`` the session yields at deploy time --
    ``model(video, audio, perturbations) -> (denoised, aux)``, unlike ``StubTransformer``
    below which emits velocity. ``onestep_core.rollout`` reads it through
    ``causal_core.denoised_from_x0_model``, which treats the model's output as the denoised
    latent directly.

    Carries a ``.velocity_model`` with a ``transformer_blocks`` attribute -- not read by this
    stub's own ``forward``, but ``causal_core.base_model`` needs *some* path to
    ``transformer_blocks`` to resolve, matching a real ``X0Model``'s shape.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.velocity_model = torch.nn.Module()
        self.velocity_model.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity()])
        self.velocity_model.inner_dim = 4

    def forward(self, video, audio, perturbations) -> tuple[torch.Tensor, None]:  # noqa: ANN001, ARG002
        return self.scale * video.latent, None


class StubTransformer(torch.nn.Module):
    """A trainable stand-in with the real ``(video, audio, perturbations) -> (v, a)`` shape.

    Deliberately linear in the latent so an expected gradient can be written down in closed
    form where a test needs one; the point of these tests is the loop's plumbing, not the
    backbone. It ignores the K/V cache entirely -- ``test_causal_core`` covers that against a
    real transformer, and a stub that honoured the cache would only be testing itself.
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

    def forward(self, video, audio, perturbations) -> tuple[torch.Tensor, None]:  # noqa: ANN001, ARG002
        self.modalities.append(video)
        return super().forward(video, audio, perturbations)


class _StubAccelerator:
    """The two ``Accelerator`` members ``train_chain`` uses, without a distributed context."""

    device = DEVICE

    @staticmethod
    def backward(loss: torch.Tensor) -> None:
        loss.backward()


def _chain(
    *, blocks: list[int] | None = None, same: bool = False, seed: int = 0, clip_start: bool = True
) -> train.Chain:
    generator = torch.Generator().manual_seed(seed)
    shape = (CHANNELS, LATENT_FRAMES, EDGE // SCALE.height, EDGE // SCALE.width)
    z_g = torch.randn(shape, generator=generator)
    z_y = z_g.clone() if same else torch.randn(shape, generator=generator)
    return train.Chain(
        source="stub/view00",
        split="train",
        actor="stub",
        seed_is_clip_start=clip_start,
        blocks=[0] if blocks is None else blocks,
        z_g=z_g,
        z_y=z_y,
        fps=FPS,
        z0_base=None,
    )


def _grid(chain: train.Chain):  # noqa: ANN202
    return train.clip_grid_for(chain, GEOMETRY, device=DEVICE, latent_channels=CHANNELS)


def _cache(chain: train.Chain) -> BlockCache:
    return BlockCache.allocate(
        _grid(chain), GEOMETRY, num_layers=1, inner_dim=4, device=DEVICE, dtype=train.DTYPE
    )


def _run(chain: train.Chain, model: StubTransformer, **kwargs) -> dict:
    return train.train_chain(
        model, torch.zeros(1, 1, 8), chain, GEOMETRY, _cache(chain), _StubAccelerator(),
        sigma0=SIGMA0, seed=0, anchor_weight=0.0, latent_channels=CHANNELS, **kwargs,
    )


def _spy_on(monkeypatch: pytest.MonkeyPatch, name: str, capture: list) -> None:
    """Record the tokens handed to ``causal_core.<name>`` as ``train`` calls it."""
    real = getattr(causal_core, name)

    def spy(denoise_fn, grid, cache, tokens, *args, **kwargs):  # noqa: ANN001, ANN202
        capture.append(tokens)
        return real(denoise_fn, grid, cache, tokens, *args, **kwargs)

    monkeypatch.setattr(causal_core, name, spy)


def test_guided_init_is_the_linear_interpolant_of_the_guide(monkeypatch: pytest.MonkeyPatch) -> None:
    """SS3: the noisy state is ``(1 - sigma_0) z_g + sigma_0 eps`` -- built from the GUIDE.

    The whole trainer gap (SS1.1 "cannot build the noisy state from a different latent than the
    loss target") is this one line, so it gets its own test rather than being implied.
    """
    chain = _chain()
    noisy: list[torch.Tensor] = []
    _spy_on(monkeypatch, "denoise_block", noisy)
    _run(chain, StubTransformer())

    grid = _grid(chain)
    lo, hi = grid.token_span(*GEOMETRY.plan(LATENT_FRAMES)[0])
    z_g_tokens = grid.patchify(chain.z_g.unsqueeze(0).to(train.DTYPE))[:, lo:hi].float()
    eps = (noisy[0].float() - (1 - SIGMA0) * z_g_tokens) / SIGMA0
    assert eps.std().item() > 0.5  # a real N(0, 1) draw, not a copy of z_g
    assert torch.allclose(noisy[0].float(), (1 - SIGMA0) * z_g_tokens + SIGMA0 * eps, atol=8e-3)


def test_full_frame_mse_averages_every_token_and_channel() -> None:
    """The training objective has no subject, alpha, or disagreement weighting."""
    pred = torch.zeros(1, 16, 4)
    target = torch.ones(1, 16, 4)
    assert train.full_frame_mse(pred, target).item() == 1.0


def test_identical_guide_and_capture_reduce_to_flow_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    """SS3 identity 1: with ``z_g == z_y`` the optimal velocity is exactly ``eps - z_y``.

    This is what makes SS7.1 a strict generalisation of the shipped trainer rather than a
    separate objective -- if it ever stops holding, the guided init has drifted from the
    linear interpolant the rest of the stack assumes.
    """
    chain = _chain(same=True)
    noisy: list[torch.Tensor] = []
    _spy_on(monkeypatch, "denoise_block", noisy)
    _run(chain, StubTransformer())

    grid = _grid(chain)
    lo, hi = grid.token_span(*GEOMETRY.plan(LATENT_FRAMES)[0])
    z_y_tokens = grid.patchify(chain.z_y.unsqueeze(0).to(train.DTYPE))[:, lo:hi].float()
    eps = (noisy[0].float() - (1 - SIGMA0) * z_y_tokens) / SIGMA0
    v_star = (noisy[0].float() - z_y_tokens) / SIGMA0
    assert torch.allclose(v_star, eps - z_y_tokens, atol=3e-2)


def test_velocity_mse_equals_x0_mse_over_sigma_squared(monkeypatch: pytest.MonkeyPatch) -> None:
    """SS3 identity 2: at fixed sigma_0 the existing velocity loss already IS an x0 loss.

    So the regression term needs no loss-function change -- only a different target -- and
    predicting z0 in the loop (which is what makes the anchor and the capture directly
    comparable) costs nothing in objective terms.
    """
    chain = _chain()
    noisy: list[torch.Tensor] = []
    _spy_on(monkeypatch, "denoise_block", noisy)
    model = StubTransformer()
    _run(chain, model)

    grid = _grid(chain)
    lo, hi = grid.token_span(*GEOMETRY.plan(LATENT_FRAMES)[0])
    target = grid.patchify(chain.z_y.unsqueeze(0).to(train.DTYPE))[:, lo:hi]
    denoise_fn = causal_core.denoised_from_velocity_model(model)
    z0 = causal_core.denoise_block(
        denoise_fn, grid, _cache(chain), noisy[0], torch.zeros(1, 1, 8), SIGMA0, GEOMETRY.plan(LATENT_FRAMES)[0]
    )
    v_hat = to_velocity(noisy[0], SIGMA0, z0)
    v_star = to_velocity(noisy[0], SIGMA0, target)
    x0_mse = (z0.float() - target.float()).pow(2).mean()
    v_mse = (v_hat.float() - v_star.float()).pow(2).mean()
    assert torch.allclose(v_mse * SIGMA0**2, x0_mse, rtol=1e-3, atol=1e-6)


def test_cache_refresh_uses_the_models_own_output_and_carries_no_grad(monkeypatch: pytest.MonkeyPatch) -> None:
    """SS4.4: what enters the cache must be the MODEL's own output, detached -- not the GT.

    Feeding the GT is the teacher forcing the plan removes; detaching is what keeps peak
    activation memory at one block rather than ``K`` (SS8.1).
    """
    chain = _chain(blocks=[0, 1])
    denoised: list[torch.Tensor] = []
    refreshed: list[torch.Tensor] = []
    _spy_on(monkeypatch, "denoise_block", denoised)
    _spy_on(monkeypatch, "refresh_block", refreshed)
    _run(chain, StubTransformer())

    assert len(refreshed) == 2
    for tokens in refreshed:
        assert not tokens.requires_grad
    grid = _grid(chain)
    plan = GEOMETRY.plan(LATENT_FRAMES)
    lo, hi = grid.token_span(*plan[0])
    gt = grid.patchify(chain.z_y.unsqueeze(0).to(train.DTYPE))[:, lo:hi]
    # ...and it is NOT the ground truth, which is what teacher forcing would have put there.
    assert not torch.allclose(refreshed[0].float(), gt.float(), atol=1e-2)


def test_teacher_forcing_refreshes_with_the_gt_instead_of_self_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--teacher-forcing``: the ablation that opts back INTO the substitution SS4.4 removed.

    One tensor is the entire difference between the two regimes, and this is it. Pinned by
    spying rather than by a gradient assertion, since ``StubTransformer`` has no cross-token
    mixing and so a block's loss does not depend on what is in the cache at all.
    """
    chain = _chain(blocks=[0, 1])
    refreshed: list[torch.Tensor] = []
    _spy_on(monkeypatch, "refresh_block", refreshed)
    _run(chain, StubTransformer(), teacher_forcing=True)

    grid = _grid(chain)
    target = grid.patchify(chain.z_y.unsqueeze(0).to(train.DTYPE))
    for span, tokens in zip(GEOMETRY.plan(LATENT_FRAMES), refreshed, strict=False):
        lo, hi = grid.token_span(*span)
        assert torch.equal(tokens, target[:, lo:hi])


def test_d0_teacher_forcing_conditions_block_zero_on_clean_capture_frame() -> None:
    """D0's source is the capture, but c0 still has to bypass noising and AdaLN time.

    This tests the assembled model input, rather than inferring the condition from a pinned
    sink after refresh: the latter can pass even when block 0 itself was never conditioned.
    """
    chain = _chain(blocks=[0], seed=9)
    model = CapturingStubTransformer()
    _run(chain, model, guide_mode="d0", teacher_forcing=True)

    grid = _grid(chain)
    target = grid.patchify(chain.z_y.unsqueeze(0).to(train.DTYPE))
    denoise = next(
        modality
        for modality in model.modalities
        if not modality.kv_write and torch.count_nonzero(modality.timesteps) > 0
    )
    c0_tokens = grid.tokens_per_latent_frame
    torch.testing.assert_close(denoise.latent[:, :c0_tokens], target[:, :c0_tokens], rtol=0, atol=0)
    assert torch.count_nonzero(denoise.timesteps[:, :c0_tokens]) == 0
    assert torch.all(denoise.timesteps[:, c0_tokens:] == SIGMA0)


def test_a_mid_clip_chain_primes_the_cache_from_the_ground_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    """SS4.4's one remaining teacher-forced seam, and it should be visible as one.

    A chain that starts at block 0 needs no priming -- deployment has no predecessor there
    either. A chain that starts mid-clip gets the frames before it from the GT capture, which
    is an approximation of what a true rollout would have cached, and is why
    ``seed_is_clip_start`` is carried in the subset at all.
    """
    primed: list[torch.Tensor] = []
    real = causal_core.prime_cache

    def spy(denoise_fn, grid, cache, tokens, geometry, context, *, upto_latent_frame):  # noqa: ANN001, ANN202
        primed.append((tokens, upto_latent_frame))
        return real(denoise_fn, grid, cache, tokens, geometry, context, upto_latent_frame=upto_latent_frame)

    monkeypatch.setattr(causal_core, "prime_cache", spy)

    _run(_chain(blocks=[0]), StubTransformer())
    assert primed[-1][1] == 0  # block 0 starts at latent frame 0: nothing to prime

    chain = _chain(blocks=[2], clip_start=False)
    _run(chain, StubTransformer())
    tokens, upto = primed[-1]
    assert upto == GEOMETRY.plan(LATENT_FRAMES)[2][0]
    grid = _grid(chain)
    assert torch.equal(tokens, grid.patchify(chain.z_y.unsqueeze(0).to(train.DTYPE)))


def test_k1_chain_matches_a_single_non_ar_step() -> None:
    """SS7.1 test (i): ``K = 1`` is the teacher-forced control, and must be a plain step.

    A2 uses exactly this as its control arm, so the difference in rollout slope it reports is
    only the value of AR training if ``K = 1`` really is the non-AR baseline.
    """
    chain = _chain(seed=3)
    model_chain = StubTransformer()
    totals = _run(chain, model_chain)
    chain_grad = model_chain.scale.grad.clone()

    model_plain = StubTransformer()
    grid = _grid(chain)
    span = GEOMETRY.plan(LATENT_FRAMES)[0]
    lo, hi = grid.token_span(*span)
    guide = grid.patchify(chain.z_g.unsqueeze(0).to(train.DTYPE))
    target = grid.patchify(chain.z_y.unsqueeze(0).to(train.DTYPE))
    c0 = target[:, : grid.tokens_per_latent_frame]
    noisy = causal_core.with_clean_prefix(causal_core.noise_block(guide[:, lo:hi], SIGMA0, 0), c0)
    z0 = causal_core.denoise_block(
        causal_core.denoised_from_velocity_model(model_plain), grid, _cache(chain), noisy,
        torch.zeros(1, 1, 8), SIGMA0, span, clean_prefix_tokens=c0.shape[1],
    )
    z0 = causal_core.with_clean_prefix(z0, c0)
    loss = train.full_frame_mse(z0, target[:, lo:hi])
    loss.backward()

    assert torch.allclose(chain_grad, model_plain.scale.grad, rtol=1e-4, atol=1e-8)
    assert abs(totals["mse"] - float(loss.detach())) < 1e-5
    assert totals["anchor"] == 0.0
    assert [w["block_index"] for w in totals["per_block"]] == [0]


def test_train_chain_records_one_per_block_entry_in_chain_order() -> None:
    """SS7.4(a): "today it writes one row per chain... that effect is unobservable" -- the fix
    is a per-block breakdown alongside the existing chain-mean, in chain order, one entry per
    block actually run (not per corpus block)."""
    totals = _run(_chain(blocks=[0, 1, 2]), StubTransformer())
    assert [w["block_index"] for w in totals["per_block"]] == [0, 1, 2]
    # The chain-mean is exactly the average of the per-block entries it was built from --
    # otherwise the two views of the same chain would disagree with each other.
    assert abs(totals["mse"] - sum(w["mse"] for w in totals["per_block"]) / 3) < 1e-6


def test_d0_noises_the_capture_not_the_guide(monkeypatch: pytest.MonkeyPatch) -> None:
    """SS4.1 D0: a training-only sanity check, not a deployable arm.

    It noises ``z_y`` instead of ``z_g`` -- the noisy branch is built from the LOSS TARGET, so
    optimal denoising is exactly SS3 identity 1's ordinary flow-matching target, decoupled
    from the render entirely. This measures the architecture's capacity ceiling at sigma_0
    (compare against the measured `r`, SS0.3), not a render-correction model -- there is no
    `z_y` at inference, so `onestep_core.guide_conditionings` refuses this mode.
    """
    chain = _chain()  # same=False: z_g and z_y are genuinely different draws
    noisy: list[torch.Tensor] = []
    _spy_on(monkeypatch, "denoise_block", noisy)
    _run(chain, StubTransformer(), guide_mode="d0")

    grid = _grid(chain)
    lo, hi = grid.token_span(*GEOMETRY.plan(LATENT_FRAMES)[0])
    z_g_tokens = grid.patchify(chain.z_g.unsqueeze(0).to(train.DTYPE))[:, lo:hi].float()
    z_y_tokens = grid.patchify(chain.z_y.unsqueeze(0).to(train.DTYPE))[:, lo:hi].float()
    eps = (noisy[0].float() - (1 - SIGMA0) * z_y_tokens) / SIGMA0
    assert eps.std().item() > 0.5
    assert torch.allclose(noisy[0].float(), (1 - SIGMA0) * z_y_tokens + SIGMA0 * eps, atol=8e-3)
    assert not torch.allclose(noisy[0].float(), (1 - SIGMA0) * z_g_tokens + SIGMA0 * eps, atol=0.1)


@pytest.mark.parametrize("weight", ["0.1", "-0.1", "nan", "inf"])
def test_anchor_weight_is_disabled(weight: str) -> None:
    """2026-09-18 audit F8: no ``base_denoised.pt`` producer exists, and a fixed per-view tensor
    cannot represent the anchor for every chain/sigma/history combination that would read it --
    so any nonzero ``--anchor-weight`` is refused before the subset or model is even touched,
    regardless of guide mode."""
    with pytest.raises(SystemExit, match="anchor-weight is disabled"):
        train.main(
            [
                "--subset", "/nonexistent.json", "--output", "/nonexistent",
                "--guide-mode", "d0", "--anchor-weight", weight,
            ]
        )
    with pytest.raises(SystemExit, match="anchor-weight is disabled"):
        train.main(
            ["--subset", "/nonexistent.json", "--output", "/nonexistent", "--anchor-weight", weight]
        )


def test_a_window_chain_subset_is_refused_rather_than_reinterpreted(tmp_path) -> None:  # noqa: ANN001
    """A subset frozen before SS4.4 indexes windows that no longer exist.

    Silently reading its ``windows`` lists as block indices would train on the wrong frames --
    a window index and a block index are different numbers over the same clip.
    """
    subset = tmp_path / "old.json"
    subset.write_text(
        '{"kind": "one_step_argavatar_window_chains", "corpus_root": "/nonexistent", '
        '"chains": [], "sources": []}'
    )
    with pytest.raises(SystemExit, match=r"block-chain subset"):
        train.main(["--subset", str(subset), "--output", str(tmp_path / "out")])


def _bad_subset(tmp_path):  # noqa: ANN001, ANN202
    """A subset that fails at the very next check after the used-output guard, so these tests
    exercise only --overwrite's effect on that guard, not the rest of the launch."""
    subset = tmp_path / "old.json"
    subset.write_text(
        '{"kind": "one_step_argavatar_window_chains", "corpus_root": "/nonexistent", '
        '"chains": [], "sources": []}'
    )
    return subset


def test_a_second_launch_into_a_used_output_is_refused(tmp_path) -> None:  # noqa: ANN001
    """S3b of the 2026-09-17 cleanup plan: train.py has no resume -- step restarts at 0 every
    launch -- so an unguarded write would silently merge two runs under one step numbering."""
    output = tmp_path / "out"
    output.mkdir()
    (output / "metrics_rank0.jsonl").write_text('{"step": 0}\n')
    with pytest.raises(SystemExit, match=r"already has 1 entr"):
        train.main(["--subset", str(_bad_subset(tmp_path)), "--output", str(output)])
    assert (output / "metrics_rank0.jsonl").is_file()  # refused, not touched


def test_overwrite_does_not_touch_the_output_when_validation_fails_first(tmp_path) -> None:  # noqa: ANN001
    """2026-09-18 audit F6: validation and (would-be) archiving must be side-effect free until
    the run is actually going to happen. The old code unlinked ``metrics_rank*.jsonl`` here
    unconditionally, before this subset-kind check ran -- so ``--overwrite`` on an otherwise
    invalid launch was destructive for no benefit. Archiving now happens only after
    ``Accelerator()``/model setup, far past where this failure is raised, so nothing here should
    ever be touched by a launch that never gets that far."""
    output = tmp_path / "out"
    output.mkdir()
    (output / "metrics_rank0.jsonl").write_text('{"step": 0}\n')
    with pytest.raises(SystemExit, match=r"block-chain subset"):  # the NEXT check, past the guard
        train.main(["--subset", str(_bad_subset(tmp_path)), "--output", str(output), "--overwrite"])
    assert (output / "metrics_rank0.jsonl").is_file()  # not archived, not deleted


def test_dry_run_flag_does_not_reach_the_archive_step(tmp_path) -> None:  # noqa: ANN001
    """The archive step (F6) sits textually after ``Accelerator()``/model setup, and
    ``--dry-run`` returns well before either is created -- so a dry run can never archive or
    delete an existing --output, for any subset. This exercises that with the same early-failing
    subset the other guard tests use; a subset that reached the actual dry-run branch would
    return 0 even earlier, before ``needs_archive`` is ever consumed."""
    output = tmp_path / "out"
    output.mkdir()
    (output / "metrics_rank0.jsonl").write_text('{"step": 0}\n')
    with pytest.raises(SystemExit, match=r"block-chain subset"):
        train.main(
            ["--subset", str(_bad_subset(tmp_path)), "--output", str(output), "--overwrite", "--dry-run"]
        )
    assert (output / "metrics_rank0.jsonl").is_file()


def test_archive_existing_run_moves_prior_contents_aside(tmp_path) -> None:  # noqa: ANN001
    """F6's replacement for deleting ``metrics_rank*.jsonl``: the whole directory moves into one
    timestamped subdirectory, so a relaunch's fresh checkpoints/config never land beside a prior
    run's under the same names -- and the prior run stays recoverable instead of discarded."""
    output = tmp_path / "out"
    output.mkdir()
    (output / "metrics_rank0.jsonl").write_text('{"step": 0}\n')
    (output / "checkpoints").mkdir()
    (output / "checkpoints" / "lora_weights_step_00100.safetensors").write_bytes(b"")

    archived = train.archive_existing_run(output)

    assert archived is not None
    assert archived.parent == output
    assert archived.name.startswith("archived_")
    assert (archived / "metrics_rank0.jsonl").is_file()
    assert (archived / "checkpoints" / "lora_weights_step_00100.safetensors").is_file()
    # The directory itself is reusable immediately: nothing but the archive is left in it.
    assert [p.name for p in output.iterdir()] == [archived.name]


def test_archive_existing_run_is_a_noop_on_an_empty_directory(tmp_path) -> None:  # noqa: ANN001
    output = tmp_path / "out"
    output.mkdir()
    assert train.archive_existing_run(output) is None
    assert list(output.iterdir()) == []


def test_a_cache_too_small_for_this_clip_is_refused_before_any_forward() -> None:
    """The run allocates ONE cache; a clip that does not fit must fail before the step's
    forwards, not inside one.

    ``LayerKVCache.write`` raises on overflow, and a raise inside one rank's forward leaves it
    a round of all-gathers short of the others -- the FSDP desynchronisation that presents as
    a hang, which ``assert_rank_lockstep`` and ``prime_cache``'s unconditional forward both
    exist to rule out. This raise is reached before ``prime_cache`` is called, so every rank
    fails the same way.
    """
    chain = _chain(blocks=[0])
    deep = CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=16)
    # Sized for a 3-latent-frame clip, then handed a 7-frame one: the "first chain sized the
    # buffer" bug, reproduced.
    undersized = BlockCache.allocate(
        train.clip_grid_for(chain, deep, device=DEVICE, latent_channels=CHANNELS),
        deep, num_layers=1, inner_dim=4, device=DEVICE, dtype=train.DTYPE,
        capacity_latent_frames=3,
    )
    with pytest.raises(ValueError, match="sized from the subset's LONGEST clip"):
        train.train_chain(
            StubTransformer(), torch.zeros(1, 1, 8), chain, deep, undersized, _StubAccelerator(),
            sigma0=SIGMA0, seed=0, anchor_weight=0.0, latent_channels=CHANNELS,
        )


def test_the_chain_store_reports_the_subsets_longest_clip(tmp_path) -> None:  # noqa: ANN001
    """The number the run-wide cache allocation is sized from.

    Read from ``windows.py``'s own ``n_latent_frames`` -- which it takes from the stored master
    -- so it costs no tensor load and cannot disagree with what ``train.py`` plans over.
    """
    subset = {
        "kind": "one_step_argavatar_block_chains",
        "geometry": {"latent_time_scale": 8},
        "chains": [{"source": "a", "split": "train", "actor": "1", "blocks": [0], "seed_is_clip_start": True}],
        "sources": [
            {"relative_dir": "a", "n_latent_frames": 18},
            {"relative_dir": "b", "n_latent_frames": 28},
        ],
    }
    store = train.ChainStore(
        subset, tmp_path, split="train", objective="bg", with_anchor=False, with_guide=False,
    )
    assert store.max_latent_frames == 28


def test_the_chain_store_refuses_a_window_chain_subset(tmp_path) -> None:  # noqa: ANN001
    """The refusal lives in ``ChainStore.__init__`` (S2 of the 2026-09-17 cleanup plan), so
    every reader that constructs one directly -- not just ``train.main`` -- gets the same
    pointed error instead of ``visualize_d0``'s old ``KeyError: 'latent_time_scale'`` deep
    inside geometry setup.
    """
    subset = {
        "kind": "one_step_argavatar_window_chains",
        "chains": [],
        "sources": [],
    }
    with pytest.raises(SystemExit, match=r"block-chain subset"):
        train.ChainStore(subset, tmp_path, split="train", objective="bg", with_anchor=False, with_guide=False)


def test_a_subset_frozen_against_the_video_length_is_refused_at_startup() -> None:
    """The 2026-09-16 stale-freeze, caught before the 42 GB checkpoint load.

    A subset frozen before that date sized its block plan from the source VIDEO; a master
    consolidated from v1 per-window slices is short of the video by up to one window, so the
    subset claims one block per source the latents do not contain. `train_chain` catches it
    too, but only after minutes of startup on every rank -- and its old message blamed "a
    different geometry", which sends a reader to the flags rather than to the artifact.
    """
    subset = {
        # 19 latent frames is what the 150-frame video implies; the master holds 18.
        "sources": [{"relative_dir": "a", "n_latent_frames": 19}],
        "chains": [{"source": "a", "blocks": [6, 7, 8]}],
    }
    train.assert_subset_matches_geometry(subset, GEOMETRY)  # 19 frames -> 9 blocks, fine

    stale = {
        "sources": [{"relative_dir": "a", "n_latent_frames": 18}],
        "chains": [{"source": "a", "blocks": [6, 7, 8]}],
    }
    with pytest.raises(SystemExit, match="2026-09-16"):
        train.assert_subset_matches_geometry(stale, GEOMETRY)


def test_a_subset_whose_masters_shrank_under_it_is_refused_at_startup(tmp_path) -> None:  # noqa: ANN001
    """The staleness the internal check CANNOT see, and the one that actually happens.

    A subset frozen before 2026-09-16 is internally consistent -- `windows.py` took both the
    latent-frame count and the block plan from the source video -- so only a comparison
    against the stored master catches it. This is the real `t2.json` failure, in miniature:
    19 recorded, 18 on disk, a chain asking for block 8 of a clip that plans 8.
    """
    view = tmp_path / "Part_1" / "0001_01" / "views" / "view01_cam01"
    view.mkdir(parents=True)
    torch.save(
        {"schema_version": 2, "master": torch.zeros(CHANNELS, 18, 2, 2), "fps": 30.0},
        view / train.dataset.capture_bundle_name("bg"),
    )
    rel = "Part_1/0001_01/views/view01_cam01"
    subset = {
        "sources": [{"relative_dir": rel, "n_latent_frames": 19}],
        "chains": [{"source": rel, "blocks": [6, 7, 8]}],
    }
    # Internally consistent: plan(19) has 9 blocks, so block 8 exists as far as the subset knows.
    train.assert_subset_matches_geometry(subset, GEOMETRY)
    with pytest.raises(SystemExit, match="subset says 19 latent frames, the master holds 18"):
        train.assert_subset_matches_geometry(subset, GEOMETRY, corpus_root=tmp_path)

    # And re-freezing against the master is what makes it trainable again.
    subset["sources"][0]["n_latent_frames"] = 18
    subset["chains"] = [{"source": rel, "blocks": [5, 6, 7]}]
    train.assert_subset_matches_geometry(subset, GEOMETRY, corpus_root=tmp_path)


def test_a_source_the_capture_pass_never_encoded_is_named(tmp_path) -> None:  # noqa: ANN001
    """A missing bundle is its own error, not a crash inside torch.load."""
    subset = {
        "sources": [{"relative_dir": "Part_1/gone/views/view01_cam01", "n_latent_frames": 18}],
        "chains": [{"source": "Part_1/gone/views/view01_cam01", "blocks": [0, 1, 2]}],
    }
    with pytest.raises(SystemExit, match="does not exist, but the subset lists"):
        train.assert_subset_matches_geometry(subset, GEOMETRY, corpus_root=tmp_path)


def _write_capture_and_guide(view, *, n_latent_frames: int, fps: float = 30.0, with_guide: bool = True) -> None:  # noqa: ANN001
    view.mkdir(parents=True)
    torch.save(
        {"schema_version": 2, "master": torch.zeros(CHANNELS, n_latent_frames, 2, 2), "fps": fps},
        view / train.dataset.capture_bundle_name("bg"),
    )
    if with_guide:
        torch.save(
            {"schema_version": 2, "master": torch.zeros(CHANNELS, n_latent_frames, 2, 2), "fps": fps},
            view / train.dataset.guide_bundle_name("bg"),
        )


def test_a_missing_guide_master_is_caught_before_the_second_source_is_ever_drawn(tmp_path) -> None:  # noqa: ANN001
    """2026-09-18 audit, Stage A gap 1: ``--dry-run`` only ever draws ``store[0]``, and the old
    startup check validated only capture bundles -- so a subset whose FIRST source is complete
    but whose SECOND is missing its guide master passed both a dry run and normal startup, and
    only failed with a raw ``FileNotFoundError`` once a rank happened to draw that source's
    chain, which can be well after the 42 GB checkpoint load. This reproduces exactly that shape
    and checks it is now refused for every source, up front, by name.
    """
    ok = tmp_path / "Part_1" / "0001_01" / "views" / "view01_cam01"
    missing_guide = tmp_path / "Part_1" / "0002_01" / "views" / "view01_cam01"
    _write_capture_and_guide(ok, n_latent_frames=18)
    _write_capture_and_guide(missing_guide, n_latent_frames=18, with_guide=False)

    subset = {
        "sources": [
            {"relative_dir": "Part_1/0001_01/views/view01_cam01", "n_latent_frames": 18},
            {"relative_dir": "Part_1/0002_01/views/view01_cam01", "n_latent_frames": 18},
        ],
        "chains": [
            {"source": "Part_1/0001_01/views/view01_cam01", "blocks": [0]},
            {"source": "Part_1/0002_01/views/view01_cam01", "blocks": [0]},
        ],
    }
    with pytest.raises(SystemExit, match=r"0002_01.*does not exist"):
        train.assert_subset_matches_geometry(subset, GEOMETRY, corpus_root=tmp_path, with_guide=True)

    # d0 never reads the guide master, so the same subset is fine without --guide-mode d1.
    train.assert_subset_matches_geometry(subset, GEOMETRY, corpus_root=tmp_path, with_guide=False)


def test_a_guide_master_that_disagrees_with_its_capture_master_is_refused_at_startup(tmp_path) -> None:  # noqa: ANN001
    """The frame-count and fps agreement ``ChainStore.__getitem__`` otherwise only checks the
    first time a rank draws this particular source's chain (a ``ValueError`` deep inside a
    30-minute-old distributed run), moved to the same up-front check as F4's other bundle
    validation."""
    frame_drift = tmp_path / "Part_1" / "0003_01" / "views" / "view01_cam01"
    frame_drift.mkdir(parents=True)
    torch.save(
        {"schema_version": 2, "master": torch.zeros(CHANNELS, 18, 2, 2), "fps": 30.0},
        frame_drift / train.dataset.capture_bundle_name("bg"),
    )
    torch.save(
        {"schema_version": 2, "master": torch.zeros(CHANNELS, 17, 2, 2), "fps": 30.0},
        frame_drift / train.dataset.guide_bundle_name("bg"),
    )
    subset = {
        "sources": [{"relative_dir": "Part_1/0003_01/views/view01_cam01", "n_latent_frames": 18}],
        "chains": [{"source": "Part_1/0003_01/views/view01_cam01", "blocks": [0]}],
    }
    with pytest.raises(SystemExit, match=r"guide master .* shape .* != capture master .* shape"):
        train.assert_subset_matches_geometry(subset, GEOMETRY, corpus_root=tmp_path, with_guide=True)

    fps_drift = tmp_path / "Part_1" / "0004_01" / "views" / "view01_cam01"
    fps_drift.mkdir(parents=True)
    torch.save(
        {"schema_version": 2, "master": torch.zeros(CHANNELS, 18, 2, 2), "fps": 30.0},
        fps_drift / train.dataset.capture_bundle_name("bg"),
    )
    torch.save(
        {"schema_version": 2, "master": torch.zeros(CHANNELS, 18, 2, 2), "fps": 25.0},
        fps_drift / train.dataset.guide_bundle_name("bg"),
    )
    subset = {
        "sources": [{"relative_dir": "Part_1/0004_01/views/view01_cam01", "n_latent_frames": 18}],
        "chains": [{"source": "Part_1/0004_01/views/view01_cam01", "blocks": [0]}],
    }
    with pytest.raises(SystemExit, match=r"guide fps 25\.0 != capture fps 30\.0"):
        train.assert_subset_matches_geometry(subset, GEOMETRY, corpus_root=tmp_path, with_guide=True)


def test_a_guide_master_with_matching_frame_count_but_a_different_spatial_size_is_refused(  # noqa: ANN001
    tmp_path,
) -> None:
    """2026-09-18 audit, Stage A gap 4: the preflight guide check compared latent-FRAME count
    only, so a guide re-encoded at a different crop box -- same 18 latent frames, spatial
    (16, 32) instead of the capture's (32, 32) -- passed both a dry run and normal startup and
    was only caught deep inside `ChainStore.__getitem__`'s `z_g.shape != z_y.shape` check, after
    `Accelerator()`/model setup. The check now compares the full `[C, F, H, W]` shape."""
    view = tmp_path / "Part_1" / "0005_01" / "views" / "view01_cam01"
    view.mkdir(parents=True)
    torch.save(
        {"schema_version": 2, "master": torch.zeros(CHANNELS, 18, 32, 32), "fps": 30.0},
        view / train.dataset.capture_bundle_name("bg"),
    )
    torch.save(
        {"schema_version": 2, "master": torch.zeros(CHANNELS, 18, 16, 32), "fps": 30.0},
        view / train.dataset.guide_bundle_name("bg"),
    )
    subset = {
        "sources": [{"relative_dir": "Part_1/0005_01/views/view01_cam01", "n_latent_frames": 18}],
        "chains": [{"source": "Part_1/0005_01/views/view01_cam01", "blocks": [0]}],
    }
    with pytest.raises(SystemExit, match=r"guide master .* shape .* != capture master .* shape"):
        train.assert_subset_matches_geometry(subset, GEOMETRY, corpus_root=tmp_path, with_guide=True)


def test_multilevel_sigma_schedule_rotates_by_rank_and_step() -> None:
    """(rank + step) % len(levels): every step's batch mixes levels; every rank sees all levels."""
    levels = (0.909375, 0.725, 0.421875)
    args = argparse.Namespace(sigma0=0.725, sigma_levels=list(levels))
    assert train.training_sigmas(args) == levels
    # step=0: 4 ranks, 3 levels -- rank 3 rotates back to rank 0's level rather than needing a
    # 4th value, same as the old rank-only assignment.
    assert [train.sigma_for_rank(levels, rank, 0) for rank in range(4)] == [
        0.909375, 0.725, 0.421875, 0.909375,
    ]
    # A fixed rank walks through every level in turn as step advances.
    assert [train.sigma_for_rank(levels, 0, step) for step in range(4)] == [
        0.909375, 0.725, 0.421875, 0.909375,
    ]


def test_sigma_zero_is_refused() -> None:
    """sigma=0.0 adds no noise, so its loss and gradient are identically zero -- never trained."""
    args = argparse.Namespace(sigma0=0.725, sigma_levels=[0.909375, 0.725, 0.421875, 0.0])
    with pytest.raises(SystemExit, match=r"sigma=0\.0"):
        train.training_sigmas(args)


def test_step_zero_lora_export_requires_exactly_zero_b() -> None:
    """The saved adapter, not just PEFT's in-memory module, is a provable no-op at step 0."""
    exported = {
        "diffusion_model.block.to_q.lora_A.weight": torch.randn(2, 4, dtype=torch.bfloat16),
        "diffusion_model.block.to_q.lora_B.weight": torch.zeros(4, 2, dtype=torch.bfloat16),
    }
    train.assert_exported_lora_is_noop(exported)

    exported["diffusion_model.block.to_q.lora_B.weight"][0, 0] = 1
    with pytest.raises(RuntimeError, match="non-zero LoRA delta"):
        train.assert_exported_lora_is_noop(exported)


def test_step_zero_lora_export_requires_b_weights() -> None:
    with pytest.raises(RuntimeError, match="no lora_B weights"):
        train.assert_exported_lora_is_noop({"diffusion_model.block.to_q.lora_A.weight": torch.zeros(2, 4)})


def _fake_model() -> model_registry.RefinerModel:
    """A ``RefinerModel`` with no checkpoint behind it -- ``checkpoint_metadata`` only reads
    ``.scale_factors`` and ``.key``, so the rest can be placeholders rather than real paths."""
    return model_registry.RefinerModel(
        key="test", version=(0,), paths=None, sigmas=[], stepper_kind="euler", caps=None,
        scale_factors=SCALE, scale_factors_source="test",
    )


@pytest.fixture
def cli_subset(tmp_path, monkeypatch):  # noqa: ANN001, ANN201
    """Two independent sources: a valid first chain cannot hide a bad second chain."""
    for source in ("a", "b"):
        _write_capture_and_guide(tmp_path / source, n_latent_frames=18)
    subset = {
        "kind": "one_step_argavatar_block_chains", "corpus_root": str(tmp_path),
        "objective": "bg", "chain_length": 1,
        "sources": [{"relative_dir": name, "n_latent_frames": 18} for name in ("a", "b")],
        "chains": [
            {"source": name, "split": "train", "actor": name, "blocks": [0], "seed_is_clip_start": True}
            for name in ("a", "b")
        ],
    }
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(subset))
    monkeypatch.setattr(model_registry, "resolve", lambda _: _fake_model())
    return path


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("defect", [
    "missing_guide", "guide_shape", "guide_fps", "missing_fps", "zero_fps",
    "nan_fps", "bad_tensor", "capture_geometry",
])
def test_cli_preflight_rejects_bad_second_source_without_touching_run(  # noqa: ANN001
    tmp_path, monkeypatch, cli_subset, dry_run, defect,
) -> None:
    guide_mode = "d1" if "guide" in defect else "d0"
    path = tmp_path / "b" / (
        train.dataset.guide_bundle_name("bg") if guide_mode == "d1" else train.dataset.capture_bundle_name("bg")
    )
    record = train._load_record(path)
    if defect == "missing_guide":
        path.unlink()
        error = "does not exist"
    else:
        if defect == "guide_shape":
            record["master"] = record["master"][:, :, :1, :]
            error = "guide master .* shape"
        elif defect == "guide_fps":
            record["fps"] = 25.0
            error = "guide fps"
        elif defect == "bad_tensor":
            record["master"] = record["master"][0]
            error = "nonempty floating"
        elif defect == "capture_geometry":
            record["master"] = record["master"][:, :, :1, :]
            error = "all sources must share"
        else:
            record.pop("fps")
            if defect != "missing_fps":
                record["fps"] = 0.0 if defect == "zero_fps" else float("nan")
            error = "fps must be a finite positive"
        torch.save(record, path)
    monkeypatch.setattr(train, "Accelerator", lambda: pytest.fail("preflight reached Accelerator"))
    output = tmp_path / "run"
    output.mkdir()
    old_log = output / "metrics_rank0.jsonl"
    old_log.write_bytes(b"old run\n")
    args = ["--subset", str(cli_subset), "--output", str(output), "--overwrite", "--guide-mode", guide_mode]
    with pytest.raises(SystemExit, match=error):
        train.main(args + (["--dry-run"] if dry_run else []))
    assert old_log.read_bytes() == b"old run\n"
    assert list(output.iterdir()) == [old_log]


def test_successful_overwrite_dry_run_preserves_the_whole_run(tmp_path, monkeypatch, cli_subset) -> None:  # noqa: ANN001
    monkeypatch.setattr(train, "Accelerator", lambda: pytest.fail("dry-run reached Accelerator"))
    output = tmp_path / "run"
    output.mkdir()
    (output / "checkpoints").mkdir()
    files = [output / "config.json", output / "metrics_rank0.jsonl", output / "checkpoints" / "old.safetensors"]
    for path in files:
        path.write_bytes(b"prior contents")
    assert train.main(["--subset", str(cli_subset), "--output", str(output), "--overwrite", "--dry-run"]) == 0
    assert set(output.rglob("*")) == {*files, output / "checkpoints"}
    assert all(path.read_bytes() == b"prior contents" for path in files)


@pytest.mark.parametrize("rank", [0, 1])
def test_only_main_rank_archives_after_setup(tmp_path, monkeypatch, cli_subset, rank) -> None:  # noqa: ANN001
    """Exercise the actual CLI mutation boundary with CPU stand-ins for distributed setup."""
    events = []
    accelerator = SimpleNamespace(
        device=torch.device("cpu"), num_processes=2, process_index=rank, is_main_process=rank == 0,
        wait_for_everyone=lambda: events.append("barrier"),
        prepare=lambda *args: (events.append("prepare") or args),
    )
    monkeypatch.setattr(train, "Accelerator", lambda: accelerator)
    monkeypatch.setattr(train.prompt_cache, "get_or_build", lambda *args: None)
    monkeypatch.setattr(train, "build_transformer", lambda *args: torch.nn.Linear(2, 2))
    monkeypatch.setattr(train, "_num_blocks", lambda _: 1)
    monkeypatch.setattr(train, "_inner_dim", lambda _: 2)
    monkeypatch.setattr(train, "init_wandb", lambda *args, **kwargs: None)
    archive = train.archive_existing_run

    def record_archive(path):
        events.append("archive")
        return archive(path)

    monkeypatch.setattr(train, "archive_existing_run", record_archive)
    output = tmp_path / "run"
    output.mkdir()
    (output / "config.json").write_text("prior config")
    assert train.main(["--subset", str(cli_subset), "--output", str(output), "--overwrite", "--steps", "0"]) == 0
    if rank == 0:
        assert events == ["prepare", "barrier", "archive", "barrier", "barrier"]
        assert json.loads((output / "config.json").read_text())["loss"] == "full_frame_x0_mse"
        archived = next(output.glob("archived_*"))
        assert (archived / "config.json").read_text() == "prior config"
    else:
        assert "archive" not in events
        assert (output / "config.json").read_text() == "prior config"


def test_checkpoint_metadata_stamps_the_loss_identifier() -> None:
    """2026-09-18 audit gap 2: the loss arithmetic has been unweighted full-frame MSE since
    before this field existed, but nothing on a saved checkpoint said so -- a reader (or a
    future loss change) had no field to check against. ``onestep_avatar_loss`` closes that."""
    args = argparse.Namespace(
        sigma0=SIGMA0, sigma_levels=None, block_latent_frames=2, context_latent_frames=2,
        objective="bg", guide_mode="d1", anchor_weight=0.0, teacher_forcing=False,
        lora_rank=8, lora_alpha=8, lora_target="attn",
    )
    subset = {"chain_length": 3, "sources": []}
    metadata = train.checkpoint_metadata(args, subset, _fake_model(), step=0)
    assert metadata["onestep_avatar_loss"] == train.FULL_FRAME_X0_MSE == "full_frame_x0_mse"


# --- onestep_core: the deployment counterpart of the training loop -----------------------


def test_rollout_slices_blocks_out_of_one_master_encode() -> None:
    """SS4.4: deployment slices the clip's ONE continuous encode, exactly as training does.

    A rollout that re-encoded per window would hand the model a fresh causal keyframe at every
    block's local frame 0 -- input it was never trained on, and a mismatch that shows up as a
    quality number rather than an error.
    """
    chain = _chain()
    grid = _grid(chain)
    geometry = GEOMETRY
    cache = _cache(chain)
    model = StubTransformer()
    denoise_fn = causal_core.denoised_from_velocity_model(model)
    guide = grid.patchify(chain.z_g.unsqueeze(0).to(train.DTYPE))
    tokens, forwards = causal_core.rollout(
        denoise_fn, grid, geometry, cache, guide, torch.zeros(1, 1, 8), SIGMA0,
        first_frame_condition=guide[:, : grid.tokens_per_latent_frame],
    )
    plan = geometry.plan(LATENT_FRAMES)
    assert forwards == 2 * len(plan)  # one denoise plus one cache refresh per block
    assert tokens.shape == guide.shape


def test_rollout_result_counts_both_passes() -> None:
    """The refresh forward is real compute and must be in the number SS6 compares to `k2`."""
    result = onestep_core.RolloutResult(
        latent=torch.zeros(1), forwards=6, blocks=3, denoise_forwards=3, refresh_forwards=3
    )
    assert result.forwards == result.denoise_forwards + result.refresh_forwards


def test_guide_conditionings_accepts_only_the_deployable_arm() -> None:
    z_g = torch.zeros(1, CHANNELS, 3, 2, 2)
    assert onestep_core.guide_conditionings(z_g, "d1") == ()
    with pytest.raises(ValueError, match="unknown guide mode"):
        onestep_core.guide_conditionings(z_g, "d2")
    with pytest.raises(ValueError, match="training-only"):
        onestep_core.guide_conditionings(z_g, "d0")


def test_one_step_sigma_passes_an_on_grid_value_through_unchanged() -> None:
    """The guard must not perturb sigma0 -- only validate it -- so wiring it into ``rollout``
    cannot itself change a rollout's numeric output at an on-grid sigma0."""
    grid = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
    assert onestep_core.one_step_sigma(grid, 0.725) == pytest.approx(0.725)


def test_one_step_sigma_refuses_an_off_grid_value() -> None:
    grid = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
    with pytest.raises(ValueError, match="not on the distilled sigma grid"):
        onestep_core.one_step_sigma(grid, 0.5)


def test_rollout_refuses_an_off_grid_sigma0() -> None:
    """SS9 risk 13: ``onestep_core.rollout`` is the deploy path, so this is where an off-grid
    sigma0 must be refused rather than silently produce plausible-looking output."""
    chain = _chain()
    with pytest.raises(ValueError, match="not on the distilled sigma grid"):
        onestep_core.rollout(
            X0Stub(),
            torch.zeros(1, 1, 8),
            chain.z_g.unsqueeze(0),
            GEOMETRY,
            0.5,  # off the nine-point distilled grid
            FPS,
            first_frame_latent=chain.z_y.unsqueeze(0)[:, :, :1],
            device=DEVICE,
            latent_channels=CHANNELS,
            num_layers=1,
            inner_dim=4,
        )


def test_rollout_runs_end_to_end_at_an_on_grid_sigma0() -> None:
    """The guard sits in front of the rollout it guards -- an on-grid sigma0 still rolls out.

    ``onestep_core.rollout`` had no direct test before this stage (only ``causal_core.rollout``,
    the shared core it calls, was covered); this is also the first positive-path coverage of it.
    """
    chain = _chain()
    result = onestep_core.rollout(
        X0Stub(),
        torch.zeros(1, 1, 8),
        chain.z_g.unsqueeze(0),
        GEOMETRY,
        SIGMA0,
        FPS,
        first_frame_latent=chain.z_y.unsqueeze(0)[:, :, :1],
        device=DEVICE,
        seed=0,
        latent_channels=CHANNELS,
        num_layers=1,
        inner_dim=4,
    )
    plan = GEOMETRY.plan(LATENT_FRAMES)
    assert result.blocks == len(plan)
    assert result.forwards == result.denoise_forwards + result.refresh_forwards == 2 * len(plan)


def test_wandb_default_enabled() -> None:
    """W&B logging is enabled by default to 'onestep-avatar' in online mode."""
    args = train.parse_args(["--subset", "/fake/subset.json", "--output", "/fake/out"])
    assert args.wandb_project == "onestep-avatar"
    assert args.wandb_mode == "online"
    assert not args.no_wandb
    assert train.wandb_is_enabled(args) is True


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--no-wandb"],
        ["--wandb-mode", "disabled"],
        ["--wandb-project", ""],
        ["--wandb-project", "none"],
    ],
)
def test_wandb_disable_options(extra_args: list[str]) -> None:
    """W&B can be disabled via --no-wandb, --wandb-mode disabled, or empty/none project."""
    args = train.parse_args(["--subset", "/fake/subset.json", "--output", "/fake/out", *extra_args])
    assert train.wandb_is_enabled(args) is False
    assert train.init_wandb(args, config={}) is None


def test_init_wandb_calls_wandb_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """When enabled, init_wandb calls wandb.init with project and mode."""
    called_with = {}

    class FakeWandb:
        @staticmethod
        def init(**kwargs):  # noqa: ANN003
            called_with.update(kwargs)
            return "fake_run"

    monkeypatch.setattr(train, "wandb", FakeWandb, raising=False)
    import sys
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)

    args = train.parse_args(["--subset", "/fake/subset.json", "--output", "/fake/out"])
    run = train.init_wandb(args, config={"test_key": 123})
    assert run == "fake_run"
    assert called_with["project"] == "onestep-avatar"
    assert called_with["mode"] == "online"
    assert called_with["config"] == {"test_key": 123}
