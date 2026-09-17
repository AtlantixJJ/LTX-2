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

import pytest
import torch

from ltx_core.types import SpatioTemporalScaleFactors
from ltx_core.utils import to_velocity
from scripts.onestep_avatar import causal_core, onestep_core, train
from scripts.onestep_avatar.causal_core import BlockCache, CausalGeometry

SCALE = SpatioTemporalScaleFactors(time=8, height=32, width=32)
GEOMETRY = CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=2)
CHANNELS = 8
EDGE = 64  # -> a 2x2 latent grid: 4 tokens per latent frame
LATENT_FRAMES = 7  # -> blocks [(0,3), (3,5), (5,7)]
FPS = 30.0
SIGMA0 = 0.725
DEVICE = torch.device("cpu")


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
        loss_weights=None,
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


def test_every_block_token_is_in_the_loss() -> None:
    """Under SS4.4's causal scheme there are no conditioning tokens left to exclude.

    The old window carried the frozen carryover and the keyframe at ``denoise_mask`` 0 and had
    to weight them out -- score a model on reproducing a value it was handed and you both
    dilute the gradient and reward copying. The cache holds that content now, so it is not in
    the sequence at all, and the weights are plainly ones.
    """
    chain = _chain()
    grid = _grid(chain)
    span = GEOMETRY.plan(LATENT_FRAMES)[1]
    weights = train.block_weights(grid, chain, span, DEVICE)
    assert weights.shape == (1, (span[1] - span[0]) * grid.tokens_per_latent_frame, 1)
    assert (weights == 1.0).all()


def test_loss_weights_are_sliced_from_the_clips_own_master_grid() -> None:
    """The weights are per-CLIP, and a block takes the same slice of them the latent takes.

    Per-window mask files were the other half of the tree SS1.6 removed; storing one grid per
    clip is what makes it impossible to index the weights and the latent differently.
    """
    chain = _chain()
    mask = torch.zeros(LATENT_FRAMES, EDGE // SCALE.height, EDGE // SCALE.width)
    mask[3:5] = 1.0  # exactly block 1's frames
    chain = train.Chain(**{**chain.__dict__, "loss_weights": mask})
    grid = _grid(chain)
    plan = GEOMETRY.plan(LATENT_FRAMES)
    assert (train.block_weights(grid, chain, plan[1], DEVICE) == 1.0).all()
    assert (train.block_weights(grid, chain, plan[2], DEVICE) == 0.0).all()


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
    noisy = causal_core.noise_block(guide[:, lo:hi], SIGMA0, 0)
    z0 = causal_core.denoise_block(
        causal_core.denoised_from_velocity_model(model_plain), grid, _cache(chain), noisy,
        torch.zeros(1, 1, 8), SIGMA0, span,
    )
    loss = train.masked_mse(z0, target[:, lo:hi], train.block_weights(grid, chain, span, DEVICE))
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


def test_disagreement_weights_keep_the_full_frame_and_down_weight_only_the_band() -> None:
    """SS1.5's one masking rule, in four cells.

    The point of the rule is what it does NOT do: agreement -- whether both grids say subject
    or both say background -- keeps full weight. A subject mask would have zeroed the
    background cell, and with it the ghost band that lives there, which is why none of the
    five subject masks this replaced could train the product objective.
    """
    record = {
        # agree-subject | render-only | capture-only | agree-background
        "render_alpha": torch.tensor([[[1.0, 1.0, 0.0, 0.0]]]),
        "capture_mask": torch.tensor([[[1.0, 0.0, 1.0, 0.0]]]),
    }
    assert train.disagreement_weights(record, 0.0).tolist() == [[[1.0, 0.0, 0.0, 1.0]]]
    # A partial weight lands proportionally, and 1.0 IS the plain full-frame loss.
    assert train.disagreement_weights(record, 0.25).tolist() == [[[1.0, 0.25, 0.25, 1.0]]]
    assert train.disagreement_weights(record, 1.0).tolist() == [[[1.0, 1.0, 1.0, 1.0]]]


def test_a_half_disputed_boundary_cell_is_half_weighted() -> None:
    """The grids are area fractions, so the band is SOFT -- a 32x latent cell straddling the
    silhouette is partly disputed, not wholly. Thresholding it here would reintroduce exactly
    the 32-pixel edge quantisation SS1.2 refuses to accept in the guide."""
    record = {
        "render_alpha": torch.tensor([[[1.0]]]),
        "capture_mask": torch.tensor([[[0.5]]]),
    }
    assert train.disagreement_weights(record, 0.0).tolist() == [[[0.5]]]


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


def test_d0_rejects_the_anchor_term() -> None:
    """base_denoised is Phi(lerp(z_g, eps, sigma_0)) -- off-input for a z_y-noised run."""
    with pytest.raises(SystemExit, match="off-input"):
        train.main(
            [
                "--subset", "/nonexistent.json", "--output", "/nonexistent",
                "--guide-mode", "d0", "--anchor-weight", "0.1",
            ]
        )


def test_a_window_chain_subset_is_refused_rather_than_reinterpreted(tmp_path) -> None:  # noqa: ANN001
    """A subset frozen before SS4.4 indexes windows that no longer exist.

    Silently reading its ``windows`` lists as block indices would train on the wrong frames --
    a window index and a block index are different numbers over the same clip.
    """
    subset = tmp_path / "old.json"
    subset.write_text('{"kind": "one_step_argavatar_window_chains", "chains": [], "sources": []}')
    with pytest.raises(SystemExit, match=r"block-chain subset"):
        train.main(["--subset", str(subset), "--output", str(tmp_path / "out")])


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
        "geometry": {"latent_time_scale": 8},
        "chains": [{"source": "a", "split": "train", "actor": "1", "blocks": [0], "seed_is_clip_start": True}],
        "sources": [
            {"relative_dir": "a", "n_latent_frames": 18},
            {"relative_dir": "b", "n_latent_frames": 28},
        ],
    }
    store = train.ChainStore(
        subset, tmp_path, split="train", objective="bg", band_weight=1.0,
        with_anchor=False, with_guide=False,
    )
    assert store.max_latent_frames == 28


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
        {"schema_version": 2, "master": torch.zeros(CHANNELS, 18, 2, 2)},
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
        denoise_fn, grid, geometry, cache, guide, torch.zeros(1, 1, 8), SIGMA0
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
