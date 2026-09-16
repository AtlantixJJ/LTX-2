"""CPU tests for §4.4's causal block rollout -- small dims, a REAL transformer, no checkpoint.

The decisive one is :func:`test_cached_rollout_matches_block_causal_full_sequence`. Everything
the K/V cache buys rests on a single claim -- that a cached forward computes exactly what a
full-sequence forward under a block-causal mask computes -- and that claim is about the
``ltx_core`` attention change, not about this package's bookkeeping. So it is checked against
an actual ``LTXModel`` (2 tiny layers) rather than a stub: a stub would pass whether or not
the cache/RoPE/mask interaction is right, which is the only thing in doubt.

The rest pin the properties a wrong number would otherwise be blamed on the model for:

* the noise formula is ``GaussianNoiser``'s, not a lookalike;
* a causal block finalizes the same pixel span the `k2` window did;
* eviction keeps the pinned sink and the newest context, not an arbitrary window;
* only latent frame 0 is marked a keyframe -- the bug the per-window tools had, where every
  window past a clip's first claimed a single-pixel keyframe it did not have;
* cache priming groups retained frames by the block they were really denoised in.
"""

from __future__ import annotations

import pytest
import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.types import LatentState, SpatioTemporalScaleFactors
from scripts.onestep_avatar import causal_core
from scripts.onestep_avatar.causal_core import BlockCache, CausalGeometry, ClipGrid
from scripts.prune.core import refine_task

SCALE = SpatioTemporalScaleFactors(time=8, height=32, width=32)
CHANNELS = 8
EDGE = 64  # a 2x2 latent grid -> 4 tokens per latent frame
LATENT_FRAMES = 7
FPS = 30.0
SIGMA0 = 0.725
DEVICE = torch.device("cpu")
CONTEXT_TOKENS = 3
CONTEXT_DIM = 16


def _model() -> LTXModel:
    """A 2-layer LTXModel with every parameter deterministically initialised.

    ``LTXModel`` allocates several parameters with ``torch.empty``; left as they come, a CPU
    test would compare two forwards through uninitialised memory and could pass or NaN at
    random.
    """
    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        num_layers=2,
        cross_attention_dim=8,  # == inner_dim: caption_projection maps the text dim into it
        caption_projection=torch.nn.Linear(CONTEXT_DIM, 8),
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return model.eval()


def _geometry(context_latent_frames: int = 16) -> CausalGeometry:
    return CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=context_latent_frames)


def _grid(geometry: CausalGeometry, latent_frames: int = LATENT_FRAMES) -> ClipGrid:
    return ClipGrid.build(
        latent_frames, EDGE, EDGE, FPS, geometry, device=DEVICE, dtype=torch.float32, latent_channels=CHANNELS
    )


def _context() -> torch.Tensor:
    return torch.randn(1, CONTEXT_TOKENS, CONTEXT_DIM, generator=torch.Generator().manual_seed(1))


def test_cached_rollout_matches_block_causal_full_sequence() -> None:
    """The cache's whole premise: cached == full-sequence-with-a-causal-mask, block by block.

    Block ``i``'s denoise forward sees earlier blocks as CLEAN tokens at timestep 0 (that is
    what the refresh pass wrote) and its own tokens noisy at sigma_0. The reference builds
    exactly that sequence explicitly, masks it block-causally, and runs one ordinary forward.
    If these ever disagree, every number the AR loop produces is measuring a different model
    from the one it thinks it is.
    """
    torch.manual_seed(0)
    model = _model()
    geometry = _geometry()
    grid = _grid(geometry)
    context = _context()
    denoise_fn = causal_core.denoised_from_velocity_model(model)
    plan = geometry.plan(grid.latent_frames)
    assert len(plan) >= 3, "the test needs at least three blocks to exercise a non-trivial history"

    tokens = grid.latent_frames * grid.tokens_per_latent_frame
    clean = torch.randn(1, tokens, CHANNELS)
    noisy = torch.randn(1, tokens, CHANNELS)

    cache = BlockCache.allocate(
        grid, geometry, num_layers=len(model.transformer_blocks), inner_dim=model.inner_dim,
        device=DEVICE, dtype=torch.float32,
    )
    cache.reset()
    cached_outputs = []
    for span in plan:
        lo, hi = grid.token_span(*span)
        cached_outputs.append(
            causal_core.denoise_block(denoise_fn, grid, cache, noisy[:, lo:hi], context, SIGMA0, span)
        )
        causal_core.refresh_block(denoise_fn, grid, cache, clean[:, lo:hi], context, span)

    for index, span in enumerate(plan):
        lo, hi = grid.token_span(*span)
        prefix_end = lo
        sequence = torch.cat([clean[:, :prefix_end], noisy[:, lo:hi]], dim=1)
        spans = [(start, end, i) for i, (start, end) in enumerate(plan[: index + 1])]
        ids = causal_core.block_ids_for(spans, grid.tokens_per_latent_frame)
        # Per-token timesteps: 0 on the clean prefix, sigma_0 on the block being denoised.
        timesteps = torch.zeros(1, sequence.shape[1], 1)
        timesteps[:, prefix_end:] = SIGMA0
        modality = causal_core.block_modality(
            grid, sequence, context, SIGMA0,
            token_slices=[(0, hi)],
            attention_mask=causal_core.block_causal_mask(ids),
        )
        reference = denoise_fn(type(modality)(**{**modality.__dict__, "timesteps": timesteps}))
        torch.testing.assert_close(cached_outputs[index], reference[:, prefix_end:], rtol=2e-4, atol=2e-4)


def test_teacher_forcing_flag_refreshes_the_cache_from_the_guide() -> None:
    """``rollout(teacher_forcing=True)`` is `train_chain`'s ablation, not a no-op flag.

    Refresh is fed ``guide_tokens`` instead of the block's own denoised output -- the one
    tensor `train.py`'s ``clean = target_tokens[...] if teacher_forcing else z0.detach()``
    also switches on. Checked against a reference built the same way, and against the
    self-forced default to confirm the two regimes actually diverge.
    """
    torch.manual_seed(2)
    model = _model()
    geometry = _geometry()
    grid = _grid(geometry)
    context = _context()
    denoise_fn = causal_core.denoised_from_velocity_model(model)
    plan = geometry.plan(grid.latent_frames)
    assert len(plan) >= 2, "the test needs a second block to see what the first block's refresh fed it"

    tokens = grid.latent_frames * grid.tokens_per_latent_frame
    guide = torch.randn(1, tokens, CHANNELS)

    def _cache() -> BlockCache:
        return BlockCache.allocate(
            grid, geometry, num_layers=len(model.transformer_blocks), inner_dim=model.inner_dim,
            device=DEVICE, dtype=torch.float32,
        )

    tf_cache = _cache()
    tf_tokens, _ = causal_core.rollout(
        denoise_fn, grid, geometry, tf_cache, guide, context, SIGMA0, seed=7, blocks=plan[:2], teacher_forcing=True,
    )

    ref_cache = _cache()
    ref_cache.reset()
    lo0, hi0 = grid.token_span(*plan[0])
    noisy0 = causal_core.noise_block(guide[:, lo0:hi0], SIGMA0, 7)
    causal_core.denoise_block(denoise_fn, grid, ref_cache, noisy0, context, SIGMA0, plan[0])
    causal_core.refresh_block(denoise_fn, grid, ref_cache, guide[:, lo0:hi0], context, plan[0])
    lo1, hi1 = grid.token_span(*plan[1])
    noisy1 = causal_core.noise_block(guide[:, lo1:hi1], SIGMA0, 8)
    reference1 = causal_core.denoise_block(denoise_fn, grid, ref_cache, noisy1, context, SIGMA0, plan[1])
    torch.testing.assert_close(tf_tokens[:, lo1:hi1], reference1, rtol=2e-4, atol=2e-4)

    sf_cache = _cache()
    sf_tokens, _ = causal_core.rollout(
        denoise_fn, grid, geometry, sf_cache, guide, context, SIGMA0, seed=7, blocks=plan[:2], teacher_forcing=False,
    )
    assert not torch.allclose(tf_tokens[:, lo1:hi1], sf_tokens[:, lo1:hi1], rtol=2e-4, atol=2e-4)


def test_noise_block_matches_the_gaussian_noiser() -> None:
    """``noise_block`` is ``GaussianNoiser``'s ``lerp(clean, eps, sigma)``, not a lookalike.

    The AR loop no longer routes the init through ``create_noised_state`` (there is no
    conditioning item left to apply), so the arithmetic is pinned here instead of inherited.
    """
    clean = torch.randn(1, 12, CHANNELS)
    seed = 1234
    ours = causal_core.noise_block(clean, SIGMA0, seed)

    state = LatentState(
        latent=clean.clone(),
        denoise_mask=torch.ones(1, 12, 1),
        positions=torch.zeros(1, 3, 12, 2),
        clean_latent=clean.clone(),
    )
    theirs = GaussianNoiser(generator=torch.Generator(device="cpu").manual_seed(seed))(state, SIGMA0)
    torch.testing.assert_close(ours, theirs.latent)


def test_block_finalizes_the_deployed_stride() -> None:
    """A causal block covers the same pixel span the `k2` sliding window finalized per step.

    Not cosmetic: every §8 table compares the one-step adapter against a `k2` baseline
    measured at that stride, and a block of a different length would silently be a different
    task with a different per-step cost.
    """
    geometry = causal_core.deployed_geometry(SCALE)
    window = refine_task.deployed_geometry(SCALE)
    assert causal_core.matches_deployed_stride(geometry, window)
    assert geometry.block_latent_frames * SCALE.time == window.stride_frames


def test_block_plan_absorbs_the_keyframe_into_block_zero() -> None:
    geometry = _geometry()
    assert geometry.plan(7) == [(0, 3), (3, 5), (5, 7)]
    # A tail shorter than a block is dropped, never shortened: a short block is a different
    # condition, not a smaller one.
    assert geometry.plan(8) == [(0, 3), (3, 5), (5, 7)]
    assert geometry.plan(2) == []


def test_eviction_keeps_the_pinned_sink_and_the_newest_context() -> None:
    geometry = CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=2)
    grid = _grid(geometry, latent_frames=9)
    tokens = grid.tokens_per_latent_frame
    cache = BlockCache.allocate(
        grid, geometry, num_layers=1, inner_dim=4, device=DEVICE, dtype=torch.float32
    )
    # Write frames 0..4 as five distinguishable one-frame blocks, evicting after each.
    for frame in range(5):
        value = torch.full((1, tokens, 4), float(frame))
        cache.caches[0].write(value, value, cache.start)
        cache.evict()
    assert cache.start == (geometry.sink_latent_frames + geometry.context_latent_frames) * tokens
    kept = cache.caches[0].k[0, : cache.start : tokens, 0].tolist()
    assert kept == [0.0, 3.0, 4.0]  # the pinned frame 0, then the two newest


def test_a_deep_cache_accumulates_the_rollout_instead_of_evicting() -> None:
    """The deep-context regime: eviction is what bounds a LONG rollout, not what a short
    one spends its time doing.

    At ``context_latent_frames`` past the chain's own reach nothing is ever dropped -- the
    cache holds the pinned sink plus every frame the rollout has finalized, in order. This is
    the property the deeper settings exist for, and it is the one that would silently not
    hold if eviction's "nothing to drop yet" branch were wrong.
    """
    geometry = CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=16)
    grid = _grid(geometry, latent_frames=9)
    tokens = grid.tokens_per_latent_frame
    cache = BlockCache.allocate(
        grid, geometry, num_layers=1, inner_dim=4, device=DEVICE, dtype=torch.float32
    )
    for frame in range(5):
        value = torch.full((1, tokens, 4), float(frame))
        cache.caches[0].write(value, value, cache.start)
        cache.evict()
    assert cache.start == 5 * tokens
    assert cache.caches[0].k[0, : cache.start : tokens, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_cache_capacity_is_capped_by_the_clip_a_chain_actually_rolls_over() -> None:
    """A deep cache must not reserve K/V for frames that cannot exist.

    ~0.8 GB per latent frame per rank at the 22B geometry means over-reserving 16 frames for
    a clip with 9 is gigabytes of untouched memory, so capacity is the smaller of the
    policy's steady-state need and the clip's whole length.
    """
    deep = CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=16)
    assert deep.cache_latent_frames == 1 + 16 + 1 + 2
    assert deep.cache_latent_frames_for(9) == 9  # the clip, not the policy
    shallow = CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=2)
    # A clip longer than the policy needs is bounded by the policy, as it always was.
    assert shallow.cache_latent_frames_for(64) == shallow.cache_latent_frames


def test_context_depth_past_the_supported_maximum_is_refused() -> None:
    """A ceiling, not a suggestion: the failure it prevents is an allocation that OOMs a
    training run after the model is already resident."""
    with pytest.raises(ValueError, match="exceeds the supported maximum"):
        CausalGeometry(
            scale_factors=SCALE,
            block_latent_frames=2,
            context_latent_frames=causal_core.MAX_CONTEXT_LATENT_FRAMES + 1,
        )


def test_clip_grid_marks_only_latent_frame_zero_as_a_keyframe() -> None:
    """The per-window construction marked EVERY window's own first frame as a keyframe.

    Under §4.4's continuous encode that is false for every window past a clip's first -- those
    slots hold a regular multi-frame block. Building the grid over the master fixes it, and
    this is the assertion that says so.
    """
    grid = _grid(_geometry())
    marks = grid.keyframes_mask[0, :, 0]
    tokens = grid.tokens_per_latent_frame
    assert marks[:tokens].bool().all()
    assert not marks[tokens:].bool().any()


def test_prime_cache_groups_retained_frames_by_their_real_block() -> None:
    """Frames denoised together must be able to attend to each other when the cache is primed.

    Frame 0 and frames 1-2 are all block 0, so a priming pass that split "the sink" from "the
    context" into two mask blocks would forbid an attention edge the real rollout had.
    """
    geometry = CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=2)
    plan = geometry.plan(9)
    assert causal_core.retained_prefix_spans(plan, geometry, upto_latent_frame=3) == [(0, 3, 0)]
    # Mid-clip: the pinned frame 0 is block 0; frames 3-4 are block 1, a separate span.
    assert causal_core.retained_prefix_spans(plan, geometry, upto_latent_frame=5) == [(0, 1, 0), (3, 5, 1)]
    assert causal_core.retained_prefix_spans(plan, geometry, upto_latent_frame=0) == []


def test_prime_cache_forwards_exactly_once_whether_or_not_it_has_anything_to_prime() -> None:
    """The FSDP lockstep invariant, and the only test that would have caught the 09-16 hang.

    `prime_cache` used to return early for a clip-start chain, making its forward count depend
    on the data. Under FSDP FULL_SHARD a forward is a round of all-gathers, so ranks holding
    clip-start chains issued one collective fewer than the rest and the job DEADLOCKED -- no
    error, no traceback, just pinned GPUs. Nothing in this suite noticed, because a single
    process cannot desynchronise with itself.

    Counting forwards is therefore the assertion, not observing the cache: an empty prime must
    still cost a forward, and must still leave the cache untouched.
    """
    model = _model()
    geometry = CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=2)
    grid = _grid(geometry)
    context = _context()
    tokens = torch.randn(1, grid.tokens_per_latent_frame * LATENT_FRAMES, CHANNELS)

    calls = 0
    denoise = causal_core.denoised_from_velocity_model(model)

    def counting(modality):  # noqa: ANN001, ANN202
        nonlocal calls
        calls += 1
        return denoise(modality)

    def run(upto: int) -> int:
        nonlocal calls
        calls = 0
        cache = BlockCache.allocate(
            grid, geometry, num_layers=len(model.transformer_blocks), inner_dim=model.inner_dim,
            device=DEVICE, dtype=torch.float32,
        )
        causal_core.prime_cache(
            counting, grid, cache, tokens, geometry, context, upto_latent_frame=upto
        )
        return cache.start

    # Nothing to prime, and something to prime: the SAME number of forwards either way.
    empty_start = run(0)
    assert calls == 1, "a clip-start chain must still forward once, or FSDP ranks desync"
    primed_start = run(5)
    assert calls == 1

    # ...and the empty one still wrote nothing, which is the reason it could not simply reuse
    # the real priming path.
    assert empty_start == 0
    assert primed_start > 0


def test_rope_range_is_enforced_rather_than_extrapolated() -> None:
    """Global positions are in seconds against ``positional_embedding_max_pos[0] = 20``."""
    geometry = _geometry()
    with pytest.raises(ValueError, match="temporal RoPE range"):
        ClipGrid.build(
            1 + 20 * 30 // 8, EDGE, EDGE, 1.0, geometry, device=DEVICE, dtype=torch.float32,
            latent_channels=CHANNELS,
        )


def test_the_real_training_loop_runs_a_chain_against_a_real_transformer() -> None:
    """One end-to-end pass of ``train.train_chain``: real model, real cache, real priming.

    Everything else in this file tests ``causal_core`` in isolation and everything in
    ``test_train.py`` tests the loop against a stub. This is the join: it is the only place
    the cache's ``kv_start`` bookkeeping, the clip grid's slicing and the loop's ordering are
    exercised together against a transformer that actually reads the cache. A shape or dtype
    mismatch between the three would otherwise first appear on a 22B model on a GPU.
    """
    from scripts.onestep_avatar import train  # noqa: PLC0415 -- pulls in accelerate/peft

    model = _model().to(train.DTYPE)
    geometry = CausalGeometry(scale_factors=SCALE, block_latent_frames=2, context_latent_frames=2)
    chain = train.Chain(
        source="stub/view00", split="train", actor="stub",
        # Mid-clip on purpose: this is the path that primes the cache from the GT prefix.
        seed_is_clip_start=False, blocks=[1, 2],
        z_g=torch.randn(CHANNELS, LATENT_FRAMES, 2, 2),
        z_y=torch.randn(CHANNELS, LATENT_FRAMES, 2, 2),
        fps=FPS, loss_weights=torch.rand(LATENT_FRAMES, 2, 2), z0_base=None,
    )

    class _Accelerator:
        device = DEVICE

        @staticmethod
        def backward(loss: torch.Tensor) -> None:
            loss.backward()

    grid = train.clip_grid_for(chain, geometry, device=DEVICE, latent_channels=CHANNELS)
    cache = BlockCache.allocate(
        grid, geometry, num_layers=len(model.transformer_blocks), inner_dim=model.inner_dim,
        device=DEVICE, dtype=train.DTYPE,
    )
    totals = train.train_chain(
        model, torch.randn(1, CONTEXT_TOKENS, CONTEXT_DIM, dtype=train.DTYPE), chain, geometry,
        cache, _Accelerator(), sigma0=SIGMA0, seed=0, anchor_weight=0.0, latent_channels=CHANNELS,
    )

    assert [entry["block_index"] for entry in totals["per_block"]] == [1, 2]
    assert totals["mse"] > 0.0
    # Every parameter got a gradient: the loss really flows back through the cached attention.
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    # Steady state after eviction: the pinned sink plus `context_latent_frames`.
    kept = geometry.sink_latent_frames + geometry.context_latent_frames
    assert cache.start == kept * grid.tokens_per_latent_frame
