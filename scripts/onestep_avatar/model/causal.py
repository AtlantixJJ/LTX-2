"""The ONE implementation of "roll a causal block forward", training and deployment alike.

``plans/2026-09-10-ltx25-one-step-argavatar-lora.md`` §4.4 (revised 2026-09-14). This module
uses block-causal attention, a clean-latent cache and continuous master latents.
Three things work together:

1. **Attention is block-causal.** A token attends to its own block and every earlier one,
   never to a later one. Within a block it stays bidirectional -- the block is denoised in
   one shot, so there is nothing to order inside it.
2. **Clean blocks live in a K/V cache.** Under (1) a finished block's keys and values no
   longer depend on later tokens at a fixed conditioning state, so they are computed once and
   reused. Block causality is necessary for that reuse, but prompt AdaLN's global sigma also
   changes history K/V; zero-sigma refresh is not generally an explicit current-sigma prefix.
3. **Everything is sliced out of the clip's ONE continuous VAE encode** -- the master latent.
   There is no per-window encode and no per-window re-keyed frame 0 (§4.4's 2026-09-11 rule,
   now enforced by construction because a window is no longer a unit of anything).

**The rollout, per block.**

    denoise:  queries = block i's noisy tokens, keys = [cache | block i]      (read-only)
    loss + backward on block i
    refresh:  queries = block i's CLEAN tokens at timestep 0                  (writes cache)
    evict:    keep the pinned frame-0 sink and the last `context` latent frames

The refresh is a second forward because the K/V a later block wants are those of the
*denoised* content, and the denoising forward only ever saw the noisy version of it. It runs
under ``no_grad``, stores no activations, and is the only place the cache is written.
Teacher forcing and self-forcing differ in exactly one tensor -- what is handed to the refresh
(the capture ``z_y`` or the model's own ``ẑ₀``) -- which is the entire mechanical difference
between the two regimes.

**Why frame 0 is pinned.** Latent frame 0 is the causal VAE's single-pixel keyframe, and under
§2.0 it is also the product's given real first frame -- the background every later frame is
supposed to propagate. Keeping it in the cache for the whole rollout costs one latent frame
and hands every block direct access to the thing it is being asked to preserve.

**Global positions, and the limit on them.** Tools are built once over the clip's whole master
latent, so RoPE positions are the clip's own and the cache's keys keep the positions they were
written with. The temporal RoPE axis is in *seconds* against ``positional_embedding_max_pos[0]
= 20``, so a rollout longer than 20 s leaves the range the checkpoint was trained on. At the
corpus's ~5 s clips this is not close; a streaming deployment past 20 s needs position
re-basing, and this module should raise rather than silently extrapolate.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import torch

from ltx_core.model.transformer.kv_cache import LayerKVCache, allocate_kv_caches
from ltx_core.types import (
    SpatioTemporalScaleFactors,
)
from scripts.onestep_avatar.model import common
from scripts.onestep_avatar.model.common import (
    ClipGrid,
    block_modality,
    mix_block_noise,
    noise_block,
    with_clean_prefix,
)
from scripts.onestep_avatar.model.sampling import euler_to, validate_schedule

# §4.4: the deployed stride is 16 pixel frames, which is 2 latent frames at the VAE's
# temporal scale of 8. Keeping the block at the stride means a trained adapter denoises
# exactly the span the old rollout finalized per window.
BLOCK_LATENT_FRAMES = 2

# Clean latent frames kept in the cache BESIDES the pinned frame-0 sink. Every extra frame
# costs ~0.8 GB at the 22B geometry (48 layers x 1024 tokens x 4096 dims x 2 tensors x 2
# bytes), which is why this is a flag rather than "the whole history".
#
# 1 sink + 8 context is a RETAINED HISTORY OF 9 LATENT FRAMES -- ~72 pixel frames, ~2.4 s at
# 30 fps. Read the two numbers together: the flag counts context frames, the memory budget
# counts the whole cache, and confusing them is an off-by-one worth ~0.8 GB per rank.
#
# **This default is set by what fits, not by what would help.** Measured 2026-09-19 on 4x49 GB
# with LoRA rank 32 and the t2r2 subset: depth 15 (history 16) OOMs in BACKWARD -- not in the
# forward, and not fixed by PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -- because
# gradient recompute holds attention over the whole retained history. Depth 7 ran flat at
# ~45.1 GB of 47.4. Depth 8 sits between a measured-good and a measured-bad point: treat a
# new OOM here as the depth, not as a leak, and drop to 7.
#
# At this depth a corpus-length chain evicts nothing until it has finalized past frame 8:
# block 2 attends to c0 plus clean frames 1-4.
CONTEXT_LATENT_FRAMES = 8

# The deepest cache this scheme supports, as CONTEXT frames. 16 clean latent frames is 128
# pixel frames -- most of a corpus clip (the 137-frame tier is 18 latent frames), so at this
# depth a K-block rollout evicts nothing and the cache simply ACCUMULATES every frame the
# chain has finalized, plus the primed prefix. That is the regime the default sits in.
#
# Two things bound it, and neither is arbitrary:
#
# * Memory. ~0.8 GB per retained latent frame per rank at the 22B geometry, so a 16-frame
#   retained history is ~13 GB of K/V on top of the model -- and the attention it lengthens
#   costs more again in backward. On 4x49 GB that ceiling does NOT fit at LoRA rank 32; it is
#   reachable only on larger cards or with a smaller adapter.
# * RoPE. The temporal axis is seconds against MAX_ROPE_SECONDS; 16 latent frames is ~4.3 s
#   at 30 fps, comfortably inside it, and ClipGrid.build raises for anything that is not.
#
# Accumulation is bounded by the chain too: a K-block chain writes at most
# ``sink + primed context + K * block`` frames, so at K = 3 and block = 2 the deepest a
# rollout can actually reach is well under this ceiling unless priming fills it (a mid-clip
# chain's prefix). ``cache_latent_frames`` sizes for what a given clip can really hold, so
# raising this ceiling does not by itself reserve memory nothing will use.
MAX_CONTEXT_LATENT_FRAMES = 16

# Latent frame 0 -- the causal keyframe and the product's given first frame -- never leaves.
SINK_LATENT_FRAMES = 1

# RoPE's temporal axis is seconds against positional_embedding_max_pos[0].


@dataclass(frozen=True)
class CausalGeometry:
    """The block layout of a causal rollout over a clip's master latent.

    Block 0 absorbs the keyframe (latent frames ``[0, 1 + block)``) and every later block is
    ``block`` frames. That is the same "first block is one frame longer" shape the deployed
    window already had -- a 25-frame window at a 16-frame stride is 1 + 3x8 pixel frames --
    restated in latent frames now that the window is gone.
    """

    scale_factors: SpatioTemporalScaleFactors
    block_latent_frames: int = BLOCK_LATENT_FRAMES
    context_latent_frames: int = CONTEXT_LATENT_FRAMES
    sink_latent_frames: int = SINK_LATENT_FRAMES

    def __post_init__(self) -> None:
        if self.block_latent_frames < 1:
            raise ValueError("block_latent_frames must be >= 1")
        if self.context_latent_frames < 0:
            raise ValueError("context_latent_frames must be >= 0")
        if self.context_latent_frames > MAX_CONTEXT_LATENT_FRAMES:
            raise ValueError(
                f"context_latent_frames={self.context_latent_frames} exceeds the supported "
                f"maximum of {MAX_CONTEXT_LATENT_FRAMES} (~{MAX_CONTEXT_LATENT_FRAMES * 0.8:.0f} GB "
                f"of K/V per rank at the 22B geometry)"
            )
        if self.sink_latent_frames not in (0, 1):
            raise ValueError("sink_latent_frames is 0 or 1: there is exactly one causal keyframe")

    def plan(self, latent_frames: int) -> list[tuple[int, int]]:
        """Block bounds ``[start, end)`` in latent frames; a short tail block is dropped.

        Dropped rather than padded or shortened for the same reason ``WindowGeometry.plan``
        drops one: a differently-sized block is a different training and inference condition,
        not a smaller one, and at most ``block - 1`` latent frames are lost.
        """
        if latent_frames < 1 + self.block_latent_frames:
            return []
        blocks = [(0, 1 + self.block_latent_frames)]
        start = 1 + self.block_latent_frames
        while start + self.block_latent_frames <= latent_frames:
            blocks.append((start, start + self.block_latent_frames))
            start += self.block_latent_frames
        return blocks

    @property
    def cache_latent_frames(self) -> int:
        """Frames the cache must hold: the sink, the retained context, and one live block.

        The live block is included because the refresh pass writes it *before* eviction
        trims back -- sizing for the steady state alone overflows on every block. The extra
        ``1`` is block 0's keyframe, which is one frame longer than every later block.
        """
        return self.sink_latent_frames + self.context_latent_frames + 1 + self.block_latent_frames

    def cache_latent_frames_for(self, latent_frames: int) -> int:
        """The same capacity, capped by what a clip of ``latent_frames`` can actually reach.

        At a deep ``context_latent_frames`` the policy's headroom stops being the binding
        constraint: a rollout can only ever cache frames that exist, so reserving 16 frames
        of K/V for an 18-frame clip that a 3-block chain touches half of would cost gigabytes
        per rank for tokens that are never written. Capacity is therefore the smaller of the
        policy's steady-state need and the clip's whole length.
        """
        if latent_frames < 1:
            raise ValueError("latent_frames must be >= 1")
        return min(self.cache_latent_frames, latent_frames)

    def as_dict(self) -> dict:
        return {
            "block_latent_frames": self.block_latent_frames,
            "context_latent_frames": self.context_latent_frames,
            "sink_latent_frames": self.sink_latent_frames,
            "cache_latent_frames": self.cache_latent_frames,
            "stride_frames": self.block_latent_frames * self.scale_factors.time,
            "scale_factors": list(self.scale_factors),
        }


class BlockCache:
    """The per-layer K/V caches plus the eviction policy, as one object.

    ``start`` is simply the cache's current length: the retained context is compacted to a
    contiguous prefix after every block, so "where do my tokens sit in the cached stream" and
    "how many tokens are cached" are the same number. Positions travel with the keys (RoPE is
    applied before the write), so compaction never disturbs them.
    """

    def __init__(self, caches: list[LayerKVCache], grid: ClipGrid, geometry: CausalGeometry) -> None:
        self.caches = caches
        self.grid = grid
        self.geometry = geometry

    @classmethod
    def allocate(
        cls,
        grid: ClipGrid,
        geometry: CausalGeometry,
        *,
        num_layers: int,
        inner_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        capacity_latent_frames: int | None = None,
    ) -> BlockCache:
        # Sized against the longest clip this cache will ever hold, not against the policy
        # alone: at a deep cache a clip can be shorter than the policy's headroom and the
        # difference is pure reserved memory.
        #
        # ``capacity_latent_frames`` exists because a caller that allocates ONCE for a whole
        # run (``train.py``) must not size the buffer from whichever clip happened to come
        # first. With the cap taken from an 18-latent-frame clip and a 28-frame clip arriving
        # later, the refresh that follows a full prime overflows and ``LayerKVCache.write``
        # raises mid-forward -- on one rank only, which under FSDP is the collective
        # desynchronisation this module works to make impossible. Pass the corpus's longest
        # clip and the capacity stops depending on draw order.
        capacity = (
            geometry.cache_latent_frames_for(
                grid.latent_frames if capacity_latent_frames is None else capacity_latent_frames
            )
            * grid.tokens_per_latent_frame
        )
        return cls(
            allocate_kv_caches(
                num_layers,
                batch_size=1,
                capacity=capacity,
                inner_dim=inner_dim,
                device=device,
                dtype=dtype,
            ),
            grid,
            geometry,
        )

    @property
    def start(self) -> int:
        return self.caches[0].length

    def fits(self, latent_frames: int) -> bool:
        """Whether this cache can hold a rollout over a clip of ``latent_frames``.

        The question a caller that allocates once and reuses the buffer across clips has to
        ask before the forward that would overflow it -- ``LayerKVCache.write`` raises, and a
        raise inside one rank's forward is a desynchronised collective stream, not a clean
        failure.
        """
        need = self.geometry.cache_latent_frames_for(latent_frames) * self.grid.tokens_per_latent_frame
        return self.caches[0].capacity >= need

    def reset(self) -> None:
        for cache in self.caches:
            cache.reset()

    def evict(self) -> None:
        """Keep the pinned sink and the last ``context_latent_frames`` clean frames.

        At a deep setting this is a no-op for a whole chain: if the rollout has finalized
        fewer frames than the policy retains, there is nothing to drop and the cache simply
        accumulates the chain's own history. Eviction is what makes a LONG rollout bounded,
        not what a short one spends its time doing.
        """
        tokens = self.grid.tokens_per_latent_frame
        sink = self.geometry.sink_latent_frames * tokens
        context = self.geometry.context_latent_frames * tokens
        total = self.start
        keep_start = max(sink, total - context)
        if keep_start == sink and total <= sink + context:
            return  # nothing to drop yet
        spans = [(0, sink), (keep_start, total)] if sink else [(keep_start, total)]
        for cache in self.caches:
            cache.keep([span for span in spans if span[1] > span[0]])


def block_causal_mask(block_ids: torch.Tensor) -> torch.Tensor:
    """``(1, T, T)`` attention mask in [0, 1] from a per-token block index.

    ``1`` where the key's block is at or before the query's. Handed to
    ``Modality.attention_mask``, which ``TransformerArgsPreprocessor`` turns into the additive
    log-space bias the attention backends take -- no new masking machinery.
    """
    allowed = block_ids.unsqueeze(0) <= block_ids.unsqueeze(1)
    return allowed.to(torch.float32).unsqueeze(0)


def block_ids_for(spans: list[tuple[int, int, int]], tokens_per_latent_frame: int) -> torch.Tensor:
    """Per-token block index for a sequence assembled from ``(start, end, block_index)`` spans."""
    ids = [
        torch.full(((end - start) * tokens_per_latent_frame,), block_index, dtype=torch.long)
        for start, end, block_index in spans
    ]
    return torch.cat(ids) if ids else torch.zeros(0, dtype=torch.long)


def retained_prefix_spans(
    plan: list[tuple[int, int]], geometry: CausalGeometry, upto_latent_frame: int
) -> list[tuple[int, int, int]]:
    """The frames a mid-clip block would have behind it, grouped by the block they belong to.

    Returns ``(start, end, block_index)`` runs over latent frames: the pinned sink plus the
    last ``context_latent_frames`` before ``upto_latent_frame``, split wherever the real block
    plan splits them. Grouping by the *real* block index is what makes the priming mask match
    the rollout -- frames that were denoised together attended to each other bidirectionally,
    and a naive "sink is one block, the rest is another" split would forbid exactly that.
    """
    if upto_latent_frame <= 0 or not plan:
        return []
    context_start = max(geometry.sink_latent_frames, upto_latent_frame - geometry.context_latent_frames)
    retained = sorted({*range(geometry.sink_latent_frames), *range(context_start, upto_latent_frame)})
    owner = {}
    for block_index, (start, end) in enumerate(plan):
        for frame in range(start, end):
            owner[frame] = block_index

    spans: list[tuple[int, int, int]] = []
    for frame in retained:
        block_index = owner.get(frame)
        if block_index is None:
            continue  # past the planned blocks; a dropped tail frame is not context
        if spans and spans[-1][1] == frame and spans[-1][2] == block_index:
            spans[-1] = (spans[-1][0], frame + 1, block_index)
        else:
            spans.append((frame, frame + 1, block_index))
    return spans


def prime_cache(
    denoise_fn,  # noqa: ANN001
    grid: ClipGrid,
    cache: BlockCache,
    clean_tokens: torch.Tensor,
    geometry: CausalGeometry,
    context: torch.Tensor,
    *,
    upto_latent_frame: int,
) -> None:
    """Seed an empty cache with the clean frames a mid-clip block would have behind it.

    One forward over the pinned frame 0 plus the last ``context_latent_frames`` before
    ``upto_latent_frame``, block-causal among themselves and written straight into the cache.

    **This is an approximation and it is the AR loop's one remaining teacher-forced seam.** In
    a true rollout those frames' keys were computed when they were generated, against whatever
    history existed then; here they are recomputed from the ground truth against a truncated
    history. It is the same compromise §4.4's old "the chain seed takes the GT carryover" made,
    at the same place, and the escalation is the same: train whole clips (``--chain-length 0``),
    which needs no priming at all.

    **This function performs exactly ONE forward, always** -- including when there is nothing
    to prime. That is not an optimisation left on the table; it is the invariant that keeps
    data-parallel ranks in collective lockstep. See the comment on the empty-spans branch.
    """
    cache.reset()
    spans = retained_prefix_spans(geometry.plan(grid.latent_frames), geometry, upto_latent_frame)
    if not spans:
        # Nothing to write -- a chain starting at the clip start has no history behind it, and
        # its block 0 must see an EMPTY cache. But returning here would make this function's
        # FORWARD COUNT depend on the data, and under FSDP FULL_SHARD every forward is a round
        # of all-gathers. Ranks that took this branch then issue one collective fewer than the
        # ranks that did not; the run deadlocks at the next backward, with the short rank in an
        # ALLREDUCE while the others sit in a _REDUCE_SCATTER_BASE of the SAME sequence number.
        # It presents as a hang, not an error: 100 % GPU utilisation, frozen memory, and a
        # watchdog message 8 minutes later that blames CudaEventDestroy.
        #
        # Measured on t2 (2026-09-16): 11 of 40 train chains start at the clip start, so
        # P(4 ranks agree) = 0.28 and a 4-GPU run had a 72 % chance of hanging on step ONE.
        #
        # So forward anyway, over a single latent frame, with NO cache attached -- `cache=None`
        # leaves `kv_caches` unset and `kv_write` False, so this cannot write what it must not.
        # The result is discarded; only the collectives it issues matter.
        lo, hi = grid.token_span(0, 1)
        with torch.no_grad():
            denoise_fn(block_modality(grid, clean_tokens[:, lo:hi], context, 0.0, token_slices=[(lo, hi)]))
        return
    token_slices = [grid.token_span(start, end) for start, end, _ in spans]
    tokens = torch.cat([clean_tokens[:, lo:hi] for lo, hi in token_slices], dim=1)
    ids = block_ids_for(spans, grid.tokens_per_latent_frame).to(tokens.device)
    modality = block_modality(
        grid,
        tokens,
        context,
        0.0,
        token_slices=token_slices,
        cache=cache,
        kv_write=True,
        attention_mask=block_causal_mask(ids),
    )
    with torch.no_grad():
        denoise_fn(modality)


def denoise_block(
    denoise_fn,  # noqa: ANN001
    grid: ClipGrid,
    cache: BlockCache,
    noisy_tokens: torch.Tensor,
    context: torch.Tensor,
    sigma: float,
    span: tuple[int, int],
    *,
    clean_prefix_tokens: int = 0,
    kv_write: bool = False,
) -> torch.Tensor:
    """One block's denoised tokens. Reads the cache; writes only when asked.

    Read-only by default, and that default is what keeps gradient checkpointing usable on this
    pass: a cache write would be replayed by recomputation, and this is the only pass that
    stores activations at all. ``kv_write=True`` exists for ``rollout(kv_source="denoise")``,
    which trades the refresh forward away by caching what this pass already computed -- safe
    under ``no_grad`` at inference, and requiring the write to be hoisted out of the
    checkpointed region before it could be used in training.

    The read is unaffected either way: ``block_modality`` captures ``kv_start = cache.start``
    before the forward, so the block attends to the history prefix and its own write lands at
    the end of it.
    """
    return denoise_fn(
        block_modality(
            grid,
            noisy_tokens,
            context,
            sigma,
            token_slices=[grid.token_span(*span)],
            cache=cache,
            kv_write=kv_write,
            clean_prefix_tokens=clean_prefix_tokens,
        )
    )


def fusion_parity_block(
    transformer,  # noqa: ANN001
    kind: str,
    grid: ClipGrid,
    cache: BlockCache,
    noisy_tokens: torch.Tensor,
    context: torch.Tensor,
    sigma: float,
    span: tuple[int, int],
    *,
    clean_prefix_tokens: int,
) -> torch.Tensor:
    """Run the one-block x0/velocity path used by fused-versus-PEFT parity checks.

    Keeping the denoiser selection and clean-prefix restoration beside ``denoise_block``
    prevents diagnostic code from becoming a second model-execution owner. ``kind`` is the
    training representation (``velocity``) or deployment representation (``x0``); both use
    the same block geometry and cache inputs.
    """
    if kind == "velocity":
        denoise_fn = common.denoised_from_velocity_model(transformer)
    elif kind == "x0":
        denoise_fn = common.denoised_from_x0_model(transformer)
    else:
        raise ValueError(f"unknown fusion-parity denoiser kind: {kind!r}")
    with torch.no_grad():
        output = denoise_block(
            denoise_fn,
            grid,
            cache,
            noisy_tokens,
            context,
            sigma,
            span,
            clean_prefix_tokens=clean_prefix_tokens,
        )
    return common.with_clean_prefix(output, noisy_tokens[:, :clean_prefix_tokens])


def denoise_with_clean_history(
    denoise_fn,  # noqa: ANN001
    grid: ClipGrid,
    geometry: CausalGeometry,
    clean_history: torch.Tensor,
    noisy_tokens: torch.Tensor,
    context: torch.Tensor,
    sigma: float,
    span: tuple[int, int],
    *,
    clean_prefix_tokens: int = 0,
    joint_window: bool = False,
) -> torch.Tensor:
    """Explicit causal reference with retained clean history recomputed at this sigma.

    History token timesteps remain zero while the global sigma (including prompt AdaLN)
    matches the current denoise call. ``joint_window`` lets clean history attend to the
    current noisy block as a separate information-access diagnostic. No K/V cache is used.
    """
    history_spans = retained_prefix_spans(geometry.plan(grid.latent_frames), geometry, span[0])
    slices = [grid.token_span(start, end) for start, end, _ in history_spans]
    slices.append(grid.token_span(*span))
    history = [clean_history[:, lo:hi] for lo, hi in slices[:-1]]
    prefix_length = sum(item.shape[1] for item in history)
    sequence = torch.cat([*history, noisy_tokens], dim=1)
    ids = block_ids_for(
        [*history_spans, (*span, geometry.plan(grid.latent_frames).index(span))], grid.tokens_per_latent_frame
    ).to(sequence.device)
    modality = block_modality(
        grid,
        sequence,
        context,
        sigma,
        token_slices=slices,
        # With no history, this is a single bidirectional block. An all-visible
        # dense mask wastes quadratic memory and can OOM a whole-clip forward.
        attention_mask=None if joint_window or not history_spans else block_causal_mask(ids),
        clean_prefix_tokens=prefix_length + clean_prefix_tokens,
        sigma_dtype=torch.float32 if span == (0, grid.latent_frames) else None,
    )
    return denoise_fn(modality)[:, prefix_length:]


def refresh_block(
    denoise_fn,  # noqa: ANN001
    grid: ClipGrid,
    cache: BlockCache,
    clean_tokens: torch.Tensor,
    context: torch.Tensor,
    span: tuple[int, int],
) -> None:
    """Write one block's CLEAN K/V into the cache, then evict down to the retained context.

    The single point where teacher forcing and self-forcing differ: the caller passes the
    ground-truth capture tokens or the model's own ``ẑ₀``. Nothing else in the loop knows
    which regime it is in.
    """
    modality = block_modality(
        grid,
        clean_tokens,
        context,
        0.0,
        token_slices=[grid.token_span(*span)],
        cache=cache,
        kv_write=True,
    )
    with torch.no_grad():
        denoise_fn(modality)
    cache.evict()


def rollout(  # noqa: PLR0912, PLR0913, PLR0915 -- one AR block loop; every branch is a documented arm
    denoise_fn,  # noqa: ANN001
    grid: ClipGrid,
    geometry: CausalGeometry,
    cache: BlockCache | None,
    guide_tokens: torch.Tensor,
    context: torch.Tensor,
    sigma: float,
    *,
    seed: int = 42,
    blocks: list[tuple[int, int]] | None = None,
    teacher_forcing: bool = False,
    teacher_tokens: torch.Tensor | None = None,
    first_frame_condition: torch.Tensor | None = None,
    block_epsilons: list[torch.Tensor] | None = None,
    schedule: list[float] | tuple[float, ...] | None = None,
    kv_source: str = "refresh",
    history_mode: str = "cache",
) -> tuple[torch.Tensor, int]:
    """Roll the whole clip forward one block at a time; return ``(tokens, forwards)``.

    ``guide_tokens`` is the patchified master guide latent -- the one continuous encode, never
    a per-block re-encode. The deployment counterpart of the training loop, and deliberately
    the same three calls in the same order, so a train/deploy mismatch would have to be a
    change to this function rather than a divergence between two of them.

    ``teacher_forcing`` mirrors ``train.py``'s ``train_chain`` ablation of the same name:
    ``refresh`` is fed the **ground-truth target** for the completed block instead of the
    model's own denoised output. Off by default, which is the self-forced regime a real
    deployment has to use (there is no ground truth at inference). A checkpoint trained with
    ``--teacher-forcing`` never saw its own denoising errors accumulate in the cache, so probing
    it self-forced evaluates an input distribution training never produced; pass
    ``teacher_forcing=True`` to roll it out the way it was trained.

    ``teacher_tokens`` is that target, and it is **required** whenever ``teacher_forcing`` is
    set. It used to be implicit: the refresh read ``guide_tokens``, the tensor the block was
    noised from. Those are the same tensor for D0 only, where the noising source *is* ``z_y``;
    for D1 that silently teacher-forced on the render guide, which is not the target, and
    evaluated a regime training never ran without raising ([G2]). D0 callers pass their
    ``z_y`` for both arguments and say so; D1 callers pass ``z_g`` as the guide and ``z_y``
    here. Making it explicit is the whole fix -- an implicit default would restore the bug
    for the next caller.
    """
    if first_frame_condition is None:
        raise ValueError("first_frame_condition is required: supply the clean latent-frame-0 tokens as c0")
    if first_frame_condition.shape[1] != grid.tokens_per_latent_frame:
        raise ValueError("first_frame_condition must contain exactly one latent frame of tokens")
    if teacher_forcing and teacher_tokens is None:
        raise ValueError(
            "teacher_forcing=True requires an explicit teacher_tokens target (z_y). Refreshing "
            "from guide_tokens is correct for D0 only, where the noising source is already the "
            "target; for D1 it teacher-forces on the render. Pass z_y explicitly."
        )
    if teacher_tokens is not None and teacher_tokens.shape != guide_tokens.shape:
        raise ValueError(
            f"teacher_tokens {tuple(teacher_tokens.shape)} must match guide_tokens {tuple(guide_tokens.shape)}"
        )
    if kv_source not in ("refresh", "denoise"):
        raise ValueError(f"kv_source must be 'refresh' or 'denoise', got {kv_source!r}")
    if history_mode not in ("cache", "recompute", "joint"):
        raise ValueError(f"history_mode must be 'cache', 'recompute' or 'joint', got {history_mode!r}")
    if history_mode != "cache" and kv_source != "refresh":
        raise ValueError("explicit history modes require kv_source='refresh'")
    if history_mode == "cache" and cache is None:
        raise ValueError("history_mode='cache' requires a BlockCache")
    if teacher_forcing and kv_source == "denoise":
        raise ValueError(
            "teacher_forcing=True is incompatible with kv_source='denoise': teacher forcing "
            "means caching the ground-truth target, which the denoising pass never saw"
        )
    levels = (float(sigma), 0.0) if schedule is None else validate_schedule(schedule)
    if levels[0] != float(sigma):
        raise ValueError(f"schedule must start at sigma={sigma}, got {levels[0]}")
    plan = geometry.plan(grid.latent_frames) if blocks is None else blocks
    if history_mode != "cache" and plan != geometry.plan(grid.latent_frames)[: len(plan)]:
        raise ValueError("explicit history requires contiguous blocks starting at latent frame 0")
    if cache is not None:
        cache.reset()
    if block_epsilons is not None and len(block_epsilons) != len(plan):
        raise ValueError(f"received {len(block_epsilons)} block epsilons for a {len(plan)}-block rollout")
    out = torch.zeros_like(guide_tokens)
    clean_history = torch.zeros_like(guide_tokens) if history_mode != "cache" else None
    forwards = 0
    for index, span in enumerate(plan):
        lo, hi = grid.token_span(*span)
        c0 = first_frame_condition if span[0] == 0 else None
        clean_source = guide_tokens[:, lo:hi]
        noisy_source = (
            noise_block(clean_source, sigma, seed + index)
            if block_epsilons is None
            else mix_block_noise(clean_source, block_epsilons[index], sigma)
        )
        state = with_clean_prefix(noisy_source, c0)
        prefix = 0 if c0 is None else c0.shape[1]
        denoised = state
        last = len(levels) - 2  # index of the final denoising call
        for step, (level, next_level) in enumerate(pairwise(levels)):
            if clean_history is None:
                assert cache is not None
                prediction = denoise_block(
                    denoise_fn,
                    grid,
                    cache,
                    state,
                    context,
                    level,
                    span,
                    clean_prefix_tokens=prefix,
                    kv_write=(kv_source == "denoise" and step == last),
                )
            else:
                prediction = denoise_with_clean_history(
                    denoise_fn,
                    grid,
                    geometry,
                    clean_history,
                    state,
                    context,
                    level,
                    span,
                    clean_prefix_tokens=prefix,
                    joint_window=history_mode == "joint",
                )
            denoised = with_clean_prefix(
                prediction,
                c0,
            )
            forwards += 1
            if next_level > 0.0:
                # Advance only the generated tokens; c0 is clean at every level by contract and
                # stepping it would re-noise the one input the product guarantees.
                state = with_clean_prefix(euler_to(state, denoised, level, next_level), c0)
        out[:, lo:hi] = denoised
        clean = teacher_tokens[:, lo:hi] if teacher_forcing else denoised
        if clean_history is not None:
            clean_history[:, lo:hi] = clean.detach()
        elif kv_source == "refresh":
            assert cache is not None
            refresh_block(denoise_fn, grid, cache, clean, context, span)
            forwards += 1
        else:
            assert cache is not None
            # The write happened inside the last denoise call; only eviction is still owed,
            # and refresh_block is the only other place that calls it.
            cache.evict()
    return out, forwards


@torch.no_grad()
def sample(  # noqa: PLR0913 -- explicit mode inputs and history diagnostics
    predict_x0,  # noqa: ANN001
    context: torch.Tensor,
    grid: ClipGrid,
    source: torch.Tensor,
    c0: torch.Tensor,
    *,
    transformer: torch.nn.Module,
    geometry: CausalGeometry,
    schedule: list[float],
    seed: int,
    epsilon: torch.Tensor,
    blocks: list[tuple[int, int]] | None = None,
    teacher_tokens: torch.Tensor | None = None,
    teacher_forcing: bool = False,
    history_mode: str = "cache",
    kv_source: str = "refresh",
) -> tuple[torch.Tensor, dict[str, int]]:
    """Slice one saved comparison-noise array and use the existing causal rollout."""
    levels = validate_schedule(schedule)
    if epsilon.shape != source.shape or not torch.isfinite(epsilon).all():
        raise ValueError("comparison noise must be finite and match the source tokens")
    plan = geometry.plan(grid.latent_frames) if blocks is None else blocks
    if not plan or plan != geometry.plan(grid.latent_frames)[: len(plan)]:
        raise ValueError("generation requires consecutive complete blocks starting at zero")
    base = common.base_model(transformer)
    cache = None
    if history_mode == "cache":
        cache = BlockCache.allocate(
            grid,
            geometry,
            num_layers=len(base.transformer_blocks),
            inner_dim=base.inner_dim,
            device=source.device,
            dtype=source.dtype,
        )
    slices = [epsilon[:, slice(*grid.token_span(*span))] for span in plan]
    tokens, forwards = rollout(
        predict_x0,
        grid,
        geometry,
        cache,
        source,
        context,
        levels[0],
        seed=seed,
        blocks=plan,
        teacher_forcing=teacher_forcing,
        teacher_tokens=teacher_tokens,
        first_frame_condition=c0,
        block_epsilons=slices,
        schedule=levels,
        history_mode=history_mode,
        kv_source=kv_source,
    )
    refresh = len(plan) if history_mode == "cache" and kv_source == "refresh" else 0
    return tokens, {
        "denoise_calls": len(plan) * (len(levels) - 1),
        "prime_calls": 0,
        "refresh_calls": refresh,
        "model_calls": forwards,
    }


def deployed_geometry(scale_factors: SpatioTemporalScaleFactors, **overrides: int) -> CausalGeometry:
    """The geometry every caller should use unless it is explicitly sweeping one.

    Shared by training, deployment, probes and benchmarks. Block size and cache
    depth are explicit avatar settings, independent of pruning.
    """
    return CausalGeometry(scale_factors=scale_factors, **overrides)


def plan_samples(
    latent_frames: int,
    geometry: CausalGeometry,
    *,
    blocks_per_sample: int,
    stride: int | None = None,
    clip_start_only: bool = False,
) -> list[list[int]]:
    """Group complete consecutive blocks; preserve the original sample-stride rule."""
    if blocks_per_sample < 1 or (stride is not None and stride < 1):
        raise ValueError("blocks_per_sample and stride must be positive")
    count = len(geometry.plan(latent_frames))
    starts = range(0, count - blocks_per_sample + 1, blocks_per_sample if stride is None else stride)
    samples = [list(range(start, start + blocks_per_sample)) for start in starts]
    return samples[:1] if clip_start_only else samples


def train_sample(  # noqa: PLR0913 -- explicit tensors, cache policy and gradient scaling define this mode call
    transformer: torch.nn.Module,
    context: torch.Tensor,
    grid: ClipGrid,
    capture: torch.Tensor,
    guide: torch.Tensor | None,
    geometry: CausalGeometry,
    blocks: list[int],
    backward: Callable[[torch.Tensor], None],
    *,
    sigma: float,
    seed: int,
    cache: BlockCache | None = None,
    guide_mode: str = "d1",
    teacher_forcing: bool = False,
    accumulation: int = 1,
    timing: bool = False,
    capacity_latent_frames: int | None = None,
) -> dict:
    """Prime once, then denoise/backward immediately and refresh between blocks."""
    if accumulation < 1 or not 0 < sigma <= 1:
        raise ValueError("accumulation must be positive and sigma must be in (0, 1]")
    plan = geometry.plan(grid.latent_frames)
    if not blocks or blocks != list(range(blocks[0], blocks[0] + len(blocks))):
        raise ValueError("training requires a nonempty consecutive block sequence")
    if blocks[0] < 0 or blocks[-1] >= len(plan):
        raise ValueError("selected blocks are outside the stored master")
    expected = grid.latent_frames * grid.tokens_per_latent_frame
    if capture.ndim != 3 or capture.shape[0] != 1 or capture.shape[1] != expected:
        raise ValueError("capture must be [1, F*H*W, C] on the video grid")
    source = common.source_for(capture, guide, guide_mode)
    c0 = capture[:, : grid.tokens_per_latent_frame]
    denoise_fn = common.denoised_from_velocity_model(transformer)
    if cache is None:
        model = common.base_model(transformer)
        cache = BlockCache.allocate(
            grid,
            geometry,
            num_layers=len(model.transformer_blocks),
            inner_dim=model.inner_dim,
            device=capture.device,
            dtype=capture.dtype,
            capacity_latent_frames=capacity_latent_frames,
        )
    elif cache.grid.tokens_per_latent_frame != grid.tokens_per_latent_frame:
        raise ValueError("tokens per latent frame differ from the allocated cache geometry")
    elif not (len(blocks) == 1 and blocks[0] == 0) and not cache.fits(grid.latent_frames):
        raise ValueError("cache must be sized from the subset's LONGEST clip")
    logger = logging.getLogger("onestep_avatar.causal")
    prime_started = time.time()
    prime_cache(denoise_fn, grid, cache, capture, geometry, context, upto_latent_frame=plan[blocks[0]][0])
    if timing:
        logger.info("timing | prime_cache(upto=%d): %.2fs", plan[blocks[0]][0], time.time() - prime_started)
    totals = {
        "loss": 0.0,
        "mse": 0.0,
        "per_block": [],
        "prime_calls": 1,
        "denoise_calls": len(blocks),
        "backward_calls": len(blocks),
        "refresh_calls": len(blocks) - 1,
    }
    k = len(blocks)
    for block_index in blocks:
        span = plan[block_index]
        lo, hi = grid.token_span(*span)
        started = time.time()
        first_frame = c0 if span[0] == 0 else None
        noisy = common.with_clean_prefix(common.noise_block(source[:, lo:hi], sigma, seed + block_index), first_frame)
        prediction = denoise_block(
            denoise_fn,
            grid,
            cache,
            noisy,
            context,
            sigma,
            span,
            clean_prefix_tokens=0 if first_frame is None else first_frame.shape[1],
        )
        prediction = common.with_clean_prefix(prediction, first_frame)
        denoised_at = time.time()
        mse = common.full_frame_mse(prediction, capture[:, lo:hi])
        backward(mse / (k * accumulation))
        backward_at = time.time()
        value = float(mse.detach())
        totals["loss"] += value / k
        totals["mse"] += value / k
        totals["per_block"].append({"block_index": block_index, "mse": value})
        clean = capture[:, lo:hi] if teacher_forcing else prediction.detach()
        if block_index != blocks[-1]:
            refresh_block(denoise_fn, grid, cache, clean, context, span)
        if timing:
            logger.info(
                "timing | block %d (span %d:%d): denoise %.2fs backward %.2fs refresh %.2fs",
                block_index,
                span[0],
                span[1],
                denoised_at - started,
                backward_at - denoised_at,
                time.time() - backward_at,
            )
        del prediction, mse
    totals["cache"] = cache
    return totals
