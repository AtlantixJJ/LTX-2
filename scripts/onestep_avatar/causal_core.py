"""The ONE implementation of "roll a causal block forward", training and deployment alike.

``plans/2026-09-10-ltx25-one-step-argavatar-lora.md`` §4.4 (revised 2026-09-14). This module
replaces the sliding-window-with-a-frozen-carryover construction that ``refine_core`` owns for
the `k2` refiner. Three things change together, and they are one change, not three:

1. **Attention is block-causal.** A token attends to its own block and every earlier one,
   never to a later one. Within a block it stays bidirectional -- the block is denoised in
   one shot, so there is nothing to order inside it.
2. **Clean blocks live in a K/V cache.** Under (1) a finished block's keys and values no
   longer depend on anything that comes after it, so they are computed once and reused for
   the rest of the rollout instead of being re-forwarded inside every later window. That
   equivalence is exactly why (1) is a precondition for (2): with bidirectional attention a
   context token's K/V differ in every window and nothing is cacheable.
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

from dataclasses import dataclass

import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.model.transformer.kv_cache import LayerKVCache, allocate_kv_caches
from ltx_core.model.transformer.modality import Modality
from ltx_core.tools import VideoLatentTools
from ltx_core.types import (
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    VideoPixelShape,
)
from ltx_core.utils import to_denoised
from scripts.prune.core import refine_core

# §4.4: the deployed stride is 16 pixel frames, which is 2 latent frames at the VAE's
# temporal scale of 8. Keeping the block at the stride means a trained adapter denoises
# exactly the span the old rollout finalized per window.
BLOCK_LATENT_FRAMES = 2

# Clean latent frames kept in the cache besides the pinned frame-0 sink. Every extra frame
# costs ~0.8 GB at the 22B geometry (48 layers x 1024 tokens x 4096 dims x 2 tensors x 2
# bytes), which is why this is a flag rather than "the whole history".
CONTEXT_LATENT_FRAMES = 2

# The deepest cache this scheme supports. 16 clean latent frames is 128 pixel frames -- most
# of a corpus clip (the 137-frame tier is 18 latent frames), so at this depth a K-block
# rollout evicts nothing and the cache simply ACCUMULATES every frame the chain has
# finalized, plus the primed prefix. That is the regime the deeper settings exist for.
#
# Two things bound it, and neither is arbitrary:
#
# * Memory. ~0.8 GB per retained latent frame per rank at the 22B geometry, so 16 frames is
#   ~13 GB of K/V on top of the model -- real on a 49 GB card, and the reason this is a
#   ceiling rather than a default.
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
MAX_ROPE_SECONDS = 20.0


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


def pixel_frames_for(latent_frames: int, time_scale: int) -> int:
    """Pixel frames a continuous encode of ``latent_frames`` latent frames covers."""
    return (latent_frames - 1) * time_scale + 1


@dataclass(frozen=True)
class ClipGrid:
    """A clip's token grid: the tools, RoPE positions and keyframe marks, built ONCE.

    Everything a block forward needs is a slice of this, which is what makes the positions
    global and the keyframe mark truthful. The old per-window ``tools_for_window`` restarted
    the time axis at 0 for every window *and* had ``VideoLatentTools._first_frame_keyframes_mask``
    mark every window's own first latent frame as a single-pixel keyframe -- which is false
    for every window past a clip's first once windows are sliced out of one continuous encode.
    Building the grid over the master fixes both at once.
    """

    tools: VideoLatentTools
    positions: torch.Tensor  # (1, 3, T, 2) global patch bounds
    keyframes_mask: torch.Tensor  # (1, T, 1), non-zero only on latent frame 0
    denoise_mask: torch.Tensor  # (1, T, 1) all ones; sliced and scaled into per-token timesteps
    latent_frames: int
    tokens_per_latent_frame: int
    fps: float

    @classmethod
    def build(
        cls,
        latent_frames: int,
        height: int,
        width: int,
        fps: float,
        geometry: CausalGeometry,
        *,
        device: torch.device,
        dtype: torch.dtype,
        latent_channels: int = 128,
    ) -> ClipGrid:
        """``height``/``width`` are PIXEL dimensions of the crop, as ``tools_for_window`` takes them."""
        pixel_frames = pixel_frames_for(latent_frames, geometry.scale_factors.time)
        duration = pixel_frames / float(fps)
        if duration > MAX_ROPE_SECONDS:
            raise ValueError(
                f"a {duration:.1f}s clip exceeds the model's temporal RoPE range "
                f"({MAX_ROPE_SECONDS}s at positional_embedding_max_pos[0]); a causal rollout uses "
                f"GLOBAL positions, so this would extrapolate rather than wrap. Split the clip or "
                f"implement position re-basing before going past it"
            )
        shape = VideoLatentShape.from_pixel_shape(
            VideoPixelShape(batch=1, frames=pixel_frames, height=height, width=width, fps=float(fps)),
            latent_channels=latent_channels,
            scale_factors=geometry.scale_factors,
        )
        if shape.frames != latent_frames:
            raise ValueError(f"pixel shape implies {shape.frames} latent frames, expected {latent_frames}")
        tools = VideoLatentTools(
            VideoLatentPatchifier(patch_size=1), shape, float(fps), scale_factors=geometry.scale_factors
        )
        state = tools.create_initial_state(device=device, dtype=dtype)
        tokens = state.latent.shape[1]
        return cls(
            tools=tools,
            positions=state.positions,
            keyframes_mask=state.keyframes_mask,
            denoise_mask=state.denoise_mask,
            latent_frames=latent_frames,
            tokens_per_latent_frame=tokens // latent_frames,
            fps=float(fps),
        )

    def token_span(self, latent_start: int, latent_end: int) -> tuple[int, int]:
        return latent_start * self.tokens_per_latent_frame, latent_end * self.tokens_per_latent_frame

    def patchify(self, latent: torch.Tensor) -> torch.Tensor:
        """``(1, C, F, H, W)`` master latent -> ``(1, T, C)`` tokens on the grid's own layout."""
        return self.tools.patchifier.patchify(latent)

    def unpatchify_block(self, tokens: torch.Tensor, latent_frames: int) -> torch.Tensor:
        """``(1, L, C)`` block tokens -> ``(1, C, F, H, W)``, using the grid's spatial shape."""
        shape = self.tools.target_shape
        channels = tokens.shape[-1]
        return (
            tokens.transpose(1, 2)
            .reshape(1, channels, latent_frames, shape.height, shape.width)
            .contiguous()
        )


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


def noise_block(clean_tokens: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """``lerp(clean, eps, sigma)`` -- ``GaussianNoiser``'s own formula, on one block's tokens.

    Reimplemented here rather than routed through ``create_noised_state`` because there is no
    conditioning item left to apply: the carryover that needed one is now the cache. The
    arithmetic is pinned against ``GaussianNoiser`` by ``tests/test_causal_core.py`` so the
    two cannot drift.
    """
    generator = torch.Generator(device=clean_tokens.device).manual_seed(seed)
    eps = torch.randn(
        *clean_tokens.shape, device=clean_tokens.device, dtype=clean_tokens.dtype, generator=generator
    )
    return torch.lerp(clean_tokens.float(), eps.float(), sigma).to(clean_tokens.dtype)


def block_modality(
    grid: ClipGrid,
    tokens: torch.Tensor,
    context: torch.Tensor,
    sigma: float,
    *,
    token_slices: list[tuple[int, int]],
    cache: BlockCache | None = None,
    kv_write: bool = False,
    attention_mask: torch.Tensor | None = None,
) -> Modality:
    """One forward's ``Modality``, assembled from slices of the clip grid.

    ``token_slices`` are half-open token ranges of the clip, concatenated in order -- one
    range for an ordinary block, several for the priming call that seeds a cache from the
    pinned frame 0 plus a non-adjacent stretch of context.
    """
    device = tokens.device
    positions = torch.cat([grid.positions[:, :, lo:hi] for lo, hi in token_slices], dim=2)
    keyframes = torch.cat([grid.keyframes_mask[:, lo:hi] for lo, hi in token_slices], dim=1)
    denoise = torch.cat([grid.denoise_mask[:, lo:hi] for lo, hi in token_slices], dim=1)
    sigma_tensor = torch.tensor([sigma], device=device, dtype=tokens.dtype)
    return Modality(
        latent=tokens,
        sigma=sigma_tensor,
        timesteps=denoise * sigma,
        positions=positions,
        context=context,
        context_mask=None,
        attention_mask=attention_mask,
        keyframes_mask=keyframes,
        kv_caches=None if cache is None else cache.caches,
        kv_start=0 if cache is None else cache.start,
        kv_write=kv_write,
    )


def base_model(module: torch.nn.Module) -> torch.nn.Module:
    """Peel a wrapper off to reach the ``LTXModel`` underneath -- whichever of the two wrapper
    families this package actually produces.

    * **Training-time**: FSDP/DDP and PEFT wrap the velocity model itself, reachable through
      ``.module`` / ``.base_model`` / ``.model``.
    * **Deploy-time**: the inference session hands back a bare
      ``X0Model(velocity_model=<LTXModel>)`` -- no FSDP, no PEFT (the LoRA is fused at load,
      not applied as an adapter) -- reachable through ``.velocity_model`` alone.

    One implementation since S1(7) of the 2026-09-17 cleanup plan: ``train.py`` had this exact
    depth-capped walk as ``_base_model``, and ``onestep_core.rollout``, ``visualize_d0`` and
    ``bench_forward`` each carried their own copy of an unbounded
    ``while not hasattr(base, "transformer_blocks") and hasattr(base, "velocity_model")`` loop
    that only ever handled the second shape. The two algorithms agreed at every existing call
    site only because each site's wrapper matches exactly one of the two families -- checked
    directly (``tests/test_causal_core.py::test_base_model_agrees_with_both_retired_walks``)
    rather than assumed, since a wrapper matching a *different* attribute here would silently
    size the K/V cache from the wrong module.
    """
    model = module
    for _ in range(8):
        if hasattr(model, "transformer_blocks"):
            return model
        for attribute in ("module", "base_model", "model", "velocity_model"):
            inner = getattr(model, attribute, None)
            if isinstance(inner, torch.nn.Module):
                model = inner
                break
        else:
            break
    raise TypeError(f"cannot find the LTXModel inside {type(module).__name__}")


def denoised_from_velocity_model(model):  # noqa: ANN001, ANN201 -- a PEFT-wrapped LTXModel
    """Adapter for the training side, where the transformer emits velocity."""

    def call(modality: Modality) -> torch.Tensor:
        velocity, _ = model(video=modality, audio=None, perturbations=None)
        return to_denoised(modality.latent, velocity, modality.timesteps)

    return call


def denoised_from_x0_model(model):  # noqa: ANN001, ANN201 -- the X0Model the session yields
    """Adapter for the deployment side, where the session hands back an ``X0Model``."""

    def call(modality: Modality) -> torch.Tensor:
        denoised, _ = model(video=modality, audio=None, perturbations=None)
        return denoised

    return call


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
) -> torch.Tensor:
    """One block's denoised tokens. Reads the cache, writes nothing.

    Read-only is what keeps gradient checkpointing usable on this pass: a cache write would
    be replayed by recomputation, and this is the only pass that stores activations at all.
    """
    return denoise_fn(
        block_modality(
            grid,
            noisy_tokens,
            context,
            sigma,
            token_slices=[grid.token_span(*span)],
            cache=cache,
            kv_write=False,
        )
    )


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


def rollout(
    denoise_fn,  # noqa: ANN001
    grid: ClipGrid,
    geometry: CausalGeometry,
    cache: BlockCache,
    guide_tokens: torch.Tensor,
    context: torch.Tensor,
    sigma: float,
    *,
    seed: int = 42,
    blocks: list[tuple[int, int]] | None = None,
    teacher_forcing: bool = False,
) -> tuple[torch.Tensor, int]:
    """Roll the whole clip forward one block at a time; return ``(tokens, forwards)``.

    ``guide_tokens`` is the patchified master guide latent -- the one continuous encode, never
    a per-block re-encode. The deployment counterpart of the training loop, and deliberately
    the same three calls in the same order, so a train/deploy mismatch would have to be a
    change to this function rather than a divergence between two of them.

    ``teacher_forcing`` mirrors ``train.py``'s ``train_chain`` ablation of the same name:
    ``refresh`` is fed ``guide_tokens`` -- the clean source the block was noised from -- instead
    of the model's own denoised output. Off by default, which is the self-forced regime a real
    deployment has to use (there is no ground truth at inference). A checkpoint trained with
    ``--teacher-forcing`` never saw its own denoising errors accumulate in the cache, so probing
    it self-forced evaluates an input distribution training never produced; pass
    ``teacher_forcing=True`` to roll it out the way it was trained.
    """
    cache.reset()
    plan = geometry.plan(grid.latent_frames) if blocks is None else blocks
    out = torch.zeros_like(guide_tokens)
    forwards = 0
    for index, span in enumerate(plan):
        lo, hi = grid.token_span(*span)
        noisy = noise_block(guide_tokens[:, lo:hi], sigma, seed + index)
        denoised = denoise_block(denoise_fn, grid, cache, noisy, context, sigma, span)
        out[:, lo:hi] = denoised
        clean = guide_tokens[:, lo:hi] if teacher_forcing else denoised
        refresh_block(denoise_fn, grid, cache, clean, context, span)
        forwards += 2
    return out, forwards


def deployed_geometry(scale_factors: SpatioTemporalScaleFactors, **overrides: int) -> CausalGeometry:
    """The geometry every caller should use unless it is explicitly sweeping one.

    Cross-checked against ``refine_task.deployed_geometry`` by ``tests/test_causal_core.py``:
    the block is the `k2` rollout's own stride, so an adapter trained here finalizes the same
    span per step that the baseline it is measured against does.
    """
    return CausalGeometry(scale_factors=scale_factors, **overrides)


def matches_deployed_stride(geometry: CausalGeometry, window: refine_core.WindowGeometry) -> bool:
    """Whether a causal block finalizes the same pixel span the `k2` window did."""
    return geometry.block_latent_frames * geometry.scale_factors.time == window.stride_frames
