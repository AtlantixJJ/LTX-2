"""SS7.1's bespoke autoregressive LoRA loop for the one-step avatar renderer.

``plans/2026-09-10-ltx25-one-step-argavatar-lora.md`` SS7.1 explains why this is not a
``ltx-trainer`` strategy: ``Trainer._training_step`` runs exactly one transformer forward
per step and the strategy interface does not own the forward, so a ``K``-block AR chain
cannot be expressed as one. Only the *step* is ours -- model loading, LoRA injection,
FSDP preparation and checkpoint plumbing are all reused from ``ltx_trainer``.

**Revised 2026-09-14 (SS4.4).** The sliding window with a frozen carryover latent frame is
gone. The scheme is now block-causal attention plus a clean-latent K/V cache, over latents
sliced from the clip's ONE continuous VAE encode:

1. **Causal.** A token attends to its own block and every earlier block, never a later one.
2. **Cached.** Because of (1) a finished block's keys and values are final, so they are
   computed once -- by a clean, no-grad ``refresh`` forward -- and every later block reads
   them out of the cache instead of re-forwarding that content inside its own window.
3. **Master latents.** ``z_g``/``z_y`` are the whole clip's continuous encodes, read straight
   from the corpus; blocks are token slices of them. There is no per-window precompute tree
   and no per-window re-keyed frame 0 to manufacture.

``causal_core.py`` owns all three -- it is the single implementation the deployment rollout
(``onestep_core.rollout``) uses too, so training and deployment cannot drift.

Three things this loop does that the shared trainer cannot:

1. **The noisy state is built from a different latent than the loss target.** The init is
   the ARGAvatar guide ``z_g`` and the target is the capture ``z_y`` (SS3). When
   ``z_g == z_y`` the target reduces exactly to ``eps - z_y``, the ordinary flow-matching
   target -- ``tests/test_train.py`` pins that, so this is a strict generalisation of what
   the trainer already does rather than a parallel objective.
2. **The context a block is conditioned on is the model's own previous output**, not the
   ground truth, by default. ``--teacher-forcing`` swaps exactly one tensor -- what is handed
   to the ``refresh`` forward -- for the GT capture, as an ablation arm to isolate how much of
   the measured drift (-48.98 dB / 100 chunks on `k2`) is exposure bias versus everything else
   the AR loop changes. It is not expected to be the production setting.
3. **The rollout is the deployed one.** Every forward goes through ``causal_core``, the same
   calls ``onestep_core.rollout`` makes, so RoPE positions, the frame-0 keyframe, the cached
   context and the eviction policy are identical to inference by construction.

sigma_0 is fixed (SS4.2, default 0.725): the distilled checkpoint is a deterministic map on a
9-point grid, not a continuum, so there is no sampler in this loop at all. sigma_0, ``K``, the
causal geometry and the subset hash go into the checkpoint metadata, because a fixed-sigma
adapter loaded at another sigma, run multi-step, or deployed at a different cache depth fails
silently (SS9 risk 13).

Run from ``LTX-2`` in the ``ltx`` env. Two or three GPUs is a preliminary-scale run -- drop
the rank rather than the chain length, since ``K`` is what the loop exists to exercise::

    accelerate launch --config_file scripts/onestep_avatar/configs/fsdp_2gpu.yaml \\
      -m scripts.onestep_avatar.train \\
      --subset ../expr/onestep_avatar/windows/t2.json \\
      --output ../expr/onestep_avatar/runs/prelim --lora-rank 8 --steps 200
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from peft.utils.other import fsdp_auto_wrap_policy
from safetensors.torch import save_file

from ltx_core.tools import VideoLatentTools
from ltx_trainer.model_loader import load_transformer
from scripts.onestep_avatar import causal_core, dataset
from scripts.onestep_avatar.causal_core import BlockCache, CausalGeometry, ClipGrid
from scripts.prune.core import model_registry, refine_task
from scripts.prune.data import prompt_cache

LOGGER = logging.getLogger("onestep_avatar.train")


@contextlib.contextmanager
def timed(label: str) -> Iterator[None]:
    """Bracket a startup phase with a begin/end line carrying its wall duration.

    Startup here is minutes of silent 42 GB checkpoint I/O followed by a collective
    (``prepare``), and both look identical to a hang from outside. Worse, the failure mode
    that actually happens is ONE straggling or dead rank, not a uniformly slow run -- so
    this logs on every rank rather than the main process, and ``ltx_trainer``'s ``[rank N]``
    prefix is the whole point of it. The lines are unconditional because they are a few per
    run; the per-step breakdown, which is per-block, is behind ``--timing``.

    **The prefix is ``timing |``, not ``[timing]``, on purpose.** ``ltx_trainer`` installs a
    ``RichHandler``, which reads ``[...]`` as console markup and silently DROPS an unknown
    tag -- the bracketed form vanished from the log entirely and no grep for it ever matched.
    (``ltx_trainer``'s own ``[rank N]`` survives only because its format string escapes it.)
    """
    LOGGER.info("timing | %s: begin", label)
    started = time.time()
    yield
    LOGGER.info("timing | %s: done in %.1fs", label, time.time() - started)


DTYPE = torch.bfloat16

# SS4.2: the deployed operating point, with a validated k2 baseline. NOT swept -- the distilled
# grid admits only {0.421875, 0.725, 0.909375}, and 0.725 is the one where the base model
# already moves texture by about the right amount (28.59 dB output-vs-input on 2.3).
DEFAULT_SIGMA0 = 0.725

# The corpus's per-view products live beside the source video, and `dataset` owns the
# objective -> filename mapping for every reader and writer of them (SS1.2). A path literal
# at a call site is how the two halves of this pipeline have desynced before.
LOSS_MASK_GRIDS = dataset.LOSS_MASK_GRIDS_NAME

# Two named target sets. "attn" is the trainer's own default projection set; "attn_ffn" adds
# the feed-forward projections, which is the A2 sweep's second axis (SS A2 "attn-only vs
# attn+FFN"). Named here rather than passed as a free list so a run's arm is one word in the
# checkpoint metadata.
LORA_TARGETS = {
    "attn": ["to_k", "to_q", "to_v", "to_out.0"],
    "attn_ffn": ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
}


@dataclass(frozen=True)
class Chain:
    """``K`` consecutive causal blocks of one clip -- SS4.4's training sample.

    The latents here are the clip's **master** encodes, not per-block slices: the blocks are
    token ranges of them, and the cache needs the clip's whole token grid anyway to keep RoPE
    positions global. A clip is ~5 MB of bf16 latents at the 1024**2 geometry, so holding the
    master costs less than the per-window tree it replaces (which stored every frame twice,
    once per overlapping window).
    """

    source: str
    split: str
    actor: str
    seed_is_clip_start: bool
    blocks: list[int]
    z_g: torch.Tensor | None  # [C, F, H, W] master guide (ARGAvatar composite render); None when not loaded (d0)
    z_y: torch.Tensor  # [C, F, H, W] master capture (the loss target)
    fps: float
    loss_weights: torch.Tensor | None  # [F, h, w] per-cell loss weights over the whole clip
    z0_base: torch.Tensor | None  # [C, F, H, W] frozen-base one-step output, for the anchor


def _load_record(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _master(bundle: dict, path: Path) -> torch.Tensor:
    """The clip's continuous encode out of a schema-2 bundle, with a pointed error on v1.

    A schema-1 bundle holds per-window slices and no master. Reassembling one is a real
    operation but it is ``precompute.py --consolidate``'s job, not a silent fallback here:
    a reader that quietly reconstructs is a second producer of the tensor the trainer learns
    from, which is the exact failure shape SS7.3 names.
    """
    version = bundle.get("schema_version")
    if version != 2 or "master" not in bundle:
        raise SystemExit(
            f"{path}: schema_version={version} holds per-window slices, not the clip's master "
            f"latent. Run `python -m scripts.onestep_avatar.precompute --consolidate` over the "
            f"corpus first -- it rebuilds the master from the slices without touching the VAE"
        )
    return bundle["master"]


class ChainStore:
    """Reads ``windows.py``'s frozen subset against the corpus's own per-view bundles.

    Lazy per chain, and the unit of laziness is now the **clip**: a clip's master latents are
    ~5 MB of bf16 each, and a T3 subset is thousands of clips, so nothing is held resident.
    The subset JSON is the only thing parsed up front -- it is also the only place the split
    lives, so a held-out actor cannot leak into training by a path convention.
    """

    def __init__(
        self,
        subset: dict,
        corpus_root: Path,
        *,
        split: str,
        objective: str,
        band_weight: float,
        with_anchor: bool,
        with_guide: bool,
    ) -> None:
        self.subset = subset
        self.root = corpus_root
        self.objective = objective
        self.band_weight = band_weight
        self.capture_bundle = dataset.capture_bundle_name(objective)
        self.guide_bundle = dataset.guide_bundle_name(objective)
        self.with_anchor = with_anchor
        self.with_guide = with_guide
        self.chains = [chain for chain in subset["chains"] if chain["split"] == split]
        if not self.chains:
            raise SystemExit(f"subset has no chains in split {split!r}")
        self.sources = {record["relative_dir"]: record for record in subset["sources"]}

    def __len__(self) -> int:
        return len(self.chains)

    def __getitem__(self, i: int) -> Chain:
        chain = self.chains[i]
        view = self.root / chain["source"]
        capture = _load_record(view / self.capture_bundle)
        z_y = _master(capture, view / self.capture_bundle)
        # Guide-mode d0 never reads z_g (train_chain uses z_y as both source and target), so
        # skip requiring the guide bundle to exist for callers that only run d0 -- e.g. the
        # D0 sanity probe, which must work against capture-only precompute output.
        z_g = None
        if self.with_guide:
            guide = _load_record(view / self.guide_bundle)
            z_g = _master(guide, view / self.guide_bundle)
            if z_g.shape != z_y.shape:
                raise ValueError(f"{chain['source']}: guide {tuple(z_g.shape)} != capture {tuple(z_y.shape)}")
            if capture["fps"] != guide["fps"]:
                raise ValueError(f"{chain['source']}: guide fps {guide['fps']} != capture fps {capture['fps']}")

        # band_weight == 1.0 IS the plain full-frame loss, so the grids are not even read:
        # the weights would be all ones by construction (SS1.5). Same for d0: the band is the
        # render/capture DISAGREEMENT, which is undefined with no render in play (`with_guide`
        # is False) -- d0 must work against capture-only precompute, same as z_g above.
        loss_weights = None
        if self.with_guide and self.band_weight < 1.0:
            loss_weights = disagreement_weights(_load_record(view / LOSS_MASK_GRIDS), self.band_weight)
            if loss_weights.shape[0] != z_y.shape[1]:
                raise ValueError(
                    f"{chain['source']}: loss-mask grid has {loss_weights.shape[0]} latent frames, "
                    f"the master latent has {z_y.shape[1]}"
                )

        z0_base = None
        if self.with_anchor:
            z0_base = _master(_load_record(view / "base_denoised.pt"), view / "base_denoised.pt")

        return Chain(
            source=chain["source"],
            split=chain["split"],
            actor=chain["actor"],
            seed_is_clip_start=bool(chain["seed_is_clip_start"]),
            blocks=list(chain["blocks"]),
            z_g=z_g,
            z_y=z_y,
            fps=float(capture["fps"]),
            loss_weights=loss_weights,
            z0_base=z0_base,
        )


def disagreement_weights(record: dict, band_weight: float) -> torch.Tensor:
    """SS1.5's masking rule: full frame, down-weighted on the silhouette disagreement band.

    The corpus stores two coverage grids per view, deliberately uncombined -- ``render_alpha``
    (where the model is asked to paint) and ``capture_mask`` (where the target is meaningful).
    They disagree by exactly the SSB1 IoU gap, and that disagreement is the ONE region a loss
    should not trust: at IoU 0.77 an unweighted loss trains the model to reproduce the
    *render's* silhouette against a photograph.

    Everything else stays in the loss at full weight, and that is the whole point of the rule.
    The five subject masks this replaced (``render``/``capture``/``union``/``intersection``,
    and ``none``) were all pre-product: each gave weight zero everywhere outside the subject
    at time ``t``, which is exactly where the ghost band (``mask_0`` minus ``mask_t``)
    lives (SS1.2).
    A subject-masked loss therefore cannot teach the model to repair the ghost -- the region
    SS1.2 calls the learning signal -- however it is combined, so there was nothing to keep.

    The band is the SOFT symmetric difference ``|render_alpha - capture_mask|``: both grids
    are area fractions at latent resolution, so a boundary cell that is half-covered in one
    and fully covered in the other is half-disputed, not wholly. ``band_weight`` is what a
    fully disputed cell is worth -- 0.0 excludes the band outright (the default), 1.0 is a
    plain unweighted full-frame loss.

    Both objectives use this one rule and this one code path. Under ``white`` the background
    is white on both sides and there is no ghost band, so the loss is dominated by the
    subject; the rule does not change, only what dominates it (SS1.5).
    """
    render, capture = record["render_alpha"].float(), record["capture_mask"].float()
    band = (render - capture).abs()
    return 1.0 - (1.0 - band_weight) * band


def _as_token_weights(mask_5d: torch.Tensor, tools: VideoLatentTools) -> torch.Tensor:
    """Latent-grid coverage ``[1, 1, F, H, W]`` -> per-token weights ``[1, seq, 1]``.

    Patchified through the model's own patchifier rather than a reshape, so the mask lands on
    the same tokens the latent does for any patch size.
    """
    return tools.patchifier.patchify(mask_5d).mean(dim=-1, keepdim=True)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted MSE over tokens, normalised by the weight mass.

    Normalising (rather than averaging over all tokens) keeps the loss scale independent of
    how much of the frame the subject occupies, so a wide crop and a tight one contribute
    comparably instead of the tight one dominating.
    """
    error = (pred.float() - target.float()).pow(2)
    weighted = error * weights
    return weighted.sum() / weights.expand_as(error).sum().clamp(min=1e-8)


def clip_grid_for(
    chain: Chain, geometry: CausalGeometry, *, device: torch.device, latent_channels: int
) -> ClipGrid:
    """The clip's token grid -- positions, keyframe marks, tools -- built once per chain."""
    _, latent_frames, height, width = chain.z_y.shape
    return ClipGrid.build(
        latent_frames,
        height * geometry.scale_factors.height,
        width * geometry.scale_factors.width,
        chain.fps,
        geometry,
        device=device,
        dtype=DTYPE,
        latent_channels=latent_channels,
    )


def block_weights(
    grid: ClipGrid, chain: Chain, span: tuple[int, int], device: torch.device
) -> torch.Tensor:
    """Per-token loss weights for one block: all ones, times SS1.5's band weighting if any.

    Every token in a block is predicted now. Under the old window there were conditioning
    tokens (the frozen carryover, the keyframe) carrying ``denoise_mask`` 0 that had to be
    excluded; the cache holds that content instead, so it is not in the sequence at all and
    there is nothing to exclude.
    """
    start, end = span
    tokens = (end - start) * grid.tokens_per_latent_frame
    weights = torch.ones(1, tokens, 1, device=device, dtype=torch.float32)
    if chain.loss_weights is None:
        return weights
    coverage = chain.loss_weights[start:end].to(device=device, dtype=torch.float32)
    return weights * _as_token_weights(coverage.unsqueeze(0).unsqueeze(0), grid.tools)


def assert_rank_lockstep(accelerator: Accelerator, planned_forwards: int, source: str) -> None:
    """Refuse a step whose ranks would issue different numbers of transformer forwards.

    Under FSDP FULL_SHARD a forward is a round of all-gathers, so ranks that run a different
    NUMBER of them desynchronise the collective stream and the job deadlocks -- silently, at
    100 % GPU utilisation, until a watchdog fires minutes later blaming something unrelated.
    That cost a long debugging session on 2026-09-16, when ``prime_cache`` skipped its forward
    for clip-start chains (fixed there; see its empty-spans branch).

    This runs BEFORE the step's forwards, while the ranks are still in step from the previous
    optimizer update, so the gather here is itself safely matched. Checking afterwards cannot
    work: by then the mismatched collective has already been enqueued and this gather would
    join the pile-up rather than report it.

    The count is ``1 prime + K denoise + K refresh``. What it is does not matter -- only that
    every rank computes the same one.
    """
    if accelerator.num_processes == 1:
        return
    counts = accelerator.gather(
        torch.tensor([planned_forwards], device=accelerator.device, dtype=torch.long)
    ).tolist()
    if len(set(counts)) > 1:
        raise SystemExit(
            f"rank forward counts disagree for this step: {counts} (this rank: "
            f"{planned_forwards}, chain from {source}). Every rank must run the same number of "
            f"transformer forwards per step or FSDP's collectives desynchronise and the job "
            f"hangs instead of failing. Something made a forward conditional on the data -- "
            f"chain length, priming, or an early return on a short clip."
        )


def train_chain(  # noqa: PLR0913, PLR0915 -- one AR training step is defined by all of these, timing included
    transformer: torch.nn.Module,
    context: torch.Tensor,
    chain: Chain,
    geometry: CausalGeometry,
    cache: BlockCache | None,
    accelerator: Accelerator,
    *,
    sigma0: float,
    seed: int,
    anchor_weight: float,
    latent_channels: int,
    guide_mode: str = "d1",
    teacher_forcing: bool = False,
    timing: bool = False,
) -> dict[str, float]:
    """SS4.4's chain: ``K`` denoise forwards, ``K`` backwards, ``K`` refreshes, one optimizer step.

    Per block, in this order and for these reasons:

    ``denoise`` reads the cache and writes nothing, so gradient checkpointing stays valid on
    the only pass that stores activations. ``backward`` runs immediately, which is what keeps
    peak activation memory at a **single** block rather than ``K`` of them. ``refresh`` then
    forwards the block's clean latent under ``no_grad`` to put its keys and values in the
    cache for every later block, and evicts down to the retained context.

    ``teacher_forcing`` hands ``refresh`` the ground-truth capture instead of the block's own
    ``ẑ₀``. That one tensor is the entire difference between the two regimes -- nothing else
    in the loop, and nothing in ``causal_core``, knows which is in play.

    ``timing`` logs the wall time of each of those four phases per block. The numbers are
    honest without an explicit ``cuda.synchronize`` only because ``float(mse.detach())``
    already forces one inside the same block -- do not move that read without revisiting
    this, or the phases will start reporting queue time instead of compute.

    The cache is primed from the ground truth for blocks before the chain's first (see
    ``causal_core.prime_cache``); a chain that starts at block 0 needs no priming, which is
    what ``seed_is_clip_start`` records.
    """
    device = accelerator.device
    grid = clip_grid_for(chain, geometry, device=device, latent_channels=latent_channels)
    plan = geometry.plan(grid.latent_frames)
    if max(chain.blocks) >= len(plan):
        raise ValueError(
            f"{chain.source}: chain asks for block {max(chain.blocks)} but the clip plans "
            f"{len(plan)} under {geometry.as_dict()}; the subset was frozen under a different geometry"
        )

    denoise_fn = causal_core.denoised_from_velocity_model(transformer)
    z_y = chain.z_y.unsqueeze(0).to(device=device, dtype=DTYPE)
    # SS4.1 d0 is a training-only sanity arm: it noises the CAPTURE instead of the guide, so
    # the objective reduces to ordinary flow matching on real video (SS3 identity 1), decoupled
    # from the render entirely. `onestep_core.guide_conditionings` refuses it because there is
    # no z_y at inference to noise. z_g is only touched in the d1 branch -- ChainStore does not
    # load it for d0 runs, so chain.z_g may be None here.
    if guide_mode == "d0":
        source = z_y
    elif guide_mode == "d1":
        source = chain.z_g.unsqueeze(0).to(device=device, dtype=DTYPE)
    else:
        raise ValueError(f"unknown guide mode {guide_mode!r}; expected 'd0' or 'd1'")
    guide_tokens = grid.patchify(source)
    target_tokens = grid.patchify(z_y)
    base_tokens = (
        grid.patchify(chain.z0_base.unsqueeze(0).to(device=device, dtype=DTYPE))
        if chain.z0_base is not None
        else None
    )

    if cache is None:
        cache = BlockCache.allocate(
            grid,
            geometry,
            num_layers=_num_blocks(transformer),
            inner_dim=_inner_dim(transformer),
            device=device,
            dtype=DTYPE,
        )
    elif cache.grid.tokens_per_latent_frame != grid.tokens_per_latent_frame:
        # The cache is allocated once for the whole run, against the fixed 1024**2 crop
        # (SS4.5). A clip at a different spatial size would make every kv_start off by a
        # frame's worth of tokens -- silently, since the buffer is big enough either way.
        raise ValueError(
            f"{chain.source}: {grid.tokens_per_latent_frame} tokens per latent frame, but the "
            f"cache was allocated for {cache.grid.tokens_per_latent_frame}; the corpus is "
            f"supposed to be one geometry (SS4.5)"
        )
    prime_started = time.time()
    causal_core.prime_cache(
        denoise_fn, grid, cache, target_tokens, geometry, context, upto_latent_frame=plan[chain.blocks[0]][0]
    )
    if timing:
        LOGGER.info(
            "timing |   prime_cache(upto=%d): %.2fs",
            plan[chain.blocks[0]][0], time.time() - prime_started,
        )

    totals = {"loss": 0.0, "mse": 0.0, "anchor": 0.0}
    per_block: list[dict[str, float]] = []
    k = len(chain.blocks)
    for block_index in chain.blocks:
        span = plan[block_index]
        lo, hi = grid.token_span(*span)
        block_started = time.time()
        noisy = causal_core.noise_block(guide_tokens[:, lo:hi], sigma0, seed + block_index)
        z0 = causal_core.denoise_block(denoise_fn, grid, cache, noisy, context, sigma0, span)
        denoised_at = time.time()

        weights = block_weights(grid, chain, span, device)
        mse = masked_mse(z0, target_tokens[:, lo:hi], weights)
        loss = mse
        anchor = torch.zeros((), device=device)
        if anchor_weight > 0.0:
            if base_tokens is None:
                raise ValueError("--anchor-weight > 0 but this view has no base_denoised.pt")
            # SS4.3 row 2 / SS2.3(3): the risk here is ERODING sharpness Phi already has, not
            # failing to synthesise it. Pulling toward the frozen model's own output on the
            # same input is the cheapest thing that targets that directly.
            anchor = masked_mse(z0, base_tokens[:, lo:hi], weights)
            loss = loss + anchor_weight * anchor

        accelerator.backward(loss / k)
        backward_at = time.time()
        totals["loss"] += float(loss.detach()) / k
        totals["mse"] += float(mse.detach()) / k
        totals["anchor"] += float(anchor.detach()) / k
        per_block.append(
            {"block_index": block_index, "mse": float(mse.detach()), "anchor": float(anchor.detach())}
        )

        clean = target_tokens[:, lo:hi] if teacher_forcing else z0.detach()
        causal_core.refresh_block(denoise_fn, grid, cache, clean, context, span)
        if timing:
            LOGGER.info(
                "timing |   block %d (span %d:%d): denoise %.2fs backward %.2fs refresh %.2fs (total %.2fs)",
                block_index, span[0], span[1],
                denoised_at - block_started,
                backward_at - denoised_at,
                time.time() - backward_at,
                time.time() - block_started,
            )
        del z0, weights, loss, mse, anchor
    totals["per_block"] = per_block
    return totals


def _base_model(transformer: torch.nn.Module) -> torch.nn.Module:
    """Peel FSDP/DDP and PEFT wrappers off to reach the ``LTXModel`` the cache is sized from.

    The cache needs two numbers the wrappers do not expose -- the block count and the inner
    dimension -- and asking the checkpoint config for them instead would be a second source
    of truth for the model that is actually resident.
    """
    model = transformer
    for _ in range(8):
        if hasattr(model, "transformer_blocks"):
            return model
        for attribute in ("module", "base_model", "model"):
            inner = getattr(model, attribute, None)
            if isinstance(inner, torch.nn.Module):
                model = inner
                break
        else:
            break
    raise TypeError(f"cannot find the LTXModel inside {type(transformer).__name__}")


def _num_blocks(transformer: torch.nn.Module) -> int:
    return len(_base_model(transformer).transformer_blocks)


def _inner_dim(transformer: torch.nn.Module) -> int:
    return _base_model(transformer).inner_dim


def build_transformer(
    model: model_registry.RefinerModel, args: argparse.Namespace, accelerator: Accelerator
) -> torch.nn.Module:
    """Load the frozen bf16 backbone and inject LoRA -- ``ltx_trainer``'s own plumbing."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Loading straight onto the GPU, not via host RAM: the full bf16 checkpoint is 42 GB and
    # three ranks staging it on the host would need 126 GB of a machine that has ~139 GB free
    # here. FSDP shards in place afterwards, so the 42 GB is transient and fits a 49 GB card.
    init_device = args.init_device if args.init_device != "cuda" else f"cuda:{local_rank}"
    transformer = load_transformer(
        checkpoint_path=model.paths.transformer(), device=init_device, dtype=DTYPE, video_only=True
    )
    transformer.requires_grad_(False)
    transformer = get_peft_model(
        transformer,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=LORA_TARGETS[args.lora_target],
            lora_dropout=0.0,
            init_lora_weights=True,
        ),
    )
    if accelerator.distributed_type == DistributedType.FSDP:
        # FSDP needs one dtype per flat parameter, and PEFT makes the adapters fp32 against a
        # bf16 base. This policy wraps the trainable leaves separately, which is what lets the
        # base stay bf16 instead of being promoted to a full fp32 host copy before sharding.
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(transformer)
    transformer.get_base_model().set_gradient_checkpointing(not args.no_gradient_checkpointing)
    return transformer


def causal_geometry(args: argparse.Namespace, model: model_registry.RefinerModel) -> CausalGeometry:
    return causal_core.deployed_geometry(
        model.scale_factors,
        block_latent_frames=args.block_latent_frames,
        context_latent_frames=args.context_latent_frames,
    )


def checkpoint_metadata(
    args: argparse.Namespace, subset: dict, model: model_registry.RefinerModel, step: int
) -> dict[str, str]:
    """SS7.1 / SS9 risk 13: a fixed-sigma adapter must not be loadable off-condition.

    sigma_0, ``K`` and now the **causal geometry** are recorded so ``refine_task``'s ``ONE_STEP``
    schedule can refuse a checkpoint whose sigma disagrees, a multi-step run, or a rollout at a
    different block/cache depth -- all of which would otherwise fail silently. The cache depth
    belongs here for the same reason sigma does: an adapter trained with two frames of cached
    context is a different function from one trained with six, and nothing downstream can tell
    by looking at the weights.
    """
    sigma_levels = training_sigmas(args)
    geometry = causal_geometry(args, model)
    return {
        # ``mixed`` deliberately prevents a fixed-sigma deployment loader from accepting a
        # multi-level adapter as though it were calibrated for just one noise level.
        "onestep_avatar_sigma0": repr(args.sigma0) if args.sigma_levels is None else "mixed",
        "onestep_avatar_sigma_levels": ",".join(repr(sigma) for sigma in sigma_levels),
        "onestep_avatar_chain_length": str(subset["chain_length"]),
        "onestep_avatar_schedule": "ONE_STEP",
        "onestep_avatar_attention": "block_causal",
        "onestep_avatar_block_latent_frames": str(geometry.block_latent_frames),
        "onestep_avatar_context_latent_frames": str(geometry.context_latent_frames),
        "onestep_avatar_sink_latent_frames": str(geometry.sink_latent_frames),
        "onestep_avatar_subset_sha256": hashlib.sha256(
            json.dumps(subset["sources"], sort_keys=True).encode()
        ).hexdigest(),
        "onestep_avatar_objective": args.objective,
        "onestep_avatar_disagreement_weight": repr(args.disagreement_weight),
        "onestep_avatar_guide_mode": args.guide_mode,
        "onestep_avatar_anchor_weight": repr(args.anchor_weight),
        "onestep_avatar_teacher_forcing": str(args.teacher_forcing),
        "model_key": model.key,
        "lora_rank": str(args.lora_rank),
        "lora_alpha": str(args.lora_alpha),
        "lora_target": args.lora_target,
        "step": str(step),
    }


def training_sigmas(args: argparse.Namespace) -> tuple[float, ...]:
    """Resolve the fixed or per-rank noise schedule once, with strict bounds.

    ``sigma=0.0`` is refused: it noises nothing, so the "denoised" target IS the input
    (identity) and the loss/gradient are both exactly zero (measured: the multilevel D0 run's
    sigma=0.0 quarter sat at 0.0 mse for all 200 steps). A rank assigned that level would train
    on nothing for the whole run, which is a training bug, not a sanity arm -- the ceiling
    measurement that wants sigma=0.0 is ``visualize_d0.py``'s fixed evaluation-time probe grid,
    not something this loop should ever spend a rank optimizing.
    """
    values = (args.sigma0,) if args.sigma_levels is None else tuple(args.sigma_levels)
    if not values or any(not 0.0 < sigma <= 1.0 for sigma in values):
        raise SystemExit(
            "--sigma-levels must contain one or more values in (0, 1] -- sigma=0.0 adds no "
            "noise, so its loss/gradient are identically zero and it must not be trained"
        )
    if len(set(values)) != len(values):
        raise SystemExit("--sigma-levels must not repeat a noise level")
    return values


def sigma_for_rank(sigmas: tuple[float, ...], rank: int, step: int) -> float:
    """This rank's noise level at this step: ``(rank + step) % len(sigmas)``.

    Two things the old per-step cycle (every rank sharing one level, changing each step) got
    wrong at once: it aliased into a sawtooth loss curve at the step's own period, since a
    step's mean-across-ranks loss was always a single-level loss rather than a genuine batch
    average; and a purely per-rank assignment (``rank % len(sigmas)``, no ``step`` term) never
    trains the levels beyond ``world_size`` at all when ``world_size < len(sigmas)`` -- a
    silent coverage gap, not just a variance one. Offsetting by ``rank`` keeps every single
    step's batch mixing whatever levels are present among the ranks (fixing the first
    problem); advancing by ``step`` walks every rank through the full level set over the run
    (fixing the second).
    """
    return sigmas[(rank + step) % len(sigmas)]


def init_wandb(args: argparse.Namespace, *, config: dict) -> object | None:
    """Create one online W&B run on rank 0; all ranks still participate in metric gathers."""
    if args.wandb_project is None:
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - environment/setup error
        raise SystemExit("--wandb-project requires the wandb package in the active environment") from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or args.output.name,
        mode=args.wandb_mode,
        config=config,
    )


def rank_mean(accelerator: Accelerator, values: list[float]) -> list[float]:
    """All-rank mean for W&B, keeping one chart point per optimiser step."""
    local = torch.tensor(values, device=accelerator.device, dtype=torch.float32)
    gathered = accelerator.gather(local).reshape(accelerator.num_processes, -1)
    return gathered.mean(dim=0).cpu().tolist()


def save_lora(
    transformer: torch.nn.Module,
    accelerator: Accelerator,
    out_dir: Path,
    step: int,
    metadata: dict[str, str],
    *,
    verify_noop: bool = False,
) -> Path | None:
    """Gather and write the adapter in the trainer's own ComfyUI-compatible layout.

    ``verify_noop`` is only for the pre-optimizer step-0 checkpoint. PEFT's default
    LoRA initialization makes A random and B exactly zero, so the product B @ A --
    and therefore the adapter delta -- must be exactly zero. Verify the *exported*
    state rather than a module attribute: that covers the actual tensors handed to
    inference, including FSDP's gathered representation.
    """
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(transformer)
    if not accelerator.is_main_process:
        return None
    unwrapped = accelerator.unwrap_model(transformer, keep_torch_compile=False)
    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    state_dict = get_peft_model_state_dict(unwrapped, state_dict=state_dict if is_fsdp else None)
    state_dict = {f"diffusion_model.{k.replace('base_model.model.', '', 1)}": v for k, v in state_dict.items()}
    state_dict = {k: v.to(torch.bfloat16).contiguous() for k, v in state_dict.items()}
    if verify_noop:
        assert_exported_lora_is_noop(state_dict)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lora_weights_step_{step:05d}.safetensors"
    save_file(state_dict, path, metadata=metadata)
    return path


def assert_exported_lora_is_noop(state_dict: dict[str, torch.Tensor]) -> None:
    """Raise unless an exported, newly-created LoRA has an exactly-zero B projection.

    A zero B is the standard LoRA no-op initialization: A may be random, but B @ A
    is zero. Checking only B catches changed PEFT initialization without rejecting
    the intended random A initialization.
    """
    b_weights = {name: value for name, value in state_dict.items() if ".lora_B" in name}
    if not b_weights:
        raise RuntimeError("step-0 LoRA export has no lora_B weights; cannot prove it is a no-op")
    nonzero = [name for name, value in b_weights.items() if torch.count_nonzero(value).item()]
    if nonzero:
        raise RuntimeError(
            "refusing to write a purported step-0 checkpoint with a non-zero LoRA delta: "
            + ", ".join(nonzero[:5])
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", type=Path, required=True, help="windows.py's frozen subset JSON")
    p.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Corpus root holding the per-view master latents. Defaults to the subset's own "
        "`corpus_root`, which is where precompute.py wrote them -- pass this only to read a "
        "relocated copy.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    p.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA0)
    p.add_argument(
        "--sigma-levels", type=float, nargs="+", default=None,
        help="Assign one rotating level per rank per step ((rank + step) %% len(levels)). "
        "Overrides --sigma0; useful for a multilevel distilled adapter. sigma=0.0 is refused "
        "-- it trains on nothing (see training_sigmas).",
    )
    p.add_argument(
        "--block-latent-frames", type=int, default=causal_core.BLOCK_LATENT_FRAMES,
        help="Latent frames denoised per causal block (SS4.4). The default is the deployed "
        "16-pixel-frame stride; changing it changes what a trained adapter finalizes per step.",
    )
    p.add_argument(
        "--context-latent-frames", type=int, default=causal_core.CONTEXT_LATENT_FRAMES,
        help=f"Clean latent frames kept in the K/V cache besides the pinned frame-0 sink, up "
        f"to {causal_core.MAX_CONTEXT_LATENT_FRAMES}. Each one costs ~0.8 GB per rank at the "
        f"22B geometry and lengthens every block's attention, so this is the compute/quality "
        f"knob of the scheme. Past roughly the chain's own reach the cache stops evicting and "
        f"simply ACCUMULATES the whole rollout's history -- at the default K=3 and a 2-frame "
        f"block that is 6 finalized frames plus the primed prefix. Recorded in the checkpoint "
        f"metadata: an adapter trained at one depth is a different function at another.",
    )
    p.add_argument("--split", choices=("train", "held_out"), default="train")
    p.add_argument("--lora-rank", type=int, default=8, help="2-3 GPU preliminary runs drop this, never K")
    p.add_argument("--lora-alpha", type=int, default=None, help="default: equal to --lora-rank")
    p.add_argument("--lora-target", choices=sorted(LORA_TARGETS), default="attn")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--objective",
        choices=dataset.OBJECTIVES,
        default=dataset.DEFAULT_OBJECTIVE,
        help="SS1.2. bg (default): the product -- guide composited over the clip's real first "
        "frame, target the unmatted capture. white: both sides on white, which isolates the "
        "subject-texture gap from the background question. Selects which pair of bundles is "
        "read; must match the objective the subset was frozen against.",
    )
    p.add_argument(
        "--disagreement-weight",
        type=float,
        default=0.0,
        help="SS1.5. Loss weight of the render-vs-capture silhouette disagreement band, the "
        "one region a full-frame loss should not trust. 0.0 (default) excludes it; 1.0 is a "
        "plain unweighted full-frame loss. Everything else -- subject, background, and the "
        "ghost band SS1.2 calls the learning signal -- always stays at full weight.",
    )
    p.add_argument(
        "--guide-mode",
        choices=("d0", "d1"),
        default="d1",
        help="SS4.1. d1: the plan's D1a -- the guide reaches the model only as the noised "
        "init. Cheapest, what the causal rollout deploys today, and the control every other "
        "arm has to beat. d0: SANITY ONLY, not deployable -- noises the capture z_y instead "
        "of the guide, reducing to ordinary flow-matching on real video (SS3 identity 1). "
        "Measures the architecture's capacity ceiling at sigma_0, to compare against the "
        "measured r; onestep_core.guide_conditionings refuses this mode because there is no "
        "z_y at inference. (The old `d2` extra-token hybrid is gone: it was dropped as an arm "
        "2026-09-13 for costing 1.05x k2, and its clean reference tokens have no place in a "
        "causal sequence -- they would be future context.)",
    )
    p.add_argument("--anchor-weight", type=float, default=0.0, help="SS4.3 row 2; needs base_denoised.pt")
    p.add_argument(
        "--teacher-forcing",
        action="store_true",
        help="Ablation arm: the cache refresh is fed the ground-truth capture latent instead "
        "of this block's own generation. Isolates exposure-bias drift from everything else "
        "the AR loop changes; NOT the production setting (deployment always carries the "
        "model's own output).",
    )
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument(
        "--save-initial",
        action="store_true",
        help="Also save a step-0 checkpoint of the untrained, LoRA-injected model before the "
        "loop starts. `init_lora_weights=True` zero-inits B, so this adapter should decode "
        "identically to the frozen base -- the point is to make that provable rather than "
        "assumed. Off by default: normal runs don't need the extra checkpoint write.",
    )
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument(
        "--timing",
        action="store_true",
        help="log a per-phase breakdown of every step (chain load, prime, per-block denoise/"
             "backward/refresh, optimizer). Startup stages are always timed; this is the "
             "per-step detail, and it is per-block, so it is off by default.",
    )
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--wandb-project", default=None, help="Enable online W&B logging to this project.")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--init-device", default="cuda", help="'cuda' (default, avoids host-RAM staging) or 'cpu'")
    p.add_argument("--dry-run", action="store_true", help="report the plan and the data shapes, load no model")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915 -- one linear training script.
    args = parse_args(argv)
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_rank
    if not 0.0 <= args.disagreement_weight <= 1.0:
        raise SystemExit("--disagreement-weight is a loss weight in [0, 1]")
    sigmas = training_sigmas(args)
    if args.guide_mode == "d0" and args.anchor_weight > 0.0:
        # base_denoised (SS4.3 row 2) is Phi(lerp(z_g, eps, sigma_0)) -- the frozen model's
        # output on the GUIDE-noised input. d0 noises z_y instead, so the anchor would be
        # pulling this run toward an output computed on an input it never sees.
        raise SystemExit(
            "--guide-mode d0 is noised from z_y; its anchor target would be off-input. "
            "Drop --anchor-weight."
        )
    if args.guide_mode == "d0" and args.disagreement_weight != 0.0:
        # SS1.5's band is render_t (-) capture_t -- undefined with no render, which is exactly
        # d0's point (SS1.3: "reducing to ordinary flow-matching on real video"). d0 must work
        # against capture-only precompute (ChainStore skips z_g the same way), so a non-default
        # weight here would ask for grids d0 has no business reading.
        raise SystemExit(
            "--guide-mode d0 has no guide render, so there is no render/capture disagreement "
            "band to weight. Drop --disagreement-weight (0.0, its default, already means "
            "'no band' for d0)."
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    subset = json.loads(args.subset.read_text())
    if subset.get("kind") != "one_step_argavatar_block_chains":
        raise SystemExit(
            f"{args.subset} is not a block-chain subset (kind={subset.get('kind')!r}). Re-freeze "
            f"it with `python -m scripts.onestep_avatar.windows` -- the window-chain subsets "
            f"predate SS4.4's causal scheme and index windows that no longer exist"
        )
    # SS1.2: a subset is surveyed and content-pinned against ONE objective's artifacts, so
    # training the other one against it would read bundles the freeze never saw. Subsets
    # frozen before the objective existed are `bg` by construction -- that is what was on
    # disk -- so they are read as such rather than refused.
    subset_objective = subset.get("objective", dataset.DEFAULT_OBJECTIVE)
    if subset_objective != args.objective:
        raise SystemExit(
            f"{args.subset} was frozen against objective {subset_objective!r} but this run asks "
            f"for {args.objective!r}. Re-freeze with `windows.py --objective {args.objective}`"
        )
    model = model_registry.resolve(args.model)
    geometry = causal_geometry(args, model)
    corpus_root = args.corpus_root or Path(subset["corpus_root"])
    store = ChainStore(
        subset,
        corpus_root,
        split=args.split,
        objective=args.objective,
        band_weight=args.disagreement_weight,
        with_anchor=args.anchor_weight > 0.0,
        with_guide=args.guide_mode != "d0",
    )

    if args.dry_run:
        chain = store[0]
        grid_frames = chain.z_y.shape[1]
        print(  # noqa: T201 -- CLI's requested plan.
            json.dumps(
                {
                    "chains": len(store),
                    "chain_length": subset["chain_length"],
                    "geometry": geometry.as_dict(),
                    "deployed_stride_match": causal_core.matches_deployed_stride(
                        geometry, refine_task.deployed_geometry(model.scale_factors)
                    ),
                    "first_chain": {
                        "source": chain.source,
                        "actor": chain.actor,
                        "seed_is_clip_start": chain.seed_is_clip_start,
                        "blocks": chain.blocks,
                        "master_shape": list(chain.z_y.shape),
                        "planned_blocks": len(geometry.plan(grid_frames)),
                        "fps": chain.fps,
                    },
                    "sigma0": args.sigma0,
                    "sigma_levels": list(sigmas),
                    "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "target": args.lora_target},
                },
                indent=2,
            )
        )
        return 0

    # No explicit mixed_precision: the accelerate config decides, and the 2/3-GPU configs are
    # copies of the trainer's own, so this loop runs under the same policy the shipped trainer
    # does rather than a second one of its own.
    with timed("Accelerator() / process group"):
        accelerator = Accelerator()
    device = accelerator.device
    world, rank = accelerator.num_processes, accelerator.process_index

    with timed("prompt cache (text encoder)"):
        context = prompt_cache.get_or_build(model, refine_task.REFINE_PROMPT, DTYPE, device)

    with timed("transformer load + LoRA injection"):
        transformer = build_transformer(model, args, accelerator)
    trainable = [p for p in transformer.parameters() if p.requires_grad]
    # Counted BEFORE `prepare`: FSDP with `use_orig_params=True` reshapes each parameter to
    # this rank's shard in place, so the same expression afterwards reports total/world_size
    # and reads like a model half the size.
    trainable_total = sum(p.numel() for p in trainable)
    num_blocks, inner_dim = _num_blocks(transformer), _inner_dim(transformer)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    with timed("accelerator.prepare (FSDP shard)"):
        transformer, optimizer = accelerator.prepare(transformer, optimizer)
    if accelerator.is_main_process:
        LOGGER.info(
            "trainable params: %s total, %s per rank across %d (%s rank %d, alpha %d)",
            f"{trainable_total:,}",
            f"{trainable_total // world:,}",
            world,
            args.lora_target,
            args.lora_rank,
            args.lora_alpha,
        )
        LOGGER.info("causal geometry: %s", json.dumps(geometry.as_dict()))

    # Chains are sharded by rank rather than by an accelerate DataLoader: a sample here is a
    # variable-length chain of tensors, not a collatable batch, and FSDP is data-parallel over
    # ranks, so a deterministic stride is both simpler and reproducible with no sampler state.
    # Every rank runs the SAME number of steps, so the shard is truncated to the common length.
    per_rank = len(store) // world
    if per_rank == 0:
        raise SystemExit(f"{len(store)} chains cannot be split across {world} ranks")
    order = list(range(len(store)))

    args.output.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        (args.output / "config.json").write_text(
            json.dumps(
                {**vars(args), "world_size": world, "causal_geometry": geometry.as_dict()},
                indent=2,
                default=str,
            )
        )
    wandb_run = init_wandb(
        args,
        config={**vars(args), "world_size": world, "sigma_levels": list(sigmas), **geometry.as_dict()},
    ) if accelerator.is_main_process else None
    log_path = args.output / f"metrics_rank{rank}.jsonl"
    log_file = log_path.open("a")

    if args.save_initial:
        with timed("save_initial (step-0 adapter)"):
            path = save_lora(
                transformer, accelerator, args.output / "checkpoints", 0,
                checkpoint_metadata(args, subset, model, 0),
                verify_noop=True,
            )
        if path is not None:
            LOGGER.info("saved initial (untrained) checkpoint %s", path)

    generator = torch.Generator().manual_seed(args.seed)
    # One cache allocation for the whole run: capacity depends only on the geometry and the
    # (fixed) 1024**2 crop, so reallocating per chain would just churn ~2 GB of VRAM.
    cache: BlockCache | None = None
    step = 0
    started = time.time()
    while step < args.steps:
        epoch_order = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]
        shard = epoch_order[rank * per_rank : (rank + 1) * per_rank]
        for chain_index in shard:
            if step >= args.steps:
                break
            lr = args.lr * min(1.0, (step + 1) / max(args.warmup_steps, 1))
            for group in optimizer.param_groups:
                group["lr"] = lr
            sigma0 = sigma_for_rank(sigmas, rank, step)

            step_started = time.time()
            # Unconditional for the FIRST chain only: it is the one that pays the corpus read
            # and the ~2 GB cache allocation, so "the run printed the geometry and then went
            # quiet" -- what the 09-15 4-GPU launch log looks like -- is decided here, before
            # any --timing opt-in could have been remembered.
            first_chain = step == 0
            verbose = args.timing or first_chain
            label = "chain load (first: corpus read + cache alloc)" if first_chain else f"chain load {chain_index}"
            with timed(label) if verbose else contextlib.nullcontext():
                chain = store[chain_index]
                if cache is None:
                    grid = clip_grid_for(
                        chain, geometry, device=device, latent_channels=model.caps.latent_channels
                    )
                    cache = BlockCache.allocate(
                        grid, geometry, num_layers=num_blocks, inner_dim=inner_dim, device=device, dtype=DTYPE
                    )
            loaded_at = time.time()
            # 1 prime + K denoise + K refresh. Checked here, before any of them run.
            assert_rank_lockstep(accelerator, 1 + 2 * len(chain.blocks), chain.source)
            totals = train_chain(
                transformer,
                context,
                chain,
                geometry,
                cache,
                accelerator,
                sigma0=sigma0,
                # Seeded per (run, block) so eps is reproducible and the same block always
                # gets the same noise -- which is also what makes a cached frozen-base output
                # (the anchor term) correspond to this exact input.
                seed=args.seed * 100003 + chain_index * 101,
                anchor_weight=args.anchor_weight,
                latent_channels=model.caps.latent_channels,
                guide_mode=args.guide_mode,
                teacher_forcing=args.teacher_forcing,
                timing=verbose,
            )
            chained_at = time.time()
            grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if verbose:
                # clip_grad_norm_ is a collective, so this line is also the cheapest read on
                # whether one rank is lagging the others -- they cannot leave it separately.
                LOGGER.info(
                    "timing | step %d: load %.2fs chain %.2fs optimizer %.2fs (total %.2fs)",
                    step, loaded_at - step_started, chained_at - loaded_at,
                    time.time() - chained_at, time.time() - step_started,
                )

            if step % args.log_every == 0:
                per_block = totals.pop("per_block")
                record = {
                    "step": step,
                    "rank": rank,
                    "lr": lr,
                    "sigma0": sigma0,
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "elapsed_s": round(time.time() - started, 1),
                    "source": chain.source,
                    **{k: round(v, 6) for k, v in totals.items()},
                    # SS7.4(a): one entry per block IN THIS CHAIN, in order, so a reader can
                    # plot loss against position without re-deriving it from the chain-mean.
                    "per_block": [
                        {"chain_position": i, **{k: round(v, 6) for k, v in w.items()}}
                        for i, w in enumerate(per_block)
                    ],
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if args.wandb_project is not None:
                    mean_loss, mean_mse, mean_anchor, mean_grad_norm = rank_mean(
                        accelerator,
                        [totals["loss"], totals["mse"], totals["anchor"], float(grad_norm)],
                    )
                    block_mse = rank_mean(accelerator, [block["mse"] for block in per_block])
                    if accelerator.is_main_process:
                        wandb_run.log(
                            {
                                "train/loss": mean_loss,
                                "train/mse": mean_mse,
                                "train/anchor": mean_anchor,
                                "train/grad_norm": mean_grad_norm,
                                "train/lr": lr,
                                "train/sigma0": sigma0,
                                "train/elapsed_s": record["elapsed_s"],
                                "train/steps_per_s": step / max(record["elapsed_s"], 1e-8),
                                **{f"train/window_{i}_mse": value for i, value in enumerate(block_mse)},
                            },
                            step=step,
                        )
                if accelerator.is_main_process:
                    LOGGER.info(
                        "step %d/%d loss %.5f mse %.5f anchor %.5f lr %.2e %.1fs",
                        step, args.steps, totals["loss"], totals["mse"], totals["anchor"], lr,
                        time.time() - started,
                    )
            if step % args.save_every == 0 or step == args.steps:
                path = save_lora(
                    transformer, accelerator, args.output / "checkpoints", step,
                    checkpoint_metadata(args, subset, model, step),
                )
                if path is not None:
                    LOGGER.info("saved %s", path)

    log_file.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if wandb_run is not None:
            wandb_run.finish()
        LOGGER.info("done: %d steps in %.1f min", step, (time.time() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
