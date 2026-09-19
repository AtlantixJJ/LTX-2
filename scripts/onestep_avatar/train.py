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
      --subset ../expr/onestep_avatar/windows/t2r2.json \\
      --output ../expr/onestep_avatar/runs/prelim --lora-rank 8 --steps 200

accelerate launch --config_file scripts/onestep_avatar/configs/fsdp_4gpu.yaml -m scripts.onestep_avatar.train --subset ../expr/onestep_avatar/windows/t2r2.json --output ../expr/onestep_avatar/runs/prelim --lora-rank 64 --steps 200
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
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
    z0_base: torch.Tensor | None  # [C, F, H, W] frozen-base one-step output, for the anchor


def _load_record(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_training_master(path: Path) -> tuple[torch.Tensor, float]:
    """Check the tensor and timebase used by both startup and lazy chain loading."""
    record = _load_record(path)
    master = dataset.load_master(path, bundle=record)
    if (
        not isinstance(master, torch.Tensor)
        or master.ndim != 4
        or not master.is_floating_point()
        or any(size <= 0 for size in master.shape)
    ):
        raise SystemExit(f"{path}: master must be a nonempty floating [C, F, H, W] tensor")
    fps = record.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise SystemExit(f"{path}: fps must be a finite positive number, got {fps!r}")
    return master, float(fps)


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
        with_anchor: bool,
        with_guide: bool,
    ) -> None:
        if subset.get("kind") != "one_step_argavatar_block_chains":
            raise SystemExit(
                f"not a block-chain subset (kind={subset.get('kind')!r}). Re-freeze it with "
                f"`python -m scripts.onestep_avatar.windows` -- a window-chain subset predates "
                f"SS4.4's causal scheme, indexes windows that no longer exist, and its chains "
                f"carry `windows`, not `blocks` (its `geometry` also has no `latent_time_scale`)"
            )
        self.subset = subset
        self.root = corpus_root
        self.objective = objective
        self.capture_bundle = dataset.capture_bundle_name(objective)
        self.guide_bundle = dataset.guide_bundle_name(objective)
        self.with_anchor = with_anchor
        self.with_guide = with_guide
        self.chains = [chain for chain in subset["chains"] if chain["split"] == split]
        if not self.chains:
            raise SystemExit(f"subset has no chains in split {split!r}")
        self.sources = {record["relative_dir"]: record for record in subset["sources"]}
        # The longest clip any chain in this subset can land on. The K/V cache is allocated
        # ONCE for the whole run, and its capacity is capped by the clip it is sized against
        # (causal_core.cache_latent_frames_for) -- so sizing it from whichever chain came
        # first would make the buffer depend on the shuffle. windows.py already records each
        # source's latent-frame count, read from the stored master, so this costs no I/O.
        self.max_latent_frames = max(
            int(record["n_latent_frames"]) for record in subset["sources"]
        )

    def __len__(self) -> int:
        return len(self.chains)

    def __getitem__(self, i: int) -> Chain:
        chain = self.chains[i]
        view = self.root / chain["source"]
        z_y, fps = _load_training_master(view / self.capture_bundle)
        # Guide-mode d0 never reads z_g (train_chain uses z_y as both source and target), so
        # skip requiring the guide bundle to exist for callers that only run d0 -- e.g. the
        # D0 sanity probe, which must work against process_gt_latent precompute output.
        z_g = None
        if self.with_guide:
            z_g, guide_fps = _load_training_master(view / self.guide_bundle)
            if z_g.shape != z_y.shape:
                raise ValueError(f"{chain['source']}: guide {tuple(z_g.shape)} != capture {tuple(z_y.shape)}")
            if fps != guide_fps:
                raise ValueError(f"{chain['source']}: guide fps {guide_fps} != capture fps {fps}")

        z0_base = None
        if self.with_anchor:
            base_path = view / "base_denoised.pt"
            z0_base = dataset.load_master(base_path, bundle=_load_record(base_path))

        return Chain(
            source=chain["source"],
            split=chain["split"],
            actor=chain["actor"],
            seed_is_clip_start=bool(chain["seed_is_clip_start"]),
            blocks=list(chain["blocks"]),
            z_g=z_g,
            z_y=z_y,
            fps=fps,
            z0_base=z0_base,
        )


# The one regression-loss identifier this package produces (2026-09-18 audit, binding
# decision: unweighted full-frame loss). Stamped into checkpoint metadata and run config.json
# rather than left implicit, so an artifact can be told apart from a hypothetical future loss
# convention by reading its own record instead of by the date it was written.
FULL_FRAME_X0_MSE = "full_frame_x0_mse"


def full_frame_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain mean squared error over every predicted token and channel.

    Training deliberately has no silhouette, alpha, or disagreement weighting: every pixel
    of the objective's continuous capture encode is part of the target. This is
    ``FULL_FRAME_X0_MSE``.
    """
    return (pred.float() - target.float()).pow(2).mean()


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


def assert_subset_matches_geometry(
    subset: dict,
    geometry: CausalGeometry,
    *,
    corpus_root: Path | None = None,
    objective: str = dataset.DEFAULT_OBJECTIVE,
    with_guide: bool = False,
) -> None:
    """Refuse an untrainable subset at STARTUP, in checks that catch different staleness.

    ``train_chain``/``ChainStore.__getitem__`` catch the same things per chain, but only once
    the run has paid the prompt cache, a 42 GB checkpoint load and ``accelerator.prepare`` --
    minutes per rank, to learn that the subset was never trainable. And per-chain is not even
    every SOURCE: ``--dry-run`` only ever draws ``store[0]``, so a subset whose first source is
    fine and whose Nth source is missing its guide master sails through a dry run and normal
    startup, and only fails when a rank happens to draw chain N -- which can be well after
    ``Accelerator()``, model load and FSDP `prepare` (2026-09-18 audit, Stage A gap 1). All
    checks here run before any of that, over EVERY selected source, not just the first drawn.

    1. **Internal** (free): every chain's blocks must exist in the plan implied by its
       source's recorded ``n_latent_frames``. Catches a ``--block-latent-frames`` that
       disagrees with the freeze.

    2. **Against the bundles** (``corpus_root`` given): require nonempty floating 4D masters,
       finite positive FPS, and common channel/spatial geometry across sources. Each source's
       recorded ``n_latent_frames`` must match what its master latent actually holds. This is the one
       that matters in practice, and (1) cannot see it -- a subset frozen before 2026-09-16
       is *internally* consistent, because ``windows.py`` sized both the count and the plan
       from the source VIDEO's length. A master consolidated out of v1 per-window slices stops
       at the last whole window and is short of the video by up to one window (a 150-frame clip
       stores 137 pixel frames = 18 latent, not 19), so the subset claims one block per source
       the latents do not contain. It is a stale artifact, not a geometry flag.

    3. **The guide master, for every source** (``with_guide`` -- i.e. ``--guide-mode d1``, the
       default): the guide bundle must exist, and must agree with the capture master on the
       FULL ``[C, F, H, W]`` shape (not just latent-frame count -- a guide re-encoded at a
       different resolution can match on frame count alone) and on fps, exactly what
       ``ChainStore.__getitem__`` otherwise only discovers the first time a rank draws that
       particular source's chain.

    (2) and (3) read bundles for every DISTINCT source (~6 MB each; disk-bound, minutes at
    full-corpus scale) -- ``--skip-subset-check`` opts out of both, at the cost of finding out
    at step 0 (or later, mid-run, for a source no early chain happens to draw) instead.
    """
    planned = {
        record["relative_dir"]: len(geometry.plan(int(record["n_latent_frames"])))
        for record in subset["sources"]
    }
    stale = sorted(
        {
            (chain["source"], planned[chain["source"]], max(chain["blocks"]))
            for chain in subset["chains"]
            if max(chain["blocks"]) >= planned[chain["source"]]
        }
    )
    if stale:
        listed = "\n  ".join(
            f"{source}: plans {blocks} blocks (0-{blocks - 1}), a chain asks for block {asked}"
            for source, blocks, asked in stale[:5]
        )
        raise SystemExit(
            f"{len(stale)} of {len(planned)} source(s) in this subset index blocks the causal "
            f"geometry {geometry.as_dict()} does not plan:\n  {listed}"
            + (f"\n  ... and {len(stale) - 5} more" if len(stale) > 5 else "")
            + f"\n\n{_REFREEZE_HINT}"
        )

    if corpus_root is None:
        return
    bundle_name = dataset.capture_bundle_name(objective)
    guide_name = dataset.guide_bundle_name(objective)
    drifted: list[tuple[str, int, int]] = []
    guide_problems: list[str] = []
    spatial_shape: tuple[int, int, int] | None = None
    for record in subset["sources"]:
        bundle = corpus_root / record["relative_dir"] / bundle_name
        if not bundle.is_file():
            raise SystemExit(
                f"{bundle} does not exist, but the subset lists {record['relative_dir']} as a "
                f"source. Re-run `precompute.py --process_gt_latent --objective {objective}` for it, "
                f"or re-freeze the subset against what is actually on disk"
            )
        capture, capture_fps = _load_training_master(bundle)
        actual = capture.shape[1]
        current_shape = (capture.shape[0], capture.shape[2], capture.shape[3])
        if spatial_shape is not None and current_shape != spatial_shape:
            raise SystemExit(
                f"{bundle}: capture [C, H, W] {current_shape} != {spatial_shape}; "
                "all sources must share one channel/spatial geometry for the training cache"
            )
        spatial_shape = current_shape
        if actual != int(record["n_latent_frames"]):
            drifted.append((record["relative_dir"], int(record["n_latent_frames"]), actual))

        if with_guide:
            guide_path = corpus_root / record["relative_dir"] / guide_name
            if not guide_path.is_file():
                raise SystemExit(
                    f"{guide_path} does not exist, but the subset lists {record['relative_dir']} "
                    f"as a source and --guide-mode d1 needs its guide master. Re-run "
                    f"`precompute.py --objective {objective}` (the paired pass) for it, or "
                    f"re-freeze the subset without this source, or pass --guide-mode d0"
                )
            guide, guide_fps = _load_training_master(guide_path)
            if guide.shape != capture.shape:
                guide_problems.append(
                    f"{record['relative_dir']}: guide master ({guide_name}) shape "
                    f"{tuple(guide.shape)} != capture master ({bundle_name}) shape {tuple(capture.shape)}"
                )
            elif capture_fps != guide_fps:
                guide_problems.append(
                    f"{record['relative_dir']}: guide fps {guide_fps} != capture fps {capture_fps}"
                )
    if drifted:
        listed = "\n  ".join(
            f"{source}: subset says {recorded} latent frames, the master holds "
            f"{actual} "
            f"({len(geometry.plan(recorded))} blocks frozen vs "
            f"{len(geometry.plan(actual)) if actual else 0} real)"
            for source, recorded, actual in drifted[:5]
        )
        raise SystemExit(
            f"{len(drifted)} of {len(planned)} source(s) have master latents that disagree with "
            f"what this subset was frozen against:\n  {listed}"
            + (f"\n  ... and {len(drifted) - 5} more" if len(drifted) > 5 else "")
            + f"\n\n{_REFREEZE_HINT}"
        )
    if guide_problems:
        listed = "\n  ".join(guide_problems[:5])
        raise SystemExit(
            f"{len(guide_problems)} of {len(planned)} source(s) have a guide master that "
            f"disagrees with its capture master -- ChainStore would otherwise only discover "
            f"this the first time a rank drew that source's chain:\n  {listed}"
            + (f"\n  ... and {len(guide_problems) - 5} more" if len(guide_problems) > 5 else "")
            + f"\n\nRe-run `precompute.py --objective {objective}` (the paired pass) for the "
            f"affected source(s), or re-freeze the subset without them."
        )


_REFREEZE_HINT = (
    "If you did not change --block-latent-frames, this is a SUBSET frozen before 2026-09-16, "
    "when windows.py sized its block plan from the source VIDEO rather than from the stored "
    "master latent (a consolidated master ends at the last whole window, so it is short of the "
    "video by up to one window). Re-freeze it:\n"
    "  python -m scripts.onestep_avatar.windows --name <name> [--max-actors N] "
    "--require-guide --chain-length K --min-holdout-actors M\n"
    "The tail frames are genuinely absent from the latents -- re-freezing makes the subset "
    "honest, it does not recover them."
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

    ``prime_cache`` is called for EVERY chain, unconditionally. A chain that starts at block 0
    has nothing to prime and the call writes nothing -- but it still forwards, because a
    forward count that depends on the data desynchronises FSDP's collectives and hangs the
    job. See ``causal_core.prime_cache``'s empty-spans branch; ``assert_rank_lockstep`` counts
    on that call being unconditional here.
    """
    device = accelerator.device
    grid = clip_grid_for(chain, geometry, device=device, latent_channels=latent_channels)
    plan = geometry.plan(grid.latent_frames)
    if max(chain.blocks) >= len(plan):
        raise ValueError(
            f"{chain.source}: chain asks for block {max(chain.blocks)} but this clip's master "
            f"latent ({grid.latent_frames} latent frames) plans only {len(plan)} blocks "
            f"(0-{len(plan) - 1}) under {geometry.as_dict()}. Either --block-latent-frames "
            f"differs from the freeze, or -- far more likely -- the subset predates 2026-09-16 "
            f"and was frozen against the source VIDEO's length rather than the stored master's. "
            f"Re-freeze it with windows.py (`assert_subset_matches_geometry` catches this at "
            f"startup for a subset whose recorded n_latent_frames is itself stale)"
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
    elif not cache.fits(grid.latent_frames):
        # Checked HERE rather than left to LayerKVCache.write: an overflow raises inside one
        # rank's forward, and a rank that leaves a forward early has issued one round of
        # all-gathers fewer than the others -- the FSDP desynchronisation assert_rank_lockstep
        # and prime_cache's empty-spans branch both exist to prevent. This raise happens
        # before any forward of the step, where it is still a clean failure on every rank.
        raise ValueError(
            f"{chain.source}: a {grid.latent_frames}-latent-frame clip needs a "
            f"{geometry.cache_latent_frames_for(grid.latent_frames)}-frame K/V cache but the "
            f"run allocated {cache.caches[0].capacity // grid.tokens_per_latent_frame} frames; "
            f"the cache must be sized from the subset's LONGEST clip, not from one chain's"
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

        mse = full_frame_mse(z0, target_tokens[:, lo:hi])
        loss = mse
        anchor = torch.zeros((), device=device)
        if anchor_weight > 0.0:
            if base_tokens is None:
                raise ValueError("--anchor-weight > 0 but this view has no base_denoised.pt")
            # SS4.3 row 2 / SS2.3(3): the risk here is ERODING sharpness Phi already has, not
            # failing to synthesise it. Pulling toward the frozen model's own output on the
            # same input is the cheapest thing that targets that directly.
            anchor = full_frame_mse(z0, base_tokens[:, lo:hi])
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
        del z0, loss, mse, anchor
    totals["per_block"] = per_block
    return totals


def _num_blocks(transformer: torch.nn.Module) -> int:
    return len(causal_core.base_model(transformer).transformer_blocks)


def _inner_dim(transformer: torch.nn.Module) -> int:
    return causal_core.base_model(transformer).inner_dim


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

    ``onestep_avatar_loss`` records ``FULL_FRAME_X0_MSE`` explicitly (2026-09-18 audit, gap 2)
    rather than leaving the loss convention implicit: the arithmetic has been full-frame MSE
    since before this field existed, but nothing on a saved artifact said so, and a reader
    (or a future loss change) had no field to check against.
    """
    sigma_levels = training_sigmas(args)
    geometry = causal_geometry(args, model)
    return {
        # ``mixed`` deliberately prevents a fixed-sigma deployment loader from accepting a
        # multi-level adapter as though it were calibrated for just one noise level.
        "onestep_avatar_loss": FULL_FRAME_X0_MSE,
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
        import wandb  # noqa: PLC0415 -- optional dependency, imported only when requested.
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


def archive_existing_run(output: Path) -> Path | None:
    """Move everything already in ``output`` aside before ``--overwrite`` reuses the directory.

    The old behaviour deleted only ``metrics_rank*.jsonl``, so a relaunch's fresh checkpoints
    and probes landed beside a prior run's under the same directory name with nothing to say
    which run produced which (F6). Archiving the whole directory into one timestamped
    subdirectory keeps that provenance intact and recoverable instead of silently mixed or
    discarded. Returns ``None`` (and does nothing) if there is nothing to move.
    """
    entries = [entry for entry in output.iterdir() if not entry.name.startswith("archived_")]
    if not entries:
        return None
    archive_dir = output / f"archived_{time.strftime('%Y%m%d_%H%M%S')}"
    archive_dir.mkdir()
    for entry in entries:
        entry.rename(archive_dir / entry.name)
    return archive_dir


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
    p.add_argument(
        "--overwrite", action="store_true",
        help="Relaunch into a --output that already holds a run: the whole prior directory "
        "(metrics, checkpoints, config.json, everything) is moved aside into an "
        "'archived_<timestamp>/' subdirectory before this run creates anything, rather than "
        "being deleted or left to coexist under one step numbering. There is no resume -- step "
        "always restarts at 0 -- so a used --output is otherwise refused outright.",
    )
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
        help="Save checkpoints at step 0 and step 1, independent of --save-every. Step 0 is "
        "the untrained, LoRA-injected model; `init_lora_weights=True` zero-inits B, so it "
        "must decode identically to the frozen base. Step 1 is the first optimizer update. "
        "Off by default: normal runs do not need these extra checkpoint writes.",
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
    p.add_argument(
        "--skip-subset-check",
        action="store_true",
        help="Skip the startup check that each source's master latent holds the number of "
        "frames the subset was frozen against. That check reads one ~6 MB bundle per distinct "
        "source -- negligible for a review tier, disk-bound at full-corpus scale. Skipping it "
        "does not make a stale subset trainable; it moves failures to chain loading, after "
        "the 42 GB checkpoint load.",
    )
    p.add_argument("--dry-run", action="store_true", help="report the plan and the data shapes, load no model")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915 -- one linear training script.
    args = parse_args(argv)
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_rank
    sigmas = training_sigmas(args)
    if args.anchor_weight != 0.0:
        # 2026-09-18 audit F8: no `base_denoised.pt` producer exists anywhere in the corpus,
        # and even if one did, a single frozen per-view tensor cannot BE "the anchor" for every
        # chain that reaches that view -- noise depends on chain index, sigma can vary between
        # ranks/steps, and primed vs. clip-start chains carry different history into the same
        # block. Refuse before the 42 GB checkpoint load rather than failing per-chain inside
        # `ChainStore`/`train_chain` once a run is already minutes into startup.
        raise SystemExit(
            "nonzero --anchor-weight is disabled: the anchor path is unsupported under the current "
            "contract (see plans/2026-09-18-onestep-avatar-audit-and-fix-plan.md F8) -- no "
            "`base_denoised.pt` producer exists, and a fixed per-view tensor cannot represent "
            "the anchor for every chain/sigma/history combination that would read it. Train "
            "with --anchor-weight 0.0 (the default) until the offline-teacher objective this "
            "flag was for is actually defined and produced."
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # S3b of the 2026-09-17 cleanup plan, revised by the 2026-09-18 audit's F6: a used --output
    # is refused rather than silently merged, exactly as before. What changed is WHEN the used
    # directory is actually touched -- see `needs_archive` below, and its consumption right
    # before this run's own config.json/log file are created. This check itself is read-only
    # (a filesystem glob), so every rank evaluating it identically before `Accelerator()` costs
    # nothing and cannot itself race.
    if args.output.exists() and not args.output.is_dir():
        raise SystemExit(f"{args.output}: --output must be a directory")
    existing_entries = sorted(args.output.glob("*")) if args.output.is_dir() else []
    if existing_entries and not args.overwrite:
        raise SystemExit(
            f"{args.output} already has {len(existing_entries)} entr{'y' if len(existing_entries) == 1 else 'ies'} "
            f"from a previous launch, and this run's config.json/logs would land beside them. "
            f"train.py has no resume -- step restarts at 0 every launch -- so writing into a "
            f"used directory would either silently merge two runs under one step numbering "
            f"(metrics) or misattribute old checkpoints/probes to this run. Pass --overwrite to "
            f"archive the existing directory (moved aside, not deleted) before this run starts."
        )
    needs_archive = bool(existing_entries) and args.overwrite

    subset = json.loads(args.subset.read_text())
    # The block-chain `kind` check lives in ChainStore.__init__ now (S2 of the 2026-09-17
    # cleanup plan), so both readers of the subset contract -- this one and visualize_d0.py's
    # direct ChainStore construction -- get the same refusal instead of one of them raising a
    # raw KeyError deep inside geometry setup.
    #
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
        with_anchor=args.anchor_weight > 0.0,
        with_guide=args.guide_mode != "d0",
    )

    # Before the Accelerator, the prompt cache and the 42 GB checkpoint: a subset that cannot
    # be trained should cost seconds to find out, not a full startup on every rank.
    assert_subset_matches_geometry(
        subset,
        geometry,
        corpus_root=None if args.skip_subset_check else corpus_root,
        objective=args.objective,
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

    if needs_archive:
        # F6: archiving (not deleting) happens here rather than at argument-parsing time, for
        # two reasons. First, everything above this line -- subset validation, `--dry-run` --
        # is now side-effect free: a bad subset or a dry run leaves a used --output untouched,
        # where the old code deleted metrics_rank*.jsonl unconditionally before either check
        # ran. Second, `Accelerator()` now exists, so ranks can be coordinated: every process
        # races on the SAME preflight (the glob above), but only the main process touches the
        # filesystem, bracketed by barriers so no rank opens its log file into a directory
        # still being archived and no rank starts training before the archive is visible to it.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            archived = archive_existing_run(args.output)
            if archived is not None:
                LOGGER.info("archived previous run to %s", archived)
        accelerator.wait_for_everyone()

    args.output.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        (args.output / "config.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "world_size": world,
                    "causal_geometry": geometry.as_dict(),
                    "loss": FULL_FRAME_X0_MSE,
                },
                indent=2,
                default=str,
            )
        )
    wandb_run = init_wandb(
        args,
        config={
            **vars(args),
            "world_size": world,
            "sigma_levels": list(sigmas),
            "loss": FULL_FRAME_X0_MSE,
            **geometry.as_dict(),
        },
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
                        grid, geometry, num_layers=num_blocks, inner_dim=inner_dim, device=device,
                        dtype=DTYPE,
                        # The subset's longest clip, not this chain's: one allocation serves
                        # every chain in the run, so a capacity capped by the first clip drawn
                        # would overflow on a longer one at a deep --context-latent-frames.
                        capacity_latent_frames=store.max_latent_frames,
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
                    # 0.0 rather than float(None): accelerate's clip_grad_norm_ returns None
                    # for some distributed types, and rank_mean is a COLLECTIVE -- a TypeError
                    # on one rank here would hang the others in the gather it never joins.
                    mean_loss, mean_mse, mean_anchor, mean_grad_norm = rank_mean(
                        accelerator,
                        [
                            totals["loss"], totals["mse"], totals["anchor"],
                            float(grad_norm) if grad_norm is not None else 0.0,
                        ],
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
                                **{f"train/block_{i}_mse": value for i, value in enumerate(block_mse)},
                            },
                            step=step,
                        )
                if accelerator.is_main_process:
                    LOGGER.info(
                        "step %d/%d loss %.5f mse %.5f anchor %.5f lr %.2e %.1fs",
                        step, args.steps, totals["loss"], totals["mse"], totals["anchor"], lr,
                        time.time() - started,
                    )
            if step % args.save_every == 0 or step == args.steps or (args.save_initial and step == 1):
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
