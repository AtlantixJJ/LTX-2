"""Shared distributed LoRA setup and updates for avatar training.

See doc/training/engine.md. Mode and data integration remain in progress;
this extraction preserves the original training calculations and save order.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from peft.utils.other import fsdp_auto_wrap_policy

from ltx_trainer.model_loader import load_transformer
from scripts.onestep_avatar import windows
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.corpus import subset as video_lists
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters, backbone, bidirectional, common
from scripts.onestep_avatar.model import causal as causal_core
from scripts.onestep_avatar.model.causal import BlockCache, CausalGeometry
from scripts.onestep_avatar.model.common import FULL_FRAME_X0_MSE, ClipGrid
from scripts.onestep_avatar.training import checkpoints as sampling
from scripts.onestep_avatar.training import numerics, resources, runtime
from scripts.onestep_avatar.training.checkpoints import load_stage_init, save_lora
from scripts.onestep_avatar.training.config import (
    SIGMA_SAMPLING,
    BidirectionalSettings,
    CausalSettings,
    RunSettings,
    parse_args,
    select_frame_plan,
    sigma_for_rank,
    training_noise_seed,
    training_sigmas,
)
from scripts.prune.core import model_registry
from scripts.prune.core.session import DEFAULT_PROMPT
from scripts.prune.data import prompt_cache

LOGGER = logging.getLogger("onestep_avatar.train")


def _launch_evidence(settings: RunSettings, job_digest: str | None) -> tuple[dict | None, Path | None]:
    """Read dispatch authority before model setup; never infer a missing launch."""
    from scripts.onestep_avatar.execution import queue  # noqa: PLC0415 -- public canonical launch authority
    from scripts.onestep_avatar.execution.queue_protocol import LAUNCH_ENV  # noqa: PLC0415 -- model-free protocol
    from scripts.onestep_avatar.training.config import parse_settings  # noqa: PLC0415

    path = os.environ.get(LAUNCH_ENV)
    if path is None:
        if job_digest is not None:
            raise ValueError("queued training requires original dispatch launch evidence")
        return None, None
    launch = queue.read_training_launch(Path(path))
    if launch["schema_version"] != 2:
        raise ValueError("current typed training requires schema-two numerical launch evidence")
    numerics.require_environment(launch["numerical_environment"])
    if launch["job"]["sha256"] != job_digest:
        raise ValueError("dispatch launch identity differs from queued attempt")
    requested = parse_settings(launch["job"]["arguments"])
    ignored = {"base_identity", "parent_contract", "world_size", "preview_record"}
    def normalized(value: object) -> object:
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items() if key not in ignored}
        return value
    if normalized(requested.as_dict()) != normalized(settings.as_dict()):
        raise ValueError("dispatch training arguments differ from actual settings")
    return launch, Path(path).resolve()


def _check_launch_current(launch: dict | None, path: Path | None) -> None:
    if launch is not None:
        from scripts.onestep_avatar.execution import queue  # noqa: PLC0415 -- same preflight/publication authority
        if queue.read_training_launch(path) != launch:
            raise ValueError("original dispatch launch changed during training")


def _launch_precision(launch: dict) -> str:
    import base64  # noqa: PLC0415 -- original bytes, never today's requested precision

    import yaml  # noqa: PLC0415 -- installed Accelerate YAML dependency
    return yaml.safe_load(base64.b64decode(launch["accelerate_config_bytes_base64"]))["mixed_precision"]


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

# Two named target sets. "attn" is the trainer's own default projection set; "attn_ffn" adds
# the feed-forward projections, which is the A2 sweep's second axis (SS A2 "attn-only vs
# attn+FFN"). Named here rather than passed as a free list so a run's arm is one word in the
# checkpoint metadata.


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


def window_start_for(seed: int, *, step: int, rank: int, slot: int, latent_frames: int, window: int) -> int:
    """The latent frame a random-window sample starts at: i.i.d. uniform on ``0 .. F - window``.

    Seeded by (noise seed, update, rank, slot) through its own generator, so it is reproducible
    and independent of the sigma draw and the epsilon stream.
    """
    if not 1 <= window <= latent_frames:
        raise ValueError(f"window of {window} latent frames does not fit a {latent_frames}-frame clip")
    rng = random.Random(f"onestep_avatar.window:{seed}:{step}:{rank}:{slot}")
    return rng.randrange(latent_frames - window + 1)


def window_chain(chain: Chain, start: int, window: int) -> Chain:
    """``chain`` restricted to latent frames ``[start, start + window)`` of its masters.

    The window is treated as a clip of its own: its frame 0 becomes ``c0`` (``train_chain``
    reads ``c0`` from token frame 0 of the target) and RoPE positions restart at 0, as for a
    supplied first image at deployment. For ``start > 0`` that ``c0`` is a stored latent
    encoding 8 pixel frames, not the single-frame keyframe a real first image encodes to
    (known gap G9).
    """
    if start < 0 or start + window > chain.z_y.shape[1]:
        raise ValueError(f"{chain.source}: window [{start}, {start + window}) outside {chain.z_y.shape[1]} frames")

    def cut(master: torch.Tensor | None) -> torch.Tensor | None:
        return None if master is None else master[:, start : start + window].contiguous()

    return replace(chain, z_y=cut(chain.z_y), z_g=cut(chain.z_g))


def check_random_window(args: argparse.Namespace, store: "ChainStore") -> None:
    """Refuse a random window the subset or geometry cannot honour, before any model load.

    The window must be exactly the trained span -- one whole-clip block plus ``c0``
    (``--block-latent-frames + 1``) -- and only whole-clip, one-block, clip-start chains are
    windowed, so the FSDP forward count and the placeholder cache stay as for ``s = 0``.
    Every source must hold at least the window.
    """
    window = args.random_window_latent_frames
    if window is None:
        return
    if window != args.block_latent_frames + 1:
        raise SystemExit(
            f"--random-window-latent-frames {window} must equal --block-latent-frames + 1 "
            f"({args.block_latent_frames + 1}): the window is the one whole-clip block plus c0"
        )
    if not all(len(chain["blocks"]) == 1 and chain["seed_is_clip_start"] for chain in store.chains):
        raise SystemExit("--random-window-latent-frames needs a clip-start, one-block-per-chain (K=1) subset")
    short = [rec["relative_dir"] for rec in store.subset["sources"] if int(rec["n_latent_frames"]) < window]
    if short:
        raise SystemExit(f"{len(short)} source(s) shorter than the {window}-frame window, e.g. {short[0]}")


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
        self.max_latent_frames = max(int(record["n_latent_frames"]) for record in subset["sources"])

    def __len__(self) -> int:
        return len(self.chains)

    def __getitem__(self, i: int) -> Chain:
        chain = self.chains[i]
        view = self.root / chain["source"]
        z_y, fps = dataset.load_training_master(view / self.capture_bundle)
        # Guide-mode d0 never reads z_g (train_chain uses z_y as both source and target), so
        # skip requiring the guide bundle to exist for callers that only run d0 -- e.g. the
        # D0 sanity probe, which must work against process_gt_latent precompute output.
        z_g = None
        if self.with_guide:
            z_g, guide_fps = dataset.load_training_master(view / self.guide_bundle)
            if z_g.shape != z_y.shape:
                raise ValueError(f"{chain['source']}: guide {tuple(z_g.shape)} != capture {tuple(z_y.shape)}")
            if fps != guide_fps:
                raise ValueError(f"{chain['source']}: guide fps {guide_fps} != capture fps {fps}")

        return Chain(
            source=chain["source"],
            split=chain["split"],
            actor=chain["actor"],
            seed_is_clip_start=bool(chain["seed_is_clip_start"]),
            blocks=list(chain["blocks"]),
            z_g=z_g,
            z_y=z_y,
            fps=fps,
        )


# The one regression-loss identifier this package produces (2026-09-18 audit, binding
# decision: unweighted full-frame loss). Stamped into checkpoint metadata and run config.json
# rather than left implicit, so an artifact can be told apart from a hypothetical future loss
# convention by reading its own record instead of by the date it was written.


def clip_grid_for(chain: Chain, geometry: CausalGeometry, *, device: torch.device, latent_channels: int) -> ClipGrid:
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

    The count is ``1 prime + K denoise + (K - 1) refresh``. The last refresh is skipped.
    Every rank must compute the same count.
    """
    if accelerator.num_processes == 1:
        return
    counts = accelerator.gather(torch.tensor([planned_forwards], device=accelerator.device, dtype=torch.long)).tolist()
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
        record["relative_dir"]: len(geometry.plan(int(record["n_latent_frames"]))) for record in subset["sources"]
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
        capture, capture_fps = dataset.load_training_master(bundle)
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
            guide, guide_fps = dataset.load_training_master(guide_path)
            if guide.shape != capture.shape:
                guide_problems.append(
                    f"{record['relative_dir']}: guide master ({guide_name}) shape "
                    f"{tuple(guide.shape)} != capture master ({bundle_name}) shape {tuple(capture.shape)}"
                )
            elif capture_fps != guide_fps:
                guide_problems.append(f"{record['relative_dir']}: guide fps {guide_fps} != capture fps {capture_fps}")
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


def train_chain(  # noqa: PLR0913 -- temporary original caller interface during migration
    transformer: torch.nn.Module,
    context: torch.Tensor,
    chain: Chain,
    geometry: CausalGeometry,
    cache: BlockCache | None,
    accelerator: Accelerator,
    *,
    sigma0: float,
    seed: int,
    latent_channels: int,
    guide_mode: str = "d1",
    teacher_forcing: bool = False,
    timing: bool = False,
    accumulation: int = 1,
) -> dict:
    """Temporary old-sample adapter; model execution belongs entirely to the causal mode.

    Remove this interface when engine membership/frame-plan integration completes.
    """
    grid = clip_grid_for(chain, geometry, device=accelerator.device, latent_channels=latent_channels)
    capture = grid.patchify(chain.z_y.unsqueeze(0).to(device=accelerator.device, dtype=DTYPE))
    guide = (
        None if chain.z_g is None else grid.patchify(chain.z_g.unsqueeze(0).to(device=accelerator.device, dtype=DTYPE))
    )
    result = causal_core.train_sample(
        transformer,
        context,
        grid,
        capture,
        guide,
        geometry,
        chain.blocks,
        accelerator.backward,
        sigma=sigma0,
        seed=seed,
        cache=cache,
        guide_mode=guide_mode,
        teacher_forcing=teacher_forcing,
        timing=timing,
        accumulation=accumulation,
    )
    result = {key: result[key] for key in ("loss", "mse", "per_block")}
    return result


def _num_blocks(transformer: torch.nn.Module) -> int:
    return len(common.base_model(transformer).transformer_blocks)


def _inner_dim(transformer: torch.nn.Module) -> int:
    return common.base_model(transformer).inner_dim


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
    # PEFT draws lora_A from the global RNG. Seeding it identically on every rank makes the
    # initialization one reproducible function of --init-seed instead of a per-rank mixture.
    transformer = adapters.attach(transformer, rank=args.lora_rank, alpha=args.lora_alpha,
                                  target=args.lora_target, init_seed=args.init_seed)
    if args.init_adapter is not None:
        load_stage_init(transformer, args.init_adapter)
    if accelerator.distributed_type == DistributedType.FSDP:
        # FSDP needs one dtype per flat parameter, and PEFT makes the adapters fp32 against a
        # bf16 base. This policy wraps the trainable leaves separately, which is what lets the
        # base stay bf16 instead of being promoted to a full fp32 host copy before sharding.
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(transformer)
        if isinstance(args, RunSettings):
            plugin = accelerator.state.fsdp_plugin
            from torch.distributed.fsdp.wrap import CustomPolicy  # noqa: PLC0415 -- typed FSDP only

            peft_policy = plugin.auto_wrap_policy

            def typed_policy(module: torch.nn.Module) -> bool | dict:
                if not peft_policy(module=module, recurse=False, nonwrapped_numel=0):
                    return False
                parameters = list(module.parameters())
                if parameters and all(parameter.requires_grad for parameter in parameters):
                    return {"mixed_precision": None}
                return True

            plugin.auto_wrap_policy = CustomPolicy(typed_policy)
            if plugin.mixed_precision_policy is not None:
                # FSDP otherwise recursively rounds sigma/timesteps/positions in Modality.
                plugin.mixed_precision_policy = replace(
                    plugin.mixed_precision_policy, cast_root_forward_inputs=False
                )
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

    sigma_0, ``K`` and now the **causal geometry** are recorded so ``sampling``'s ``ONE_STEP``
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
    base = args.base_identity
    return {
        "onestep_avatar_base_variant": base["base_variant"],
        "onestep_avatar_base_transformer_file": base["base_transformer_file"],
        "onestep_avatar_base_transformer_fingerprint": base["base_transformer_fingerprint"],
        # How history K/V are computed: the cached clean refresh at global sigma zero (G7).
        "onestep_avatar_history_computation": "cached_refresh_global_sigma0",
        "onestep_avatar_subset_full_sha256": windows.subset_sha256(subset),
        "onestep_avatar_noise_policy": args.noise_policy,
        "onestep_avatar_init_seed": str(args.init_seed),
        "onestep_avatar_data_seed": str(args.data_seed),
        "onestep_avatar_noise_seed": str(args.noise_seed),
        "onestep_avatar_chains_per_update": str(args.chains_per_rank * args.world_size),
        "onestep_avatar_parent_adapter": "" if args.init_adapter is None else str(args.init_adapter),
        # ``mixed`` deliberately prevents a fixed-sigma deployment loader from accepting a
        # multi-level adapter as though it were calibrated for just one noise level.
        "onestep_avatar_loss": FULL_FRAME_X0_MSE,
        "onestep_avatar_sigma0": repr(args.sigma0) if args.sigma_levels is None else "mixed",
        "onestep_avatar_sigma_levels": ",".join(repr(sigma) for sigma in sigma_levels),
        "onestep_avatar_sigma_sampling": SIGMA_SAMPLING,
        "onestep_avatar_window": (
            "clip_start"
            if getattr(args, "random_window_latent_frames", None) is None
            else f"random_start_v1:{args.random_window_latent_frames}"
        ),
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
        "onestep_avatar_teacher_forcing": str(args.teacher_forcing),
        "onestep_avatar_first_frame_conditioning": "clean_c0_v1",
        "model_key": model.key,
        "lora_rank": str(args.lora_rank),
        "lora_alpha": str(args.lora_alpha),
        "lora_target": args.lora_target,
        "step": str(step),
    }


# How a multi-level run picks each sample's noise level. Stamped into checkpoint metadata and
# config.json. Runs before 2026-10-05 used the deterministic rotation
# ``sigmas[(rank + step) % len(sigmas)]`` instead (no stamp); that rule is retired.


def wandb_is_enabled(args: argparse.Namespace) -> bool:
    """True if W&B logging is enabled for this run."""
    if getattr(args, "no_wandb", False):
        return False
    if not args.wandb_project or str(args.wandb_project).strip().lower() in ("", "none"):
        return False
    return getattr(args, "wandb_mode", "online") != "disabled"


def init_wandb(args: argparse.Namespace, *, config: dict) -> object | None:
    """Create one online W&B run on rank 0; all ranks still participate in metric gathers."""
    if not wandb_is_enabled(args):
        return None
    try:
        import wandb  # noqa: PLC0415 -- optional dependency, imported only when requested.
    except ImportError as exc:  # pragma: no cover - environment/setup error
        raise SystemExit(
            f"W&B logging is enabled (project: {args.wandb_project!r}), but the wandb package "
            "is not installed in the active environment. Install wandb or pass --no-wandb."
        ) from exc
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


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915 -- one linear training script.
    args = parse_args(argv)
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_rank
    for name in ("init_seed", "data_seed", "noise_seed"):
        if getattr(args, name) is None:
            setattr(args, name, args.seed)
    if args.chains_per_rank < 1:
        raise SystemExit("--chains-per-rank must be >= 1")
    if args.init_adapter is not None and not args.init_adapter.is_file():
        raise SystemExit(f"--init-adapter does not exist: {args.init_adapter}")
    sigmas = training_sigmas(args)
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
    model = backbone.resolve(args.model, args.variant)
    args.base_identity = backbone.identity(model.paths.transformer(), args.variant, args.model)
    if args.init_adapter is not None:
        parent = sampling.read_adapter_metadata(args.init_adapter)
        if (
            parent.get("onestep_avatar_base_transformer_fingerprint")
            != args.base_identity["base_transformer_fingerprint"]
        ):
            raise SystemExit(f"--init-adapter {args.init_adapter} was trained on different base weights")
        if (parent.get("lora_rank"), parent.get("lora_target")) != (str(args.lora_rank), args.lora_target):
            raise SystemExit(f"--init-adapter {args.init_adapter} has a different LoRA rank/target")
    geometry = causal_geometry(args, model)
    corpus_root = args.corpus_root or Path(subset["corpus_root"])
    store = ChainStore(
        subset,
        corpus_root,
        split=args.split,
        objective=args.objective,
        with_guide=args.guide_mode != "d0",
    )
    check_random_window(args, store)

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
                    "base": args.base_identity,
                    "subset_full_sha256": windows.subset_sha256(subset),
                    "noise_policy": args.noise_policy,
                    "seeds": {"init": args.init_seed, "data": args.data_seed, "noise": args.noise_seed},
                    "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "target": args.lora_target},
                },
                indent=2,
            )
        )
        return 0

    if args.reserve_gpu_gib > 0:
        # Operational, not experimental: on a shared machine another job's scheduler can see
        # this card as free during the minutes-long 42 GB load and claim it, OOM-ing this rank
        # at start-up. Allocating and freeing a block leaves it RESERVED in PyTorch's caching
        # allocator (nothing here calls empty_cache), so the card reads as taken from the first
        # second; the load and training then reuse that reserved memory.
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        reserve = torch.empty(int(args.reserve_gpu_gib * 2**30), dtype=torch.uint8, device=f"cuda:{local}")
        del reserve
    # No explicit mixed_precision: the accelerate config decides, and the 2/3-GPU configs are
    # copies of the trainer's own, so this loop runs under the same policy the shipped trainer
    # does rather than a second one of its own.
    with timed("Accelerator() / process group"):
        accelerator = Accelerator()
    device = accelerator.device
    world, rank = accelerator.num_processes, accelerator.process_index
    args.world_size = world

    with timed("prompt cache (text encoder)"):
        context = prompt_cache.get_or_build(model, DEFAULT_PROMPT, DTYPE, device)

    with timed("transformer load + LoRA injection"):
        transformer = build_transformer(model, args, accelerator)
    trainable = [p for p in transformer.parameters() if p.requires_grad]
    # Log what LoRA actually attached to: a video-only model whose target names silently
    # missed a projection would otherwise train a smaller adapter than the run claims.
    lora_modules = sorted(
        {name.rsplit(".lora_", 1)[0] for name, p in transformer.named_parameters() if p.requires_grad}
    )
    target_counts = {
        target: sum(1 for name in lora_modules if name.endswith(target))
            for target in adapters.LORA_TARGETS[args.lora_target]
    }
    # Counted BEFORE `prepare`: FSDP with `use_orig_params=True` reshapes each parameter to
    # this rank's shard in place, so the same expression afterwards reports total/world_size
    # and reads like a model half the size.
    trainable_total = sum(p.numel() for p in trainable)
    num_blocks, inner_dim = _num_blocks(transformer), _inner_dim(transformer)
    # Stated explicitly rather than inherited from the library defaults (they coincide today).
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
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
        LOGGER.info("LoRA modules: %d (%s)", len(lora_modules), json.dumps(target_counts))
        LOGGER.info("base: %s", json.dumps(args.base_identity))

    # Chains are sharded by rank rather than by an accelerate DataLoader: a sample here is a
    # variable-length chain of tensors, not a collatable batch, and FSDP is data-parallel over
    # ranks, so a deterministic stride is both simpler and reproducible with no sampler state.
    # Every rank runs the SAME number of steps, so the shard is truncated to the common length.
    # Small tiers (the two-pair overfit control) can have fewer chains than one update needs.
    # Tile the chain list so every rank gets a chain: under the fresh noise policy each copy
    # still draws its own epsilon (the rank is in the seed), so a tiled update is several
    # noise draws of the same chains, not duplicated samples. Recorded as `chain_tiling`.
    tiling = max(1, math.ceil(world * args.chains_per_rank / len(store)))
    per_rank = len(store) * tiling // world
    per_rank -= per_rank % args.chains_per_rank
    if per_rank == 0:
        raise SystemExit(
            f"{len(store)} chains cannot be split across {world} ranks x {args.chains_per_rank} chains per rank"
        )
    order = list(range(len(store))) * tiling

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
                    "sigma_sampling": SIGMA_SAMPLING,
                    "subset_full_sha256": windows.subset_sha256(subset),
                    "lora_modules": len(lora_modules),
                    "lora_target_counts": target_counts,
                    "trainable_params": trainable_total,
                    "chains_per_update": world * args.chains_per_rank,
                    "chain_tiling": tiling,
                    "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
                },
                indent=2,
                default=str,
            )
        )
    wandb_run = (
        init_wandb(
            args,
            config={
                **vars(args),
                "world_size": world,
                "sigma_levels": list(sigmas),
                "loss": FULL_FRAME_X0_MSE,
                **geometry.as_dict(),
            },
        )
        if accelerator.is_main_process
        else None
    )
    log_path = args.output / f"metrics_rank{rank}.jsonl"
    log_file = log_path.open("a")

    if args.save_initial:
        with timed("save_initial (step-0 adapter)"):
            path = save_lora(
                transformer,
                accelerator,
                args.output / "checkpoints",
                0,
                checkpoint_metadata(args, subset, model, 0),
                verify_noop=True,
            )
        if path is not None:
            LOGGER.info("saved initial (untrained) checkpoint %s", path)

    generator = torch.Generator().manual_seed(args.data_seed)
    single_block_chains = all(len(chain["blocks"]) == 1 and chain["seed_is_clip_start"] for chain in store.chains)
    # One cache allocation for the whole run: capacity depends only on the geometry and the
    # (fixed) 1024**2 crop, so reallocating per chain would just churn ~2 GB of VRAM.
    cache: BlockCache | None = None
    step = 0
    started = time.time()
    while step < args.steps:
        epoch_order = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]
        shard = epoch_order[rank * per_rank : (rank + 1) * per_rank]
        groups = [shard[i : i + args.chains_per_rank] for i in range(0, len(shard), args.chains_per_rank)]
        for group_indices in groups:
            if step >= args.steps:
                break
            lr = args.lr * min(1.0, (step + 1) / max(args.warmup_steps, 1))
            for group in optimizer.param_groups:
                group["lr"] = lr
            sigma0 = sigma_for_rank(sigmas, rank, step, args.noise_seed)

            step_started = time.time()
            step_totals: dict[str, float] = {"loss": 0.0, "mse": 0.0}
            per_block: list[dict[str, float]] = []
            sources: list[str] = []
            window_starts: list[int] = []
            for slot, chain_index in enumerate(group_indices):
                # Unconditional for the FIRST chain only: it is the one that pays the corpus read
                # and the ~2 GB cache allocation, so "the run printed the geometry and then went
                # quiet" -- what the 09-15 4-GPU launch log looks like -- is decided here, before
                # any --timing opt-in could have been remembered.
                first_chain = step == 0 and slot == 0
                verbose = args.timing or first_chain
                label = "chain load (first: corpus read + cache alloc)" if first_chain else f"chain load {chain_index}"
                with timed(label) if verbose else contextlib.nullcontext():
                    chain = store[chain_index]
                    window_start = 0
                    if args.random_window_latent_frames is not None:
                        window_start = window_start_for(
                            args.noise_seed,
                            step=step,
                            rank=rank,
                            slot=slot,
                            latent_frames=chain.z_y.shape[1],
                            window=args.random_window_latent_frames,
                        )
                        chain = window_chain(chain, window_start, args.random_window_latent_frames)
                    if cache is None:
                        grid = clip_grid_for(chain, geometry, device=device, latent_channels=model.caps.latent_channels)
                        cache = BlockCache.allocate(
                            grid,
                            geometry,
                            num_layers=num_blocks,
                            inner_dim=inner_dim,
                            device=device,
                            dtype=DTYPE,
                            # The subset's longest clip, not this chain's: one allocation serves
                            # every chain in the run, so a capacity capped by the first clip drawn
                            # would overflow on a longer one at a deep --context-latent-frames.
                            # A K=1 subset (whole-clip training) never writes the cache: a clip-start
                            # chain primes nothing and its only refresh is the skipped last one. Its
                            # denoise still reads an (empty) cache, so a one-frame placeholder serves;
                            # sizing it for the clip would reserve ~14.5 GB/rank at block 16.
                            capacity_latent_frames=1 if single_block_chains else store.max_latent_frames,
                        )
                loaded_at = time.time()
                # 1 prime + K denoise + (K - 1) refresh -- the last refresh is skipped. Checked
                # here, before any of them run.
                assert_rank_lockstep(accelerator, 2 * len(chain.blocks), chain.source)
                totals = train_chain(
                    transformer,
                    context,
                    chain,
                    geometry,
                    cache,
                    accelerator,
                    sigma0=sigma0,
                    seed=training_noise_seed(args, step=step, rank=rank, slot=slot, chain_index=chain_index),
                    latent_channels=model.caps.latent_channels,
                    guide_mode=args.guide_mode,
                    teacher_forcing=args.teacher_forcing,
                    timing=verbose,
                    accumulation=len(group_indices),
                )
                for key in step_totals:
                    step_totals[key] += totals[key] / len(group_indices)
                per_block.extend(totals["per_block"])
                sources.append(chain.source)
                window_starts.append(window_start)
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
                    step,
                    loaded_at - step_started,
                    chained_at - loaded_at,
                    time.time() - chained_at,
                    time.time() - step_started,
                )
            totals = {**step_totals, "per_block": per_block}

            if step % args.log_every == 0:
                per_block = totals.pop("per_block")
                record = {
                    "step": step,
                    "rank": rank,
                    "lr": lr,
                    "sigma0": sigma0,
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "elapsed_s": round(time.time() - started, 1),
                    "source": chain.source if len(sources) == 1 else sources,
                    "window_start": window_starts[0] if len(window_starts) == 1 else window_starts,
                    **{k: round(v, 6) for k, v in totals.items()},
                    # SS7.4(a): one entry per block IN THIS CHAIN, in order, so a reader can
                    # plot loss against position without re-deriving it from the chain-mean.
                    "per_block": [
                        {"chain_position": i, **{k: round(v, 6) for k, v in w.items()}} for i, w in enumerate(per_block)
                    ],
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if wandb_is_enabled(args):
                    # 0.0 rather than float(None): accelerate's clip_grad_norm_ returns None
                    # for some distributed types, and rank_mean is a COLLECTIVE -- a TypeError
                    # on one rank here would hang the others in the gather it never joins.
                    mean_loss, mean_mse, mean_grad_norm = rank_mean(
                        accelerator,
                        [
                            totals["loss"],
                            totals["mse"],
                            float(grad_norm) if grad_norm is not None else 0.0,
                        ],
                    )
                    block_mse = rank_mean(accelerator, [block["mse"] for block in per_block])
                    if accelerator.is_main_process and wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": mean_loss,
                                "train/mse": mean_mse,
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
                        "step %d/%d loss %.5f mse %.5f lr %.2e %.1fs",
                        step,
                        args.steps,
                        totals["loss"],
                        totals["mse"],
                        lr,
                        time.time() - started,
                    )
            if step % args.save_every == 0 or step == args.steps or (args.save_initial and step == 1):
                path = save_lora(
                    transformer,
                    accelerator,
                    args.output / "checkpoints",
                    step,
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


def read_preview_inputs(path: Path, settings: RunSettings) -> dict:  # noqa: PLR0912 -- ordered pinned input gates
    """Pin all preview inputs before model setup; no execution occurs here."""
    record = json.loads(path.read_text())
    if record.get("schema_version") != 2 or record.get("kind") != "onestep_avatar.preview_inputs":
        raise ValueError("preview inputs require a version-two fixed record")
    if record.get("mode") != settings.mode:
        raise ValueError("preview mode differs from the training mode")
    files = record.get("input_files")
    required = {"subset", "capture", "first_image", "text", "noise"}
    if settings.guide_mode == "d1":
        required.add("guide")
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("preview record lacks fixed capture/guide/image/text/noise inputs")
    for role, identity in files.items():
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            raise ValueError(f"preview {role} identity is malformed")
        source = Path(identity["path"])
        if not source.is_absolute() or sha256(source) != identity.get("sha256"):
            raise ValueError(f"preview {role} file changed or lacks an absolute pinned path")
        if role in required - {"subset"}:
            tensor_hash = identity.get("tensor_sha256")
            if not isinstance(tensor_hash, str) or len(tensor_hash) != 64:
                raise ValueError(f"preview {role} requires the actual selected tensor identity")
    arguments = record.get("evaluation_arguments")
    if not isinstance(arguments, list) or not all(isinstance(value, str) for value in arguments):
        raise ValueError("preview evaluation arguments must be a list of strings")
    if any(value.split("=", 1)[0] in {"--output", "--checkpoint", "--dry-run"} for value in arguments):
        raise ValueError("preview executor owns checkpoint/output arguments")
    from scripts.onestep_avatar.evaluate import parse_args as parse_evaluation  # noqa: PLC0415 -- no model session

    args = parse_evaluation([*arguments, "--output", str(settings.output / "previews")])
    if args.cfg != 1.0:
        negative = files.get('negative_text')
        if not isinstance(negative, dict) or not isinstance(negative.get('tensor_sha256'), str):
            raise ValueError('guided preview requires pinned negative text')
    if args.mode != settings.mode or args.guide_mode != settings.guide_mode:
        raise ValueError("preview arguments differ from the recorded training task")
    if args.subset.resolve() != Path(files["subset"]["path"]).resolve():
        raise ValueError("preview arguments use an unpinned video list")
    if args.noise_file is None or args.noise_file.resolve() != Path(files["noise"]["path"]).resolve():
        raise ValueError("preview arguments must consume their pinned noise file")
    if args.schedule != record.get("schedule"):
        raise ValueError("preview arguments differ from the exact recorded schedule")
    from scripts.onestep_avatar.previews import check_preview_reference_bundle  # noqa: PLC0415 -- saved pixels only

    check_preview_reference_bundle(record)
    record["sha256"] = video_lists.record_hash(record)
    return record


def enqueue_preview(checkpoint: Path, record: dict, output: Path) -> Path:
    """Publish an idempotent pending job only for a completed checkpoint."""
    marker = json.loads(checkpoint.with_suffix(".complete.json").read_text())
    if marker.get("state") != "complete" or marker.get("sha256") != sha256(checkpoint):
        raise ValueError("preview checkpoint is incomplete or changed")
    identity = hashlib.sha256((marker["sha256"] + record["sha256"]).encode()).hexdigest()
    job = {
        "schema_version": 2,
        "kind": "onestep_avatar.preview_job",
        "id": identity,
        "state": "pending",
        "checkpoint": {**marker, "path": str(checkpoint.resolve())},
        "fixed_inputs": record,
        "output": str((output / "previews" / identity).resolve()),
    }
    destination = output / "preview_jobs" / f"{identity}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing.get("fixed_inputs") != record or existing.get("checkpoint") != job["checkpoint"]:
            raise ValueError("preview job identity collides with different pinned inputs")
        return destination
    dataset.atomic_write(destination, lambda temporary: temporary.write_text(json.dumps(job, indent=2) + "\n"))
    return destination


def prepare_run(  # noqa: PLR0912 -- fail all input/parent conditions before runtime setup
    settings: RunSettings,
    *, require_fresh_output: bool = True,
) -> tuple[dataset.ClipStore, dict, model_registry.RefinerModel, bool]:
    """Check the new runtime's inputs and parent before model or output mutation."""
    expected_type = BidirectionalSettings if settings.mode == "bidirectional" else CausalSettings
    if settings.mode not in {"bidirectional", "causal"} or not isinstance(settings.mode_settings, expected_type):
        raise ValueError("run mode and typed mode settings disagree")
    if settings.preview_inputs is not None:
        settings.preview_record = read_preview_inputs(settings.preview_inputs, settings)
    if settings.output.exists() and not settings.output.is_dir():
        raise ValueError("output must be a directory")
    used = settings.output.is_dir() and any(settings.output.iterdir())
    if require_fresh_output and used and not settings.overwrite:
        raise ValueError("output already has a run; explicit overwrite archives it after all checks and setup succeed")
    membership = json.loads(settings.subset.read_text())
    video_lists.validate_membership(membership)
    if membership["objective"] != settings.objective:
        raise ValueError("requested background differs from the fixed video list")
    store = dataset.ClipStore(membership, settings.corpus_root)
    store.verify(require_guide=settings.guide_mode == "d1")
    specification = backbone.resolve(settings.model, settings.variant)
    from scripts.onestep_avatar.corpus.precompute import file_fingerprint  # noqa: PLC0415 -- producer identity

    vae_fingerprint = file_fingerprint(Path(specification.paths.video_vae()))
    for source_id, source in store.sources.items():
        for role in (("capture", "guide") if settings.guide_mode == "d1" else ("capture",)):
            if source.get(f"{role}_encode_record", {}).get("vae_fingerprint") != vae_fingerprint:
                raise ValueError(f"{source_id}: {role} encoding VAE differs from the selected base VAE")
    levels = training_sigmas(settings)
    if settings.variant == "distilled" and any(
        not any(abs(level - v) < 1e-9 for v in specification.sigmas) for level in levels
    ):
        raise ValueError("requested training levels are not on the selected distilled base grid")
    plan = select_frame_plan(settings, membership, specification.scale_factors)
    for sample in plan["samples"]:
        if sample["split"] != settings.split:
            continue
        source = store.sources[sample["source"]]
        if source["shape"][0] != specification.caps.latent_channels:
            raise ValueError(f"{sample['source']}: encoded channel count differs from the selected base")
        mode = settings.mode_settings
        frames = (
            sum(end - start for start, end in sample["ranges"])
            if settings.mode == "bidirectional" or mode.start_policy == "random"
            else sample["ranges"][-1][1]
        )
        seconds = common.pixel_frames_for(frames, specification.scale_factors.time) / float(source["fps"])
        if seconds > common.MAX_ROPE_SECONDS:
            raise ValueError(f"{sample['source']}: selected model positions exceed the 20-second range")
    settings.base_identity = backbone.identity(
        specification.paths.transformer(), settings.variant, settings.model, full_hash=True
    )
    if settings.init_adapter is not None:
        parent = sampling.read_contract(settings.init_adapter)
        sampling.validate_adapter_tensors(settings.init_adapter, parent)
        if parent["model"]["base_sha256"] != settings.base_identity["base_transformer_sha256"]:
            raise ValueError("parent adapter uses different base weights")
        if parent["model"]["version"] != settings.model or parent["model"]["variant"] != settings.variant:
            raise ValueError("parent adapter uses a different base version or variant")
        adapter = parent["adapter"]
        if (adapter["rank"], adapter["alpha"], adapter["target"]) != (
            settings.lora_rank,
            settings.lora_alpha,
            settings.lora_target,
        ):
            raise ValueError("parent adapter has different LoRA rank, alpha or targets")
        if parent["mode"] != settings.mode and not settings.allow_cross_mode_init:
            raise ValueError("cross-mode parent initialization requires explicit --allow-cross-mode-init")
        settings.parent_contract = {
            "path": str(settings.init_adapter),
            "sha256": sha256(settings.init_adapter),
            "original_mode": parent["mode"],
            "base_sha256": parent["model"]["base_sha256"],
            "rank": adapter["rank"],
            "alpha": adapter["alpha"],
            "target": adapter["target"],
            "initialization": "fresh_optimizer_and_random_state",
            "calibration_transferred": False,
        }
    return store, plan, specification, bool(used)


def verify_training_conditions(job: dict, checkpoint: Path, contract: dict) -> None:  # noqa: PLR0915 -- sequential evidence gates
    """Recheck queued scientific evidence without models, updates or output mutation."""
    from scripts.onestep_avatar.training.config import parse_settings  # noqa: PLC0415 -- typed command owner

    settings = parse_settings(job["arguments"])
    store, plan, _specification, _used = prepare_run(settings, require_fresh_output=False)
    settings.world_size = job["processes"]
    expected_checkpoint = settings.output / "checkpoints" / f"lora_weights_step_{settings.steps:05d}.safetensors"
    if checkpoint.resolve() != expected_checkpoint.resolve():
        raise ValueError("queue training final checkpoint path differs from requested run")
    expected_contract = sampling.make_contract(settings, store.membership, plan, settings.steps)
    expected_contract["adapter"]["tensor_shapes"] = contract["adapter"]["tensor_shapes"]
    if contract != expected_contract:
        raise ValueError("queue training adapter contract differs from requested scientific settings")
    config_path, plan_path = settings.output / "config.json", settings.output / "frame_plan.json"
    resolved = json.loads(config_path.read_text())
    software.check_current(resolved.get("software"))
    from scripts.onestep_avatar.execution import queue  # noqa: PLC0415 -- current launch authority
    launch = resolved.get("queue_launch")
    queue.verify_training_launch(launch, job)
    runtime.validate(resolved.get("runtime"), settings.world_size, _launch_precision(launch), native=True,
                     numerical_policy=True)
    budget = resources.read_budget(settings.resource_budget)
    if resolved.get("resource_budget") != budget:
        raise ValueError("training resource budget differs from original run")
    resource_evidence = {}
    if budget is not None:
        measurements, _journal_identities = resources.read_records(settings.output, settings.world_size)
        snapshot, resource_evidence = resources.read_records(settings.output, settings.world_size, step=settings.steps)
        if measurements != snapshot:
            raise ValueError("final checkpoint resource snapshot differs from rank journals")
        phases = resources.training_phases(settings)
        resources.validate_records(measurements, settings.world_size, phases, budget)
    saved_plan = json.loads(plan_path.read_text())
    sample_count = sum(s["split"] == settings.split for s in plan["samples"])
    tensor_shapes = contract["adapter"]["tensor_shapes"]
    modules = {name.rsplit(".lora_", 1)[0] for name in tensor_shapes}
    expected = {
        **settings.as_dict(), "membership_sha256": store.membership["sha256"],
        "frame_plan_sha256": plan["sha256"], "samples": sample_count,
        "loss": FULL_FRAME_X0_MSE, "samples_per_update": settings.world_size * settings.chains_per_rank,
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
        "queue_job_sha256": job["sha256"], "producer_source_sha256": sha256(Path(__file__)),
        "software": software.capture("training", settings.mode),
        "resource_budget": budget,
        "sample_tiling": max(1, math.ceil(settings.world_size * settings.chains_per_rank / sample_count)),
        "trainable_params": sum(math.prod(shape) for shape in tensor_shapes.values()),
        "lora_modules": len(modules),
        "lora_target_counts": {
            target: sum(name.endswith(target) for name in modules)
                for target in adapters.LORA_TARGETS[settings.lora_target]
        },
    }
    # Resolve the same JSON representation used by the producer (paths and tuples).
    expected = json.loads(json.dumps(expected, default=str))
    if saved_plan != plan or any(key not in resolved or resolved[key] != value for key, value in expected.items()):
        raise ValueError("queue training saved configuration or frame plan differs from requested run")
    marker = queue.read_training_marker(checkpoint, settings.steps)
    software.check_current(marker.get("software"))
    identities = {
        "queue_job_sha256": job["sha256"], "producer_source_sha256": expected["producer_source_sha256"],
        "software": expected["software"],
        "queue_launch": launch, "runtime": resolved["runtime"],
        "resource_budget": budget, "resource_evidence": resource_evidence,
        "training_record": {"config_sha256": sha256(config_path), "frame_plan_sha256": sha256(plan_path)},
    }
    if settings.consumer_trace:
        identities["consumer_trace_evidence"] = _read_consumer_evidence(
            settings.output, settings.steps, settings.world_size,
            None if resolved.get("queue_launch_path") is None else Path(resolved["queue_launch_path"]),
            job["sha256"], resolved.get("queue_attempt_token"), settings.chains_per_rank,
        )
    if any(marker.get(key) != value for key, value in identities.items()):
        raise ValueError("queue training completed marker has different run provenance")
    if settings.save_update_state:
        text = resolved.get("update_text", {})
        text_path = settings.output / "update_states/text.pt"
        if text.get("path") != str(text_path.resolve()) or text.get("sha256") != sha256(text_path):
            raise ValueError("training update text evidence differs")
        for step in range(1, settings.steps + 1):
            path = settings.output / "update_states" / f"step_{step:05d}.pt"
            state = json.loads(path.with_suffix(".json").read_text())
            if (state.get("step") != step or state.get("sha256") != sha256(path)
                    or state.get("shapes") != tensor_shapes or state.get("software") != expected["software"]
                    or state.get("world_size") != settings.world_size
                    or state.get("accumulation") != settings.chains_per_rank
                    or state.get("optimizer") != expected["optimizer"]
                    or not isinstance(state.get("grad_norm"), (int, float))
                    or not math.isfinite(state["grad_norm"])):
                raise ValueError("training Adam update evidence differs")


def tokens_for_sample(
    video: dataset.EncodedVideo,
    sample: dict,
    settings: RunSettings,
    specification: model_registry.RefinerModel,
    device: torch.device,
    *,
    step: int,
    rank: int,
    slot: int,
) -> tuple:
    """Build one mode's independent segment or original-position causal prefix."""
    mode = settings.mode_settings
    start = 0
    end = sample["ranges"][-1][1]
    if settings.mode == "bidirectional" or mode.start_policy == "random":
        start, end = bidirectional.plan_samples(
            video.z_y.shape[1],
            span_latent_frames=mode.span_latent_frames,
            start_policy=mode.start_policy,
            seed=settings.noise_seed,
            step=step,
            rank=rank,
            slot=slot,
        )
    capture = video.z_y[:, start:end].unsqueeze(0).to(device=device, dtype=DTYPE)
    guide = None if video.z_g is None else video.z_g[:, start:end].unsqueeze(0).to(device=device, dtype=DTYPE)
    grid = common.ClipGrid.build(
        capture.shape[2],
        capture.shape[3] * specification.scale_factors.height,
        capture.shape[4] * specification.scale_factors.width,
        video.fps,
        specification,
        device=device,
        dtype=DTYPE,
        latent_channels=specification.caps.latent_channels,
    )
    return grid, grid.patchify(capture), None if guide is None else grid.patchify(guide), start, end


def _read_consumer_evidence(
    output: Path, step: int, world: int, launch_path: Path | None,
    job: str | None, token: str | None, accumulation: int | None = None,
) -> dict[str, str]:
    """Require actual complete observations, immutable bytes and the original launch."""
    from scripts.onestep_avatar.training.consumer_trace import validate  # noqa: PLC0415 -- selected diagnostic
    evidence = {}
    for rank in range(world):
        path = output / "consumer_traces" / f"step_{step:05d}" / f"rank{rank}.json"
        data = path.read_bytes()
        record = json.loads(data)
        validate(record, {"rank": rank, "world_size": world, "queue_job_sha256": job,
                          "queue_attempt_token": token,
                          "launch_sha256": None if launch_path is None else sha256(launch_path)})
        if accumulation is not None:
            expected_visits = [(update, slot) for update in range(1, step + 1) for slot in range(accumulation)]
            if [(sample["step"], sample["slot"]) for sample in record["samples"]] != expected_visits:
                raise ValueError("consumer trace sample coverage differs from completed updates")
        digest = hashlib.sha256(data).hexdigest()
        if sha256(path) != digest:
            raise ValueError("consumer trace changed during verification")
        evidence[str(path.resolve())] = digest
    return evidence


def run_settings(settings: RunSettings) -> int:
    """Record queued startup failures around the typed runtime, never retry here."""
    from scripts.onestep_avatar.training.startup import StartupEvents  # noqa: PLC0415 -- queue protocol only

    with StartupEvents() as events:
        return _run_settings(settings, events)


def _run_settings(settings: RunSettings, events) -> int:  # noqa: ANN001 -- startup event owner
    with contextlib.ExitStack() as lifecycle:
        return _run_settings_body(settings, events, lifecycle)


def _run_settings_body(settings: RunSettings, events, lifecycle: contextlib.ExitStack) -> int:  # noqa: ANN001, PLR0912, PLR0915 -- linear runtime and startup boundary
    """Run the typed two-mode engine; previews execute in their own model sessions."""
    launch, launch_path = _launch_evidence(settings, events.job)
    budget = resources.read_budget(settings.resource_budget)
    store, plan, specification, needs_archive = prepare_run(settings)
    producer_software = software.capture("training", settings.mode)
    samples = [sample for sample in plan["samples"] if sample["split"] == settings.split]
    resolved = {
        **settings.as_dict(),
        "membership_sha256": store.membership["sha256"],
        "frame_plan_sha256": plan["sha256"],
        "samples": len(samples),
        "loss": FULL_FRAME_X0_MSE,
        "queue_job_sha256": events.job,
        "queue_attempt_token": events.token,
        "producer_source_sha256": sha256(Path(__file__)),
        "software": producer_software,
        "queue_launch": launch,
        "queue_launch_path": None if launch_path is None else str(launch_path),
        "resource_budget": budget,
    }
    if settings.dry_run:
        print(json.dumps(resolved, indent=2, default=str))  # noqa: T201 -- requested CLI plan
        return 0
    applied_numerics = numerics.apply(environment_required=launch is not None)
    if settings.reserve_gpu_gib > 0:
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        reserve = torch.empty(int(settings.reserve_gpu_gib * 2**30), dtype=torch.uint8, device=f"cuda:{local}")
        del reserve
    with timed("Accelerator() / process group"):
        accelerator = Accelerator()
    settings.world_size = accelerator.num_processes
    device = accelerator.device
    rank = accelerator.process_index
    if launch is not None:
        runtime.check_accelerator(accelerator, launch["job"]["processes"], _launch_precision(launch),
                                  distributed_type=DistributedType.FSDP)
    _check_launch_current(launch, launch_path)
    measurements = []
    def preserve_measurement(record: dict) -> None:
        measurements.append(record)
        LOGGER.info("resource | %s", json.dumps(record, sort_keys=True))
        if settings.output.is_dir() and not needs_archive:
            dataset.atomic_write(settings.output / f"resources_rank{rank}.jsonl",
                                 lambda temporary: temporary.write_text(
                                     "".join(json.dumps(item) + "\n" for item in measurements)))
        if record["state"] != "passed":
            raise ValueError(record["error"])

    @contextlib.contextmanager
    def measured(phase: str) -> Iterator[None]:
        if budget is None:
            yield
            return
        observation = resources.Phase(device, phase, rank, budget)
        try:
            observation.start()
            yield
        except BaseException as error:
            if observation.started is None:
                raise
            record = observation.finish(error=f"{type(error).__name__}: {error}")
            measurements.append(record)
            LOGGER.error("resource | %s", json.dumps(record, sort_keys=True))
            if settings.output.is_dir() and not needs_archive:
                dataset.atomic_write(settings.output / f"resources_rank{rank}.jsonl",
                                     lambda temporary: temporary.write_text(
                                         "".join(json.dumps(item) + "\n" for item in measurements)))
            raise
        else:
            preserve_measurement(observation.finish())
    with measured("load"):
        software.check_current(producer_software)
        with timed("prompt cache (text encoder)"):
            context = prompt_cache.get_or_build(specification, DEFAULT_PROMPT, DTYPE, device)
        with timed("transformer load + LoRA injection"):
            software.check_current(producer_software)
            transformer = build_transformer(specification, settings, accelerator)
        trainable = [p for p in transformer.parameters() if p.requires_grad]
        lora_modules = sorted(
            {name.rsplit(".lora_", 1)[0] for name, p in transformer.named_parameters() if p.requires_grad}
        )
        target_counts = {
            target: sum(name.endswith(target) for name in lora_modules)
                for target in adapters.LORA_TARGETS[settings.lora_target]
        }
        trainable_count = sum(p.numel() for p in trainable)
        optimizer = torch.optim.AdamW(trainable, lr=settings.lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
        with timed("accelerator.prepare (FSDP shard)"):
            transformer, optimizer = accelerator.prepare(transformer, optimizer)
    applied_runtime = runtime.gather(runtime.capture(transformer, accelerator, common.SIGMA_PRECISION), accelerator)
    runtime.validate(applied_runtime, settings.world_size, accelerator.mixed_precision,
                     numerical_policy=applied_numerics)
    if launch is not None:
        runtime.validate(applied_runtime, settings.world_size, _launch_precision(launch), native=True)
    resolved["runtime"] = applied_runtime
    trace = None
    if settings.consumer_trace:
        from scripts.onestep_avatar.training.consumer_trace import Trace  # noqa: PLC0415 -- selected diagnostic
        trace = lifecycle.enter_context(Trace(transformer, {"rank": rank, "world_size": settings.world_size,
                                   "queue_job_sha256": events.job, "queue_attempt_token": events.token,
                                   "launch_sha256": None if launch_path is None else sha256(launch_path)},
                                   failure_path=settings.output / f"consumer_trace_failed_rank{rank}.json"))
    world = settings.world_size
    tiling = max(1, math.ceil(world * settings.chains_per_rank / len(samples)))
    per_rank = len(samples) * tiling // world
    per_rank -= per_rank % settings.chains_per_rank
    if per_rank == 0:
        raise ValueError("sample plan cannot supply the requested ranks and accumulation")
    order = list(range(len(samples))) * tiling
    software.check_current(producer_software)
    _check_launch_current(launch, launch_path)
    resources.check_budget(budget)
    if needs_archive:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            archive_existing_run(settings.output)
        accelerator.wait_for_everyone()
        needs_archive = False
    if accelerator.is_main_process:
        settings.output.mkdir(parents=True, exist_ok=True)
        software.check_current(producer_software)
        _check_launch_current(launch, launch_path)
        resources.check_budget(budget)
        if settings.save_update_state:
            text_path = settings.output / "update_states" / "text.pt"
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_cpu = context.detach().cpu().contiguous()
            dataset.atomic_write(text_path, lambda temporary: torch.save(text_cpu, temporary))
            text_digest = hashlib.sha256(text_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
            resolved["update_text"] = {"path": str(text_path.resolve()), "sha256": sha256(text_path),
                                       "tensor_sha256": text_digest,
                                       "shape": list(text_cpu.shape), "dtype": str(text_cpu.dtype)}
        resolved.update(
            world_size=world,
            trainable_params=trainable_count,
            lora_modules=len(lora_modules),
            lora_target_counts=target_counts,
            samples_per_update=world * settings.chains_per_rank,
            sample_tiling=tiling,
            optimizer={"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
        )
        dataset.atomic_write(
            settings.output / "config.json",
            lambda temporary: temporary.write_text(json.dumps(resolved, indent=2, default=str) + "\n"),
        )
        dataset.atomic_write(
            settings.output / "frame_plan.json",
            lambda temporary: temporary.write_text(json.dumps(plan, indent=2) + "\n"),
        )
    accelerator.wait_for_everyone()
    if budget is not None:
        dataset.atomic_write(settings.output / f"resources_rank{rank}.jsonl",
                             lambda temporary: temporary.write_text(
                                 "".join(json.dumps(item) + "\n" for item in measurements)))
    accelerator.wait_for_everyone()
    wandb_run = init_wandb(settings, config=resolved) if accelerator.is_main_process else None

    def save(step: int, *, verify_noop: bool = False) -> None:
        numerics.validate(numerics.capture(), required=True, expected=applied_numerics)
        software.check_current(producer_software)
        _check_launch_current(launch, launch_path)
        resources.check_budget(budget)
        contract = sampling.make_contract(settings, store.membership, plan, step)
        with measured(f"export:{step}"):
            path = save_lora(
                transformer,
                accelerator,
                settings.output / "checkpoints",
                step,
                {sampling.CONTRACT_KEY: json.dumps(contract, sort_keys=True)},
                verify_noop=verify_noop,
            )
            if trace is not None:
                trace.write(settings.output / "consumer_traces" / f"step_{step:05d}" / f"rank{rank}.json")
        if budget is not None:
            resources.save_snapshot(settings.output, rank, step)
        accelerator.wait_for_everyone()
        if path is not None:
            software.check_current(producer_software)
            _check_launch_current(launch, launch_path)
            resources.check_budget(budget)
            resource_evidence = {}
            consumer_evidence = (_read_consumer_evidence(settings.output, step, world, launch_path,
                                 events.job, events.token, settings.chains_per_rank)
                                 if settings.consumer_trace else {})
            if budget is not None:
                records, resource_evidence = resources.read_records(settings.output, world, step=step)
                phases = resources.training_phases(settings, step=step)
                resources.validate_records(records, world, phases, budget)
            dataset.atomic_write(
                path.with_suffix(".complete.json"),
                lambda temporary: temporary.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "step": step,
                            "path": str(path),
                            "sha256": sha256(path),
                            "state": "complete",
                            "queue_job_sha256": events.job,
                            "producer_source_sha256": resolved["producer_source_sha256"],
                            "software": producer_software,
                            "queue_launch": launch,
                            "runtime": applied_runtime,
                            "resource_budget": budget,
                            "resource_evidence": resource_evidence,
                            "consumer_trace_evidence": consumer_evidence,
                            "training_record": {
                                "config_sha256": sha256(settings.output / "config.json"),
                                "frame_plan_sha256": sha256(settings.output / "frame_plan.json"),
                            },
                        },
                        indent=2,
                    )
                    + "\n"
                ),
            )
            if settings.preview_record is not None:
                try:
                    enqueue_preview(path, settings.preview_record, settings.output)
                except Exception:  # Optional preview failures cannot interrupt distributed training.
                    LOGGER.exception("preview enqueue failed; completed checkpoint remains valid: %s", path)
        accelerator.wait_for_everyone()

    if settings.save_initial:
        save(0, verify_noop=settings.init_adapter is None)
    mode = settings.mode_settings
    geometry = (
        causal_core.CausalGeometry(specification.scale_factors, mode.block_latent_frames, mode.context_latent_frames)
        if isinstance(mode, CausalSettings)
        else None
    )
    longest = max(s["ranges"][-1][1] for s in samples)
    cache = None
    generator = torch.Generator().manual_seed(settings.data_seed)
    levels = training_sigmas(settings)
    step = 0
    started = time.time()
    events.begin_updates()
    with (settings.output / f"metrics_rank{rank}.jsonl").open("a") as log:
        while step < settings.steps:
            epoch_order = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]
            shard = epoch_order[rank * per_rank : (rank + 1) * per_rank]
            for begin in range(0, len(shard), settings.chains_per_rank):
                if step >= settings.steps:
                    break
                indices = shard[begin : begin + settings.chains_per_rank]
                numerics.validate(numerics.capture(), required=True, expected=applied_numerics)
                lr = settings.lr * min(1.0, (step + 1) / max(settings.warmup_steps, 1))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                sigma = sigma_for_rank(levels, rank, step, settings.noise_seed)
                with measured(f"update:{step + 1}"):
                    update_started = time.time()
                    metrics = {"loss": 0.0, "mse": 0.0}
                    details = []
                    counts = {"prime": 0, "denoise": 0, "backward": 0, "refresh": 0}
                    for slot, index in enumerate(indices):
                        sample = samples[index]
                        video = store.load(sample["source"], require_guide=settings.guide_mode == "d1")
                        grid, capture, guide, start, end = tokens_for_sample(
                            video, sample, settings, specification, device, step=step, rank=rank, slot=slot
                        )
                        expected = 1 if settings.mode == "bidirectional" else 2 * len(sample["blocks"])
                        assert_rank_lockstep(accelerator, expected, sample["source"])
                        noise_seed = training_noise_seed(settings, step=step, rank=rank, slot=slot, chain_index=index)
                        trace_scope = (contextlib.nullcontext() if trace is None else
                                       trace.sample(mode=settings.mode, step=step + 1, slot=slot, index=index))
                        with trace_scope:
                            if settings.mode == "bidirectional":
                                result = bidirectional.train_sample(
                                    transformer,
                                    context,
                                    grid,
                                    capture,
                                    guide,
                                    accelerator.backward,
                                    sigma=sigma,
                                    seed=noise_seed,
                                    guide_mode=settings.guide_mode,
                                    accumulation=len(indices),
                                )
                            else:
                                result = causal_core.train_sample(
                                    transformer,
                                    context,
                                    grid,
                                    capture,
                                    guide,
                                    geometry,
                                    sample["blocks"],
                                    accelerator.backward,
                                    sigma=sigma,
                                    seed=noise_seed,
                                    cache=cache,
                                    guide_mode=settings.guide_mode,
                                    teacher_forcing=mode.teacher_forcing,
                                    accumulation=len(indices),
                                    timing=settings.timing,
                                    capacity_latent_frames=longest,
                                )
                                cache = result.pop("cache")
                        for key in metrics:
                            metrics[key] += result[key] / len(indices)
                        for key in counts:
                            counts[key] += result.get(key + "_calls", 0)
                        details.append(
                            {
                                "source": sample["source"],
                                "start": start,
                                "end": end,
                                "ranges": [[start, end]]
                                if settings.mode == "bidirectional" or mode.start_policy == "random"
                                else sample["ranges"],
                                "noise_seed": noise_seed,
                                "per_block": result.get("per_block", []),
                            }
                        )
                    grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), settings.max_grad_norm)
                    optimizer.step()
                    if settings.save_update_state:
                        from scripts.onestep_avatar.training.update_state import save_adam_state  # noqa: PLC0415
                        software.check_current(producer_software)
                        state_path = settings.output / "update_states" / f"step_{step + 1:05d}.pt"
                        shapes = save_adam_state(transformer, optimizer, accelerator, state_path, step + 1)
                        if shapes is not None:
                            software.check_current(producer_software)
                            state_json = json.dumps({"step": step + 1, "sha256": sha256(state_path), "shapes": shapes,
                                                     "grad_norm": float(grad_norm), "world_size": world,
                                                     "accumulation": settings.chains_per_rank,
                                                     "optimizer": resolved["optimizer"],
                                                     "software": producer_software}, indent=2) + "\n"
                            dataset.atomic_write(state_path.with_suffix(".json"),
                                                 lambda temporary, payload=state_json: temporary.write_text(payload))
                    optimizer.zero_grad(set_to_none=True)
                step += 1
                record = {
                    "schema_version": 2,
                    "step": step,
                    "rank": rank,
                    "mode": settings.mode,
                    "lr": lr,
                    "sigma0": sigma,
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "elapsed_s": time.time() - started,
                    "update_s": time.time() - update_started,
                    **metrics,
                    "samples": details,
                    "call_counts": counts,
                }
                if settings.save_update_state:
                    record["text_tensor_sha256"] = hashlib.sha256(
                        context.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                log.write(json.dumps(record) + "\n")
                log.flush()
                if step % settings.log_every == 0 and wandb_is_enabled(settings):
                    mean = rank_mean(
                        accelerator,
                        [metrics["loss"], metrics["mse"], float(grad_norm) if grad_norm is not None else 0.0],
                    )
                    if accelerator.is_main_process and wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": mean[0],
                                "train/mse": mean[1],
                                "train/grad_norm": mean[2],
                                "train/lr": lr,
                                "train/sigma0": sigma,
                                "train/elapsed_s": record["elapsed_s"],
                            },
                            step=step,
                        )
                if step % settings.save_every == 0 or step == settings.steps or (settings.save_initial and step == 1):
                    save(step)
    if accelerator.is_main_process and wandb_run is not None:
        wandb_run.finish()
    accelerator.wait_for_everyone()
    return 0
