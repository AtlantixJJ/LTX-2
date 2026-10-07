"""Parse training options and preserve seeded sigma/noise selection.

See doc/training/config.md for checks and mode ownership. The initial extraction
retains the old parser while explicit mode and membership integration proceeds.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import dataset
from scripts.onestep_avatar.model import backbone
from scripts.onestep_avatar.model import causal as causal_core
from scripts.prune.core import model_registry

SIGMA_SAMPLING = "iid_uniform_v1"

DEFAULT_SIGMA0 = 0.725


LORA_TARGETS = {
    "attn": ["to_k", "to_q", "to_v", "to_out.0"],
    "attn_ffn": ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
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


def training_noise_seed(args: argparse.Namespace, *, step: int, rank: int, slot: int, chain_index: int) -> int:
    """The base seed of one chain visit's epsilon; ``train_chain`` adds the block index.

    ``fresh``: a function of (noise seed, update, rank, accumulation slot) only -- so a chain
    revisited in a later epoch gets new noise, and two candidates launched with the same seeds
    and topology draw the identical stream whatever their sigma or arm. ``fixed_per_chain`` is
    the historical rule (one epsilon per chain for the whole run), kept as the debug control
    that separates memorising a noise draw from learning the correction.
    """
    if args.noise_policy == "fixed_per_chain":
        return args.noise_seed * 100003 + chain_index * 101
    return ((args.noise_seed * 1_000_003 + step) * 1009 + rank) * 131 + slot * 17


def sigma_for_rank(sigmas: tuple[float, ...], rank: int, step: int, seed: int) -> float:
    """This rank's noise level at this step: an i.i.d. uniform draw from ``sigmas``.

    Each (update, rank) draws independently from its own seeded generator, so a run is
    reproducible and the draw is independent of the epsilon stream, the arm and the data
    order. Levels are mixed within an update (four independent draws) and every level is
    reached with probability 1/len(sigmas) per sample; unlike the retired rotation, exposure
    per level is equal only in expectation. A single-level run returns that level.
    """
    if len(sigmas) == 1:
        return sigmas[0]
    rng = random.Random(f"onestep_avatar.sigma:{seed}:{step}:{rank}")
    return sigmas[rng.randrange(len(sigmas))]


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
        "--overwrite",
        action="store_true",
        help="Relaunch into a --output that already holds a run: the whole prior directory "
        "(metrics, checkpoints, config.json, everything) is moved aside into an "
        "'archived_<timestamp>/' subdirectory before this run creates anything, rather than "
        "being deleted or left to coexist under one step numbering. There is no resume -- step "
        "always restarts at 0 -- so a used --output is otherwise refused outright.",
    )
    p.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    p.add_argument(
        "--variant",
        choices=backbone.VARIANTS,
        default=backbone.DEFAULT_VARIANT,
        help="Backbone weights: the distilled transformer (default, the historical recipes) or "
        "dev. Resolved by backbone.py and stamped by file fingerprint, so a dev adapter is "
        "refused on distilled weights at probe time.",
    )
    p.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA0)
    p.add_argument(
        "--sigma-levels",
        type=float,
        nargs="+",
        default=None,
        help="Train across these levels: each rank draws one level per update i.i.d. uniformly "
        "(seeded by --noise-seed, update and rank). Overrides --sigma0. sigma=0.0 is refused "
        "-- it trains on nothing (see training_sigmas).",
    )
    p.add_argument(
        "--random-window-latent-frames",
        type=int,
        default=None,
        help="Train each sample on this many latent frames starting at a random latent frame "
        "(uniform, seeded by --noise-seed, update, rank and slot); the window's frame 0 becomes c0. "
        "Default: the clip from its start. Every source must have at least this many frames.",
    )
    p.add_argument(
        "--block-latent-frames",
        type=int,
        default=causal_core.BLOCK_LATENT_FRAMES,
        help="Latent frames denoised per causal block (SS4.4). The default is the deployed "
        "16-pixel-frame stride; changing it changes what a trained adapter finalizes per step.",
    )
    p.add_argument(
        "--context-latent-frames",
        type=int,
        default=causal_core.CONTEXT_LATENT_FRAMES,
        help=f"Clean latent frames kept in the K/V cache BESIDES the pinned frame-0 sink, up "
        f"to {causal_core.MAX_CONTEXT_LATENT_FRAMES}. The default "
        f"{causal_core.CONTEXT_LATENT_FRAMES} is a retained history of "
        f"{causal_core.CONTEXT_LATENT_FRAMES + causal_core.SINK_LATENT_FRAMES} latent frames "
        f"(c0 plus clean frames 1-{causal_core.CONTEXT_LATENT_FRAMES}). Each one costs ~0.8 GB "
        f"per rank at the 22B geometry and lengthens every block's attention, so this is the "
        f"compute/quality knob of the scheme. Past roughly the chain's own reach the cache "
        f"stops evicting and simply ACCUMULATES the whole rollout's history -- at the default "
        f"depth a K=3, 2-frame-block chain never evicts at all. Recorded in the checkpoint "
        f"metadata: an adapter trained at one depth is a different function at another.",
    )
    p.add_argument("--split", choices=("train", "held_out", "validation", "test"), default="train")
    p.add_argument("--lora-rank", type=int, default=8, help="2-3 GPU preliminary runs drop this, never K")
    p.add_argument("--lora-alpha", type=int, default=None, help="default: equal to --lora-rank")
    p.add_argument("--lora-target", choices=sorted(LORA_TARGETS), default="attn")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42, help="default for the three seeds below")
    p.add_argument("--init-seed", type=int, default=None, help="LoRA initialization seed (default --seed)")
    p.add_argument("--data-seed", type=int, default=None, help="chain-order seed (default --seed)")
    p.add_argument("--noise-seed", type=int, default=None, help="training-epsilon stream seed (default --seed)")
    p.add_argument(
        "--noise-policy",
        choices=("fresh", "fixed_per_chain"),
        default="fresh",
        help="fresh (default): epsilon depends on (noise seed, update, rank, slot), so a chain "
        "revisited in a later epoch gets new noise and matched candidates share one stream. "
        "fixed_per_chain: the historical rule -- every visit to a chain reuses one epsilon; "
        "kept only as the overfit debug control.",
    )
    p.add_argument(
        "--chains-per-rank",
        type=int,
        default=1,
        help="Chains each rank accumulates into one optimizer update (chains per update = this x "
        "world size). Lets a 2-GPU run match a 4-chain update without changing K.",
    )
    p.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help="Start a NEW stage from this adapter's LoRA weights (fresh optimizer/scheduler/RNG, "
        "parent recorded). Not a resume.",
    )
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
        "measured r; the product infer CLI refuses this mode because there is no "
        "z_y at inference. (The old `d2` extra-token hybrid is gone: it was dropped as an arm "
        "2026-09-13 for costing 1.05x k2, and its clean reference tokens have no place in a "
        "causal sequence -- they would be future context.)",
    )
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
    p.add_argument(
        "--wandb-project",
        default="onestep-avatar",
        help="Enable online W&B logging to this project (default: 'onestep-avatar'). "
        "Pass --no-wandb, set --wandb-mode disabled, or pass an empty string to disable.",
    )
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
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
    p.add_argument(
        "--reserve-gpu-gib",
        type=float,
        default=0.0,
        help="Operational: reserve this much GPU memory per rank at start-up (allocate+free into the "
        "caching allocator) so a shared machine's other schedulers do not claim the card during load. "
        "Requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, or the reservation fragments the allocator.",
    )
    p.add_argument("--dry-run", action="store_true", help="report the plan and the data shapes, load no model")
    return p.parse_args(argv)


@dataclass(frozen=True)
class BidirectionalSettings:
    span_latent_frames: int | None = None
    start_policy: str = "clip_start"
    attention: str = "full_bidirectional"


@dataclass(frozen=True)
class CausalSettings:
    block_latent_frames: int = 2
    blocks_per_sample: int = 3
    context_latent_frames: int = 8
    teacher_forcing: bool = False
    span_latent_frames: int | None = None
    start_policy: str = "clip_start"
    attention: str = "block_causal"


@dataclass
class RunSettings:
    mode: str
    subset: Path
    output: Path
    mode_settings: BidirectionalSettings | CausalSettings
    corpus_root: Path | None = None
    frame_plan: Path | None = None
    preview_inputs: Path | None = None
    preview_record: dict | None = None
    model: str = "2.5"
    variant: str = "distilled"
    objective: str = "bg"
    guide_mode: str = "d1"
    split: str = "train"
    sigma0: float = DEFAULT_SIGMA0
    sigma_levels: tuple[float, ...] | None = None
    seed: int = 42
    init_seed: int = 42
    data_seed: int = 42
    noise_seed: int = 42
    noise_policy: str = "fresh"
    chains_per_rank: int = 1
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_target: str = "attn"
    lr: float = 1e-4
    warmup_steps: int = 20
    steps: int = 200
    max_grad_norm: float = 1.0
    save_every: int = 100
    log_every: int = 1
    save_initial: bool = False
    init_adapter: Path | None = None
    allow_cross_mode_init: bool = False
    overwrite: bool = False
    dry_run: bool = False
    timing: bool = False
    no_gradient_checkpointing: bool = False
    init_device: str = "cuda"
    reserve_gpu_gib: float = 0.0
    wandb_project: str = "onestep-avatar"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    no_wandb: bool = False
    base_identity: dict = field(default_factory=dict)
    parent_contract: dict | None = None
    world_size: int = 1

    def as_dict(self) -> dict:
        """Save common and selected-mode fields; there are no causal fields in bidirectional settings."""
        return asdict(self)


def _mode_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> BidirectionalSettings | CausalSettings:
    """Distinguish explicitly supplied causal options from omitted defaults."""
    causal_options = (
        args.block_latent_frames,
        args.blocks_per_sample,
        args.context_latent_frames,
        args.teacher_forcing,
    )
    if args.mode == "bidirectional":
        if any(value is not None for value in causal_options):
            parser.error("bidirectional mode rejects block, cache, and history options")
        mode_settings = BidirectionalSettings(args.span_latent_frames, args.start_policy)
    else:
        block = 2 if args.block_latent_frames is None else args.block_latent_frames
        count = 3 if args.blocks_per_sample is None else args.blocks_per_sample
        depth = 8 if args.context_latent_frames is None else args.context_latent_frames
        if block < 1 or count < 1 or not 0 <= depth <= causal_core.MAX_CONTEXT_LATENT_FRAMES:
            parser.error("causal block count/length must be positive; context must be within the supported limit")
        if args.start_policy == "random" and (count != 1 or args.span_latent_frames != block + 1):
            parser.error("random-start causal segments require one block and span = block length + 1")
        mode_settings = CausalSettings(
            block, count, depth, bool(args.teacher_forcing), args.span_latent_frames, args.start_policy
        )
    return mode_settings


def _check_common_settings(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Check numerical and adapter settings before any file or model access."""
    if args.span_latent_frames is not None and args.span_latent_frames < 2:
        parser.error("a video segment needs at least two encoded frames")
    args.lora_alpha = args.lora_rank if args.lora_alpha is None else args.lora_alpha
    if args.lora_rank < 1 or args.lora_alpha != args.lora_rank:
        parser.error("LoRA rank must be positive; the supported application requires alpha = rank")
    if args.chains_per_rank < 1 or args.steps < 0 or args.warmup_steps < 0:
        parser.error("accumulation must be positive; steps and warmup must be nonnegative")
    if args.log_every < 1 or args.save_every < 1:
        parser.error("log and save intervals must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        parser.error("learning rate and gradient bound must be finite and positive")
    if not math.isfinite(args.reserve_gpu_gib) or args.reserve_gpu_gib < 0:
        parser.error("GPU reservation must be finite and nonnegative")
    training_sigmas(args)


def parse_settings(argv: list[str] | None = None) -> RunSettings:
    """Resolve typed explicit modes before accessing files or loading any model."""
    parser = argparse.ArgumentParser(description="Train avatar LoRA weights with an explicit attention mode.")
    parser.add_argument("--mode", choices=("bidirectional", "causal"), required=True)
    parser.add_argument("--subset", type=Path, required=True, help="Version-two fixed video list.")
    parser.add_argument("--output", type=Path, required=True)
    for option in ("corpus-root", "frame-plan", "init-adapter", "preview-inputs"):
        parser.add_argument("--" + option, type=Path)
    parser.add_argument("--model", choices=model_registry.SUPPORTED_MODELS, default="2.5")
    parser.add_argument("--variant", choices=backbone.VARIANTS, default=backbone.DEFAULT_VARIANT)
    parser.add_argument("--objective", choices=dataset.OBJECTIVES, default=dataset.DEFAULT_OBJECTIVE)
    parser.add_argument("--guide-mode", choices=("d0", "d1"), default="d1")
    parser.add_argument("--split", choices=("train", "held_out", "validation", "test"), default="train")
    parser.add_argument("--span-latent-frames", type=int)
    parser.add_argument("--start-policy", choices=("clip_start", "random"), default="clip_start")
    for option in ("block-latent-frames", "blocks-per-sample", "context-latent-frames"):
        parser.add_argument("--" + option, type=int)
    parser.add_argument("--teacher-forcing", action="store_true", default=None)
    parser.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA0)
    parser.add_argument("--sigma-levels", type=float, nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    for option in ("init-seed", "data-seed", "noise-seed"):
        parser.add_argument("--" + option, type=int)
    parser.add_argument("--noise-policy", choices=("fresh", "fixed_per_chain"), default="fresh")
    for option, default in (
        ("chains-per-rank", 1),
        ("lora-rank", 8),
        ("warmup-steps", 20),
        ("steps", 200),
        ("save-every", 100),
        ("log-every", 1),
    ):
        parser.add_argument("--" + option, type=int, default=default)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-target", choices=sorted(LORA_TARGETS), default="attn")
    for option, default in (("lr", 1e-4), ("max-grad-norm", 1.0), ("reserve-gpu-gib", 0.0)):
        parser.add_argument("--" + option, type=float, default=default)
    for option in (
        "allow-cross-mode-init",
        "overwrite",
        "dry-run",
        "timing",
        "save-initial",
        "no-gradient-checkpointing",
        "no-wandb",
    ):
        parser.add_argument("--" + option, action="store_true")
    parser.add_argument("--init-device", default="cuda")
    parser.add_argument("--wandb-project", default="onestep-avatar")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args(argv)
    mode_settings = _mode_from_args(args, parser)
    _check_common_settings(args, parser)
    for name in ("init_seed", "data_seed", "noise_seed"):
        if getattr(args, name) is None:
            setattr(args, name, args.seed)
    payload = vars(args).copy()
    for key in (
        "block_latent_frames",
        "blocks_per_sample",
        "context_latent_frames",
        "teacher_forcing",
        "span_latent_frames",
        "start_policy",
    ):
        payload.pop(key)
    if payload["sigma_levels"] is not None:
        payload["sigma_levels"] = tuple(payload["sigma_levels"])
    return RunSettings(**payload, mode_settings=mode_settings)


def window_start_draw(seed: int) -> dict:
    """Describe the original random-window stream for plans and adapter evidence."""
    return {"rule": "iid_uniform_v1", "seed": seed, "key": "onestep_avatar.window:{seed}:{step}:{rank}:{slot}"}


def build_frame_plan(  # noqa: PLR0912 -- keep each mode's frame-selection checks together
    settings: RunSettings,
    membership: dict,
    scale_factors: SpatioTemporalScaleFactors,
) -> dict:
    """Select deterministic mode samples over a checked fixed video list."""
    from scripts.onestep_avatar import subset  # noqa: PLC0415 -- the converter also uses mode geometry
    from scripts.onestep_avatar.model import bidirectional  # noqa: PLC0415

    subset.validate_membership(membership)
    if settings.objective != membership["objective"]:
        raise ValueError("requested background differs from the fixed video list")
    mode = settings.mode_settings
    selected = [source for source in membership["sources"] if source["split"] == settings.split]
    if not selected:
        raise ValueError(f"fixed video list has no selected videos in group {settings.split}")
    shapes = {(source["shape"][0], *source["shape"][2:]) for source in selected}
    if len(shapes) != 1:
        raise ValueError("selected videos must share channel and spatial dimensions")
    samples = []
    geometry = None
    if isinstance(mode, CausalSettings):
        geometry = causal_core.CausalGeometry(scale_factors, mode.block_latent_frames, mode.context_latent_frames)
    for source in selected:
        frames = int(source["n_latent_frames"])
        if mode.span_latent_frames is not None and frames < mode.span_latent_frames:
            raise ValueError(f"{source['relative_dir']}: shorter than the requested video segment")
        base = {"source": source["relative_dir"], "actor": source["actor"], "split": source["split"]}
        if isinstance(mode, BidirectionalSettings):
            start, end = bidirectional.plan_samples(frames, span_latent_frames=mode.span_latent_frames)
            if end - start < 2:
                raise ValueError("a bidirectional training segment needs at least two encoded frames")
            samples.append({**base, "ranges": [[start, end]], "independent_segment": True})
        else:
            length = frames if mode.span_latent_frames is None else mode.span_latent_frames
            groups = causal_core.plan_samples(length, geometry, blocks_per_sample=mode.blocks_per_sample)
            if not groups:
                raise ValueError(f"{source['relative_dir']}: too short for the requested block sequence")
            bounds = geometry.plan(length)
            for blocks in groups:
                samples.append(
                    {
                        **base,
                        "blocks": blocks,
                        "ranges": [list(bounds[b]) for b in blocks],
                        "seed_is_clip_start": blocks[0] == 0,
                        "independent_segment": mode.start_policy == "random",
                    }
                )
    record = {
        "schema_version": 2,
        "kind": subset.FRAME_PLAN_KIND,
        "mode": settings.mode,
        "membership_sha256": membership["sha256"],
        "mode_settings": asdict(mode),
        "samples": samples,
        "selection_rule": "mode_rules_v2",
        "split": settings.split,
    }
    if mode.start_policy == "random":
        record["start_draw"] = window_start_draw(settings.noise_seed)
    if geometry is not None:
        record["geometry"] = geometry.as_dict()
    record["sha256"] = subset.record_hash(record)
    return record


def select_frame_plan(  # noqa: PLR0912 -- check every provenance field before returning a saved plan
    settings: RunSettings,
    membership: dict,
    scale_factors: SpatioTemporalScaleFactors,
) -> dict:
    """Check an explicit reproduction plan, or build a new declared selection rule."""
    from scripts.onestep_avatar import subset  # noqa: PLC0415

    if settings.frame_plan is None:
        return build_frame_plan(settings, membership, scale_factors)
    plan = json.loads(settings.frame_plan.read_text())
    if plan.get("schema_version") != 2 or plan.get("kind") != subset.FRAME_PLAN_KIND:
        raise ValueError("expected a version-two frame selection plan")
    if plan.get("sha256") != subset.record_hash(plan):
        raise ValueError("frame-plan content differs from its hash")
    if plan.get("membership_sha256") != membership["sha256"] or plan.get("mode") != settings.mode:
        raise ValueError("frame-plan membership or mode differs from the request")
    sources = {s["relative_dir"]: s for s in membership["sources"]}
    samples = [sample for sample in plan["samples"] if sample["split"] == settings.split]
    if not samples:
        raise ValueError("frame plan has no selected samples in the requested group")
    mode = settings.mode_settings
    expected_draw = window_start_draw(settings.noise_seed) if mode.start_policy == "random" else None
    if plan.get("start_draw") != expected_draw:
        raise ValueError("frame-plan random start draw differs from the request")
    geometry = None
    if isinstance(mode, CausalSettings):
        geometry = causal_core.CausalGeometry(scale_factors, mode.block_latent_frames, mode.context_latent_frames)
        expected = geometry.as_dict()
        for key in ("block_latent_frames", "context_latent_frames", "sink_latent_frames"):
            if plan.get("geometry", {}).get(key) != expected[key]:
                raise ValueError(f"frame-plan {key} differs from the request")
    for sample in samples:
        source = sources.get(sample["source"])
        if source is None or sample["actor"] != source["actor"] or sample["split"] != source["split"]:
            raise ValueError("frame-plan video/person/group is not in the fixed video list")
        ranges = sample.get("ranges", [])
        if not ranges or any(not 0 <= start < end <= source["n_latent_frames"] for start, end in ranges):
            raise ValueError("frame-plan ranges are outside the stored master")
        if isinstance(mode, CausalSettings):
            blocks = sample.get("blocks", [])
            if mode.start_policy == "random" and (
                blocks != [0] or ranges != [[0, mode.block_latent_frames + 1]]
                or sample.get("seed_is_clip_start") is not True
            ):
                raise ValueError("frame-plan random causal segment must be an independent block-zero template")
            if len(blocks) != mode.blocks_per_sample or blocks != list(range(blocks[0], blocks[0] + len(blocks))):
                raise ValueError("frame-plan causal block sequence differs from the request")
            actual_bounds = geometry.plan(source["n_latent_frames"])
            if any(block < 0 or block >= len(actual_bounds) for block in blocks):
                raise ValueError("frame-plan block indices are outside the stored master")
            if ranges != [list(actual_bounds[block]) for block in blocks]:
                raise ValueError("frame-plan ranges disagree with the recorded block indices")
        elif len(ranges) != 1 or "blocks" in sample:
            raise ValueError("bidirectional plans contain one segment and no causal blocks")
        else:
            length = source["n_latent_frames"] if mode.span_latent_frames is None else mode.span_latent_frames
            if ranges != [[0, length]]:
                raise ValueError("bidirectional frame-plan coverage differs from the requested segment rule")
    # Preserve the exact original record/hash; runtime filtering does not rewrite it.
    return plan
