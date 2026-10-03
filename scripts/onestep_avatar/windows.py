#!/usr/bin/env python
"""Freeze a training subset: causal block chains, an actor-disjoint split, and a sha256 pin.

The B2 stage of ``plans/2026-09-10-ltx25-one-step-argavatar-lora.md`` (SS7.2) that turns
"every view the capture pass has encoded" into "the exact blocks this run trains on".
Its output JSON is the **only** thing ``train.py`` reads from this module -- the same
producer/consumer discipline the crop box already has.

Runs in the ``ltx`` env, like everything here except ``build_guidance.py``. It did run under
``argavatar`` until the 2026-09-15 consolidation, when the block plan stopped being
transcribed and started being ``causal_core.CausalGeometry.plan`` itself; importing that pulls
in torch, which is the whole cost of having one definition instead of two.

Four jobs, all of them things the plan says must not be left to the training loop:

1. **Chain blocks.** SS4.4's AR training sample is ``K`` *consecutive* causal blocks of one
   source at the deployed stride, so the cached context the model reads forward is the one
   deployment would give it. Chains never straddle sources.
2. **Split by bare actor id.** SS5.0: actor ids are not unique across ``Part_*``, so the
   split key is the bare id -- an actor that appears in both parts lands wholly on one
   side under either interpretation of who they are.
3. **Exclude the clipped views.** SS4.5's 0.4 % whose subject union is wider than the
   canvas: their box genuinely cuts the subject, so the loss target is wrong, not merely
   tight. ``geometry.effective_pad_factor < 1.0`` names them.
4. **Pin by content.** SS5.0 / SS0.4: the capture manifest fingerprints sources by
   ``size;mtime_ns``, which detects a re-copy but is not a frozen content digest. This
   module hashes the selected ``rgb.mp4`` (and each guide render) with sha256, so a
   mid-sweep re-sync of the corpus is detected rather than silently retrained on.

    conda activate ltx
    python -m scripts.onestep_avatar.windows --name t2 --max-actors 8 --require-guide
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import causal_core, dataset, geometry
from scripts.onestep_avatar.dataset import ClipRef
from scripts.onestep_avatar.hashing import sha256

# The block plan depends on scale factors only through the pixel<->latent time ratio -- the
# latent frame count `plan_blocks` is called with already has that applied (see
# `dataset.capture_master_latent_frames`, the one producer). Spatial factors are irrelevant to
# it, so the deployed 8/32/32 is stated once here rather than threaded through a module that
# never touches a latent.
_SCALE_FACTORS = SpatioTemporalScaleFactors(time=8, height=32, width=32)

SCHEMA_VERSION = 1

# The deployed causal block layout, re-exported from ``causal_core`` -- NOT a second spelling
# of it. SS4.4 (2026-09-14): the unit is a causal BLOCK of latent frames, not a 25-frame
# sliding window. A block is the deployed stride -- 2 latent frames = 16 pixel frames -- and
# block 0 additionally absorbs latent frame 0, the causal VAE's single-pixel keyframe.
#
# These were transcribed here until 2026-09-15, with a test pinning the two spellings
# together, because this module ran in the corpus tree and ``causal_core`` (torch, ltx_core)
# in the model tree and the two could not import each other. Consolidating the package removed
# the seam: the rollout geometry now has exactly one definition, so a subset can no longer be
# frozen against a block plan the trainer does not use.
BLOCK_LATENT_FRAMES = causal_core.BLOCK_LATENT_FRAMES
CONTEXT_LATENT_FRAMES = causal_core.CONTEXT_LATENT_FRAMES
SINK_LATENT_FRAMES = causal_core.SINK_LATENT_FRAMES

DEFAULT_OUTPUT_ROOT = dataset.WORKSPACE_ROOT / "expr" / "onestep_avatar" / "windows"
# SS5.0: 12 is the floor the 09-05 plan's 3-subject holdout failed; at 126 actors a 20 %
# split gives 25, so the floor binds only on small tiers.
DEFAULT_HOLDOUT_FRACTION = 0.2
MIN_HOLDOUT_ACTORS = 12


def plan_blocks(latent_frames: int, block_latent_frames: int = BLOCK_LATENT_FRAMES) -> list[tuple[int, int]]:
    """Causal block bounds in latent frames -- ``causal_core.CausalGeometry.plan`` itself.

    Block 0 is ``[0, 1 + block)``: it absorbs latent frame 0, the causal keyframe, which is
    the product's given real first frame (SS1.2) and stays pinned in the K/V cache for the
    whole rollout. Every later block is exactly ``block`` frames, and a tail shorter than one
    block is dropped rather than shortened -- a short block is a different condition, not a
    smaller one.

    A thin wrapper rather than a reimplementation: the subset this module freezes indexes
    blocks that ``train.py`` then slices out of a master latent, so the two must agree by
    construction, not by a test that notices afterwards.
    """
    return causal_core.CausalGeometry(
        scale_factors=_SCALE_FACTORS, block_latent_frames=block_latent_frames
    ).plan(latent_frames)


def chain_blocks(n_blocks: int, chain_length: int, chain_stride: int | None = None) -> list[list[int]]:
    """Group consecutive block indices into ``K``-block AR chains (SS4.4).

    ``chain_stride`` defaults to ``chain_length`` (disjoint chains, every block seen once
    per epoch). A smaller stride overlaps chains, which buys more *exposure* samples per
    source at the cost of re-seeing the same blocks -- worth it only on a small tier.
    A trailing remainder shorter than ``chain_length`` is dropped rather than padded:
    a short chain is a different training condition, not a smaller one.
    """
    if chain_length < 1:
        raise ValueError("chain_length must be >= 1")
    stride = chain_length if chain_stride is None else chain_stride
    if stride < 1:
        raise ValueError("chain_stride must be >= 1")
    return [
        list(range(start, start + chain_length))
        for start in range(0, n_blocks - chain_length + 1, stride)
    ]


def split_actors(
    actors: list[str], *, holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION, min_holdout: int = MIN_HOLDOUT_ACTORS
) -> tuple[list[str], list[str]]:
    """Deterministic actor-disjoint split -- by a hash of the id, never by position.

    Sorting by id and slicing would correlate the split with ingest order (and therefore
    with capture session, rig calibration and clip length); hashing the id decorrelates it
    while staying reproducible with no seed to record. ``min_holdout`` is a floor, not a
    target: it is dropped only when there are too few actors to honour it, and the caller
    is told so rather than silently getting a 1-actor held-out set.
    """
    unique = sorted(set(actors))
    if len(unique) < 2:
        raise ValueError(f"cannot split {len(unique)} actor(s) into train/held_out")
    wanted = max(round(len(unique) * holdout_fraction), min(min_holdout, len(unique) - 1))
    wanted = min(wanted, len(unique) - 1)
    ranked = sorted(unique, key=lambda actor: hashlib.sha256(actor.encode()).hexdigest())
    held_out = sorted(ranked[:wanted])
    train = sorted(ranked[wanted:])
    return train, held_out


@dataclass(frozen=True)
class SourceRecord:
    """One (clip, view) that survived every gate, with its provenance."""

    relative_dir: str
    clip: str
    actor: str
    view_idx: int
    n_frames: int
    fps: float
    n_latent_frames: int
    n_blocks: int
    box_xyxy: list[float]
    effective_pad_factor: float
    rgb_sha256: str
    guide_sha256: str | None
    # Full input provenance (G6/G3): the masters the trainer actually reads, and the guide's
    # render sidecar. Empty/None when frozen with --skip-hash or when the artifact is absent.
    capture_latent_sha256: str = ""
    guide_latent_sha256: str | None = None
    guide_sidecar_sha256: str | None = None
    # The common span every consumer uses (loss, evaluation, decode, noise, references), and
    # the unused tail it leaves -- recorded rather than padded or silently trimmed.
    span_latent_frames: int | None = None
    unused_tail_latent_frames: int = 0


def _view_dirs(clip: ClipRef) -> list[tuple[int, Path]]:
    out = []
    for view in clip.meta()["views"]:
        idx = int(view["view_idx"])
        out.append((idx, clip.view_dir(idx)))
    return out


def survey_source(
    clip: ClipRef, view_idx: int, manifest: dataset.CaptureManifest, *, require_guide: bool,
    objective: str = dataset.DEFAULT_OBJECTIVE,
    require_guide_latent: bool = False,
) -> tuple[SourceRecord | None, str]:
    """Apply every gate to one view; return its record or the reason it was dropped.

    The gates are ordered cheapest-first and each one is a *named* exclusion rather than a
    silent skip -- the counts end up in the frozen manifest so a shrinking subset is
    visible instead of mysterious.
    """
    view_dir = clip.view_dir(view_idx)
    relative = str(view_dir.resolve().relative_to(manifest.root.resolve()))
    if not manifest.has(view_dir):
        return None, "no_capture_bundle_planned"
    bundle = view_dir / dataset.capture_bundle_name(objective)
    if not bundle.is_file():
        return None, "capture_not_encoded_yet"
    guide = view_dir / dataset.render_name(objective)
    if require_guide and not guide.is_file():
        return None, "no_guide_render"
    if require_guide_latent:
        # The render MP4 existing is not D1 readiness (G6): the trainer reads the guide
        # LATENT, and that latent must come from a render composited under the current contract.
        if not (view_dir / dataset.guide_bundle_name(objective)).is_file():
            return None, "no_guide_latent"
        sidecar = view_dir / dataset.render_metadata_name(objective)
        if not sidecar.is_file():
            return None, "no_guide_sidecar"
        if json.loads(sidecar.read_text()).get("compositing_version") != dataset.GUIDE_COMPOSITING_VERSION:
            return None, "stale_guide_compositing"

    box = manifest.box_for(view_dir)
    record = np.load(clip.bbox_path(view_idx), allow_pickle=True).item()
    view_meta = clip.view_meta(view_idx)
    canonical = geometry.canonical_crop_box(
        record["xyxy"], record["valid"], int(view_meta["width"]), int(view_meta["height"]), manifest.pad_factor
    )
    if tuple(round(v) for v in canonical) != tuple(round(v) for v in box):
        # The same guard build_guidance.py carries: the manifest is what the target latents
        # were encoded with, so a bbox.npy that has moved underneath it means a re-ingest.
        return None, "bbox_moved_since_encode"

    pad = geometry.effective_pad_factor(box, record["xyxy"], record["valid"])
    if pad < 1.0:
        return None, "clipped_subject"

    # From the BUNDLE, not from `latent_frames_for(clip.n_frames())`. A master consolidated
    # out of v1 per-window slices stops at the last whole window, so the video's frame count
    # overstates it by up to one window -- and a block plan sized from the video names blocks
    # the latents do not contain. See `dataset.capture_master_latent_frames`.
    latent_frames = dataset.capture_master_latent_frames(bundle)
    if latent_frames is None:
        return None, "capture_bundle_not_consolidated"
    blocks = plan_blocks(latent_frames)
    if not blocks:
        return None, "too_short_for_one_block"

    return (
        SourceRecord(
            relative_dir=relative,
            clip=clip.name,
            actor=clip.actor_id(),
            view_idx=view_idx,
            n_frames=clip.n_frames(),
            fps=clip.fps(),
            n_latent_frames=latent_frames,
            n_blocks=len(blocks),
            box_xyxy=[float(v) for v in box],
            effective_pad_factor=float(pad),
            rgb_sha256="",  # filled in by the (expensive) hashing pass, over survivors only
            guide_sha256=None,
        ),
        "ok",
    )


def build(
    root: Path,
    *,
    views: list[int] | None,
    require_guide: bool,
    objective: str,
    max_actors: int | None,
    chain_length: int,
    chain_stride: int | None,
    holdout_fraction: float,
    min_holdout: int,
    hash_workers: int,
    skip_hash: bool,
    require_guide_latent: bool = False,
    actor_splits: dict[str, list[str]] | None = None,
    clip_start_only: bool = False,
    span_latent_frames: int | None = None,
    max_views_per_clip: int | None = None,
    clips_allowed: list[str] | None = None,
    block_latent_frames: int = BLOCK_LATENT_FRAMES,
    hash_latents_only: bool = False,
) -> dict:
    """Freeze the subset. With ``actor_splits`` the split is the given explicit lists.

    ``actor_splits`` (``{"train": [...], "validation": [...], "test": [...]}``) replaces the
    hashed two-way split with a predeclared three-way one, frozen before any candidate output
    is seen; every source of an actor outside the lists is dropped as ``actor_not_in_split``.
    ``span_latent_frames`` freezes one common latent span per source (shorter sources are
    dropped, never padded) and records the unused tail; ``clip_start_only`` keeps only the
    chain starting at block 0, so no chain needs ground-truth cache priming.
    """
    manifest = dataset.CaptureManifest.load(root)
    clips = dataset.list_clips(root)
    dropped: dict[str, int] = {}
    survivors: list[tuple[ClipRef, SourceRecord]] = []

    for clip in clips:
        if clips_allowed is not None and clip.name not in clips_allowed:
            dropped["clip_not_selected"] = dropped.get("clip_not_selected", 0) + 1
            continue
        for view_idx, _ in _view_dirs(clip):
            if views is not None and view_idx not in views:
                continue
            record, reason = survey_source(
                clip, view_idx, manifest, require_guide=require_guide, objective=objective,
                require_guide_latent=require_guide_latent,
            )
            if record is not None and span_latent_frames is not None:
                if record.n_latent_frames < span_latent_frames:
                    record, reason = None, "shorter_than_span"
                else:
                    record = SourceRecord(
                        **{
                            **asdict(record),
                            "n_blocks": len(plan_blocks(span_latent_frames, block_latent_frames)),
                            "span_latent_frames": span_latent_frames,
                            "unused_tail_latent_frames": record.n_latent_frames - span_latent_frames,
                        }
                    )
            if record is None:
                dropped[reason] = dropped.get(reason, 0) + 1
            else:
                survivors.append((clip, record))

    if not survivors:
        raise SystemExit(
            f"no (clip, view) survived the gates under {root}: {json.dumps(dropped, sort_keys=True)}"
        )

    # Actor selection happens BEFORE hashing: sha256 over rgb.mp4 is ~1 GB/s at best and the
    # T2 tier needs 8 actors' worth, not 3360 views' worth.
    actors = sorted({record.actor for _, record in survivors})
    if max_actors is not None and len(actors) > max_actors:
        # Same hash ordering the split uses, so a --max-actors subset is a prefix of the
        # full corpus's own ordering rather than an independently arbitrary one.
        actors = sorted(sorted(actors, key=lambda a: hashlib.sha256(a.encode()).hexdigest())[:max_actors])
        survivors = [(clip, record) for clip, record in survivors if record.actor in actors]

    if actor_splits is not None:
        listed = [actor for members in actor_splits.values() for actor in members]
        if len(listed) != len(set(listed)):
            raise SystemExit(f"an actor appears in more than one split: {actor_splits}")
        missing = sorted(set(listed) - set(actors))
        if missing:
            raise SystemExit(f"split actors with no surviving source: {missing}")
        outside = [record for _, record in survivors if record.actor not in set(listed)]
        if outside:
            dropped["actor_not_in_split"] = dropped.get("actor_not_in_split", 0) + len(outside)
        survivors = [(clip, record) for clip, record in survivors if record.actor in set(listed)]
        splits = {name: sorted(members) for name, members in actor_splits.items()}
    else:
        train_actors, held_out_actors = split_actors(
            actors, holdout_fraction=holdout_fraction, min_holdout=min_holdout
        )
        splits = {"train": train_actors, "held_out": held_out_actors}

    if max_views_per_clip is not None:
        kept: list[tuple[ClipRef, SourceRecord]] = []
        per_clip: dict[str, int] = {}
        for clip, record in sorted(survivors, key=lambda item: (item[1].clip, item[1].view_idx)):
            if per_clip.get(record.clip, 0) >= max_views_per_clip:
                dropped["over_views_per_clip"] = dropped.get("over_views_per_clip", 0) + 1
                continue
            per_clip[record.clip] = per_clip.get(record.clip, 0) + 1
            kept.append((clip, record))
        survivors = kept

    if not skip_hash:
        survivors = _hash_survivors(survivors, root, hash_workers, objective, latents_only=hash_latents_only)

    split_of = {actor: name for name, members in splits.items() for actor in members}

    chains = []
    for _, record in survivors:
        source_chains = chain_blocks(record.n_blocks, chain_length, chain_stride)
        if clip_start_only:
            source_chains = source_chains[:1]
        for blocks in source_chains:
            chains.append(
                {
                    "source": record.relative_dir,
                    "split": split_of[record.actor],
                    "actor": record.actor,
                    "blocks": blocks,
                    # SS4.4: a chain that does not start at block 0 has its K/V cache PRIMED
                    # from the ground truth, which is this scheme's one teacher-forced seam.
                    # A chain starting at block 0 needs no priming -- deployment has no
                    # predecessor there either -- so the flag is worth carrying.
                    "seed_is_clip_start": blocks[0] == 0,
                }
            )

    records = [record for _, record in survivors]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "one_step_argavatar_block_chains",
        "corpus_root": str(root),
        "capture_manifest": {
            "pad_factor": manifest.pad_factor,
            "edge": manifest.edge,
            "sha256": sha256(root / dataset.CAPTURE_MANIFEST_NAME),
        },
        "geometry": {
            "attention": "block_causal",
            "block_latent_frames": block_latent_frames,
            "context_latent_frames": CONTEXT_LATENT_FRAMES,
            "sink_latent_frames": SINK_LATENT_FRAMES,
            "stride_frames": block_latent_frames * _SCALE_FACTORS.time,
            "latent_time_scale": _SCALE_FACTORS.time,
            "edge": manifest.edge,
        },
        "chain_length": chain_length,
        "chain_stride": chain_length if chain_stride is None else chain_stride,
        "content_pinned": not skip_hash,
        "content_pin_scope": "none" if skip_hash else ("latent_masters_only" if hash_latents_only else "raw_and_latents"),
        "requires_guide": require_guide,
        # SS1.2: which objective's artifacts this subset was surveyed against. train.py reads
        # it rather than taking its own --objective on faith, so a subset frozen against the
        # composite guides cannot be trained as if it were the white-background pair.
        "objective": objective,
        "excluded": dropped,
        "splits": splits,
        "clip_start_only": clip_start_only,
        "span_latent_frames": span_latent_frames,
        "requires_guide_latent": require_guide_latent,
        "sources": [asdict(record) for record in records],
        "chains": chains,
        "counts": {
            "actors": sum(len(members) for members in splits.values()),
            "sources": len(records),
            "blocks": sum(record.n_blocks for record in records),
            "chains": len(chains),
            "train_chains": sum(1 for chain in chains if chain["split"] == "train"),
            **{
                f"{name}_chains": sum(1 for chain in chains if chain["split"] == name)
                for name in splits
            },
        },
    }


def subset_sha256(subset: dict) -> str:
    """The identity of a whole frozen subset: sources, chains, splits, span and objective.

    ``train.checkpoint_metadata`` used to hash ``subset['sources']`` only, so two subsets with
    different chains or splits shared one "identity" (G3). This is the one spelling of the
    full digest; training stamps it and the probe can compare it.
    """
    keys = ("objective", "sources", "chains", "splits", "span_latent_frames", "clip_start_only", "chain_length")
    payload = {key: subset.get(key) for key in keys}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _hash_survivors(
    survivors: list[tuple[ClipRef, SourceRecord]], root: Path, workers: int, objective: str,
    latents_only: bool = False,
) -> list[tuple[ClipRef, SourceRecord]]:
    """sha256 every selected ``rgb.mp4`` and guide render, in parallel.

    This is the SS5.0 pin, and it is the expensive part of this module -- a 4096x3000 h264
    source is hundreds of MB. It runs over the *selected* subset only, after actor
    selection has shrunk it.
    """

    def digests(item: tuple[ClipRef, SourceRecord]) -> tuple[str, str | None]:
        _, record = item
        view_dir = root / record.relative_dir
        guide = view_dir / dataset.render_name(objective)
        guide_latent = view_dir / dataset.guide_bundle_name(objective)
        sidecar = view_dir / dataset.render_metadata_name(objective)
        if latents_only:
            # Corpus-scale freeze: pin only the ~5 MB masters the trainer reads; hashing every
            # raw rgb.mp4 (hundreds of MB each) per freeze is impractical at 3,000+ views.
            return {
                "rgb_sha256": "",
                "guide_sha256": None,
                "capture_latent_sha256": sha256(view_dir / dataset.capture_bundle_name(objective)),
                "guide_latent_sha256": sha256(guide_latent) if guide_latent.is_file() else None,
                "guide_sidecar_sha256": None,
            }
        return {
            "rgb_sha256": sha256(view_dir / "rgb.mp4"),
            "guide_sha256": sha256(guide) if guide.is_file() else None,
            "capture_latent_sha256": sha256(view_dir / dataset.capture_bundle_name(objective)),
            "guide_latent_sha256": sha256(guide_latent) if guide_latent.is_file() else None,
            "guide_sidecar_sha256": sha256(sidecar) if sidecar.is_file() else None,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(digests, survivors))
    return [
        (clip, SourceRecord(**{**asdict(record), **hashes}))
        for (clip, record), hashes in zip(survivors, results, strict=True)
    ]


def verify(subset: dict) -> list[str]:
    """Re-hash every pinned source and report the ones that have changed.

    Cheap insurance to run before a training launch: SS5.0's "two copies of the capture
    data exist and the ingest is still running" is exactly the situation where a subset
    silently stops describing what is on disk.
    """
    if not subset.get("content_pinned"):
        return ["subset was frozen with --skip-hash; there is nothing to verify against"]
    root = Path(subset["corpus_root"])
    problems = []
    for record in subset["sources"]:
        view_dir = root / record["relative_dir"]
        rgb = view_dir / "rgb.mp4"
        if not rgb.is_file():
            problems.append(f"{record['relative_dir']}: rgb.mp4 is gone")
            continue
        if record["rgb_sha256"] and sha256(rgb) != record["rgb_sha256"]:
            problems.append(f"{record['relative_dir']}: rgb.mp4 content changed since the freeze")
        guide = view_dir / dataset.render_name(subset.get("objective", dataset.DEFAULT_OBJECTIVE))
        if record["guide_sha256"] is not None:
            if not guide.is_file():
                problems.append(f"{record['relative_dir']}: {guide.name} is gone")
            elif sha256(guide) != record["guide_sha256"]:
                problems.append(f"{record['relative_dir']}: guide render changed since the freeze")
        objective = subset.get("objective", dataset.DEFAULT_OBJECTIVE)
        for key, name in (
            ("capture_latent_sha256", dataset.capture_bundle_name(objective)),
            ("guide_latent_sha256", dataset.guide_bundle_name(objective)),
            ("guide_sidecar_sha256", dataset.render_metadata_name(objective)),
        ):
            if record.get(key):
                path = view_dir / name
                if not path.is_file() or sha256(path) != record[key]:
                    problems.append(f"{record['relative_dir']}: {name} changed or is gone since the freeze")
    return problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-root", type=Path, default=dataset.DEFAULT_CORPUS_ROOT)
    p.add_argument("--name", default="subset", help="output file stem under --output-root")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--views", type=int, nargs="+", default=None, help="restrict to these view indices")
    p.add_argument("--require-guide", action="store_true", help="keep only views whose guide render already exists")
    p.add_argument(
        "--objective",
        choices=dataset.OBJECTIVES,
        default=dataset.DEFAULT_OBJECTIVE,
        help="SS1.2. Which objective's artifacts to survey and pin: bg (default) or white. "
        "Recorded in the subset, and train.py refuses a mismatch.",
    )
    p.add_argument("--max-actors", type=int, default=None, help="T2-style tier cap, applied before hashing")
    p.add_argument("--chain-length", type=int, default=3, help="K: causal blocks per AR training sample (SS4.4)")
    p.add_argument("--chain-stride", type=int, default=None, help="default: chain-length (disjoint chains)")
    p.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    p.add_argument("--min-holdout-actors", type=int, default=MIN_HOLDOUT_ACTORS)
    p.add_argument("--hash-workers", type=int, default=8)
    p.add_argument("--skip-hash", action="store_true", help="skip the sha256 pin (development only -- SS5.0 requires it)")
    p.add_argument("--verify", type=Path, default=None, help="re-hash an existing subset JSON and exit")
    p.add_argument(
        "--require-guide-latent", action="store_true",
        help="keep only views whose guide LATENT exists and whose render sidecar is current under "
        "GUIDE_COMPOSITING_VERSION (implies the D1 readiness G6 asks for, not just a render MP4)",
    )
    p.add_argument("--train-actors", nargs="+", default=None, help="explicit train actor ids (all three lists together)")
    p.add_argument("--validation-actors", nargs="+", default=None)
    p.add_argument("--test-actors", nargs="+", default=None)
    p.add_argument("--clip-start-only", action="store_true", help="keep only each source's block-0 chain (no priming)")
    p.add_argument("--span-latent-frames", type=int, default=None, help="freeze one common latent span per source")
    p.add_argument("--max-views-per-clip", type=int, default=None, help="keep the lowest-index N views per clip")
    p.add_argument("--clips", nargs="+", default=None, help="restrict to these clip names (e.g. Part_2_0007_01)")
    p.add_argument("--hash-latents-only", action="store_true",
                   help="pin only the latent masters the trainer reads (corpus-scale freezes); raw videos are not hashed")
    p.add_argument(
        "--block-latent-frames", type=int, default=BLOCK_LATENT_FRAMES,
        help="Causal block size the chains are planned with. With --span-latent-frames N and block N-1, "
        "block 0 is the whole span: one bidirectional block per clip (whole-clip training, K=1).",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.verify is not None:
        problems = verify(json.loads(args.verify.read_text()))
        for problem in problems:
            print(f"CHANGED: {problem}")  # noqa: T201
        print(f"{len(problems)} problem(s)")  # noqa: T201
        return 1 if problems else 0

    explicit = (args.train_actors, args.validation_actors, args.test_actors)
    if any(v is not None for v in explicit) and any(v is None for v in explicit):
        p.error("--train-actors, --validation-actors and --test-actors go together")
    actor_splits = (
        None
        if args.train_actors is None
        else {"train": args.train_actors, "validation": args.validation_actors, "test": args.test_actors}
    )
    subset = build(
        args.corpus_root,
        require_guide_latent=args.require_guide_latent,
        actor_splits=actor_splits,
        clip_start_only=args.clip_start_only,
        span_latent_frames=args.span_latent_frames,
        max_views_per_clip=args.max_views_per_clip,
        clips_allowed=args.clips,
        block_latent_frames=args.block_latent_frames,
        hash_latents_only=args.hash_latents_only,
        views=args.views,
        require_guide=args.require_guide,
        objective=args.objective,
        max_actors=args.max_actors,
        chain_length=args.chain_length,
        chain_stride=args.chain_stride,
        holdout_fraction=args.holdout_fraction,
        min_holdout=args.min_holdout_actors,
        hash_workers=args.hash_workers,
        skip_hash=args.skip_hash,
    )
    summary = {
        "counts": subset["counts"],
        "excluded": subset["excluded"],
        "splits": {k: len(v) for k, v in subset["splits"].items()},
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))  # noqa: T201
        return 0
    out = args.output_root / f"{args.name}.json"
    dataset.atomic_write(out, lambda temp: temp.write_text(json.dumps(subset, indent=2)))
    print(json.dumps(summary, indent=2))  # noqa: T201
    print(f"wrote {out}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
