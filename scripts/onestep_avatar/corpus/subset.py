"""Create checked fixed-video membership; frame selection belongs to training settings.

See doc/corpus/subset.md. Survey saved crop records and continuous masters on CPU,
assign person groups before model work, and publish only a new content-pinned list.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
from pathlib import Path

import torch

from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.hashing import sha256

SCHEMA_VERSION = 2
KIND = "onestep_avatar_membership"
FRAME_PLAN_KIND = "onestep_avatar_frame_plan"


def record_hash(record: dict) -> str:
    """Hash exact JSON content, excluding the saved digest itself."""
    payload = {key: value for key, value in record.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def membership_hash(membership: dict) -> str:
    """Relocating the corpus does not change the fixed videos or their groups."""
    return record_hash({key: value for key, value in membership.items() if key not in {"corpus_root", "sha256"}})


def validate_membership(membership: dict) -> None:
    """Check schema, source IDs, people, groups and record integrity before loading models."""
    if membership.get("schema_version") != SCHEMA_VERSION or membership.get("kind") != KIND:
        raise ValueError("expected a version-two onestep_avatar_membership record")
    if membership.get("objective") not in dataset.OBJECTIVES:
        raise ValueError("membership must record bg or white")
    forbidden = {"geometry", "chains", "chain_length", "chain_stride", "attention", "context_latent_frames"}
    if forbidden.intersection(membership):
        raise ValueError("fixed video membership cannot contain causal frame settings")
    sources = membership.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("membership must contain selected videos")
    split_of = {}
    for group, actors in membership.get("splits", {}).items():
        for actor in actors:
            if actor in split_of:
                raise ValueError(f"actor {actor} appears in multiple groups")
            split_of[actor] = group
    seen = set()
    for source in sources:
        name = source["relative_dir"]
        if name in seen or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"invalid or duplicate video ID {name}")
        seen.add(name)
        if source.get("split") != split_of.get(source["actor"]):
            raise ValueError(f"{name}: person/group does not match the fixed split lists")
        if any(key in source for key in ("n_blocks", "span_latent_frames", "unused_tail_latent_frames")):
            raise ValueError(f"{name}: frame selection belongs in the frame plan")
    if membership.get("sha256") != membership_hash(membership):
        raise ValueError("membership content does not match its SHA-256")


def split_actors(actors: list[str], held_out_fraction: float = 0.1, min_held_out: int = 1) -> dict[str, list[str]]:
    """Group person IDs independently of clip discovery, with at least one training person."""
    if (
        isinstance(held_out_fraction, bool)
        or not isinstance(held_out_fraction, (int, float))
        or not math.isfinite(held_out_fraction)
        or not 0 <= held_out_fraction <= 1
        or type(min_held_out) is not int
        or min_held_out < 0
    ):
        raise ValueError("held-out fraction must be in [0, 1] and its minimum a nonnegative integer")
    ordered = _actor_order(actors)
    if len(ordered) < 2:
        raise ValueError("automatic grouping requires at least two people; supply explicit group lists")
    count = min(max(round(len(ordered) * held_out_fraction), min(min_held_out, len(ordered) - 1)), len(ordered) - 1)
    return {"train": ordered[count:], "held_out": ordered[:count]}


def _actor_order(actors: list[str]) -> list[str]:
    if any(not isinstance(actor, str) or not actor for actor in actors):
        raise ValueError("person IDs must be nonempty strings")
    return sorted(set(actors), key=lambda actor: (hashlib.sha256(actor.encode()).hexdigest(), actor))


def _check_splits(splits: dict[str, list[str]], available: set[str]) -> dict[str, str]:
    if not isinstance(splits, dict) or not splits:
        raise ValueError("explicit groups must be a nonempty record")
    split_of = {}
    for group, actors in splits.items():
        if (
            not isinstance(group, str)
            or not group
            or not isinstance(actors, list)
            or any(not isinstance(actor, str) or not actor for actor in actors)
        ):
            raise ValueError("group names and person IDs must be nonempty strings")
        for actor in actors:
            if actor in split_of:
                raise ValueError(f"person {actor} appears more than once in the group lists")
            if actor not in available:
                raise ValueError(f"person {actor} has no available selected video")
            split_of[actor] = group
    if not split_of:
        raise ValueError("group lists must select at least one person")
    return split_of


def _source_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 4
        or relative.parts[2] != "views"
        or str(relative) != name
    ):
        raise ValueError(f"invalid relative video ID {name!r}")
    view = root / relative
    if not view.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"video {name!r} resolves outside the corpus")
    return view


def _one_view_membership(root: Path, objective: str, source: dict) -> dict:
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "corpus_root": str(root),
        "objective": objective,
        "sources": [{**source, "split": "selected"}],
        "splits": {"selected": [source["actor"]]},
    }
    record["sha256"] = membership_hash(record)
    return record


def pin_content(
    root: Path,
    name: str,
    objective: str,
    *,
    require_guide: bool = False,
    crop_manifest: dataset.CaptureManifest | None = None,
) -> dict:
    """Pin one current view, checking its saved crop and the shared paired-master rules."""
    if objective not in dataset.OBJECTIVES:
        raise ValueError("membership objective must be bg or white")
    root = Path(root).resolve()
    view = _source_path(root, name)
    metadata_path = view.parent.parent / "meta.json"
    metadata_hash = sha256(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    person = metadata["actor"]["id"]
    if type(person) not in (int, str) or not str(person).strip():
        raise ValueError("clip metadata must identify a person")
    actor = str(person)
    if metadata.get("_pipeline", {}).get("state") != "done":
        raise ValueError("clip metadata must identify a person and complete pipeline state")
    manifest = crop_manifest or dataset.CaptureManifest.load(root)
    capture = view / dataset.capture_bundle_name(objective)
    capture_hash = sha256(capture)
    shape, encoding = _encode_record(capture)
    if encoding.get("source") != name or encoding.get("objective") != objective:
        raise ValueError("capture source or objective differs from the selected view")
    if (
        encoding.get("box_xyxy") != list(manifest.box_for(view))
        or encoding.get("edge") != manifest.edge
        or not encoding.get("vae_fingerprint")
        or not encoding.get("input_fingerprint")
        or type(encoding.get("encode_contract_version")) is not int
        or encoding["encode_contract_version"] < 1
        or type(encoding.get("pixel_frames")) is not int
        or encoding["pixel_frames"] < 1
    ):
        raise ValueError("capture crop, VAE or encoding contract is missing or differs from the crop manifest")
    record = {
        "relative_dir": name,
        "actor": actor,
        "fps": encoding["fps"],
        "n_latent_frames": shape[1],
        "shape": shape,
        "capture_encode_record": encoding,
        "capture_latent_sha256": capture_hash,
        "rgb_sha256": sha256(view / "rgb.mp4"),
        "clip_metadata_sha256": metadata_hash,
    }
    if require_guide:
        guide = view / dataset.guide_bundle_name(objective)
        record["guide_latent_sha256"] = sha256(guide)
        _, record["guide_encode_record"] = _encode_record(guide)
        record["guide_sha256"] = sha256(view / dataset.render_name(objective))
        record["guide_sidecar_sha256"] = sha256(view / dataset.render_metadata_name(objective))
    dataset.ClipStore(_one_view_membership(root, objective, record)).load(name, require_guide=require_guide)
    _check_source_pins(root, objective, record)
    return record


def _check_source_pins(root: Path, objective: str, source: dict) -> None:
    view = _source_path(root, source["relative_dir"])
    paths = {
        "rgb_sha256": view / "rgb.mp4",
        "capture_latent_sha256": view / dataset.capture_bundle_name(objective),
        "clip_metadata_sha256": view.parent.parent / "meta.json",
    }
    if source.get("guide_latent_sha256"):
        paths.update(
            guide_latent_sha256=view / dataset.guide_bundle_name(objective),
            guide_sha256=view / dataset.render_name(objective),
            guide_sidecar_sha256=view / dataset.render_metadata_name(objective),
        )
    if any(sha256(path) != source[key] for key, path in paths.items()):
        raise ValueError(f"{source['relative_dir']}: input content changed during membership preparation")


def survey_sources(
    root: Path, objective: str, *, require_guide: bool = False, source_ids: list[str] | None = None
) -> tuple[list[dict], dict[str, str], str]:
    """Read available views and record exclusions before person grouping."""
    root = Path(root).resolve()
    manifest_path = root / dataset.CAPTURE_MANIFEST_NAME
    manifest_hash = sha256(manifest_path)
    crop_manifest = dataset.CaptureManifest.load(root)
    if source_ids is None:
        names = [
            str(view.relative_to(root))
            for clip in dataset.list_clips(root, done_only=False)
            for view in sorted((clip.dir / "views").glob("view*"))
            if view.is_dir()
        ]
    else:
        if (
            not source_ids
            or any(not isinstance(name, str) or not name for name in source_ids)
            or len(source_ids) != len(set(source_ids))
        ):
            raise ValueError("explicit source IDs must be nonempty and unique")
        names = sorted(source_ids)
    sources, excluded = [], {}
    for name in names:
        try:
            sources.append(pin_content(root, name, objective, require_guide=require_guide, crop_manifest=crop_manifest))
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            EOFError,
            pickle.UnpicklingError,
            SystemExit,
        ) as exc:
            reason = str(exc).replace(str(root), "<corpus>")
            if source_ids is not None:
                raise ValueError(f"selected video {name} is unavailable: {reason}") from exc
            excluded[name] = reason
    if sha256(manifest_path) != manifest_hash:
        raise ValueError("crop manifest changed during membership preparation")
    return sources, excluded, manifest_hash


def build_membership(
    root: Path,
    *,
    objective: str = dataset.DEFAULT_OBJECTIVE,
    require_guide: bool = False,
    source_ids: list[str] | None = None,
    splits: dict[str, list[str]] | None = None,
    max_people: int | None = None,
    held_out_fraction: float = 0.1,
    min_held_out: int = 1,
) -> dict:
    """Create an immutable video list from current files without choosing model frames."""
    if objective not in dataset.OBJECTIVES or type(require_guide) is not bool:
        raise ValueError("membership requires objective bg or white and an explicit guide requirement")
    if max_people is not None and (type(max_people) is not int or max_people < 1):
        raise ValueError("max_people must be a positive integer")
    root = Path(root).resolve()
    sources, excluded, manifest_hash = survey_sources(
        root, objective, require_guide=require_guide, source_ids=source_ids
    )
    people = _actor_order([source["actor"] for source in sources])
    if max_people is not None:
        people = people[:max_people]
    available = set(people)
    declared = copy.deepcopy(splits) if splits is not None else split_actors(people, held_out_fraction, min_held_out)
    split_of = _check_splits(declared, available)
    selected = []
    for source in sources:
        actor = source["actor"]
        if actor not in available or actor not in split_of:
            excluded[source["relative_dir"]] = "person_not_selected"
        else:
            selected.append({**source, "split": split_of[actor]})
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "corpus_root": str(root),
        "objective": objective,
        "capture_manifest": {"sha256": manifest_hash},
        "sources": selected,
        "splits": declared,
        "excluded": excluded,
        "requires_guide_latent": require_guide,
        "content_pin_scope": "raw_and_latents",
        "selection": {
            "source_ids": sorted(source_ids) if source_ids is not None else None,
            "max_people": max_people,
            "group_rule": "explicit_lists" if splits is not None else "sha256_person_ids",
            "held_out_fraction": held_out_fraction if splits is None else None,
            "min_held_out": min_held_out if splits is None else None,
            "minimum_held_out_satisfied": (len(declared["held_out"]) >= min_held_out) if splits is None else None,
        },
    }
    record["sha256"] = membership_hash(record)
    verify_prepared_membership(record)
    return record


def verify_prepared_membership(record: dict) -> None:
    """Recheck the selected producer bytes before a new list is published."""
    validate_membership(record)
    root = Path(record["corpus_root"])
    if sha256(root / dataset.CAPTURE_MANIFEST_NAME) != record["capture_manifest"]["sha256"]:
        raise ValueError("crop manifest changed during membership preparation")
    for source in record["sources"]:
        _check_source_pins(root, record["objective"], source)
    dataset.ClipStore(record).verify(require_guide=record["requires_guide_latent"])


def _encode_record(path: Path) -> tuple[list[int], dict]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    master, _ = dataset.load_training_master(path, bundle=bundle)
    if not torch.isfinite(master).all():
        raise ValueError(f"{path}: master contains nonfinite values")
    fields = (
        "schema_version",
        "encode_contract_version",
        "source",
        "objective",
        "vae_fingerprint",
        "input_fingerprint",
        "fps",
        "pixel_frames",
        "box_xyxy",
        "edge",
        "compositing_version",
    )
    return list(master.shape), {key: bundle[key] for key in fields if key in bundle}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--corpus-root", type=Path, default=dataset.DEFAULT_CORPUS_ROOT)
    parser.add_argument("--objective", choices=dataset.OBJECTIVES, default=dataset.DEFAULT_OBJECTIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-guide", action="store_true")
    parser.add_argument("--source", action="append")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--max-people", type=int)
    parser.add_argument("--held-out-fraction", type=float, default=0.1)
    parser.add_argument("--min-held-out", type=int, default=1)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("membership output must be a new file")
    splits_hash = sha256(args.splits) if args.splits is not None else None
    splits = json.loads(args.splits.read_text()) if args.splits is not None else None
    membership = build_membership(
        args.corpus_root,
        objective=args.objective,
        require_guide=args.require_guide,
        source_ids=args.source,
        splits=splits,
        max_people=args.max_people,
        held_out_fraction=args.held_out_fraction,
        min_held_out=args.min_held_out,
    )

    def publish(temporary: Path) -> None:
        temporary.write_text(json.dumps(membership, indent=2) + "\n")
        verify_prepared_membership(membership)
        if args.splits is not None and sha256(args.splits) != splits_hash:
            raise ValueError("explicit group input changed during membership preparation")
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("membership output appeared during preparation")

    dataset.atomic_write(args.output, publish, replace_existing=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
