"""Create checked fixed-video membership; frame selection belongs to training settings.

See doc/corpus/subset.md. Survey saved crop records and continuous masters on CPU,
assign person groups before model work, and publish only a new content-pinned list.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import functools
import hashlib
import io
import json
import math
import multiprocessing
import pickle
from collections.abc import Callable
from pathlib import Path

import torch

from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.hashing import sha256

SCHEMA_VERSION = 2
KIND = "onestep_avatar_membership"
FRAME_PLAN_KIND = "onestep_avatar_frame_plan"
CSV_FORMAT = "onestep_avatar_membership_sources_csv_v1"
DEFAULT_WORKERS = 8


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


def _flatten(record: dict, prefix: str = "") -> dict:
    """Nested nonempty records become dotted columns; lists and scalars are leaf cells."""
    flat = {}
    for key, value in record.items():
        if "." in key:
            raise ValueError(f"membership field {key!r} cannot be a table column")
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _unflatten(flat: dict) -> dict:
    record: dict = {}
    for column, value in flat.items():
        *parents, leaf = column.split(".")
        node = record
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = value
    return record


def summarize_exclusions(record: dict) -> dict:
    """Replace the per-view exclusion list by counts per reason, as the CSV form stores it.

    Nothing downstream reads individual exclusions; the counts keep what was dropped and
    why. The content hash is recomputed because exclusions are part of the hashed record.
    """
    summary = {key: value for key, value in record.items() if key not in {"excluded", "sha256"}}
    counts: dict[str, int] = {}
    for reason in record.get("excluded", {}).values():
        counts[reason] = counts.get(reason, 0) + 1
    summary["excluded_counts"] = dict(sorted(counts.items()))
    summary["sha256"] = membership_hash(summary)
    return summary


def sources_csv_path(path: Path) -> Path:
    """The per-view table that sits beside a membership JSON: same stem, ``.csv``."""
    return path.with_suffix(".csv")


def membership_files(record: dict, path: Path) -> tuple[str, str]:
    """Split a membership into its small JSON text and its pure per-view CSV text.

    The CSV has one header row and one row per selected view, holding every per-view
    field as a column (nested records as dotted names). String columns hold the raw
    string, other columns one JSON value; an empty cell means the field is absent from
    that view. The JSON holds the list-level fields and names, hashes and types the CSV,
    so pinning the JSON bytes pins the whole list; ``sha256`` stays the content hash.
    """
    if "excluded" in record:
        raise ValueError("summarize per-view exclusions before writing the table form")
    flats = [_flatten(source) for source in record["sources"]]
    names = list(dict.fromkeys(column for flat in flats for column in flat))
    if any(other.startswith(f"{name}.") for name in names for other in names):
        raise ValueError("a membership field is a record in one view and a value in another")
    columns = [
        {"name": name, "type": "str" if all(isinstance(f[name], str) for f in flats if name in f) else "json"}
        for name in names
    ]
    if any(flat.get(c["name"]) == "" for flat in flats for c in columns if c["type"] == "str"):
        raise ValueError("an empty string field cannot be stored in a membership CSV")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(names)
    for flat in flats:
        writer.writerow(
            ""
            if c["name"] not in flat
            else flat[c["name"]] if c["type"] == "str" else json.dumps(flat[c["name"]], separators=(",", ":"))
            for c in columns
        )
    table = buffer.getvalue()
    meta = {key: value for key, value in record.items() if key != "sources"}
    meta["sources_csv"] = {
        "format": CSV_FORMAT,
        "path": sources_csv_path(path).name,
        "sha256": hashlib.sha256(table.encode()).hexdigest(),
        "rows": len(flats),
        "columns": columns,
    }
    return json.dumps(meta, indent=2) + "\n", table


def read_sources_csv(directory: Path, table: dict) -> list[dict]:
    if table.get("format") != CSV_FORMAT or Path(table["path"]).name != table["path"]:
        raise ValueError(f"membership sources must be a {CSV_FORMAT} file beside the JSON")
    data = (directory / table["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != table["sha256"]:
        raise ValueError(f"membership table {table['path']} does not match its SHA-256")
    columns = table["columns"]
    reader = csv.reader(io.StringIO(data.decode(), newline=""))
    if next(reader, None) != [column["name"] for column in columns]:
        raise ValueError("membership CSV header differs from its declared columns")
    sources = []
    for row in reader:
        if len(row) != len(columns):
            raise ValueError("membership CSV row has the wrong number of cells")
        sources.append(_unflatten({
            column["name"]: cell if column["type"] == "str" else json.loads(cell)
            for column, cell in zip(columns, row, strict=True)
            if cell != ""
        }))
    if len(sources) != table["rows"]:
        raise ValueError(f"membership table {table['path']} has the wrong row count")
    return sources


def load_membership(path: Path) -> dict:
    """Read a membership JSON; its per-view rows come from the CSV it names, when present."""
    path = Path(path)
    record = json.loads(path.read_text())
    if "sources_csv" not in record:
        return record
    table = record.pop("sources_csv")
    record["sources"] = read_sources_csv(path.parent, table)
    validate_membership(record)
    return record


def write_membership(path: Path, record: dict, check: Callable[[], None] = lambda: None) -> None:
    """Publish the new CSV, then the JSON that pins it; refuse any existing file.

    ``check`` runs after both are written and before the JSON appears. On failure the CSV
    this call published is removed, so no partial list is left behind.
    """
    meta, table = membership_files(record, path)
    table_path = sources_csv_path(path)
    for target in (table_path, path):
        if target.exists() or target.is_symlink():
            raise ValueError(f"membership output {target} must be a new file")
    dataset.atomic_write(table_path, lambda temporary: temporary.write_text(table), replace_existing=False)
    try:

        def publish(temporary: Path) -> None:
            temporary.write_text(meta)
            check()

        dataset.atomic_write(path, publish, replace_existing=False)
    except BaseException:
        table_path.unlink(missing_ok=True)
        raise
    if load_membership(path) != record:
        raise ValueError("published membership files do not reproduce the checked record")


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


def _map(function: Callable, items: list, workers: int) -> list:
    """Order-preserving map; views are independent, so pinning parallelizes across processes."""
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(min(workers, len(items)), mp_context=context) as pool:
        return list(pool.map(function, items, chunksize=16))


def _try_pin(
    root: Path, objective: str, require_guide: bool, crop_manifest: dataset.CaptureManifest, name: str
) -> tuple[dict | None, str | None]:
    try:
        return pin_content(root, name, objective, require_guide=require_guide, crop_manifest=crop_manifest), None
    except (
        OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError, SystemExit
    ) as exc:
        # Name the failing view generically so identical failures group in the exclusion counts.
        return None, str(exc).replace(str(root / name), "<view>").replace(str(root), "<corpus>")


def _pins_ok(root: Path, objective: str, source: dict) -> str | None:
    try:
        _check_source_pins(root, objective, source)
    except ValueError as exc:
        return str(exc)
    return None


def survey_sources(
    root: Path,
    objective: str,
    *,
    require_guide: bool = False,
    source_ids: list[str] | None = None,
    workers: int = 1,
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
    pin = functools.partial(_try_pin, root, objective, require_guide, crop_manifest)
    for name, (source, reason) in zip(names, _map(pin, names, workers), strict=True):
        if source is not None:
            sources.append(source)
        elif source_ids is not None:
            raise ValueError(f"selected video {name} is unavailable: {reason}")
        else:
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
    workers: int = 1,
    verify: bool = True,
) -> dict:
    """Create an immutable video list from current files without choosing model frames.

    ``verify=False`` leaves the content recheck to the caller, which must run
    ``verify_prepared_membership`` before publishing (the CLI does it at publication).
    """
    if objective not in dataset.OBJECTIVES or type(require_guide) is not bool:
        raise ValueError("membership requires objective bg or white and an explicit guide requirement")
    if max_people is not None and (type(max_people) is not int or max_people < 1):
        raise ValueError("max_people must be a positive integer")
    root = Path(root).resolve()
    sources, excluded, manifest_hash = survey_sources(
        root, objective, require_guide=require_guide, source_ids=source_ids, workers=workers
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
    if verify:
        verify_prepared_membership(record, workers=workers)
    return record


def verify_prepared_membership(record: dict, workers: int = 1) -> None:
    """Recheck the selected producer bytes before a new list is published."""
    validate_membership(record)
    root = Path(record["corpus_root"])
    if sha256(root / dataset.CAPTURE_MANIFEST_NAME) != record["capture_manifest"]["sha256"]:
        raise ValueError("crop manifest changed during membership preparation")
    check = functools.partial(_pins_ok, root, record["objective"])
    failure = next((reason for reason in _map(check, record["sources"], workers) if reason), None)
    if failure:
        raise ValueError(failure)
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
    parser.add_argument("--output", type=Path, required=True, help="new .json path; the per-view .csv goes beside it")
    parser.add_argument("--require-guide", action="store_true")
    parser.add_argument("--source", action="append")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--max-people", type=int)
    parser.add_argument("--held-out-fraction", type=float, default=0.1)
    parser.add_argument("--min-held-out", type=int, default=1)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="CPU processes for pinning views")
    args = parser.parse_args(argv)
    if args.output.suffix != ".json" or any(
        path.exists() or path.is_symlink() for path in (args.output, sources_csv_path(args.output))
    ):
        raise SystemExit("membership output must be a new file: a .json path with no .csv of the same name")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
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
        workers=args.workers,
        verify=False,
    )
    membership = summarize_exclusions(membership)

    def check() -> None:
        verify_prepared_membership(membership, workers=args.workers)
        if args.splits is not None and sha256(args.splits) != splits_hash:
            raise ValueError("explicit group input changed during membership preparation")

    write_membership(args.output, membership, check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
