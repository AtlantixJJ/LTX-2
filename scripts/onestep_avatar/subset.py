"""Fixed video membership and explicit conversion of original block-chain subsets.

See doc/subset.md. Conversion reads and pins existing producer artifacts and saves
new records. It never re-encodes masters or overwrites original scientific inputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import dataset
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model.causal import CausalGeometry

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
        raise ValueError("expected a version-two onestep_avatar_membership record; convert the old subset")
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


def _pin(path: Path, previous: str | None) -> str:
    actual = sha256(path)
    if previous and actual != previous:
        raise ValueError(f"{path}: content differs from the original subset pin")
    return actual


def _encode_record(path: Path) -> tuple[list[int], dict]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    master, _ = dataset.load_training_master(path, bundle=bundle)
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


def _convert_source(original: dict, root: Path, objective: str, split_of: dict, require_guide: bool) -> dict:
    """Read and pin one unchanged producer output, with its original person/group."""
    name = original["relative_dir"]
    view = root / name
    if original["actor"] not in split_of:
        raise ValueError(f"{name}: actor missing from the original split lists")
    record = {
        key: copy.deepcopy(value)
        for key, value in original.items()
        if key not in {"n_blocks", "span_latent_frames", "unused_tail_latent_frames"}
    }
    record["split"] = split_of[record["actor"]]
    capture = view / dataset.capture_bundle_name(objective)
    shape, encoding = _encode_record(capture)
    if shape[1] != original["n_latent_frames"] or encoding["fps"] != original["fps"]:
        raise ValueError(f"{name}: original frame count or frame rate differs from the master")
    if encoding.get("objective", "bg") != objective:
        raise ValueError(f"{name}: capture background differs from the original subset")
    record["shape"] = shape
    record["capture_encode_record"] = encoding
    record["capture_latent_sha256"] = _pin(capture, original.get("capture_latent_sha256"))
    guide = view / dataset.guide_bundle_name(objective)
    if require_guide or original.get("guide_latent_sha256"):
        guide_shape, guide_record = _encode_record(guide)
        if guide_shape != shape or guide_record["fps"] != encoding["fps"]:
            raise ValueError(f"{name}: guide/capture shape or frame rate differs")
        for key in ("objective", "source", "box_xyxy", "vae_fingerprint", "edge", "encode_contract_version"):
            if key not in guide_record or guide_record[key] != encoding.get(key):
                raise ValueError(f"{name}: guide/capture {key} differs or is not recorded")
        render_hash = _pin(view / dataset.render_name(objective), original.get("guide_sha256"))
        if guide_record.get("input_fingerprint") != render_hash:
            raise ValueError(f"{name}: guide master does not encode the current render")
        sidecar = view / dataset.render_metadata_name(objective)
        render = json.loads(sidecar.read_text())
        if render.get("compositing_version") != dataset.GUIDE_COMPOSITING_VERSION:
            raise ValueError(f"{name}: outdated guide compositing record")
        record["guide_encode_record"] = guide_record
        record["guide_latent_sha256"] = _pin(guide, original.get("guide_latent_sha256"))
        record["guide_sidecar_sha256"] = _pin(sidecar, original.get("guide_sidecar_sha256"))
    # Existing raw pins stay exact. Check them rather than changing them silently.
    for key, artifact in (("rgb_sha256", "rgb.mp4"), ("guide_sha256", dataset.render_name(objective))):
        if original.get(key):
            _pin(view / artifact, original[key])
    return record


def from_saved_probe(path: Path) -> dict:
    """Pin one recorded paired probe source, preserving its original encoded identities."""
    digest = sha256(path)
    probe = json.loads(path.read_text())
    if (probe.get("kind") != "d1_paired_source_probe" or probe.get("objective") not in dataset.OBJECTIVES
            or not probe.get("videos")):
        raise ValueError("replay membership requires a saved paired probe manifest")
    art = probe["videos"][0]["artifacts"]
    fields = ("view", "capture", "guide", "capture_sha256", "guide_sha256", "fps")
    if any(any(row["artifacts"].get(key) != art.get(key) for key in fields) for row in probe["videos"]):
        raise ValueError("saved probe rows identify different input sources")
    objective = probe["objective"]
    capture, guide = Path(art["capture"]).resolve(), Path(art["guide"]).resolve()
    if (capture.parent != Path(art["view"]).resolve() or guide.parent != capture.parent
            or capture.name != dataset.capture_bundle_name(objective)
            or guide.name != dataset.guide_bundle_name(objective)):
        raise ValueError("saved probe paths do not identify the recorded paired masters")
    if sha256(capture) != art["capture_sha256"] or sha256(guide) != art["guide_sha256"]:
        raise ValueError("saved probe encoded input bytes changed")
    shape, record = _encode_record(capture)
    name = record["source"]
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 4:
        raise ValueError("saved probe capture has an invalid relative source")
    root = capture.parent
    for _ in relative.parts:
        root = root.parent
    if (root / relative).resolve() != capture.parent or record["fps"] != art["fps"]:
        raise ValueError("saved probe source or frame rate differs from the capture record")
    metadata = capture.parent.parent.parent / "meta.json"
    actor = str(json.loads(metadata.read_text())["actor"]["id"])
    original = {"relative_dir": name, "actor": actor, "fps": art["fps"], "n_latent_frames": shape[1],
                "capture_latent_sha256": art["capture_sha256"], "guide_latent_sha256": art["guide_sha256"],
                "rgb_sha256": sha256(capture.parent / "rgb.mp4")}
    source = _convert_source(original, root, objective, {actor: "historical"}, require_guide=True)
    membership = {"schema_version": SCHEMA_VERSION, "kind": KIND, "corpus_root": str(root),
                  "objective": objective, "sources": [source], "splits": {"historical": [actor]},
                  "capture_manifest": {"sha256": sha256(root / dataset.CAPTURE_MANIFEST_NAME)},
                  "original_probe_file_sha256": digest, "current_clip_metadata_sha256": sha256(metadata),
                  "group_rule": "historical_replay_only_no_training_split", "requires_guide_latent": True,
                  "content_pin_scope": "original_encoded_inputs_and_current_raw_metadata", "excluded": {}}
    if sha256(path) != digest:
        raise ValueError("saved probe manifest changed during membership preparation")
    membership["sha256"] = membership_hash(membership)
    validate_membership(membership)
    return membership


def convert_legacy(old: dict, *, original_file_sha256: str, require_guide: bool = False) -> tuple[dict, dict]:
    """Preserve exact old video groups, content pins, block indices, and selected ranges."""
    if old.get("kind") != "one_step_argavatar_block_chains":
        raise ValueError("conversion requires an original block-chain subset")
    root = Path(old["corpus_root"])
    objective = old.get("objective", dataset.DEFAULT_OBJECTIVE)
    split_of = {}
    for group, actors in old["splits"].items():
        for actor in actors:
            if actor in split_of:
                raise ValueError(f"actor {actor} appears in multiple original groups")
            split_of[actor] = group
    sources = []
    original_sources = {record["relative_dir"]: record for record in old["sources"]}
    sources = [_convert_source(record, root, objective, split_of, require_guide) for record in old["sources"]]
    membership = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "corpus_root": str(root),
        "objective": objective,
        "capture_manifest": copy.deepcopy(old["capture_manifest"]),
        "splits": copy.deepcopy(old["splits"]),
        "sources": sources,
        "excluded": copy.deepcopy(old.get("excluded", {})),
        "content_pin_scope": old.get("content_pin_scope", "raw_and_latents"),
        "original_subset_file_sha256": original_file_sha256,
        "requires_guide_latent": require_guide or old.get("requires_guide_latent", False),
    }
    membership["sha256"] = membership_hash(membership)
    validate_membership(membership)
    geometry = old["geometry"]
    scale = SpatioTemporalScaleFactors(time=geometry["latent_time_scale"], height=32, width=32)
    layout = CausalGeometry(
        scale,
        block_latent_frames=geometry["block_latent_frames"],
        context_latent_frames=geometry["context_latent_frames"],
        sink_latent_frames=geometry["sink_latent_frames"],
    )
    samples = []
    for index, chain in enumerate(old["chains"]):
        record = original_sources[chain["source"]]
        if chain["actor"] != record["actor"] or chain["split"] != split_of[record["actor"]]:
            raise ValueError(f"original chain {index}: person/group differs from its video")
        length = record.get("span_latent_frames") or old.get("span_latent_frames") or record["n_latent_frames"]
        blocks = layout.plan(length)
        if not chain["blocks"] or any(block < 0 or block >= len(blocks) for block in chain["blocks"]):
            raise ValueError(f"original chain {index}: block index is outside the stored master")
        samples.append(
            {
                **copy.deepcopy(chain),
                "original_chain_index": index,
                "ranges": [list(blocks[block]) for block in chain["blocks"]],
            }
        )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": FRAME_PLAN_KIND,
        "mode": "causal",
        "membership_sha256": membership["sha256"],
        "original_subset_file_sha256": original_file_sha256,
        "geometry": copy.deepcopy(geometry),
        "samples": samples,
        "original_source_records": copy.deepcopy(old["sources"]),
        "selection_rule": "exact_original_block_indices",
    }
    plan["sha256"] = record_hash(plan)
    return membership, plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convert", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-plan-output", type=Path, required=True)
    parser.add_argument("--require-guide", action="store_true")
    args = parser.parse_args(argv)
    paths = [args.convert.resolve(), args.output.resolve(), args.frame_plan_output.resolve()]
    if len(set(paths)) != 3 or args.output.exists() or args.frame_plan_output.exists():
        raise SystemExit("conversion requires two distinct new output files; originals and used outputs are refused")
    old = json.loads(args.convert.read_text())
    membership, plan = convert_legacy(old, original_file_sha256=sha256(args.convert), require_guide=args.require_guide)
    for destination, record in ((args.output, membership), (args.frame_plan_output, plan)):
        dataset.atomic_write(
            destination, lambda temporary, record=record: temporary.write_text(json.dumps(record, indent=2) + "\n")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
