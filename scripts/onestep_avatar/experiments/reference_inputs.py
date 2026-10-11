"""Prepare truthful RGB references from original membership; see doc/experiments/reference_inputs.md."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from scripts.onestep_avatar import media
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import spec_protocol
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import backbone

ENTRY = "scripts/onestep_avatar/experiments/reference_inputs.py"
EXTRAS = ("scripts/onestep_avatar/experiments/__init__.py",
          "scripts/onestep_avatar/experiments/conditioning_check.py", ENTRY, spec_protocol.ENTRY)


def checked_source(store: dataset.ClipStore, source_id: str) -> tuple[dict, dict]:
    """Duplicate only the exact recorded crop for the public RGB reader's source-row schema."""
    source = store.sources[source_id]
    manifest_path = store.root / dataset.CAPTURE_MANIFEST_NAME
    digest = sha256(manifest_path)
    if digest != store.membership["capture_manifest"]["sha256"]:
        raise ValueError("reference crop manifest differs from original membership")
    manifest = dataset.CaptureManifest.load(store.root)
    box = list(manifest.box_for(store.root / source_id))
    for role in ("capture", "guide"):
        record = source.get(role + "_encode_record")
        if role == "guide" and record is None:
            continue
        if (not isinstance(record, dict) or record.get("source") != source_id
                or record.get("objective") != store.objective or record.get("box_xyxy") != box
                or record.get("edge") != manifest.edge):
            raise ValueError("reference source crop, objective or edge differs from saved producer")
    if "box_xyxy" in source and source["box_xyxy"] != box:
        raise ValueError("reference top-level crop differs from original producer")
    row = copy.deepcopy(source)
    row["box_xyxy"] = box
    evidence = {"source": source_id, "objective": store.objective, "box_xyxy": box,
                "edge": manifest.edge, "capture_manifest": {"path": str(manifest_path), "sha256": digest},
                "original_membership_sha256": store.membership["sha256"],
                "derivation": "exact recorded capture-manifest crop equals original capture and guide encode records"}
    return row, evidence


def scientific_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("subset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--encoded-frames", type=int, required=True)
    parser.add_argument("--guide-mode", choices=("d0", "d1"), default="d1")
    parser.add_argument("--model", choices=("2.5",), default="2.5")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args(argv)


def prepare(args: argparse.Namespace, *, fresh: bool = True) -> dict:
    if fresh and args.output.exists():
        raise ValueError("reference preparation requires a fresh output")
    membership = json.loads(args.subset.read_text())
    subset.verify_prepared_membership(membership)
    store = dataset.ClipStore(membership)
    if args.source not in store.sources:
        raise ValueError("reference source is absent from original membership")
    row, crop = checked_source(store, args.source)
    if not 1 <= args.encoded_frames <= row["n_latent_frames"]:
        raise ValueError("reference span differs from the original continuous master")
    if args.guide_mode == "d1" and not row.get("guide_latent_sha256"):
        raise ValueError("D1 reference requires a checked guide")
    video = store.load(args.source, require_guide=bool(row.get("guide_latent_sha256")))
    specification = backbone.resolve(args.model, "dev")
    from scripts.onestep_avatar.corpus import precompute  # noqa: PLC0415 -- original producer fingerprint

    vae = Path(specification.paths.video_vae())
    if precompute.file_fingerprint(vae) != row["capture_encode_record"]["vae_fingerprint"]:
        raise ValueError("reference decoder differs from original capture VAE")
    view = store.root / args.source
    paths = [args.subset, store.root / dataset.CAPTURE_MANIFEST_NAME, vae,
             view / dataset.capture_bundle_name(store.objective), view / "rgb.mp4", view / "bbox.npy"]
    if store.objective == "white":
        paths.append(view / dataset.CAPTURE_MASK_NAME)
    if row.get("guide_latent_sha256"):
        paths.extend(view / name for name in (dataset.guide_bundle_name(store.objective),
                     dataset.render_name(store.objective), dataset.render_metadata_name(store.objective)))
    return {"store": store, "source": row, "crop": crop, "video": video, "specification": specification,
            "input_files": {str(path): sha256(path) for path in paths},
            "software": software.capture("decoding", extra_sources=EXTRAS)}


def verify(args: argparse.Namespace) -> dict:
    prepared = prepare(args, fresh=False)
    crop = {"crop_adapter": prepared["crop"], "input_files": prepared["input_files"],
            "software": prepared["software"]}
    if json.loads((args.output / "source_crop.json").read_text()) != crop:
        raise ValueError("saved reference crop derivation or input pins differ")
    panels, producer = media.load_training_references(args.output / "references.json")
    source, store = prepared["source"], prepared["store"]
    vae_hash = sha256(Path(prepared["specification"].paths.video_vae()))
    shape = [1, source["shape"][0], args.encoded_frames, *source["shape"][2:]]
    frames = (args.encoded_frames - 1) * 8 + 1
    expected = {"source": args.source, "objective": store.objective, "fps": source["fps"],
        "source_frames": list(range(frames)), "capture_encoding_sha256": source["capture_latent_sha256"],
        "guide_rgb_sha256": source.get("guide_sha256"), "vae_sha256": vae_hash, "decode_seed": args.seed,
        "decoder_settings": media.native_decoder_settings(),
        "decode_key": media.decode_key(source["capture_latent_sha256"], vae_hash, shape,
                                       "native_decode_video", args.seed, media.native_decoder_settings()),
        "membership_file_sha256": sha256(args.subset), "membership_sha256": store.membership["sha256"],
        "source_crop_adapter": prepared["crop"], "software": prepared["software"]}
    if any(producer.get(key) != value for key, value in expected.items()):
        raise ValueError("saved reference producer differs from original membership or decoder")
    edge = source["capture_encode_record"]["edge"]
    for panel in panels:
        if panel.pixels is not None and (list(panel.pixels.shape) != [frames, 3, edge, edge]
                                        or not torch.isfinite(panel.pixels).all()):
            raise ValueError("saved reference geometry differs")
    if not torch.equal(panels[0].pixels, media.recorded_capture_rgb(source, store.root, store.objective,
                                                                  args.encoded_frames)):
        raise ValueError("saved reference capture pixels differ from original crop/objective")
    if source.get("guide_sha256") and not torch.equal(panels[2].pixels,
        media.recorded_guide_rgb(source, store.root, store.objective, args.encoded_frames)):
        raise ValueError("saved reference guide pixels differ from original producer")
    return {"state": "passed", "scope": "original membership RGB references; current VAE decoding",
            "source": args.source, "encoded_frames": args.encoded_frames, "quality_acceptance": None}


def execute(args: argparse.Namespace) -> dict:
    prepared = prepare(args)
    session = media.open_decoder_session(args.model, args.gpu_id, script="onestep_avatar.reference_inputs")
    with session.decoder() as decoder, torch.inference_mode():
        panels, producer = media.prepare_training_references(session, decoder, prepared["source"],
            prepared["store"].root, prepared["store"].objective, args.encoded_frames, args.seed,
            require_guide=args.guide_mode == "d1")
    if any(sha256(Path(path)) != digest for path, digest in prepared["input_files"].items()):
        raise ValueError("reference input bytes changed during native decoding")
    software.check_current(prepared["software"])
    producer.update(membership_file_sha256=sha256(args.subset),
                    membership_sha256=prepared["store"].membership["sha256"],
                    source_crop_adapter=prepared["crop"], software=prepared["software"])
    media.save_training_references(panels, producer, args.output)
    (args.output / "source_crop.json").write_text(json.dumps({"crop_adapter": prepared["crop"],
        "input_files": prepared["input_files"], "software": prepared["software"]}, indent=2, allow_nan=False) + "\n")
    return verify(args)
