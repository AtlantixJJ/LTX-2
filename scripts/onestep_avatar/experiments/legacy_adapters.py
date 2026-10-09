"""Derive legacy adapter metadata from checked original evidence without tensor changes.

See doc/experiments/legacy_adapters.md. This explicit CPU conversion owns historical
classification; shared checkpoint readers and digest/tensor checks stay in training.
Keep original records and the historical canonical_rule attribution unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path


def convert_legacy_adapter(  # noqa: PLR0912, PLR0915 -- explicit original evidence and publication gates
    source: Path,
    destination: Path,
    record: dict,
    *,
    config_path: Path,
    subset_path: Path,
    membership_path: Path,
    frame_plan_path: Path,
    base_path: Path,
) -> dict:
    """Check original conditions and derive metadata while preserving every tensor byte."""
    import copy  # noqa: PLC0415
    import hashlib  # noqa: PLC0415
    import os  # noqa: PLC0415
    import struct  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from ltx_core.types import SpatioTemporalScaleFactors  # noqa: PLC0415
    from scripts.onestep_avatar.corpus import dataset, subset  # noqa: PLC0415 -- same lazy caller scope
    from scripts.onestep_avatar.hashing import sha256  # noqa: PLC0415
    from scripts.onestep_avatar.model import backbone  # noqa: PLC0415
    from scripts.onestep_avatar.model.causal import CausalGeometry  # noqa: PLC0415
    from scripts.onestep_avatar.training import checkpoints  # noqa: PLC0415 -- public shared checks
    from scripts.onestep_avatar.training.config import window_start_draw  # noqa: PLC0415

    if destination.exists() or destination.resolve() == source.resolve():
        raise ValueError("legacy conversion requires a fresh derived adapter path")
    paths = [source, config_path, subset_path, membership_path, frame_plan_path, base_path]
    # Refuse ambiguous flat records before reading tens of GiB of base weights.
    hashes = {str(path.resolve()): sha256(path) for path in paths[:-1]}
    config, original, membership, plan = [json.loads(path.read_text()) for path in paths[1:5]]
    meta = checkpoints.read_adapter_metadata(source)
    if checkpoints.CONTRACT_KEY in meta:
        raise ValueError("legacy conversion refuses an already classified adapter")
    converted = copy.deepcopy(record)
    checkpoints.validate_adapter_tensors(source, converted)
    sampling = converted["training"]["sigma_sampling"]
    if config.get("sigma_sampling") != sampling or meta.get("onestep_avatar_sigma_sampling") != sampling:
        raise ValueError("legacy sigma sampling evidence is missing or differs")
    window = config.get("random_window_latent_frames")
    if window is not None and (type(window) is not int or window < 2):
        raise ValueError("legacy random window length is invalid")
    for key, value in {
        "schedule": "ONE_STEP",
        "attention": "block_causal",
        "history_computation": "cached_refresh_global_sigma0",
        "parent_adapter": "",
        "window": "clip_start" if window is None else f"random_start_v1:{window}",
    }.items():
        if meta.get(f"onestep_avatar_{key}") != value:
            raise ValueError(f"legacy {key} evidence is missing or differs")
    subset.validate_membership(membership)
    old_hash = hashes[str(subset_path.resolve())]
    canonical_old_hash = subset.subset_sha256(original)
    if (
        config.get("subset_full_sha256") != canonical_old_hash
        or meta.get("onestep_avatar_subset_full_sha256") != canonical_old_hash
        or membership.get("original_subset_file_sha256") != old_hash
        or plan.get("original_subset_file_sha256") != old_hash
    ):
        raise ValueError("legacy conversion original subset evidence differs")
    if (
        membership.get("sha256") != subset.membership_hash(membership)
        or plan.get("sha256") != subset.record_hash(plan)
        or converted["data"]["membership_sha256"] != membership["sha256"]
        or converted["data"]["frame_plan_sha256"] != plan["sha256"]
        or plan.get("membership_sha256") != membership["sha256"]
    ):
        raise ValueError("legacy conversion derived data hashes differ")
    selected = [sample for sample in plan["samples"] if sample["split"] == config["split"]]
    expected_indices = [index for index, chain in enumerate(original["chains"]) if chain["split"] == config["split"]]
    if [sample.get("original_chain_index") for sample in selected] != expected_indices:
        raise ValueError("legacy conversion selected chain inventory differs")
    coverage = [{"source": sample["source"], "ranges": sample["ranges"]} for sample in selected]
    if not selected or coverage != converted["data"]["coverage"]:
        raise ValueError("legacy conversion selected frame coverage differs")
    if config.get("skip_subset_check") is not False or original.get("content_pinned") is not True:
        raise ValueError("legacy conversion requires checked original data pins")
    geometry = original["geometry"]
    layout = CausalGeometry(
        SpatioTemporalScaleFactors(geometry["latent_time_scale"], 32, 32),
        geometry["block_latent_frames"],
        geometry["context_latent_frames"],
    )
    old_sources = {item["relative_dir"]: item for item in original["sources"]}
    new_sources = {item["relative_dir"]: item for item in membership["sources"]}
    for sample in selected:
        index = sample.get("original_chain_index")
        if type(index) is not int or not 0 <= index < len(original["chains"]):
            raise ValueError("legacy conversion original chain mapping is missing")
        chain = original["chains"][index]
        if any(
            sample.get(key) != chain.get(key)
            for key in ("source", "actor", "split", "blocks", "seed_is_clip_start")
        ):
            raise ValueError("legacy conversion original chain mapping differs")
        old_source, new_source = old_sources[sample["source"]], new_sources[sample["source"]]
        checkpoints.require_digest(old_source.get("capture_latent_sha256"), "original capture")
        if converted["task"]["guide_mode"] == "d1":
            checkpoints.require_digest(old_source.get("guide_latent_sha256"), "original guide")
        view = Path(membership["corpus_root"]) / sample["source"]
        roles = [("capture", dataset.capture_bundle_name(converted["task"]["objective"]))]
        if converted["task"]["guide_mode"] == "d1":
            roles.append(("guide", dataset.guide_bundle_name(converted["task"]["objective"])))
        for role, filename in roles:
            path = view / filename
            digest = sha256(path)
            if digest != old_source[f"{role}_latent_sha256"]:
                raise ValueError("legacy conversion original encoded data changed")
            master, fps = dataset.load_training_master(path)
            if list(master.shape) != new_source["shape"] or fps != old_source["fps"]:
                raise ValueError("legacy conversion master geometry or fps differs")
            del master
            if path not in paths:
                paths.append(path)
                hashes[str(path.resolve())] = digest
        length = (
            old_source.get("span_latent_frames") or original.get("span_latent_frames") or old_source["n_latent_frames"]
        )
        expected_ranges = [list(layout.plan(length)[block]) for block in chain["blocks"]]
        if sample["ranges"] != expected_ranges:
            raise ValueError("legacy conversion original frame ranges differ")
        for key in ("actor", "fps", "n_latent_frames", "capture_latent_sha256", "guide_latent_sha256"):
            if key in old_source and new_source.get(key) != old_source[key]:
                raise ValueError("legacy conversion original video pins differ")
        if [new_source["shape"][0], *new_source["shape"][2:]] != [
            converted["shape"][key] for key in ("channels", "height", "width")
        ]:
            raise ValueError("legacy conversion encoded geometry differs")
    if config.get("init_adapter") is not None:
        raise ValueError("legacy parent evidence needs explicit additional conversion support")
    start_policy = "clip_start" if window is None else "random"
    if converted["mode_settings"]["start_policy"] != start_policy:
        raise ValueError("legacy start policy differs")
    expected_draw = None if window is None else window_start_draw(config["noise_seed"])
    if plan.get("start_draw") != expected_draw:
        raise ValueError("legacy frame-plan random start draw differs")
    if window is not None:
        if window != geometry["block_latent_frames"] + 1 or any(
            sample["blocks"] != [0] or sample.get("seed_is_clip_start") is not True for sample in selected
        ):
            raise ValueError("legacy random window needs independent block-zero samples")
        expected_selection = checkpoints.random_segment_selection(
            coverage, {source: item["n_latent_frames"] for source, item in new_sources.items()}, window,
            config["noise_seed"],
        )
        if converted["data"]["segment_selection"] != expected_selection:
            raise ValueError("legacy random segment selection differs from the checked masters")
    if config.get("anchor_weight") != 0 or meta.get("onestep_avatar_anchor_weight") not in ("0", "0.0"):
        raise ValueError("legacy conversion requires a verified unanchored loss")
    conditions = {
        "guide_mode": converted["task"]["guide_mode"],
        "objective": converted["task"]["objective"],
        "loss": converted["task"]["loss"],
        "first_frame_conditioning": converted["task"]["first_frame_conditioning"],
        "noise_policy": converted["training"]["noise_policy"],
    }
    if "global_sigma_dtype" in converted["training"]:
        conditions["global_sigma_dtype"] = converted["training"]["global_sigma_dtype"]
    for key, value in conditions.items():
        if meta.get(f"onestep_avatar_{key}") != str(value):
            raise ValueError(f"legacy adapter {key} differs")
        if key != "first_frame_conditioning" and config.get(key) != value:
            raise ValueError(f"legacy config {key} differs")
    for key in ("rank", "alpha", "target"):
        if config.get(f"lora_{key}") != converted["adapter"][key] or meta.get(f"lora_{key}") != str(
            converted["adapter"][key]
        ):
            raise ValueError(f"legacy LoRA {key} differs")
    if meta.get("step") != str(converted["adapter"]["step"]) or converted["adapter"]["parent"] is not None:
        raise ValueError("legacy adapter step or parent differs")
    levels = config.get("sigma_levels") or [config["sigma0"]]
    if converted["training"]["sigma_levels"] != levels or meta.get("onestep_avatar_sigma_levels") != ",".join(
        repr(v) for v in levels
    ):
        raise ValueError("legacy exact sigma levels differ")
    for key in ("init", "data", "noise"):
        seed = converted["training"]["seeds"][key]
        if config.get(f"{key}_seed") != seed or meta.get(f"onestep_avatar_{key}_seed") != str(seed):
            raise ValueError(f"legacy {key} seed differs")
    hashes[str(base_path.resolve())] = sha256(base_path)
    identity = backbone.identity(base_path, config["variant"], config["model"])
    if meta.get("model_key") != config["model"] or meta.get("onestep_avatar_base_variant") != config["variant"]:
        raise ValueError("legacy base version or variant differs")
    for key in ("base_transformer_file", "base_transformer_fingerprint"):
        if config["base_identity"].get(key) != identity[key] or meta.get(f"onestep_avatar_{key}") != identity[key]:
            raise ValueError("legacy base identity differs")
    if converted["model"] != {
        "version": config["model"],
        "variant": config["variant"],
        "base_file": base_path.name,
        "base_sha256": hashes[str(base_path.resolve())],
    }:
        raise ValueError("legacy full base contract differs")
    block_count = int(meta["onestep_avatar_chain_length"])
    if meta.get("onestep_avatar_teacher_forcing") != str(config["teacher_forcing"]):
        raise ValueError("legacy forcing records differ")
    for key in ("block_latent_frames", "context_latent_frames"):
        if config.get(key) != geometry[key] or meta.get(f"onestep_avatar_{key}") != str(geometry[key]):
            raise ValueError("legacy configured geometry differs from the original subset")
    if any(len(sample["blocks"]) != block_count for sample in selected):
        raise ValueError("legacy chain length differs")
    if converted["mode"] == "bidirectional":
        span = geometry["block_latent_frames"] + 1
        if block_count != 1 or any(
            sample["blocks"] != [0] or sample["ranges"] != [[0, span]] or sample.get("seed_is_clip_start") is not True
            for sample in selected
        ):
            raise ValueError("legacy bidirectional classification needs unused-history evidence")
        if (
            converted["mode_settings"]["start_policy"] != start_policy
            or converted["mode_settings"]["span_latent_frames"] != span
        ):
            raise ValueError("legacy bidirectional frame selection differs")
    else:
        causal = converted["causal"]
        for key in ("block_latent_frames", "context_latent_frames", "sink_latent_frames"):
            if causal[key] != geometry[key] or meta.get(f"onestep_avatar_{key}") != str(causal[key]):
                raise ValueError("legacy causal geometry differs")
        if (
            causal["blocks_per_sample"] != block_count
            or converted["mode_settings"]["teacher_forcing"] != config["teacher_forcing"]
        ):
            raise ValueError("legacy causal forcing or block count differs")
    converted["conversion"] = {
        "source": str(source.resolve()),
        "source_sha256": hashes[str(source.resolve())],
        "evidence_sha256": hashes,
        "original_metadata": meta,
        "classification": "checked_original_ranges_and_conditions",
        "original_base_identity_evidence": "saved_filename_and_fast_fingerprint",
        "original_subset_identity": {
            "canonical_sha256": canonical_old_hash,
            "file_sha256": old_hash,
            "canonical_rule": "windows.subset_sha256",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".safetensors", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with source.open("rb") as reader, temporary_path.open("wb") as writer:
            prefix = reader.read(8)
            raw_header = reader.read(struct.unpack("<Q", prefix)[0])
            header = json.loads(raw_header)
            header["__metadata__"] = {checkpoints.CONTRACT_KEY: json.dumps(converted, sort_keys=True)}
            encoded_header = json.dumps(header, separators=(",", ":")).encode()
            encoded_header += b" " * (-len(encoded_header) % 8)
            writer.write(struct.pack("<Q", len(encoded_header)))
            writer.write(encoded_header)
            copied = hashlib.sha256(prefix + raw_header)
            while chunk := reader.read(8 * 1024 * 1024):
                writer.write(chunk)
                copied.update(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if copied.hexdigest() != hashes[str(source.resolve())] or any(
            sha256(path) != hashes[str(path.resolve())] for path in paths
        ):
            raise ValueError("legacy conversion evidence changed during copying")
        checkpoints.validate_adapter_tensors(temporary_path, converted)
        os.link(temporary_path, destination)  # exclusive publication; never replace another result
    finally:
        temporary_path.unlink(missing_ok=True)
    return converted


def main(argv: list[str] | None = None) -> int:
    """Convert one explicitly classified legacy adapter from saved evidence."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Derive checked version-two adapter metadata without changing tensors")
    for name in (
        "source",
        "output",
        "contract",
        "original-config",
        "original-subset",
        "membership",
        "frame-plan",
        "base",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    convert_legacy_adapter(
        args.source,
        args.output,
        json.loads(args.contract.read_text()),
        config_path=args.original_config,
        subset_path=args.original_subset,
        membership_path=args.membership,
        frame_plan_path=args.frame_plan,
        base_path=args.base,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
