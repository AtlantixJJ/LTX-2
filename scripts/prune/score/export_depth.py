"""Stream an intact-block D0 deletion into a physically shorter checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import struct
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

import torch
from safetensors import safe_open

from ltx_core.model.transformer.model_configurator import LTXVideoOnlyModelConfigurator
from scripts.prune.core import provenance
from scripts.prune.data import whole_clip

PREFIX = "model.diffusion_model.transformer_blocks."
FORMAT = "whole_clip_d0_depth_v1"
PER_LAYER_FIELDS = (
    "per_layer_video_attn1_heads", "per_layer_video_attn2_heads", "per_layer_ff_inner_dim",
    "per_layer_video_attn1_rope_head_indices", "per_layer_video_attn2_rope_head_indices",
    "per_layer_video_attn1_active_head_indices", "per_layer_video_attn2_active_head_indices",
    "per_layer_video_ffn_active_channels",
)
PURPOSES = ("diagnostic_architecture", "calibration_candidate", "no_prune_control")
QUALITY_STATUSES = ("not_evaluated", "failed_prior_functional_gate")


def _header(path: Path) -> tuple[dict, int, str]:
    """Validate the format through safetensors, then read only the JSON header."""
    with safe_open(path, framework="pt", device="cpu") as handle:
        if not handle.metadata() or "config" not in handle.metadata():
            raise ValueError("checkpoint lacks native config metadata")
    with path.open("rb") as source:
        size = struct.unpack("<Q", source.read(8))[0]
        raw = source.read(size)
    return json.loads(raw), 8 + size, hashlib.sha256(raw).hexdigest()


def _block_index(name: str) -> int | None:
    if not name.startswith(PREFIX):
        return None
    index, separator, _ = name[len(PREFIX):].partition(".")
    if not separator or not index.isdecimal():
        raise ValueError(f"invalid transformer block key: {name}")
    return int(index)


def _configuration(header: dict) -> dict:
    config = json.loads(header["__metadata__"]["config"])
    transformer = config.get("transformer", {})
    layers = transformer.get("num_layers")
    if type(layers) is not int or layers <= 0:
        raise ValueError("source num_layers must be a positive integer")
    return config


def _counts(entries: dict) -> dict[str, int]:
    return {
        "tensors": len(entries),
        "elements": sum(math.prod(entry["shape"]) for entry in entries.values()),
        "stored_bytes": sum(entry["data_offsets"][1] - entry["data_offsets"][0] for entry in entries.values()),
    }


def _architecture_counts(header: dict, config: dict) -> dict:
    """Count expected video parameters, excluding unused AV/connector tensors."""
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    with torch.device("meta"):
        model = LTXVideoOnlyModelConfigurator.from_metadata({"config": config})
    video = {}
    for name, tensor in model.state_dict().items():
        key = "model.diffusion_model." + name
        if key not in tensors or list(tensor.shape) != tensors[key]["shape"]:
            raise ValueError(f"checkpoint does not match native video architecture: {key}")
        video[key] = tensors[key]
    blocks = {key for key in tensors if _block_index(key) is not None}
    return {
        "checkpoint_tensors": _counts(tensors),
        "resident_video_parameters": _counts(video),
        "block_checkpoint_tensors": _counts({key: tensors[key] for key in blocks}),
        "block_resident_video_parameters": _counts({key: video[key] for key in blocks if key in video}),
    }


def inspect_checkpoint(source: str | Path) -> dict:
    """Inspect an unpruned source without materializing any tensor payload."""
    source = Path(source)
    header, _, header_sha = _header(source)
    config = _configuration(header)
    transformer = config["transformer"]
    if transformer.get("pruning") is not None:
        raise ValueError("already-pruned source checkpoints are not supported")
    layers = transformer["num_layers"]
    blocks = {_block_index(key) for key in header if key != "__metadata__"}
    blocks.discard(None)
    if blocks != set(range(layers)):
        raise ValueError("source transformer block keys are not contiguous or disagree with num_layers")
    for key, values in transformer.items():
        if key.startswith("per_layer_") and key not in PER_LAYER_FIELDS:
            raise ValueError(f"unsupported block-indexed config field: {key}")
        if key in PER_LAYER_FIELDS and values is not None and (
            not isinstance(values, list) or len(values) != layers
        ):
            raise ValueError(f"{key} must contain one entry per original block")
    return {
        "source_num_layers": layers,
        "source_header_sha256": header_sha,
        "source_fingerprint": provenance.checkpoint_fingerprint(source),
        "parameter_counts": _architecture_counts(header, config),
    }


def _mapping(layers: int, removed: list[int]) -> dict:
    if not isinstance(removed, list) or any(type(index) is not int for index in removed):
        raise ValueError("removed_blocks must be a list of integer block indices")
    if removed != sorted(set(removed)) or any(index < 0 or index >= layers for index in removed):
        raise ValueError("removed_blocks must be distinct, sorted original indices within num_layers")
    retained = [index for index in range(layers) if index not in set(removed)]
    if not retained:
        raise ValueError("refusing to remove every transformer block")
    compact = {original: index for index, original in enumerate(retained)}
    return {
        "removed_blocks": removed,
        "retained_blocks": retained,
        "compact_to_original": retained,
        "original_to_compact": [compact.get(index) for index in range(layers)],
    }


def create_artifact(
    baseline_root: str | Path, calibration_views: list[str], sigmas: list[float], removed_blocks: list[int],
    *, purpose: str = "diagnostic_architecture", quality_status: str = "not_evaluated",
) -> dict:
    """Bind an explicitly chosen deletion set to saved native calibration inputs."""
    root = Path(baseline_root)
    baseline = whole_clip.load_manifest(root)
    if not calibration_views or len(set(calibration_views)) != len(calibration_views):
        raise ValueError("calibration_views must be nonempty and distinct")
    if not sigmas or len(set(sigmas)) != len(sigmas) or any(sigma not in baseline["sigmas"] for sigma in sigmas):
        raise ValueError("calibration sigmas must be distinct saved native levels")
    if any((view, sigma) not in whole_clip.records(baseline) for view in calibration_views for sigma in sigmas):
        raise ValueError("calibration view/sigma pair absent from saved baseline")
    if purpose not in PURPOSES or quality_status not in QUALITY_STATUSES:
        raise ValueError("unknown depth candidate purpose or quality status")
    source = Path(baseline["model"]["transformer_path"])
    inspected = inspect_checkpoint(source)
    if inspected["source_fingerprint"] != baseline["model"]["transformer_fingerprint"]:
        raise ValueError("source checkpoint changed since the saved baseline")
    return {
        "candidate_format": FORMAT, "family": "depth", "purpose": purpose,
        "qualification": "unqualified", "quality_status": quality_status,
        "source_num_layers": inspected["source_num_layers"],
        "source_header_sha256": inspected["source_header_sha256"],
        "source_parameter_counts": inspected["parameter_counts"],
        **_mapping(inspected["source_num_layers"], removed_blocks),
        "provenance": whole_clip.native_provenance(root, baseline, calibration_views, sigmas),
    }


def read_artifact(
    path: str | Path, *, model_key: str, fingerprint: str, num_layers: int, baseline: dict | None = None,
) -> tuple[dict, str]:
    """Validate depth syntax and immutable native D0 provenance independently of width masks."""
    raw = Path(path).read_bytes()
    artifact = json.loads(raw)
    if (not isinstance(artifact, dict) or artifact.get("candidate_format") != FORMAT or
            artifact.get("family") != "depth" or
            "masks" in artifact or artifact.get("source_num_layers") != num_layers or
            type(artifact.get("source_num_layers")) is not int):
        raise ValueError("invalid native D0 depth artifact format or original layer count")
    mapping = _mapping(num_layers, artifact.get("removed_blocks"))
    if any(artifact.get(key) != value for key, value in mapping.items()):
        raise ValueError("depth artifact retained mapping differs from removed_blocks")
    stamp = artifact.get("provenance")
    if (not isinstance(stamp, dict) or stamp.get("model_key") != model_key or
            stamp.get("transformer_fingerprint") != fingerprint or stamp.get("task") != whole_clip.TASK):
        raise ValueError("depth artifact model, source fingerprint or task differs")
    views, sigmas = stamp.get("calibration_views"), stamp.get("sigmas")
    if (not isinstance(views, list) or not views or any(not isinstance(view, str) or not view for view in views) or
            len(set(views)) != len(views) or not isinstance(sigmas, list) or not sigmas or
            any(type(sigma) not in (int, float) or not math.isfinite(sigma) or not 0 < sigma <= 1
                for sigma in sigmas) or
            len(set(sigmas)) != len(sigmas)):
        raise ValueError("incomplete native D0 depth calibration scope")
    if (artifact.get("purpose") not in PURPOSES or artifact.get("quality_status") not in QUALITY_STATUSES or
            artifact.get("qualification") != "unqualified" or
            not isinstance(artifact.get("source_header_sha256"), str) or
            len(artifact["source_header_sha256"]) != 64 or
            not isinstance(artifact.get("source_parameter_counts"), dict)):
        raise ValueError("incomplete depth artifact source accounting or diagnostic status")
    whole_clip.validate_native_provenance(stamp, baseline)
    return artifact, hashlib.sha256(raw).hexdigest()


def _remap_header(header: dict, artifact: dict) -> dict:
    result = {}
    mapping = artifact["original_to_compact"]
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        block, renamed = _block_index(name), name
        if block is not None:
            compact = mapping[block]
            if compact is None:
                continue
            renamed = PREFIX + str(compact) + "." + name[len(PREFIX):].split(".", 1)[1]
        result[renamed] = copy.deepcopy(entry)
    return result


def _compact_configuration(header: dict, artifact: dict) -> dict:
    config = _configuration(header)
    transformer = config["transformer"]
    transformer["num_layers"] = len(artifact["retained_blocks"])
    for key in PER_LAYER_FIELDS:
        if transformer.get(key) is not None:
            transformer[key] = [transformer[key][index] for index in artifact["retained_blocks"]]
    return config


def _parameter_counts(compact: dict, config: dict, artifact: dict) -> dict:
    output = _architecture_counts(compact, config)
    source = artifact["source_parameter_counts"]
    return {
        "source": source, "exported": output,
        "resident_video_element_reduction": 1 - (
            output["resident_video_parameters"]["elements"] / source["resident_video_parameters"]["elements"]
        ),
        "checkpoint_payload_byte_reduction": 1 - (
            output["checkpoint_tensors"]["stored_bytes"] / source["checkpoint_tensors"]["stored_bytes"]
        ),
    }


def _source_stat(value: os.stat_result) -> tuple[int, ...]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _copy_tensor(source: BinaryIO, output: BinaryIO, start: int, size: int, chunk_bytes: int) -> None:
    source.seek(start)
    remaining = size
    while remaining:
        data = source.read(min(remaining, chunk_bytes))
        if not data:
            raise OSError("source ended while copying retained tensor")
        output.write(data)
        remaining -= len(data)


def export(  # noqa: PLR0912, PLR0915
    source: str | Path, artifact_path: str | Path, output: str | Path, *, chunk_bytes: int = 8 << 20,
) -> dict:
    """Write intact retained payloads with bounded RAM and atomic, non-overwriting publication."""
    source, artifact_path, output = Path(source), Path(artifact_path), Path(output)
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    if output.resolve() == source.resolve() or (output.exists() and output.samefile(source)):
        raise ValueError("refusing to overwrite source checkpoint or its filesystem alias")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    source_state = _source_stat(source.stat())
    inspected = inspect_checkpoint(source)
    artifact, digest = read_artifact(
        artifact_path, model_key=json.loads(artifact_path.read_bytes())["provenance"]["model_key"],
        fingerprint=inspected["source_fingerprint"], num_layers=inspected["source_num_layers"],
    )
    if (artifact["source_header_sha256"] != inspected["source_header_sha256"] or
            artifact["source_parameter_counts"] != inspected["parameter_counts"]):
        raise ValueError("depth artifact source header or parameter accounting changed")
    header, data_start, _ = _header(source)
    compact = _remap_header(header, artifact)
    config = _compact_configuration(header, artifact)
    counts = _parameter_counts(compact, config, artifact)
    pruning = {
        "family": "depth", "mode": "compact_depth", "task": whole_clip.TASK,
        "model_key": artifact["provenance"]["model_key"],
        "source_transformer_path": str(source.resolve()),
        "source_transformer_fingerprint": inspected["source_fingerprint"],
        "artifact": str(artifact_path.resolve()), "artifact_sha256": digest,
        "source_num_layers": artifact["source_num_layers"],
        **{key: artifact[key] for key in ("removed_blocks", "retained_blocks", "compact_to_original",
                                         "original_to_compact", "purpose", "qualification", "quality_status")},
        "parameter_counts": counts,
    }
    config["transformer"]["pruning"] = pruning
    metadata = dict(header["__metadata__"])
    metadata["config"] = json.dumps(config)
    plan = []
    cursor = 0
    # Preserve payload order. The key names change; the stored dtype, shape and bytes do not.
    for _name, entry in sorted(compact.items(), key=lambda item: item[1]["data_offsets"][0]):
        lo, hi = entry["data_offsets"]
        plan.append((data_start + lo, hi - lo))
        entry["data_offsets"] = [cursor, cursor + hi - lo]
        cursor += hi - lo
    compact["__metadata__"] = metadata
    encoded = json.dumps(compact, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with source.open("rb") as source_file, tempfile.NamedTemporaryFile(
            mode="wb", prefix="." + output.name + ".", suffix=".tmp", dir=output.parent, delete=False,
        ) as target:
            temporary = Path(target.name)
            if (_source_stat(os.fstat(source_file.fileno())) != source_state or
                    _source_stat(source.stat()) != source_state):
                raise ValueError("source checkpoint changed before streaming copy")
            target.write(struct.pack("<Q", len(encoded)))
            target.write(encoded)
            for start, size in plan:
                _copy_tensor(source_file, target, start, size, chunk_bytes)
            target.flush()
            os.fsync(target.fileno())
            if (_source_stat(os.fstat(source_file.fileno())) != source_state or
                    _source_stat(source.stat()) != source_state):
                raise ValueError("source checkpoint changed during streaming copy")
        if provenance.file_sha256(artifact_path) != digest:
            raise ValueError("depth artifact changed during streaming copy")
        whole_clip.validate_native_provenance(artifact["provenance"])
        if _source_stat(source.stat()) != source_state:
            raise ValueError("source checkpoint changed before publication")
        with safe_open(temporary, framework="pt", device="cpu") as handle:
            if len(handle.keys()) != len(compact) - 1:
                raise ValueError("streamed checkpoint key count differs from retained inventory")
        # A hard link publishes this completed file atomically and fails if another writer won.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "checkpoint": str(output.resolve()), "artifact": str(artifact_path.resolve()),
        "artifact_sha256": digest, "fingerprint": provenance.checkpoint_fingerprint(output),
        "source_file_bytes": source.stat().st_size, "exported_file_bytes": output.stat().st_size,
        "parameter_counts": counts, "source_num_layers": artifact["source_num_layers"],
        "exported_num_layers": len(artifact["retained_blocks"]),
        "removed_blocks": artifact["removed_blocks"], "qualification": "unqualified",
        "quality_status": artifact["quality_status"],
    }


def verify_export(
    source: str | Path, artifact_path: str | Path, exported: str | Path, *, baseline: dict | None = None,
) -> dict:
    """Bind a depth export to its artifact, source architecture and exact retained key inventory."""
    source, exported = Path(source), Path(exported)
    inspected = inspect_checkpoint(source)
    header, _, _ = _header(exported)
    config = _configuration(header)
    pruning = config["transformer"].get("pruning", {})
    artifact, digest = read_artifact(
        artifact_path, model_key=pruning.get("model_key"), fingerprint=inspected["source_fingerprint"],
        num_layers=inspected["source_num_layers"], baseline=baseline,
    )
    required = {
        "family": "depth", "mode": "compact_depth", "task": whole_clip.TASK,
        "source_transformer_fingerprint": inspected["source_fingerprint"], "artifact_sha256": digest,
        **{key: artifact[key] for key in ("source_num_layers", "removed_blocks", "retained_blocks",
                                         "compact_to_original", "original_to_compact", "purpose",
                                         "qualification", "quality_status")},
    }
    if any(pruning.get(key) != value for key, value in required.items()):
        raise ValueError("exported checkpoint does not carry this native depth artifact and source")
    if (artifact["source_header_sha256"] != inspected["source_header_sha256"] or
            artifact["source_parameter_counts"] != inspected["parameter_counts"]):
        raise ValueError("depth artifact source architecture or accounting changed")
    source_header, _, _ = _header(source)
    expected_header = _remap_header(source_header, artifact)
    actual = {key: value for key, value in header.items() if key != "__metadata__"}
    if set(actual) != set(expected_header) or any(
        actual[key]["shape"] != value["shape"] or actual[key]["dtype"] != value["dtype"]
        for key, value in expected_header.items()
    ):
        raise ValueError("depth export retained tensor inventory differs from source mapping")
    expected_config = _compact_configuration(source_header, artifact)
    expected_counts = _parameter_counts(expected_header, expected_config, artifact)
    config["transformer"].pop("pruning")
    if config != expected_config or pruning.get("parameter_counts") != expected_counts:
        raise ValueError("depth export architecture or parameter accounting differs from source mapping")
    return pruning


@contextmanager
def retained_blocks(model: torch.nn.Module, removed_blocks: list[int]) -> Iterator[None]:
    """Temporarily shorten the source model to the ordered retained-block reference."""
    core = getattr(model, "velocity_model", model)
    original = core.transformer_blocks
    mapping = _mapping(len(original), removed_blocks)
    core.transformer_blocks = torch.nn.ModuleList([original[index] for index in mapping["retained_blocks"]])
    try:
        yield
    finally:
        core.transformer_blocks = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--calibration-views", nargs="+", required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", required=True)
    parser.add_argument("--remove-blocks", type=int, nargs="*", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=PURPOSES, default="diagnostic_architecture")
    parser.add_argument("--quality-status", choices=QUALITY_STATUSES, default="not_evaluated")
    args = parser.parse_args()
    artifact = create_artifact(
        args.baseline, args.calibration_views, args.sigmas, args.remove_blocks,
        purpose=args.purpose, quality_status=args.quality_status,
    )
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    with args.artifact.open("x") as target:
        target.write(json.dumps(artifact, indent=2) + "\n")
    source = whole_clip.load_manifest(args.baseline)["model"]["transformer_path"]
    result = export(source, args.artifact, args.output)
    result["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(json.dumps(result))


if __name__ == "__main__":
    main()
