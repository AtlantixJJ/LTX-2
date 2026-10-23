"""Frozen-study identity, source identity and pinned-file verification.

Every frozen study is one JSON object whose `identity` is the digest of all
other fields. `code_identity` digests the package sources, the LTX core and
pipeline utilities and the avatar corpus/model sources; any edit changes it.
`read_frozen` checks format, identity and, by default, the code identity.
`verify_files` hashes each distinct pinned path once.
`validate_devices` checks disjoint device groups while retaining the four-rank
draw contract; actual availability is checked at native startup.
"""

import json
from pathlib import Path

from scripts.prune.core.provenance import file_sha256
from scripts.prune.core.records import digest


def code_identity():
    repo = Path(__file__).resolve().parents[3]
    paths = list((repo / "scripts/prune").rglob("*.py"))
    paths += list((repo / "packages/ltx-core/src/ltx_core").rglob("*.py"))
    paths += list((repo / "packages/ltx-pipelines/src/ltx_pipelines/utils").rglob("*.py"))
    paths += list((repo / "scripts/onestep_avatar/corpus").glob("*.py"))
    paths += list((repo / "scripts/onestep_avatar/model").glob("*.py"))
    files = {str(p.relative_to(repo)): file_sha256(p) for p in sorted(paths) if "tests" not in p.parts}
    return {"sha256": digest(files), "files": files}


def read_frozen(path, *, formats, verify_code=True):
    """Load one frozen study; reject a changed identity, an unknown format or changed sources."""
    study = json.loads(Path(path).read_text())
    identity = study.pop("identity", None)
    if study.get("format") not in formats or identity != digest(study):
        raise ValueError("frozen study identity changed")
    study["identity"] = identity
    if verify_code and study["code"] != code_identity():
        raise ValueError("frozen model code changed; create a new study")
    return study


def verify_files(items):
    seen = {}
    for item in items:
        path = item["path"]
        if path not in seen:
            seen[path] = file_sha256(path)
        if seen[path] != item.get("sha256", item.get("file_sha256")):
            raise ValueError(f"frozen input changed: {path}")


def validate_devices(config):
    """Validate frozen physical-device groups without probing CUDA during CPU preparation.

    Formal learning retains four draws/ranks. Device IDs can change in a new
    config; each group must be nonempty, unique and disjoint from the other.
    Runtime availability is checked immediately before model loading.
    """
    if type(config.get("world_size")) is not int or config["world_size"] != 4:
        raise ValueError("the frozen draw contract requires four formal ranks")
    for key in ("gpu_ids", "test_gpu_ids"):
        values = config.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value < 0 for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{key} must contain distinct nonnegative physical GPU IDs")
    if len(config["gpu_ids"]) != config["world_size"] or set(config["gpu_ids"]) & set(config["test_gpu_ids"]):
        raise ValueError("formal/test GPU groups must be disjoint and match the frozen world size")


def read_device_config(path):
    """Read only device groups for a new study, without changing its frozen parent.

    The effective groups are copied into the new study config and identity.
    Reject extra keys so this input cannot silently override the pruning method.
    """
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict) or set(values) != {"gpu_ids", "test_gpu_ids", "world_size"}:
        raise ValueError("device config requires only gpu_ids, test_gpu_ids and world_size")
    validate_devices(values)
    return values
