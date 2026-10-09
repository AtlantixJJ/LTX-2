"""Pinned checker specs and saved manifests; see doc/experiments/spec_protocol.md."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from scripts.onestep_avatar.hashing import sha256

ENTRY = "scripts/onestep_avatar/experiments/spec_protocol.py"
OVERRIDES = {
    "--spec", "--output", "--out", "--gpu-id", "--dry-run", "--help", "-h", "--verify",
    "--supervise", "--process-ledger", "--launch-record", "--execute", "--overwrite",
}


def outer_args(argv: list[str] | None, description: str) -> argparse.Namespace:
    """Leave relative paths unopened for the queue's first normalization pass."""
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu_id < 0:
        parser.error("local GPU must be nonnegative")
    args.completion = {"manifest": str(args.output.resolve() / "manifest.json")}
    return args


def read_spec(
    outer: argparse.Namespace,
    selector: str,
    parser: Callable[[list[str]], argparse.Namespace],
    *,
    operations: tuple[str, ...] = ("run",),
    output_flag: str = "--output",
) -> argparse.Namespace:
    """Decode scientific settings only, with no data/model preparation or writes."""
    def unique_fields(pairs: list[tuple[str, object]]) -> dict:
        record = {}
        for key, value in pairs:
            if key in record:
                raise ValueError("scientific specification repeats a JSON field")
            record[key] = value
        return record

    before = sha256(outer.spec)
    record = json.loads(outer.spec.read_text(), object_pairs_hook=unique_fields)
    if (not isinstance(record, dict)
            or set(record) not in ({"schema_version", "protocol", "arguments"},
                                   {"schema_version", "protocol", "arguments", "operation"})
            or type(record.get("schema_version")) is not int or record["schema_version"] != 1
            or record.get("protocol") != selector or not isinstance(record.get("arguments"), list)
            or any(not isinstance(value, str) for value in record["arguments"])
            or record.get("operation", operations[0]) not in operations):
        raise ValueError("checker requires its exact schema-one scientific specification")
    flags = [value.split("=", 1)[0] for value in record["arguments"] if value.startswith("-")]
    if any(flag in OVERRIDES for flag in flags) or len(flags) != len(set(flags)):
        raise ValueError("scientific specification contains an execution override or repeated option")
    try:
        args = parser([*record["arguments"], output_flag, str(outer.output), "--gpu-id", str(outer.gpu_id)])
    except SystemExit as error:
        raise ValueError("invalid checker scientific arguments") from error
    for value in vars(args).values():
        paths = [value] if isinstance(value, Path) else value if isinstance(value, list) else []
        if any(isinstance(path, Path) and not path.is_absolute() for path in paths):
            raise ValueError("scientific specification paths must be absolute")
    args.spec, args.output = outer.spec, outer.output
    args.gpu_id, args.dry_run = outer.gpu_id, outer.dry_run
    if sha256(outer.spec) != before:
        raise ValueError("scientific specification changed during parsing")
    args.spec_sha256, args.operation = before, record.get("operation", operations[0])
    args.completion = outer.completion
    return args


def settings(args: argparse.Namespace) -> dict:
    """Keep scientific settings separate from queue-owned execution fields."""
    def encode(value: object) -> object:
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        return value

    omitted = {"spec", "spec_sha256", "output", "out", "gpu_id", "dry_run", "completion"}
    return {name: encode(value) for name, value in vars(args).items() if name not in omitted}


def inventory(root: Path) -> dict[str, str]:
    """Hash the whole local scientific output, refusing escaped artifacts."""
    if root.is_symlink():
        raise ValueError("checker output root is a symlink")
    root = root.resolve()
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("checker artifact escapes its output")
        if path.is_file() and path != root / "manifest.json":
            result[path.relative_to(root).as_posix()] = sha256(path)
    if not result:
        raise ValueError("checker output has no scientific artifacts")
    return result


def publish(args: argparse.Namespace, selector: str) -> dict:
    """Publish only after the scientific owner has finished its saved outputs."""
    if sha256(args.spec) != args.spec_sha256:
        raise ValueError("checker scientific specification changed during execution")
    manifest = {"schema_version": 1, "protocol": selector,
                "spec": {"path": str(args.spec.resolve()), "sha256": args.spec_sha256},
                "settings": settings(args), "artifacts": inventory(args.output)}
    destination = args.output / "manifest.json"
    if destination.exists():
        raise ValueError("checker completion manifest already exists")
    with destination.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def inspect(args: argparse.Namespace, selector: str) -> dict:
    """Check saved byte identity; scientific verification remains with the checker."""
    manifest = json.loads((args.output / "manifest.json").read_text())
    expected = {"schema_version": 1, "protocol": selector,
                "spec": {"path": str(args.spec.resolve()), "sha256": sha256(args.spec)},
                "settings": settings(args), "artifacts": inventory(args.output)}
    if manifest != expected:
        raise ValueError("checker completion specification, settings or artifacts differ")
    return manifest


def evidence(manifest: dict, root: Path) -> list[Path]:
    """Expose every checked local byte for queue dependency receipts."""
    return [root / "manifest.json", *(root / name for name in sorted(manifest["artifacts"]))]
