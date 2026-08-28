"""Family-balanced calibration record selection."""

from __future__ import annotations

from pathlib import Path

from scripts.prune import chunk_states


def select(root: Path, *, split: str | None = None, limit: int | None = None, family: str | None = None,
           step_index: int | None = None) -> list[Path]:
    """Select records with an optional balanced, evenly-strided cap."""
    paths = list(chunk_states.iter_records(root, split))
    if family:
        paths = [path for path in paths if f"__{family}" in path.name]
    if step_index is not None:
        paths = [path for path in paths if f"__s{step_index}__" in path.name]
    if not paths:
        raise SystemExit(
            f"no {split or 'any'}/{family or 'any'} records under {root}; build the Phase 1 cache with "
            "`teacher --build-calibration` first"
        )
    if limit is None or limit >= len(paths):
        return paths
    if limit <= 0:
        raise ValueError("limit must be positive or None (None means all records)")
    families: dict[str, list[Path]] = {}
    for path in paths:
        families.setdefault("renoised" if "__renoised" in path.name else "on_policy", []).append(path)
    keep: set[Path] = set()
    for index, (_, group) in enumerate(sorted(families.items())):
        wanted = limit // len(families) + (1 if index < limit % len(families) else 0)
        if wanted:
            keep.update(group[::max(1, len(group) // wanted)][:wanted])
    return [path for path in paths if path in keep]
