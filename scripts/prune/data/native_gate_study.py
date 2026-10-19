"""Freeze native gate observations against the exact learned-pruning training stream."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.data.learned_head_study import code_identity
from scripts.prune.learn.checkpoints import digest, tensor_hash
from scripts.prune.score.native_gate_accumulator import atomic_json

CYCLE = [0, 9, 1, 8, 2, 7, 3, 6, 4, 5]


def verified(path: Path, expected: str) -> dict:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"source content changed: {path}")
    return {"path": str(path.resolve()), "sha256": actual}


def freeze_context(record: dict, root: Path) -> dict:
    """Source bytes_sha256 hashes raw tensor bytes, not the torch archive."""
    path = root / "configs/context.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record["path"], path)
    context = torch.load(path, map_location="cpu", weights_only=True)
    raw = hashlib.sha256(context.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    if raw != record["bytes_sha256"] or tensor_hash(context) != record["tensor_sha256"]:
        raise ValueError("frozen context tensor changed")
    return record | {"path": str(path.resolve()), "file_sha256": file_sha256(path)}


def mask_references(combined: Path, output: Path, source: dict) -> None:  # noqa: PLR0915
    # Keep the primary lineage reconstruction adjacent to its canonical checkpoint validation.
    """Bind original IDs to receipts, canonical checkpoints and unsuperseded events."""
    first = combined / "runs/01_selection_u0-500_from_r13u0"
    second = combined / "runs/02_continuation_u500-847"
    relative = "training/learned_progressive/replicate_42"
    pins = []
    masks = []
    for update in (600, 800):
        path = second / f"checks/learned_progressive/replicate_42/monitor_{update:04d}.json"
        pins.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
        masks.append(np.asarray(json.loads(path.read_text())["progressive"]["free"], dtype=bool))
    if masks[0].shape != (48, 32) or not np.array_equal(*masks) or masks[0].sum() != 1244:
        raise ValueError("primary selection/recovery masks disagree")
    path_map = json.loads((combined / "provenance/path_map.json").read_text())
    item = next(r for r in path_map["key_checkpoints"] if r["update"] == 600)
    checkpoint = combined / item["path"]
    pins.append(verified(checkpoint, item["sha256"]))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not np.array_equal(np.asarray(state["progressive"]["free"], dtype=bool), masks[0]):
        raise ValueError("primary checkpoint and monitor binary masks differ")
    del state
    alive = np.ones((48, 32), dtype=bool)
    events = []
    for update in range(50, 601, 50):
        base = first if update <= 500 else second
        path = base / relative / f"events/{update:04d}.json"
        pins.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
        event = json.loads(path.read_text())
        if event["alive_before"] != int(alive.sum()) or event["update"] != update:
            raise ValueError("primary event lineage or eligible population differs")
        rows = []
        chosen = {tuple(ids) for ids in event["selected_ids"]}
        reasons = {(r["block"], r["head"]): r["tier"] for r in event["candidates"]}
        for block, head in zip(*np.where(alive), strict=True):
            rows.append(
                {
                    "block": int(block),
                    "head": int(head),
                    "drop": event["drop"][block][head],
                    "reason": reasons[(block, head)] if (block, head) in chosen else "kept",
                }
            )
        for block, head in chosen:
            if not alive[block, head]:
                raise ValueError("primary event removes an already absent head")
            alive[block, head] = False
        events.append({"update": update, "heads": rows})
    if not np.array_equal(alive, masks[0]):
        raise ValueError("primary event reconstruction differs from canonical mask")
    references = [{"name": "clipped_reset_u600", "update": 600, "kept": masks[0].astype(int).tolist(), "pins": pins}]
    checkpoint_dir = Path(source["root"]) / "training/checkpoints"
    candidates = sorted(p for p in checkpoint_dir.glob("update_*.json") if int(p.stem.split("_")[1]) > 700)
    if candidates:
        sidecar = candidates[-1]
        receipt = json.loads(sidecar.read_text())
        path = sidecar.with_suffix(".pt")
        pin = verified(path, receipt["sha256"])
        state = torch.load(path, map_location="cpu", weights_only=False)
        mask = state["heads_mask"]
        if (
            state["study"] != source["identity"]
            or state["head_phase"] not in ("binary", "compact")
            or mask is None
            or mask.shape != (48, 32)
            or not ((mask == 0) | (mask == 1)).all()
            or int(mask.sum()) != 1384
        ):
            raise ValueError("additional r3 checkpoint lacks its verified binary head mask")
        references.append(
            {
                "name": f"r3_u{state['update']:04d}",
                "update": state["update"],
                "phase": state["head_phase"],
                "frozen": state["frozen"],
                "kept": mask.int().tolist(),
                "pins": [pin],
                "recipe": "joint head and FFN training",
            }
        )
        del state
    atomic_json(
        output / "comparison/masks.json", {"references": references, "future_arms": "A/B/C unavailable until run"}
    )
    atomic_json(output / "comparison/events.json", events)


def prepare(source_path: Path, combined: Path, plan: Path, output: Path, steps: int = 1300) -> dict:
    output = output.resolve()
    if (output / "configs/frozen_study.json").exists():
        return load(output / "configs/frozen_study.json")
    source = json.loads(source_path.read_text())
    payload = copy.deepcopy(source)
    identity = payload.pop("identity")
    if digest(payload) != identity:
        raise ValueError("training source identity changed")
    if identity != "d08091face66b6fe97c4c6f4c5f5ba1f8af697b4e0f68d43cc8792fee928a49d":
        raise ValueError("training source differs from the plan's pinned study")
    draws_path = Path(source["draws"]["path"])
    verified(draws_path, source["draws"]["sha256"])
    draws = json.loads(draws_path.read_text())
    if steps < 100 or steps > 1300 or steps % 100 or len(draws) != 5200:
        raise ValueError("native gate budget requires complete 100-step blocks of the 1300-step stream")
    if any(row["sigma_index"] != CYCLE[i % 10] or row["microbatch"] != i for i, row in enumerate(draws)):
        raise ValueError("training draw/sigma cycle differs")
    if source["config"].get("uniform_f18") is not True or source["config"]["world_size"] != 4:
        raise ValueError("source does not use four-draw F18 training")
    members = {row["relative_dir"]: row for row in source["splits"]["learning"]}
    if len(members) != 256 or len({s["actor"] for s in members.values()}) != 93:
        raise ValueError("frozen training membership differs")
    for component in source["components"].values():
        verified(Path(component["path"]), component["file_sha256"])
    context_pin = freeze_context(source["text_context"], output)
    for member in members.values():
        verified(Path(member["capture"]), member["capture_latent_sha256"])
    mask_references(combined, output, source)
    output.joinpath("provenance").mkdir(parents=True, exist_ok=True)
    plan_copy = output / "provenance/plan.md"
    shutil.copyfile(plan, plan_copy)
    atomic_json(output / "configs/source_study.json", source)
    atomic_json(output / "configs/draws.json", draws[: steps * 4])
    narrative = []
    raw = plan.read_text()
    for number in range(5):
        marker = f"### E{number}. "
        block = raw.split(marker, 1)[1].split("\n### ", 1)[0].split("\n## ", 1)[0]
        title, content = block.split("\n", 1)
        narrative.append({"id": f"E{number}", "question": title.strip(), "design": content.strip()})
    atomic_json(output / "configs/narrative.json", {"sections": narrative})
    study = {
        "format": "native_gate_training_stream_v1",
        "root": str(output),
        "source": source,
        "parent": {"path": str(source_path.resolve()), "sha256": file_sha256(source_path)},
        "steps": steps,
        "draws_per_step": 4,
        "interval": 100,
        "gpus": [4, 5, 6],
        "bins": 2048,
        "source_sigma_indices": [1, 4, 7, 9],
        "skipped_draw_indices": [i for i, d in enumerate(draws[: steps * 4]) if d["sigma_index"] not in (1, 4, 7, 9)],
        "sigmas": [source["config"]["sigmas"][i] for i in (1, 4, 7, 9)],
        "selected_draw_indices": [i for i, d in enumerate(draws[: steps * 4]) if d["sigma_index"] in (1, 4, 7, 9)],
        "components": source["components"],
        "context": context_pin,
        "plan": {"path": str(plan_copy), "sha256": file_sha256(plan_copy)},
        "code": code_identity(),
        "files": {
            str(p.relative_to(output)): file_sha256(p)
            for p in [
                output / "configs/source_study.json",
                output / "configs/draws.json",
                output / "comparison/masks.json",
                output / "comparison/events.json",
            ]
        },
    }
    study["identity"] = digest(study)
    atomic_json(output / "configs/frozen_study.json", study)
    return study


def load(path: Path) -> dict:
    study = json.loads(path.read_text())
    identity = study.pop("identity")
    if study["format"] != "native_gate_training_stream_v1" or digest(study) != identity:
        raise ValueError("frozen native gate study identity differs")
    study["identity"] = identity
    if code_identity() != study["code"]:
        raise ValueError("frozen native gate code changed")
    for relative, expected in study["files"].items():
        verified(Path(study["root"]) / relative, expected)
    verified(Path(study["plan"]["path"]), study["plan"]["sha256"])
    return study


def saved_input_root(source: dict, view: str, seed: int) -> Path:
    key = hashlib.sha256(view.encode()).hexdigest()
    roots = [Path(source["root"]), Path(source["parent"]["input_root"])]
    for root in roots:
        path = root / f"inputs/{key}/seed_{seed}"
        if (path / "manifest.json").is_file():
            return path
    # The standalone input owner will create a fresh bundle only for unconsumed draws.
    return roots[0] / f"inputs/{key}/seed_{seed}"
