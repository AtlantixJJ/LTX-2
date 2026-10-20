"""Resumable learning state: rank RNG, head-gate checkpoints and versioned state receipts.

`save`/`load` own the `head_learning_state_v1` head-gate format. Progressive
policy states are retired; `load` rejects them. `publish_state`/`read_state` own
every other versioned torch state: one payload plus a JSON receipt with its
file hash, unchanged-state hash and draw cursor. A replay must hold the same state.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.core.records import publish, publish_json, publish_payload, tensor_hash, unchanged_state_hash


def rng_state(devices=None):
    result = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        devices = [torch.cuda.current_device()] if devices is None else devices
        result["cuda_states"] = {int(d): torch.cuda.get_rng_state(d) for d in sorted(set(devices))}
    return result


def restore_rng(state):
    if "ranks" in state:
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() != state["world_size"]:
            raise ValueError("distributed checkpoint requires the recorded rank count")
        state = state["ranks"][dist.get_rank()]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    for device, value in state.get("cuda_states", {}).items():
        torch.cuda.set_rng_state(value, device=int(device))
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device=state.get("cuda_device", 0))


def save(
    path,
    *,
    identity,
    update,
    phase,
    gates,
    adapters,
    adapter_optimizer,
    gate_optimizer=None,
    lineage=None,
    rng_override=None,
    draws_per_update=2,
):
    from scripts.prune.learn.adapters import state_dict

    if type(draws_per_update) is not int or draws_per_update < 1:
        raise ValueError("checkpoint draws per update must be a positive integer")
    state = {
        "format": "head_learning_state_v1",
        "identity": identity,
        "update": update,
        "phase": phase,
        "draw_cursor": update * draws_per_update,
        "draws_per_update": draws_per_update,
        "alpha": gates.alpha.detach().cpu(),
        "mask": gates.decision,
        "adapters": state_dict(adapters),
        "adapter_optimizer": adapter_optimizer.state_dict(),
        "gate_optimizer": gate_optimizer.state_dict() if gate_optimizer else None,
        "rng": rng_override
        if rng_override is not None
        else rng_state(
            [p.device.index for module in adapters.values() for p in (module.A, module.B) if p.device.type == "cuda"]
            + ([gates.alpha.device.index] if gates.alpha.device.type == "cuda" else [])
        ),
        "lineage": lineage or {},
        "progressive": None,
    }

    def write(p):
        with p.open("wb") as stream:
            torch.save(state, stream)

    if Path(path).exists():
        old = load(path, identity)
        if (
            old["update"] != update
            or old.get("draws_per_update", 2) != draws_per_update
            or old["phase"] != phase
            or tensor_hash(old["alpha"]) != tensor_hash(state["alpha"])
            or {k: tensor_hash(v) for k, v in old["adapters"].items()}
            != {k: tensor_hash(v) for k, v in state["adapters"].items()}
        ):
            raise ValueError("checkpoint update differs")
        return Path(path)
    publish(path, write)
    publish_json(
        Path(path).with_suffix(".json"),
        {
            "identity": identity,
            "update": update,
            "phase": phase,
            "draws_per_update": draws_per_update,
            "sha256": file_sha256(path),
            "tensor_hashes": {k: tensor_hash(v) for k, v in state["adapters"].items()},
            "alpha_sha256": tensor_hash(state["alpha"]),
            "lineage": state["lineage"],
            "progressive_sha256": None,
        },
    )
    return Path(path)


def load(path, identity=None):
    receipt = json.loads(Path(path).with_suffix(".json").read_text())
    if file_sha256(path) != receipt["sha256"]:
        raise ValueError("checkpoint bytes changed")
    state = torch.load(path, map_location="cpu", weights_only=False)  # Own trusted optimizer/RNG record.
    if state.get("format") != "head_learning_state_v1" or (identity is not None and state["identity"] != identity):
        raise ValueError("checkpoint training identity changed")
    count = state.get("draws_per_update", 2)
    if (
        type(count) is not int
        or count < 1
        or state["draw_cursor"] != state["update"] * count
        or count != receipt.get("draws_per_update", 2)
        or state["identity"] != receipt["identity"]
    ):
        raise ValueError("checkpoint draw cursor or receipt differs")
    if {k: tensor_hash(v) for k, v in state["adapters"].items()} != receipt["tensor_hashes"]:
        raise ValueError("adapter inventory changed")
    if tensor_hash(state["alpha"]) != receipt["alpha_sha256"]:
        raise ValueError("checkpoint logits changed")
    if state.get("progressive") is not None:
        raise ValueError("progressive head-gate checkpoints are retired; read them from their archived source snapshot")
    return state


def verify_prefix(state, prefix_identity):
    if state["update"] != 700 or state["phase"] != "soft" or state["identity"].get("prefix") != prefix_identity:
        raise ValueError("only the matching budget-independent 700-soft prefix can branch")
    return {"source_update": 700, "source_identity": state["identity"], "shared_prefix": prefix_identity}


def validate_finalists(study, *, arm=None, seed=None):
    """Blind work requires immutable candidate hashes, rather than a marker file."""
    path = Path(study["root"]) / "provenance/finalists_frozen.json"
    record = json.loads(path.read_text())
    if (
        record.get("study") != study["identity"]
        or record.get("config_sha256") != study["config_sha256"]
        or record.get("test_actors") != study["splits"]["test_actors"]
        or record.get("status") != "frozen"
        or not record.get("candidates")
    ):
        raise ValueError("blind work requires exact frozen study, settings, actors and finalist candidates")
    for candidate in record["candidates"]:
        for label in ("checkpoint", "mask", "envelope"):
            if file_sha256(candidate[label]["path"]) != candidate[label]["sha256"]:
                raise ValueError("a frozen finalist changed before blind inference")
    if arm is not None and not any(c["arm"] == arm and c["replicate_seed"] == seed for c in record["candidates"]):
        raise ValueError("candidate was not frozen before blind inference")
    return record


def publish_state(path, payload):
    """Publish a versioned state once, with its receipt; replays must not change its content."""
    path = Path(path)
    publish_payload(path, payload)
    publish_json(
        path.with_suffix(".json"),
        dict(
            format=payload["format"],
            study=payload["study"],
            update=payload["update"],
            sha256=file_sha256(path),
            state_sha256=unchanged_state_hash(payload),
            draw_cursor=payload["draw_cursor"],
        ),
    )
    return path


def read_state(path, *, format, label, study=None, world_size=None, draw_stride=None):
    """Verify receipt bytes, state hash, study, rank coverage and draw cursor before use.

    `draw_stride` is the frozen number of draws per update; None means the saved world size.
    """
    path = Path(path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    if file_sha256(path) != receipt["sha256"]:
        raise ValueError(f"{label} checkpoint bytes changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)  # Own trusted optimizer/RNG record.
    if payload.get("format") != format or unchanged_state_hash(payload) != receipt["state_sha256"]:
        raise ValueError(f"{label} checkpoint state changed")
    if study is not None and payload["study"] != study:
        raise ValueError(f"{label} checkpoint belongs to another study")
    if world_size is not None and payload["world_size"] != world_size:
        raise ValueError(f"{label} checkpoint world size differs")
    stride = payload["world_size"] if draw_stride is None else draw_stride
    if payload["draw_cursor"] != stride * payload["update"] or len(payload["rng"]) != payload["world_size"]:
        raise ValueError(f"{label} checkpoint draw cursor or rank RNG coverage differs")
    return payload
