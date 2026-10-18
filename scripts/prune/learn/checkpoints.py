"""Immutable publication and resumable adapter/gate state for learned heads."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.prune.core.provenance import file_sha256


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return digest(
        {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    )


def publish(path, writer):
    """Atomic create. Writers never replace completed evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".publish-", suffix=path.suffix)
    os.close(fd)
    try:
        writer(Path(name))
        with open(name, "rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            if file_sha256(name) != file_sha256(path):
                raise ValueError(f"completed artifact differs: {path}") from None
    finally:
        Path(name).unlink(missing_ok=True)
    return path


def publish_json(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return publish(path, lambda p: p.write_text(raw))


def publish_tensor(path, value):
    value = value.detach().cpu().contiguous()
    if Path(path).exists():
        if tensor_hash(torch.load(path, map_location="cpu", weights_only=True)) != tensor_hash(value):
            raise ValueError(f"completed tensor differs: {path}")
        return Path(path)

    # A stable stream filename avoids archive-name-dependent torch.save bytes.
    def write(p):
        with p.open("wb") as stream:
            torch.save(value, stream)

    return publish(path, write)


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


def save_pre_event(path, *, identity, update, gates, adapters, adapter_optimizer, gate_optimizer, rng_override=None):
    """Save a disposable transaction input separately from canonical checkpoints.

    A restart replays from its last canonical checkpoint. It checks this input
    before publishing the same event again. It never treats this file as a
    completed update. Saving after the reset would lose gate Adam evidence.
    """
    from scripts.prune.data.resume_learned_heads import unchanged_state_hash
    from scripts.prune.learn.adapters import state_dict

    policy = gates.progressive
    if policy.update != update - 1 or update % policy.interval or policy.recovering:
        raise ValueError("pre-event publication requires a pending selection boundary")
    state = dict(
        format="clipped_reset_transaction_v1", stage="post_optimizer_pre_event",
        identity=identity, update=update, draw_cursor=4 * update,
        alpha=gates.alpha.detach().cpu().clone(),
        pre_event_gates=policy.compose(gates.alpha).detach().cpu().clone(),
        progressive=policy.state_dict(), adapters=state_dict(adapters),
        adapter_optimizer=adapter_optimizer.state_dict(), gate_optimizer=gate_optimizer.state_dict(),
        rng=rng_override if rng_override is not None else rng_state(),
    )
    if Path(path).exists():
        sidecar = Path(path).with_suffix(".json")
        if sidecar.exists() and json.loads(sidecar.read_text())["sha256"] != file_sha256(path):
            raise ValueError("pre-event transaction bytes changed")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if unchanged_state_hash(saved) != unchanged_state_hash(state) or saved["identity"] != identity:
            raise ValueError("replayed pre-event transaction differs")
    else:
        def write(p):
            with p.open("wb") as stream:
                torch.save(state, stream)

        publish(path, write)
    publish_json(Path(path).with_suffix(".json"), dict(
        stage=state["stage"], update=update, sha256=file_sha256(path), state_sha256=unchanged_state_hash(state),
    ))


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
        "progressive": gates.progressive.state_dict() if getattr(gates, "progressive", None) else None,
    }

    def write(p):
        with p.open("wb") as stream:
            torch.save(state, stream)

    if Path(path).exists():
        old = load(path, identity)
        if state["progressive"] is not None:
            from scripts.prune.data.resume_learned_heads import unchanged_state_hash

            if unchanged_state_hash(old) != unchanged_state_hash(state):
                raise ValueError("replayed progressive checkpoint state differs")
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
            "progressive_sha256": digest(state["progressive"]) if state["progressive"] is not None else None,
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
        type(count) is not int or count < 1
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
        from scripts.prune.learn.progressive import ProgressivePolicy

        policy = ProgressivePolicy.from_state(state["progressive"])
        if not torch.equal(policy.window[-1], state["alpha"]):
            raise ValueError("checkpoint progressive snapshot differs from logits")
        fixed = ~policy.free
        if not torch.equal(state["alpha"][fixed], policy.frozen[fixed]):
            raise ValueError("checkpoint terminal logits changed")
        if policy.update != state["update"] or digest(state["progressive"]) != receipt.get("progressive_sha256"):
            raise ValueError("checkpoint progressive cursor or history changed")
        if state["progressive"]["format"] == "progressive_clip_reset_v2":
            optimizer = state["gate_optimizer"]
            if optimizer is None:
                raise ValueError("clipped-reset checkpoint lacks its gate optimizer")
            local_step = 0 if policy.recovering else policy.update % policy.interval
            values = list(optimizer["state"].values())
            if local_step == 0:
                if values:
                    raise ValueError("clipped-reset gate Adam was not fully reset")
            elif len(values) != 1 or float(values[0]["step"]) != local_step:
                raise ValueError("clipped-reset gate Adam local cursor differs")
            if any(
                value.shape != state["alpha"].shape or not torch.isfinite(value).all() or value[fixed].any()
                for item in values for key, value in item.items() if key != "step"
            ):
                raise ValueError("clipped-reset gate Adam moments or removed coordinates differ")
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
