"""Measure a queue module in its actual child; see doc/execution/runtime.md."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, TOKEN_ENV
from scripts.onestep_avatar.hashing import sha256

CONTRACT_ENV = "ONESTEP_AVATAR_RUNTIME_CONTRACT"
CONTRACT_SHA_ENV = "ONESTEP_AVATAR_RUNTIME_CONTRACT_SHA256"


def read_contract(path: Path, digest: str) -> dict:
    """Check the immutable queue contract without loading any native model."""
    record = json.loads(path.read_text())
    if sha256(path) != digest:
        raise ValueError("queue runtime contract bytes changed")
    if (
        not isinstance(record, dict)
        or set(record) != {"schema_version", "token", "job_sha256", "budget", "generic_phase", "numerical_policy"}
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or type(record["generic_phase"]) is not bool
        or record["numerical_policy"] not in (None, "deterministic")
        or not isinstance(record["token"], str)
        or len(record["token"]) != 32
        or any(c not in "0123456789abcdef" for c in record["token"])
        or not isinstance(record["job_sha256"], str)
        or len(record["job_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in record["job_sha256"])
    ):
        raise ValueError("queue runtime contract is malformed")
    from scripts.onestep_avatar.training import resources  # noqa: PLC0415 -- native resource owner at use

    resources.check_budget(record["budget"])
    if record["budget"] is None:
        raise ValueError("queue runtime requires an effective resource budget")
    return record


def result_path(contract_path: Path) -> Path:
    return contract_path.with_name(contract_path.name + ".result.json")


def prepare_contract(
    path: Path, *, token: str, job_sha256: str, budget: dict, generic_phase: bool, numerical_policy: str | None
) -> dict[str, str]:
    """Publish one attempt binding before launch; no model session or child."""
    from scripts.onestep_avatar.execution.queue_launch import (  # noqa: PLC0415 -- avoids queue import cycle
        publish_exclusive,
    )

    publish_exclusive(
        path,
        {
            "schema_version": 1,
            "token": token,
            "job_sha256": job_sha256,
            "budget": budget,
            "generic_phase": generic_phase,
            "numerical_policy": numerical_policy,
        },
    )
    digest = sha256(path)
    read_contract(path, digest)
    return {CONTRACT_ENV: str(path.resolve()), CONTRACT_SHA_ENV: digest}


def verify_result(path: Path, digest: str) -> dict:
    """Recheck saved child measurements and policy without a native loader."""
    from scripts.onestep_avatar.training import numerics, resources  # noqa: PLC0415 -- model-free saved checks

    contract = read_contract(path, digest)
    saved = result_path(path)
    record = json.loads(saved.read_text())
    if (
        not isinstance(record, dict)
        or set(record)
        != {
            "schema_version",
            "contract_sha256",
            "token",
            "job_sha256",
            "state",
            "error",
            "numerical_policy_requested",
            "numerical_policy_observed",
            "resource_records",
        }
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["contract_sha256"] != digest
        or record["token"] != contract["token"]
        or record["job_sha256"] != contract["job_sha256"]
        or record["state"] != "passed"
        or record["error"] is not None
        or record["numerical_policy_requested"] != contract["numerical_policy"]
        or not isinstance(record["resource_records"], list)
    ):
        raise ValueError("queue runtime result differs from its successful contract")
    if contract["numerical_policy"] == "deterministic":
        numerics.validate(record["numerical_policy_observed"], required=True)
    elif record["numerical_policy_observed"] is not None:
        raise ValueError("queue runtime recorded an unrequested numerical policy")
    if contract["generic_phase"]:
        resources.validate_records(record["resource_records"], 1, ["runtime"], contract["budget"])
    elif record["resource_records"]:
        raise ValueError("checker runtime must retain the owner's phase measurements")
    return record


def run(operation: Callable[[], object]) -> object:  # noqa: PLR0912 -- one child failure and publication lifecycle
    """Call one module main; queue bindings enable measurements, never launches."""
    selected = os.environ.get(CONTRACT_ENV)
    digest = os.environ.get(CONTRACT_SHA_ENV)
    if selected is None and digest is None:
        return operation()
    if not selected or not digest:
        raise ValueError("queue runtime environment binding is incomplete")
    path = Path(selected)
    contract = read_contract(path, digest)
    if os.environ.get(TOKEN_ENV) != contract["token"] or os.environ.get(JOB_ENV) != contract["job_sha256"]:
        raise ValueError("queue runtime attempt identity differs")
    from scripts.onestep_avatar.execution.queue_launch import (  # noqa: PLC0415 -- avoids queue import cycle
        publish_exclusive,
    )
    from scripts.onestep_avatar.training import numerics, resources  # noqa: PLC0415 -- model-free saved checks

    if resources.budget_from_environment() != contract["budget"]:
        raise ValueError("queue runtime effective budget differs")
    record = {
        "schema_version": 1,
        "contract_sha256": digest,
        "token": contract["token"],
        "job_sha256": contract["job_sha256"],
        "state": "failed",
        "error": None,
        "numerical_policy_requested": contract["numerical_policy"],
        "numerical_policy_observed": None,
        "resource_records": [],
    }
    phase = None
    try:
        if contract["numerical_policy"] == "deterministic":
            record["numerical_policy_observed"] = numerics.apply(environment_required=True)
        if contract["generic_phase"]:
            phase = resources.Phase("cuda:0", "runtime", 0, contract["budget"]).start()
        try:
            value = operation()
        except SystemExit as exit_error:
            if exit_error.code not in (None, 0):
                raise
            value = 0
        if type(value) is int and value != 0:
            raise ValueError(f"queue runtime module returned nonzero status {value}")
        record["state"] = "passed"
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if phase is not None:
            observation = phase.finish(record["error"])
            record["resource_records"].append(observation)
            if observation["state"] != "passed":
                record.update(state="failed", error=observation["error"])
        publish_exclusive(result_path(path), record)
    if record["state"] != "passed":
        raise ValueError(record["error"])
    return value
