"""Shared test fixtures.

Controlled lifecycle substitutions (real launch-gate checks never use them) and the shared
saved-input fixtures and `checked_settings`, registered here once instead of
being imported into every test module that needs them.
"""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
if os.environ.get("ONESTEP_AVATAR_TEST_CUDA") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch

from scripts.onestep_avatar import PACKAGE_ROOT
from scripts.onestep_avatar.execution import completion as queue_completion
from scripts.onestep_avatar.execution import queue_launch

torch.set_num_threads(1)


if os.environ.get("ONESTEP_AVATAR_BLOCK_EXPERIMENTS") == "1":
    hook_directory = PACKAGE_ROOT / "tests/blocked_imports"
    hook_spec = importlib.util.spec_from_file_location(
        "onestep_avatar_experiment_import_blocker", hook_directory / "sitecustomize.py"
    )
    if hook_spec is None or hook_spec.loader is None:
        raise ImportError("experiment import blocker is absent")
    hook_spec.loader.exec_module(importlib.util.module_from_spec(hook_spec))
    previous_path = os.environ.get("PYTHONPATH")
    if previous_path is None or str(hook_directory) not in previous_path.split(os.pathsep):
        os.environ["PYTHONPATH"] = str(hook_directory) + ("" if previous_path is None else os.pathsep + previous_path)
    collect_ignore = ["experiments"]


@pytest.fixture
def controlled_training_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the synthetic marker test on hash/mode checks; science has its own tests."""
    from scripts.onestep_avatar.training import (  # noqa: PLC0415 -- defer loader or fixture imports until preflight and test hooks are installed
        engine,
    )

    monkeypatch.setattr(engine, "verify_training_conditions", lambda _job, _checkpoint, _contract: None)


@pytest.fixture
def controlled_evaluation_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope synthetic lifecycle records to lifecycle and generic tensor checks.

    These tests do not contain checked masters, weights or actual sampler output.
    Scientific result binding is tested separately against the production verifier.
    No production path bypasses the verifier.
    """
    from scripts.onestep_avatar import evaluate  # noqa: PLC0415 -- install the import blocker before model owners

    monkeypatch.setattr(evaluate, "verify_evaluation_conditions", lambda _arguments, _paths: None)
    monkeypatch.setattr(evaluate, "evaluation_evidence_paths", lambda _arguments, paths: paths)


@pytest.fixture
def controlled_queue_launch(monkeypatch: pytest.MonkeyPatch, controlled_queue_observation: None) -> None:  # noqa: ARG001 -- fixture installs controlled observer
    """Keep old fake-Popen tests scoped to receipts/retries, not OS registration.

    Each caller already replaces Popen with an immediate synthetic completion.
    These substitutions do not establish bootstrap/grant or process correctness.
    Real-process integration tests retain every production protocol function.
    """
    monkeypatch.setattr(queue_launch, "guard_command", lambda path: queue_launch.load_request(path)["command"])

    def registration(path: Path, child: object, _command: object, _refresh: object, **_limits: object) -> dict:
        request = queue_launch.load_request(path)
        return {"identity": {"pid": child.pid, "start_ticks": 1, "command": request["command"], "terminal": False}}

    monkeypatch.setattr(queue_launch, "wait_registration", registration)
    monkeypatch.setattr(queue_launch, "publish_grant", lambda _path: {})


@pytest.fixture
def controlled_queue_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope fake observations to receipt/guard fixtures, never native acceptance."""
    from scripts.onestep_avatar.execution import runtime, supervision  # noqa: PLC0415 -- fixture only

    def observe(child: object, **arguments: object) -> dict:
        code = child.poll()
        if code is None:
            code = child.wait(timeout=5)
        result = {"state": "passed", "error": None, "returncode": code, "workers_absent": True}
        queue_launch.publish_exclusive(arguments["evidence_path"], result)
        return result

    monkeypatch.setattr(supervision, "supervise", observe)
    monkeypatch.setattr(runtime, "verify_result", lambda *_: {})
    monkeypatch.setattr(queue_completion, "verify_attempt_evidence", lambda *_: [])


@pytest.fixture
def current_membership(tmp_path: Path) -> dict:
    from scripts.onestep_avatar.tests.support.corpus import (  # noqa: PLC0415 -- pytest fixture imports occur after CPU/blocker setup
        build_membership,
    )

    return build_membership(tmp_path)


@pytest.fixture
def checked_settings(current_membership: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    from ltx_core.types import SpatioTemporalScaleFactors  # noqa: PLC0415 -- needed only by preflight fixtures
    from scripts.onestep_avatar.corpus import precompute  # noqa: PLC0415 -- fixture-only saved-input producers
    from scripts.onestep_avatar.training import config, engine  # noqa: PLC0415 -- import after the architecture blocker

    membership = current_membership
    path = tmp_path / "membership.json"
    path.write_text(json.dumps(membership))
    settings = config.parse_settings(
        [
            "--mode",
            "bidirectional",
            "--subset",
            str(path),
            "--output",
            str(tmp_path / "output"),
            "--variant",
            "dev",
            "--objective",
            "white",
            "--guide-mode",
            "d0",
            "--span-latent-frames",
            "7",
        ]
    )
    specification = SimpleNamespace(
        scale_factors=SpatioTemporalScaleFactors(8, 32, 32),
        caps=SimpleNamespace(latent_channels=2),
        sigmas=[0.725],
        paths=SimpleNamespace(
            transformer=lambda: tmp_path / "base.safetensors", video_vae=lambda: tmp_path / "vae.safetensors"
        ),
    )
    monkeypatch.setattr(engine.backbone, "resolve", lambda *_args: specification)
    monkeypatch.setattr(precompute, "file_fingerprint", lambda _path: "saved VAE identity")
    monkeypatch.setattr(
        engine.backbone,
        "identity",
        lambda *_args, **_kwargs: {
            "base_transformer_sha256": "a" * 64,
            "base_transformer_file": "base.safetensors",
            "base_variant": "dev",
            "model_key": "2.5",
        },
    )
    monkeypatch.setattr(engine, "build_transformer", lambda *_args: pytest.fail("preflight loaded a transformer"))
    return settings, membership


@pytest.fixture(scope="session", autouse=True)
def cached_software_inputs() -> object:
    """Cache unchanged source and versions while keeping per-test mutation hooks."""
    import copy  # noqa: PLC0415 -- fixture-only imports follow the CPU/import blocker setup

    from scripts.onestep_avatar import (  # noqa: PLC0415 -- pytest fixture imports occur after CPU/blocker setup
        LTX_ROOT,
    )
    from scripts.onestep_avatar.execution import (  # noqa: PLC0415 -- pytest fixture imports occur after CPU/blocker setup
        software,
    )

    patch = pytest.MonkeyPatch()
    original_hash, original_versions = software.sha256, software.runtime_versions
    hashes = {}
    versions = None

    def digest(path: object) -> object:
        path = Path(path).resolve()
        if path.suffix != ".py" or not path.is_relative_to(LTX_ROOT):
            return original_hash(path)
        stat = path.stat()
        key = (path, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in hashes:
            hashes[key] = original_hash(path)
        return hashes[key]

    def runtime_versions() -> object:
        nonlocal versions
        if versions is None:
            versions = original_versions()
        return copy.deepcopy(versions)

    patch.setattr(software, "sha256", digest)
    patch.setattr(software, "runtime_versions", runtime_versions)
    yield
    patch.undo()


@pytest.fixture
def attempt_token() -> object:
    """Unique /proc lookup token per lifecycle case."""
    from uuid import uuid4  # noqa: PLC0415 -- fixture-only imports follow the CPU/import blocker setup

    return uuid4().hex


def pytest_configure(config: object) -> None:
    for name, description in {
        "subprocess": "starts a Python child that imports the model stack",
        "boundary": "import, software profile and entry CLI architecture checks",
        "slow": "takes more than one second on the pinned CPU runtime",
    }.items():
        config.addinivalue_line("markers", name + ": " + description)
    config._avatar_unmarked_slow = {}


def pytest_collection_modifyitems(items: object) -> None:
    import ast  # noqa: PLC0415 -- fixture-only imports follow the CPU/import blocker setup

    from scripts.onestep_avatar.tests.select import (  # noqa: PLC0415 -- pytest fixture imports occur after CPU/blocker setup
        BOUNDARY,
    )

    sources = {}
    for item in items:
        relative = str(item.path.relative_to(PACKAGE_ROOT / "tests"))
        if relative in BOUNDARY:
            item.add_marker(pytest.mark.boundary)
        if item.path not in sources:
            sources[item.path] = ast.parse(item.path.read_text())
        nodes = [
            node
            for node in ast.walk(sources[item.path])
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == (item.originalname or item.name)
        ]
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr in {"run", "Popen", "check_output"}
            for root in nodes
            for node in ast.walk(root)
        ):
            item.add_marker(pytest.mark.subprocess)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: object, call: object) -> None:  # noqa: ARG001 -- pytest hook or fixture interface
    outcome = yield
    report = outcome.get_result()
    if report.duration > 1 and not any(item.get_closest_marker(name) for name in ("slow", "subprocess")):
        item.config._avatar_unmarked_slow[item.nodeid] = report.duration


def pytest_terminal_summary(terminalreporter: object) -> None:
    rows = terminalreporter.config._avatar_unmarked_slow
    if rows:
        terminalreporter.write_sep("-", "CPU tests over 1 s without slow/subprocess markers")
        for name, duration in sorted(rows.items(), key=lambda row: -row[1]):
            terminalreporter.write_line(f"{duration:.2f}s {name}")


@pytest.fixture(autouse=True)
def isolate_attempt_token(attempt_token: str) -> object:
    from scripts.onestep_avatar.tests.support.tokens import (  # noqa: PLC0415 -- fixture-only imports follow the CPU/import blocker setup
        reset_token,
        restore_token,
    )

    handle = reset_token(attempt_token)
    yield
    restore_token(handle)


@pytest.fixture(autouse=True)
def controlled_registry_handle(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic Popen fixtures register one terminal synthetic handle.

    This substitution is confined to the existing controlled launch fixture.
    Real launch and containment tests retain the production identity observer.
    """
    if "controlled_queue_launch" not in request.fixturenames:
        return
    from scripts.onestep_avatar.execution import process_registry  # noqa: PLC0415 -- controlled observer only

    original = process_registry._identity
    monkeypatch.setattr(
        process_registry,
        "_identity",
        lambda pid: (
            {"pid": pid, "start_ticks": 1, "command": ["synthetic child"], "terminal": True}
            if pid == 99999
            else original(pid)
        ),
    )
