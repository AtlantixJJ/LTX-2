"""Static selection and pinned CPU runtime checks."""

import os

import pytest
import torch

from scripts.onestep_avatar.tests.select import PREFIX, TESTS, commands, select
from scripts.onestep_avatar.tests.support.imports import targets


def test_cpu_runtime() -> None:
    assert torch.get_num_threads() == 1
    if os.environ.get("ONESTEP_AVATAR_TEST_CUDA") != "1":
        assert not torch.cuda.is_available()


@pytest.mark.parametrize(
    ("owner", "expected"),
    [
        ("model/common.py", "test_bidirectional.py"),
        ("model/causal.py", "test_causal_core.py"),
        ("corpus/subset.py", "test_subset.py"),
        ("training/engine.py", "test_training_runtime.py"),
        ("evaluate.py", "test_evaluate.py"),
        ("media.py", "test_media.py"),
        ("execution/supervision.py", "test_supervision.py"),
        ("execution/jobs.py", "test_queue.py"),
        ("experiments/conditioning_check.py", "experiments/test_conditioning_check.py"),
    ],
)
def test_area_mapping(owner: object, expected: object) -> None:
    tests = {
        TESTS + expected: "",
        TESTS + "test_consumer.py": "from scripts.onestep_avatar." + owner[:-3].replace("/", ".") + " import function",
    }
    plan = select([PREFIX + owner], "T1", tests)
    assert TESTS + expected in plan["files"]
    assert TESTS + "test_consumer.py" in plan["files"]


@pytest.mark.parametrize(
    "owner", ["hashing.py", "__init__.py", "execution/software.py", "tests/conftest.py", "tests/support/model.py"]
)
def test_shared_owner_escalates(owner: object) -> None:
    tests = {TESTS + "test_hashing.py": "", TESTS + "test_select.py": ""}
    plan = select([PREFIX + owner], "T0", tests)
    assert plan["tier"] == "T2"
    assert plan["files"] == sorted(tests)
    assert plan["prune"]


def test_relative_and_function_imports() -> None:
    assert "scripts.onestep_avatar.model.causal" in targets(
        "def f():\n from ..model import causal", "scripts.onestep_avatar.tests.test_example"
    )


def test_changed_tests_docs_and_boundary_commands() -> None:
    tests = {TESTS + "test_select.py": ""}
    assert select([PREFIX + "doc/core_algorithm.md"], "T0", tests)["files"] == []
    plan = select([TESTS + "test_select.py"], "T0", tests, imports_changed=True)
    rows = commands(plan)
    assert rows[0][0][-2:] == ["-m", "not subprocess"]
    assert rows[-1][1] == {"ONESTEP_AVATAR_BLOCK_EXPERIMENTS": "1"}


def test_prune_and_unknown_owner() -> None:
    tests = {TESTS + "test_causal_core.py": ""}
    assert select([PREFIX + "model/causal.py"], "T1", tests)["prune"]
    assert select([PREFIX + "new_owner.py"], "T0", tests)["tier"] == "T2"
