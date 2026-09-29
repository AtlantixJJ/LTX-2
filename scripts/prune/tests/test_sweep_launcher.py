from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


STUB = r'''#!__PYTHON__
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ["SWEEP_TEST_ROOT"])
failure = os.environ.get("SWEEP_TEST_FAILURE")
if args[:2] == ["-m", "scripts.prune.core.preflight"]:
    sys.exit(1 if failure == "preflight" else 0)
if args[:2] == ["-m", "scripts.prune.score.head_scores"]:
    if failure == "score":
        sys.exit(2)
    path = root / "head_scores.json"
    path.write_text(json.dumps({"iterative": {"target_sparsity": float(args[args.index("--target-sparsity")+1])}}))
    print(path)
    sys.exit(0)
if args[:2] == ["-m", "scripts.prune.evaluate.phase1_gates"]:
    if failure == "evaluation":
        sys.exit(3)
    Path(args[args.index("--output")+1]).write_text("{}")
    sys.exit(0)
if args[0] == "-c" and "artifacts.run_dir" in args[1]:
    path = root / "expr" / "refiner_prune" / "2.5" / "sweep"
    path.mkdir(parents=True, exist_ok=True)
    print(path)
    sys.exit(0)
os.execv(sys.executable, [sys.executable, *args])
'''


@pytest.mark.parametrize("failure,expected", [("preflight", None), ("score", "score_failed"),
                                              ("evaluation", "evaluation_failed"), (None, "complete")])
def test_sweep_status_and_exit(tmp_path, failure, expected):
    script = tmp_path / "scripts" / "prune" / "run_head_sweep.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path("scripts/prune/run_head_sweep.sh"), script)
    stub = tmp_path / "python-stub"
    stub.write_text(STUB.replace("__PYTHON__", sys.executable))
    stub.chmod(0o755)
    env = {**os.environ, "LTX_PYTHON": str(stub), "SWEEP_TEST_ROOT": str(tmp_path),
           "SPARSITY_LIST": "0.05", "SWEEP_STAGGER_SECONDS": "0"}
    if failure:
        env["SWEEP_TEST_FAILURE"] = failure
    result = subprocess.run(["bash", str(script), "2.5", "0"], cwd=tmp_path, env=env,
                            text=True, capture_output=True, check=False)
    assert (result.returncode == 0) is (failure is None)
    assert ("sweep complete" in result.stdout) is (failure is None)
    manifest = tmp_path / "expr" / "refiner_prune" / "2.5" / "sweep" / "sweep_manifest.json"
    if expected is None:
        assert not manifest.exists()
    else:
        assert json.loads(manifest.read_text())["candidates"][0]["status"] == expected
