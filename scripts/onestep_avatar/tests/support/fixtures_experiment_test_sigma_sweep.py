"""Shared saved-input builders and fixtures; never a collected test module."""

import json
from pathlib import Path

from scripts.onestep_avatar.execution import jobs as queue_jobs
from scripts.onestep_avatar.hashing import sha256


def sweep_queue_job(path: Path, output: Path) -> dict:
    """Prepare the real queue command through its public parser and identity owner."""
    job = {
        "id": "matched",
        "kind": "experiment",
        "experiment": "sigma_sweep",
        "spec": str(path),
        "spec_sha256": sha256(path),
        "arguments": ["--spec", str(path), "--output", str(output)],
        "output": str(output),
        "completion": {"manifest": str(output / "manifest.json")},
    }
    jobs = path.parent / "queue_jobs.json"
    jobs.write_text(json.dumps({"schema_version": 1, "jobs": [job]}))
    return queue_jobs.prepare_jobs(jobs)[0]
