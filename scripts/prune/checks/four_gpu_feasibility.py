"""Run the native full-clip feasibility gate with a four-device block split."""

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from scripts.prune.data.learned_head_study import load_study
from scripts.prune.learn import checkpoints, heads
from scripts.prune.learn.placement import partition


class PlacedSession:
    def __init__(self, current, devices, reference_root):
        self.current = current
        self.device = current.device
        self.execution_devices = tuple(torch.device(f"cuda:{d}") for d in devices)
        self.reference_root = reference_root

    def __getattr__(self, name):
        return getattr(self.current, name)

    @contextmanager
    def transformer(self, *args, **kwargs):
        with self.current.transformer(*args, **kwargs) as model:
            with partition(model, self.execution_devices):
                yield model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--reference-study", type=Path, required=True)
    args = parser.parse_args()
    heads.numerical_policy()
    torch.set_num_threads(4)
    study = load_study(args.study)
    placement = study["config"]["native_placement"]
    devices = placement["logical_devices"]
    if devices != [0, 1, 2, 3] or torch.cuda.device_count() != 4:
        raise ValueError("test requires exactly four visible GPUs in order 0–3")
    reference = json.loads(args.reference_study.read_text())
    identity = reference.pop("identity")
    if checkpoints.digest(reference) != identity:
        raise ValueError("reference study content changed")
    for key in ("components", "splits", "text_context", "software"):
        if reference[key] != study[key]:
            raise ValueError(f"single-GPU reference differs: {key}")
    old_config = reference["config"]
    new_config = {k: v for k, v in study["config"].items() if k != "native_placement"}
    if old_config != new_config:
        raise ValueError("scientific configuration differs from single-GPU reference")
    current = heads.open_current(study, 0)
    for device in devices:
        if torch.cuda.get_device_capability(device) != torch.cuda.get_device_capability(0):
            raise ValueError("four-GPU test requires matching GPU architectures")
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    current = PlacedSession(current, devices, Path(reference["root"]))
    root = Path(study["root"])
    checkpoints.publish_json(
        root / "provenance/placement.json",
        {
            "study": study["identity"],
            "placement": placement,
            "reference_study": identity,
            "reference_file_sha256": heads.provenance.file_sha256(args.reference_study),
            "devices": [{"logical": d, "name": torch.cuda.get_device_name(d)} for d in devices],
        },
    )
    started = time.monotonic()
    try:
        heads.feasibility(study, current, started + study["config"]["wall_time_hours"] * 3600)
    except Exception as error:
        checkpoints.publish_json(
            root / "logs/four_gpu.failure.json",
            {
                "study": study["identity"],
                "error": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.monotonic() - started,
                "context": json.loads((root / "checks/feasibility_progress.json").read_text())
                if (root / "checks/feasibility_progress.json").exists()
                else None,
            },
        )
        raise
    elapsed = time.monotonic() - started
    checkpoints.publish_json(
        root / "provenance/four_gpu_completion.json",
        {
            "study": study["identity"],
            "elapsed_seconds": elapsed,
            "gpu_hours": elapsed * 4 / 3600,
        },
    )


if __name__ == "__main__":
    main()
