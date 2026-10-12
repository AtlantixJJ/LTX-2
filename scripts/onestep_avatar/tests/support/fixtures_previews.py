"""Shared saved-input builders and fixtures; never a collected test module."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.support.contracts import A, B, _contract
from scripts.onestep_avatar.training import config
from scripts.onestep_avatar.training.checkpoints import CONTRACT_KEY


def _fixed_inputs(tmp_path: Path) -> tuple:
    files = {}
    for role in ("subset", "capture", "guide", "first_image", "text", "noise"):
        path = tmp_path / role
        path.write_text(role)
        files[role] = {"path": str(path), "sha256": sha256(path), "tensor_sha256": sha256(path)}
    record = {
        "schema_version": 2,
        "kind": "onestep_avatar.preview_inputs",
        "mode": "bidirectional",
        "schedule": [0.725, 0],
        "input_files": files,
        "evaluation_arguments": [
            "--mode",
            "bidirectional",
            "--subset",
            files["subset"]["path"],
            "--noise-file",
            files["noise"]["path"],
            "--schedule",
            "0.725",
            "0",
        ],
    }
    path = tmp_path / "preview.json"
    path.write_text(json.dumps(record))
    settings = config.RunSettings(
        "bidirectional", Path(files["subset"]["path"]), tmp_path / "run", config.BidirectionalSettings()
    )
    return path, settings


def _completed(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint.safetensors"
    contract = _contract()
    contract["adapter"]["step"] = 100
    save_file({A: torch.ones(2, 4), B: torch.zeros(4, 2)}, checkpoint, metadata={CONTRACT_KEY: json.dumps(contract)})
    checkpoint.with_suffix(".complete.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "path": str(checkpoint),
                "sha256": sha256(checkpoint),
                "state": "complete",
                "step": 100,
            }
        )
    )
    return checkpoint
