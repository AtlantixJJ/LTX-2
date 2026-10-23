"""New frozen device policies may relocate ranks without changing draw semantics."""

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.prune.core import preflight
from scripts.prune.data import ffn_study, joint_study
from scripts.prune.data.study_io import read_device_config, validate_devices

ROOT = Path(__file__).resolve().parents[1]


def config():
    head = json.loads((ROOT / "configs/learned_heads_dev10_ddp_gt_uniform_f18_rank128.json").read_text())
    head.update(json.loads((ROOT / "configs/learned_ffn_dev10_ddp_joint.json").read_text()))
    return head


def test_new_ffn_and_joint_configs_accept_disjoint_relocated_groups():
    cfg = config()
    original = copy.deepcopy(cfg)
    ffn_study.validate_config(cfg)
    cfg.update(gpu_ids=[1, 2, 4, 7], test_gpu_ids=[0, 5])
    ffn_study.validate_config(cfg)
    cfg.update(format="joint_config_v3", arm="joint_clip_reset_v3", selection_updates=1500,
               rho=1 / 704, policy=dict(interval=50, late=10, earliest=200, limit=1500, recovery=200, alignment=128))
    joint_study.validate_config(cfg)
    # Validation reads old configs too and never rewrites either policy.
    ffn_study.validate_config(original)
    assert original["gpu_ids"] == [0, 1, 2, 3]


@pytest.mark.parametrize("change", [
    {"gpu_ids": [0, 0, 1, 2]}, {"gpu_ids": [0, 1, 2]}, {"gpu_ids": [0, 1, 2, True]},
    {"test_gpu_ids": []}, {"test_gpu_ids": [1, 4]}, {"test_gpu_ids": [-1]},
    {"test_gpu_ids": [5, 5]}, {"world_size": 3}, {"world_size": True},
])
def test_bad_group_does_not_reach_native_preparation(change):
    with pytest.raises(ValueError):
        validate_devices(dict(config(), **change))


def test_single_device_checks_use_config_and_refuse_busy_or_unavailable(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cfg = dict(test_gpu_ids=[5, 7])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda gpu: (45 * 2**30, 48 * 2**30))
    assert preflight.testing_gpu(cfg) == 5
    assert preflight.testing_gpu(cfg, 7) == 7
    with pytest.raises(ValueError, match="test_gpu_ids"):
        preflight.testing_gpu(cfg, 4)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda gpu: (43 * 2**30, 48 * 2**30))
    with pytest.raises(ValueError, match="44 GiB"):
        preflight.testing_gpu(cfg, 5)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 5)
    with pytest.raises(ValueError, match="unavailable"):
        preflight.testing_gpu(cfg, 5)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        preflight.testing_gpu(cfg)


def test_single_device_check_refuses_physical_id_remapping(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,4,7")
    with pytest.raises(ValueError, match="GPU IDs are physical"):
        preflight.testing_gpu(dict(test_gpu_ids=[0, 5]), 0)


def test_new_device_config_is_bounded_to_groups(tmp_path):
    path = tmp_path / "devices.json"
    groups = dict(gpu_ids=[1, 2, 4, 7], test_gpu_ids=[0, 5], world_size=4)
    path.write_text(json.dumps(groups))
    assert read_device_config(path) == groups
    for invalid in (dict(groups, gate_lr=100), dict(groups, world_size=3), {"gpu_ids": [0, 1, 2, 3]}):
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            read_device_config(path)
