"""Fixed-video conversion must preserve source data, groups and selected coverage."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.onestep_avatar import dataset, subset
from scripts.onestep_avatar.hashing import sha256


@pytest.fixture
def old_subset(tmp_path: Path) -> dict:
    sources = []
    chains = []
    for actor, group in (("1", "train"), ("2", "held_out")):
        name = f"Part_1/{actor}/views/view00"
        view = tmp_path / name
        view.mkdir(parents=True)
        (view / "rgb.mp4").write_bytes(b"saved capture bytes" + actor.encode())
        torch.save(
            {
                "schema_version": 2,
                "master": torch.arange(56).reshape(2, 7, 2, 2).float(),
                "fps": 30.0,
                "objective": "white",
                "source": name,
                "vae_fingerprint": "saved VAE identity",
                "box_xyxy": [0, 0, 64, 64],
                "edge": 64,
                "encode_contract_version": 1,
            },
            view / dataset.capture_bundle_name("white"),
        )
        sources.append(
            {
                "relative_dir": name,
                "actor": actor,
                "fps": 30.0,
                "n_latent_frames": 7,
                "n_blocks": 3,
                "rgb_sha256": sha256(view / "rgb.mp4"),
                "guide_sha256": None,
            }
        )
        chains += [
            {"source": name, "actor": actor, "split": group, "blocks": [0, 1], "seed_is_clip_start": True},
            {"source": name, "actor": actor, "split": group, "blocks": [1, 2], "seed_is_clip_start": False},
        ]
    return {
        "kind": "one_step_argavatar_block_chains",
        "schema_version": 1,
        "objective": "white",
        "corpus_root": str(tmp_path),
        "capture_manifest": {"sha256": "original crop record"},
        "splits": {"train": ["1"], "held_out": ["2"]},
        "sources": sources,
        "chains": chains,
        "geometry": {
            "latent_time_scale": 8,
            "block_latent_frames": 2,
            "context_latent_frames": 1,
            "sink_latent_frames": 1,
        },
        "chain_length": 2,
    }


def test_conversion_preserves_people_hashes_and_exact_original_ranges(old_subset: dict) -> None:
    before = copy.deepcopy(old_subset)
    membership, plan = subset.convert_legacy(old_subset, original_file_sha256="original file hash")
    assert old_subset == before
    assert membership["splits"] == before["splits"]
    for source, original in zip(membership["sources"], before["sources"], strict=True):
        assert source["relative_dir"] == original["relative_dir"]
        assert source["actor"] == original["actor"]
        assert source["rgb_sha256"] == original["rgb_sha256"]
        assert "n_blocks" not in source
        assert source["capture_latent_sha256"]
    assert plan["original_source_records"] == before["sources"]
    assert plan["samples"][0]["ranges"] == [[0, 3], [3, 5]]
    assert plan["samples"][1]["ranges"] == [[3, 5], [5, 7]]
    assert plan["samples"][1]["original_chain_index"] == 1
    assert plan["membership_sha256"] == membership["sha256"]
    assert plan["sha256"] == subset.record_hash(plan)
    membership["corpus_root"] = "/relocated/corpus"
    subset.validate_membership(membership)


def test_conversion_refuses_changed_original_bytes(old_subset: dict) -> None:
    source = old_subset["sources"][0]
    path = Path(old_subset["corpus_root"]) / source["relative_dir"] / "rgb.mp4"
    path.write_bytes(b"changed capture")
    with pytest.raises(ValueError, match="original subset pin"):
        subset.convert_legacy(old_subset, original_file_sha256="hash")


def test_conversion_refuses_changed_frame_coverage(old_subset: dict) -> None:
    old_subset["sources"][0]["n_latent_frames"] = 8
    with pytest.raises(ValueError, match="frame count"):
        subset.convert_legacy(old_subset, original_file_sha256="hash")


def test_conversion_refuses_a_person_in_two_groups(old_subset: dict) -> None:
    old_subset["splits"]["held_out"].append("1")
    with pytest.raises(ValueError, match="multiple original groups"):
        subset.convert_legacy(old_subset, original_file_sha256="hash")


def test_cli_keeps_originals_and_refuses_existing_destinations(old_subset: dict, tmp_path: Path) -> None:
    original = tmp_path / "original.json"
    original.write_text(json.dumps(old_subset))
    original_hash = sha256(original)
    membership = tmp_path / "membership.json"
    plan = tmp_path / "frame_plan.json"
    args = ["--convert", str(original), "--output", str(membership), "--frame-plan-output", str(plan)]
    assert subset.main(args) == 0
    assert sha256(original) == original_hash
    saved = json.loads(membership.read_text())
    assert saved["original_subset_file_sha256"] == original_hash
    subset.validate_membership(saved)
    with pytest.raises(SystemExit, match="new output files"):
        subset.main(args)
    assert sha256(original) == original_hash


def test_clip_store_d0_reads_no_guide_and_checks_changed_masters(old_subset: dict) -> None:
    membership, _ = subset.convert_legacy(old_subset, original_file_sha256="original file hash")
    store = dataset.ClipStore(membership)
    source_id = membership["sources"][0]["relative_dir"]
    video = store.load(source_id)
    assert video.z_g is None
    assert video.z_y.shape == (2, 7, 2, 2)
    assert video.actor == "1"
    assert video.split == "train"
    store.verify()
    with pytest.raises(ValueError, match="guide content hash is not recorded"):
        store.load(source_id, require_guide=True)
    capture_path = Path(membership["corpus_root"]) / source_id / dataset.capture_bundle_name("white")
    capture_path.write_bytes(b"changed encoded input")
    with pytest.raises(ValueError, match="encoded content changed"):
        store.verify()
