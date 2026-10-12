"""Fixed membership uses current producer records and keeps people and inputs pinned."""

from __future__ import annotations

import concurrent.futures
import copy
import json
import shutil
from pathlib import Path

import pytest
import torch

from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.support.executor import InlineExecutor


def _build(record: dict, **options: object) -> dict:
    return subset.build_membership(Path(record["corpus_root"]), objective="white", splits=record["splits"], **options)


def _guides(record: dict) -> None:
    root = Path(record["corpus_root"])
    for source in record["sources"]:
        view = root / source["relative_dir"]
        render = view / dataset.render_name("white")
        render.write_bytes(b"guide RGB " + source["actor"].encode())
        capture = torch.load(view / dataset.capture_bundle_name("white"), weights_only=True)
        torch.save(
            {**capture, "master": capture["master"] + 1, "input_fingerprint": sha256(render)},
            view / dataset.guide_bundle_name("white"),
        )
        (view / dataset.render_metadata_name("white")).write_text(
            json.dumps({"objective": "white", "compositing_version": dataset.GUIDE_COMPOSITING_VERSION})
        )


def test_membership_reads_actual_masters_and_preserves_saved_hash_rule(current_membership: dict) -> None:
    before = copy.deepcopy(current_membership)
    membership = _build(current_membership)
    assert current_membership == before
    assert membership["splits"] == before["splits"]
    assert [source["relative_dir"] for source in membership["sources"]] == sorted(
        source["relative_dir"] for source in before["sources"]
    )
    for source, original in zip(membership["sources"], before["sources"], strict=True):
        assert source["shape"] == [2, 7, 2, 2]
        assert source["n_latent_frames"] == 7
        assert source["capture_latent_sha256"] == original["capture_latent_sha256"]
        assert source["rgb_sha256"] == original["rgb_sha256"]
    assert membership["sha256"] == subset.membership_hash(membership)
    assert not {"geometry", "chains", "attention", "context_latent_frames"}.intersection(membership)
    subset.validate_membership(before)
    assert before["sha256"] == current_membership["sha256"]


def test_relocation_preserves_discovered_membership_and_saved_identity(
    current_membership: dict, tmp_path: Path
) -> None:
    root = Path(current_membership["corpus_root"])
    moved = tmp_path.parent / f"{tmp_path.name}_relocated"
    shutil.copytree(root, moved)
    left = _build(current_membership)
    right = subset.build_membership(moved, objective="white", splits=current_membership["splits"])
    assert left["sha256"] == right["sha256"]
    relocated = {**current_membership, "corpus_root": str(moved)}
    subset.validate_membership(relocated)
    dataset.ClipStore(relocated).verify()


def test_all_views_of_one_person_share_a_group_across_parts(current_membership: dict) -> None:
    root = Path(current_membership["corpus_root"])
    old_name = current_membership["sources"][0]["relative_dir"]
    new_name = "Part_2/second_clip/views/view01"
    shutil.copytree(root / old_name, root / new_name)
    shutil.copyfile(root / old_name / "../../meta.json", root / new_name / "../../meta.json")
    capture_path = root / new_name / dataset.capture_bundle_name("white")
    capture = torch.load(capture_path, weights_only=True)
    capture["source"] = new_name
    torch.save(capture, capture_path)
    manifest_path = root / dataset.CAPTURE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["windows"].append({"bundle": f"{new_name}/{capture_path.name}", "box_xyxy": [0, 0, 64, 64]})
    manifest_path.write_text(json.dumps(manifest))
    membership = _build(current_membership)
    person = [source for source in membership["sources"] if source["actor"] == "1"]
    assert len(person) == 2
    assert {source["split"] for source in person} == {"train"}


def test_automatic_grouping_uses_person_hash_order_and_clamps_minimum() -> None:
    assert subset.split_actors(["2", "1", "3", "1"], 0.5, 0) == {"train": ["2"], "held_out": ["3", "1"]}
    assert subset.split_actors(["1", "2"], 0, 100) == {"train": ["2"], "held_out": ["1"]}
    with pytest.raises(ValueError, match="at least two people"):
        subset.split_actors(["1"])


@pytest.mark.parametrize(
    ("fraction", "minimum"), [(float("nan"), 1), (-0.1, 1), (1.1, 1), (True, 1), (0.1, -1), (0.1, True)]
)
def test_invalid_grouping_settings_fail(fraction: float, minimum: int) -> None:
    with pytest.raises(ValueError, match="held-out fraction"):
        subset.split_actors(["1", "2"], fraction, minimum)


def test_person_cap_is_independent_of_discovery_order(current_membership: dict) -> None:
    membership = subset.build_membership(
        Path(current_membership["corpus_root"]), objective="white", splits={"evaluation": ["1"]}, max_people=1
    )
    assert [source["actor"] for source in membership["sources"]] == ["1"]
    assert membership["excluded"] == {current_membership["sources"][1]["relative_dir"]: "person_not_selected"}


@pytest.mark.parametrize(
    "groups", [{"train": ["1"], "test": ["1"]}, {"train": ["missing"]}, {"train": ["1", "1"]}, {"train": "1"}]
)
def test_explicit_groups_refuse_duplicates_or_missing_people(current_membership: dict, groups: dict) -> None:
    with pytest.raises(
        ValueError, match=r"appears more than once|no available selected video|must be nonempty strings"
    ):
        subset.build_membership(Path(current_membership["corpus_root"]), objective="white", splits=groups)


def test_discovery_records_unavailable_view_before_grouping(current_membership: dict) -> None:
    source = current_membership["sources"][1]
    root = Path(current_membership["corpus_root"])
    (root / source["relative_dir"] / dataset.capture_bundle_name("white")).unlink()
    membership = subset.build_membership(root, objective="white", splits={"evaluation": ["1"]})
    assert len(membership["sources"]) == 1
    assert source["relative_dir"] in membership["excluded"]
    assert str(root) not in membership["excluded"][source["relative_dir"]]
    with pytest.raises(ValueError, match=r"selected video.*unavailable"):
        subset.build_membership(
            root, objective="white", splits={"evaluation": ["2"]}, source_ids=[source["relative_dir"]]
        )


@pytest.mark.parametrize(
    "name", ["/absolute/view", "Part_1/../views/view00", "Part_1/1/view00", "Part_1/1/views/missing"]
)
def test_explicit_sources_do_not_guess_or_omit_invalid_inputs(current_membership: dict, name: str) -> None:
    with pytest.raises(ValueError, match=r"selected video.*unavailable"):
        _build(current_membership, source_ids=[name])


@pytest.mark.parametrize("defect", ["crop", "objective", "source", "nonfinite"])
def test_capture_provenance_defect_is_rejected(current_membership: dict, defect: str) -> None:
    source = current_membership["sources"][0]
    path = Path(current_membership["corpus_root"]) / source["relative_dir"] / dataset.capture_bundle_name("white")
    capture = torch.load(path, weights_only=True)
    if defect == "crop":
        capture["box_xyxy"] = [1, 0, 65, 64]
    elif defect == "nonfinite":
        capture["master"][0, 0, 0, 0] = float("nan")
    else:
        capture[defect] = "different"
    torch.save(capture, path)
    with pytest.raises(ValueError, match=r"selected video.*unavailable"):
        _build(current_membership, source_ids=[source["relative_dir"]])


def test_paired_list_uses_shared_guide_checks_and_missing_guide_fails(current_membership: dict) -> None:
    with pytest.raises(ValueError, match="unavailable"):
        _build(current_membership, require_guide=True, source_ids=[current_membership["sources"][0]["relative_dir"]])
    _guides(current_membership)
    membership = _build(current_membership, require_guide=True)
    dataset.ClipStore(membership).verify(require_guide=True)
    source = membership["sources"][0]
    sidecar = Path(membership["corpus_root"]) / source["relative_dir"] / dataset.render_metadata_name("white")
    sidecar.write_text(json.dumps({"objective": "white", "compositing_version": -1}))
    with pytest.raises(ValueError, match=r"selected video.*outdated guide compositing"):
        _build(current_membership, require_guide=True, source_ids=[source["relative_dir"]])


def test_input_change_during_survey_refuses_membership(
    current_membership: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = subset.pin_content
    changed = False

    def pin(root: Path, name: str, objective: str, **options: object) -> dict:
        nonlocal changed
        result = original(root, name, objective, **options)
        if not changed:
            (root / name / "rgb.mp4").write_bytes(b"changed during survey")
            changed = True
        return result

    monkeypatch.setattr(subset, "pin_content", pin)
    with pytest.raises(ValueError, match="input content changed"):
        _build(current_membership)


def test_clip_store_d0_reads_no_guide_and_checks_changed_masters(current_membership: dict) -> None:
    membership = _build(current_membership)
    store = dataset.ClipStore(membership)
    source = membership["sources"][0]
    video = store.load(source["relative_dir"])
    assert video.z_g is None
    assert video.z_y.shape == (2, 7, 2, 2)
    assert video.actor == "1"
    assert video.split == "train"
    store.verify()
    with pytest.raises(ValueError, match="guide content hash is not recorded"):
        store.load(source["relative_dir"], require_guide=True)
    path = Path(membership["corpus_root"]) / source["relative_dir"] / dataset.capture_bundle_name("white")
    path.write_bytes(b"changed encoded input")
    with pytest.raises(ValueError, match="encoded content changed"):
        store.verify()


def test_cli_publishes_new_list_without_changing_inputs(
    monkeypatch: object, current_membership: dict, tmp_path: Path
) -> None:
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    root = Path(current_membership["corpus_root"])
    inputs = {path: sha256(path) for path in root.rglob("*") if path.is_file()}
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps(current_membership["splits"]))
    output = tmp_path / "membership.json"
    arguments = ["--corpus-root", str(root), "--objective", "white", "--splits", str(groups), "--output", str(output)]
    assert subset.main(arguments) == 0
    membership = subset.load_membership(output)
    subset.validate_membership(membership)
    subset.verify_prepared_membership(membership)
    assert all(sha256(path) == digest for path, digest in inputs.items())
    with pytest.raises(SystemExit, match="new file"):
        subset.main(arguments)


def test_cli_changed_input_during_publication_leaves_no_list(
    current_membership: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    original = dataset.atomic_write
    root = Path(current_membership["corpus_root"])
    source = current_membership["sources"][0]

    def publish(destination: Path, writer: object, **options: object) -> None:
        (root / source["relative_dir"] / "rgb.mp4").write_bytes(b"changed before publication")
        original(destination, writer, **options)

    monkeypatch.setattr(dataset, "atomic_write", publish)
    output = tmp_path / "membership.json"
    with pytest.raises(ValueError, match="input content changed"):
        subset.main(["--corpus-root", str(root), "--objective", "white", "--output", str(output)])
    assert not output.exists()


def test_cli_dangling_output_symlink_is_refused_before_survey(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    output = tmp_path / "membership.json"
    output.symlink_to(tmp_path / "missing")
    with pytest.raises(SystemExit, match="new file"):
        subset.main(["--output", str(output)])


def test_cli_publication_race_preserves_other_writer(
    current_membership: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    original = dataset.atomic_write
    output = tmp_path / "membership.json"

    def publish(destination: Path, writer: object, **options: object) -> None:
        def peer_write(temporary: Path) -> None:
            writer(temporary)
            if destination == output:
                destination.write_bytes(b"another complete list")

        original(destination, peer_write, **options)

    monkeypatch.setattr(dataset, "atomic_write", publish)
    with pytest.raises(FileExistsError):
        subset.main(
            ["--corpus-root", current_membership["corpus_root"], "--objective", "white", "--output", str(output)]
        )
    assert output.read_bytes() == b"another complete list"
    assert not subset.sources_csv_path(output).exists()
    assert not list(tmp_path.glob(".membership.tmp.*"))


@pytest.mark.subprocess
def test_membership_is_a_small_json_and_a_pure_csv(current_membership: dict, tmp_path: Path) -> None:
    root = Path(current_membership["corpus_root"])
    out = tmp_path / "lists"
    out.mkdir()
    groups = out / "groups.json"
    groups.write_text(json.dumps(current_membership["splits"]))
    output = out / "list.json"
    arguments = ["--corpus-root", str(root), "--objective", "white", "--splits", str(groups), "--output", str(output)]
    assert subset.main([*arguments, "--workers", "2"]) == 0
    assert sorted(path.name for path in out.iterdir()) == ["groups.json", "list.csv", "list.json"]
    meta = json.loads(output.read_text())
    assert "sources" not in meta
    assert "excluded" not in meta
    lines = (out / "list.csv").read_text().splitlines()
    assert lines[0].split(",")[0] == "relative_dir"
    assert len(lines) == 1 + meta["sources_csv"]["rows"]
    membership = subset.load_membership(output)
    assert membership["sha256"] == meta["sha256"]
    assert membership["sha256"] == subset.membership_hash(membership)
    assert [s["relative_dir"] for s in membership["sources"]] == sorted(
        s["relative_dir"] for s in current_membership["sources"]
    )
    dataset.ClipStore(membership).verify()
    (out / "list.csv").write_text((out / "list.csv").read_text().replace("Part", "Prt", 1))
    with pytest.raises(ValueError, match="does not match its SHA-256"):
        subset.load_membership(output)


def test_membership_csv_round_trips_unusual_fields(current_membership: dict, tmp_path: Path) -> None:
    record = copy.deepcopy(current_membership)
    record.pop("excluded", None)
    record["sources"][0]["optional_note"] = 'a, "quoted"\nvalue'
    record["sha256"] = subset.membership_hash(record)
    output = tmp_path / "membership.json"
    subset.write_membership(output, record)
    loaded = subset.load_membership(output)
    assert loaded == record
    assert "optional_note" not in loaded["sources"][1]
    single = tmp_path / "single.json"
    single.write_text(json.dumps(record))
    assert subset.load_membership(single) == record
    with pytest.raises(ValueError, match="must be a new file"):
        subset.write_membership(output, record)
    with pytest.raises(ValueError, match="summarize"):
        subset.membership_files({**record, "excluded": {}}, tmp_path / "other.json")
