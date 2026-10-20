"""Historical V3 evaluation checks pinned bytes while allowing old source identity."""

import pytest

from scripts.prune.core import provenance, records
from scripts.prune.evaluate import joint_final


def frozen(tmp_path):
    def pin(name):
        path = tmp_path / name
        path.write_text(name)
        return dict(path=str(path), sha256=provenance.file_sha256(path))

    study = dict(
        format="joint_study_v3",
        root=str(tmp_path),
        code={"sha256": "historical source"},
        files=[pin("capture"), pin("context")],
        components={k: pin(k) for k in ("transformer", "text_encoder", "video_vae")},
        **{k: pin(k) for k in ("plan", "draws", "parent", "continuation")},
    )
    study["identity"] = records.digest(study)
    path = tmp_path / "study.json"
    records.publish_json(path, study)
    records.publish_json(tmp_path / "monitoring/owner.json", dict(status="complete"))
    return path, study


def test_completed_study_allows_historical_code(tmp_path):
    path, study = frozen(tmp_path)
    assert joint_final.load(path) == study


@pytest.mark.parametrize(
    "role",
    ["plan", "draws", "parent", "continuation", "capture", "context", "transformer", "text_encoder", "video_vae"],
)
def test_completed_study_rejects_changed_pin(tmp_path, role):
    path, _ = frozen(tmp_path)
    (tmp_path / role).write_text("changed")
    with pytest.raises(ValueError, match="frozen input changed"):
        joint_final.load(path)


def test_completed_study_requires_terminal_owner(tmp_path):
    path, _ = frozen(tmp_path)
    records.write_json(tmp_path / "monitoring/owner.json", dict(status="running"))
    with pytest.raises(ValueError, match="completed study"):
        joint_final.load(path)
