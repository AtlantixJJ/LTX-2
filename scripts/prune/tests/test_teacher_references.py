import pytest
import torch

from scripts.prune.learn.teacher_references import REQUIRED, TeacherReferences, reference_key


def test_immutable_reuse_and_capacity(tmp_path):
    identity = {k: k for k in REQUIRED}
    cache = TeacherReferences(tmp_path, disk_cap=1000, hot_cap=100)
    v = torch.ones(1, 4, 2, dtype=torch.bfloat16)
    first = cache.produce(identity, lambda: (v, v * 2))
    second = cache.produce(identity, lambda: (_ for _ in ()).throw(AssertionError("must reuse teacher")))
    assert torch.equal(first["clean"], second["clean"]) and cache.hot_bytes <= 100
    with pytest.raises(ValueError):
        reference_key({**identity, "arm": "student"})
    with pytest.raises(ValueError, match="cap"):
        cache.budget([(1000, 1000)])
    (tmp_path / reference_key(identity) / "velocity.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        cache.read(identity)
