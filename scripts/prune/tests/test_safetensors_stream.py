"""Streaming preserves real serialized bytes and rejects changed sources before publication."""

import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.prune.core import records
from scripts.prune.score import export_learned_ffn, export_pruned, safetensors_stream
from scripts.prune.tests.test_export_pruned import _checkpoint


def test_mixed_dtype_copy_matches_serializer_bytes_and_hashes(tmp_path):
    tensors = {
        "bf16": torch.arange(7).to(torch.bfloat16),
        "float": torch.randn(3, 4),
        "integer": torch.arange(2),
        "empty": torch.empty(0),
        "scalar": torch.tensor(True),
    }
    source, output = tmp_path / "source.safetensors", tmp_path / "copy.safetensors"
    save_file(tensors, str(source), metadata={"config": '{"note":"é"}'})
    shapes, metadata = safetensors_stream.inspect(source)
    inventory = safetensors_stream.write(source, output, shapes, metadata, inventory=True, chunk_bytes=3)
    assert output.read_bytes() == source.read_bytes()
    assert {k: v["sha256"] for k, v in inventory.items()} == {k: records.tensor_hash(v) for k, v in tensors.items()}


@pytest.mark.parametrize("mode", ["masked_full", "sparse", "compact", "compact_faithful"])
def test_width_stream_matches_prior_in_memory_serializer(tmp_path, mode):
    source, output, reference = [
        tmp_path / name for name in ("source.safetensors", "stream.safetensors", "prior.safetensors")
    ]
    _checkpoint(source)
    masks = {"0.attn1": [1, 0], "0.attn2": [0, 1], "0.ff": [1, 0, 1]}
    tensors, metadata = export_pruned.load(source)
    export_pruned.export_tensors(
        tensors, metadata, masks, widths=export_pruned.checkpoint_mask_widths(source), model_key="2.5", mode=mode
    )
    save_file(tensors, str(reference), metadata=metadata)
    export_pruned.export(source, masks, output, model_key="2.5", mode=mode)
    assert output.read_bytes() == reference.read_bytes()


@pytest.mark.parametrize("merge_first", [False, True])
def test_streamed_merge_preserves_family_order_and_binds_inventory(tmp_path, merge_first):
    source, output, reference = [
        tmp_path / name for name in ("source.safetensors", "stream.safetensors", "prior.safetensors")
    ]
    _checkpoint(source)
    masks = {"0.attn1": [1, 0], "0.ff": [1, 0, 1]}
    base, metadata = export_pruned.load(source)
    for key in base:
        base[key] = base[key].to(torch.bfloat16)
    save_file(base, str(source), metadata=metadata)
    shape = (3, 3) if merge_first else (2, 3)
    state = {"0.ff.up.A": torch.randn(2, shape[1]), "0.ff.up.B": torch.randn(shape[0], 2)}
    expected = {k: v.clone() for k, v in base.items()}
    if merge_first:
        key = f"{export_pruned.PREFIX}.0.ff.net.0.proj.weight"
        expected[key] = (expected[key].float() + state["0.ff.up.B"] @ state["0.ff.up.A"]).to(torch.bfloat16)
    export_pruned.export_tensors(
        expected, metadata, masks, widths=export_pruned.checkpoint_mask_widths(source), model_key="2.5", mode="compact"
    )
    if not merge_first:
        key = f"{export_pruned.PREFIX}.0.ff.net.0.proj.weight"
        expected[key] = (expected[key].float() + state["0.ff.up.B"] @ state["0.ff.up.A"]).to(torch.bfloat16)
    envelope_path = tmp_path / "envelope.json"
    envelope = {
        "family": "fixture",
        "tensor_inventory": {
            k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=records.tensor_hash(v)) for k, v in expected.items()
        },
    }
    # Pin the independent expected envelope before the writer fills its equivalent inventory.
    records.publish_json(envelope_path, envelope)
    config = json.loads(metadata["config"])
    from scripts.prune.core.provenance import file_sha256

    config["transformer"]["pruning"].update(artifact=str(envelope_path), artifact_sha256=file_sha256(envelope_path))
    metadata["config"] = json.dumps(config)
    save_file(expected, str(reference), metadata=metadata)
    export_learned_ffn.stream_merge(
        source,
        state,
        output,
        masks=masks,
        merge_first=merge_first,
        envelope={"family": "fixture"},
        envelope_path=envelope_path,
    )
    assert output.read_bytes() == reference.read_bytes()
    assert all(torch.equal(load_file(str(output))[k], v) for k, v in expected.items())


def test_changed_source_and_invalid_transform_never_publish(tmp_path):
    source, output = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    save_file({"weight": torch.ones(4)}, str(source))
    shapes, metadata = safetensors_stream.inspect(source)

    def mutate(value):
        # A fast touch can share the original filesystem timestamp. Advance
        # it explicitly so this fixture always injects a real metadata change.
        before = source.stat()
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        return value

    with pytest.raises(ValueError, match="source checkpoint changed"):
        safetensors_stream.write(source, output, shapes, metadata, transforms={"weight": mutate})
    assert not output.exists()
    with pytest.raises(ValueError, match="shape/dtype"):
        safetensors_stream.write(source, output, shapes, metadata, transforms={"weight": lambda value: value[:1]})
    assert not output.exists()
