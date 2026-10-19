"""Independent native arithmetic, prefix, resume, distribution and ranking controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ltx_core.model.transformer.attention import Attention
from scripts.prune.data.native_gate_study import freeze_context
from scripts.prune.learn.checkpoints import tensor_hash
from scripts.prune.score import native_gate_analysis as analysis
from scripts.prune.score.native_gate_accumulator import Accumulator, atomic_json, distribution_distance, save_arrays
from scripts.prune.score.native_gate_analysis import agreement, build_report
from scripts.prune.score.native_gate_observer import NativeGateObserver, summarize
from scripts.prune.score.native_head_gates import publish_shard_receipt


def sample(value: float = 1.0) -> dict[str, np.ndarray]:
    gates = np.full((36, 2), value, dtype=np.float32)
    gates[:2] = 2
    return {k: v[None] for k, v in summarize(np.zeros_like(gates), gates, 18, 2).items()}


def receipt(index: int, sigma: int = 0) -> dict:
    return {"draw_index": index, "sigma_index": sigma, "tokens_per_frame": 2, "frames": 18}


def test_reductions_against_independent_histogram() -> None:
    gates = np.linspace(0, 2, 72).reshape(36, 2)
    z = np.arange(72).reshape(36, 2) / 10
    result = summarize(z, gates, 18, 2)
    for head in range(2):
        expected, _ = np.histogram(gates[2:, head], bins=np.linspace(0, 2, 2049))
        np.testing.assert_array_equal(result["histogram"][head], expected)
    np.testing.assert_allclose(result["gate_sum"], gates[2:].sum(0))
    np.testing.assert_allclose(result["gate_square_sum"], (gates[2:] ** 2).sum(0))
    np.testing.assert_allclose(result["clean_gate_sum"], gates[:2].sum(0))
    np.testing.assert_allclose(result["logit_sum"], z[2:].sum(0))
    np.testing.assert_allclose(result["frame_gate_sum"], gates.reshape(18, 2, 2).sum(1))
    assert result["threshold_counts"][0].tolist() == (gates[2:] < 0.1).sum(0).tolist()
    with pytest.raises(ValueError, match="geometry"):
        summarize(z, gates, 17, 2)


class TinyNative(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.attn1 = Attention(query_dim=16, heads=2, dim_head=8, apply_gated_attention=True)
        block.attn2 = nn.Identity()
        self.transformer_blocks = nn.ModuleList([block])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.transformer_blocks[0].attn1(value)


def test_native_bf16_transparency(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(5)
    model = TinyNative().to(dtype=torch.bfloat16).eval()
    value = torch.randn(1, 36, 16, dtype=torch.bfloat16)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        reference = model(value)
        executed = []
        original = torch.sigmoid

        def capture(logits: torch.Tensor) -> torch.Tensor:
            result = original(logits)
            executed.append(result.clone())
            return result

        monkeypatch.setattr(torch, "sigmoid", capture)
        with NativeGateObserver(model, blocks=1, heads=2) as observer:
            observer.begin(18, 2)
            observed = model(value)
            reduced = observer.finish()
            assert len(executed) == 2
            assert executed[0].dtype == torch.bfloat16
            assert torch.equal(2 * executed[0], 2 * executed[1])
            assert torch.equal(reference, observed)
            assert reduced["histogram"].sum() == 34 * 2
            observer.begin(18, 2)
            with pytest.raises(ValueError, match="coverage"):
                observer.finish()
        assert not model.transformer_blocks[0].attn1.to_gate_logits._forward_hooks
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[key])


def test_resume_counts_and_duplicate_refusal(tmp_path: Path) -> None:
    acc = Accumulator(2, "frozen")
    acc.add(0, 0, sample(0.25), receipt(0))
    path = tmp_path / "state.npz"
    acc.save(path)
    restored = Accumulator.load(path, "frozen")
    acc.add(1, 1, sample(1.5), receipt(1, 1))
    restored.add(1, 1, sample(1.5), receipt(1, 1))
    for key in acc.arrays:
        np.testing.assert_array_equal(acc.arrays[key], restored.arrays[key])
    assert restored.counts.tolist() == [1, 1]
    restored.require_prefix(0, 2)
    with pytest.raises(ValueError, match="duplicate"):
        restored.add(1, 1, sample(), receipt(1, 1))
    with pytest.raises(ValueError, match="another frozen study"):
        Accumulator.load(path, "changed")
    with pytest.raises(ValueError, match="prefix"):
        restored.require_prefix(0, 3)


def test_shuffled_merge_and_count_conservation() -> None:
    acc = Accumulator(2, "frozen")
    second = Accumulator(2, "frozen")
    second.add(1, 1, sample(1.5), receipt(1, 1))
    first = Accumulator(2, "frozen")
    first.add(0, 0, sample(0.5), receipt(0))
    acc.merge(second)
    acc.merge(first)
    acc.require_prefix(0, 2)
    np.testing.assert_allclose(acc.arrays["gate_sum"][:, 0, 0] / acc.tokens, [0.5, 1.5])
    with pytest.raises(ValueError, match="duplicate"):
        acc.merge(first)
    invalid = sample()
    invalid["histogram"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="conservation"):
        Accumulator(2, "frozen").add(0, 0, invalid, receipt(0))


def test_distances_and_ranking_ties() -> None:
    low = np.zeros(2048)
    high = low.copy()
    low[0], high[-1] = 1, 1
    assert distribution_distance(low, low) == 0
    assert distribution_distance(low, high) == pytest.approx(2047 / 1024)
    kept = np.array([[False, True, True, False], [True, False, True, True]])
    perfect = agreement(kept.astype(float), kept)
    assert perfect["roc_auc"] == perfect["average_precision"] == 1
    assert perfect["intersection"] == 3
    tied = agreement(np.ones_like(kept, dtype=float), kept)
    assert tied["roc_auc"] == 0.5
    assert tied["average_precision"] == pytest.approx(3 / 8)
    assert tied["selected"] == [0, 1, 2]
    assert agreement((~kept).astype(float), kept)["roc_auc"] == 0


def test_original_head_permutation_preserves_agreement_and_selected_ids() -> None:
    scores = np.array([[0.2, 0.8, 0.1, 1.9], [0.7, 1.1, 0.3, 1.5]])
    kept = np.array([[False, True, False, True], [True, True, False, True]])
    permutation = [2, 0, 3, 1]
    original = agreement(scores, kept)
    permuted = agreement(scores[:, permutation], kept[:, permutation])
    for key in ("roc_auc", "average_precision", "intersection", "recall", "jaccard", "within_block_intersection"):
        assert original[key] == permuted[key]
    original_ids = np.arange(scores.size).reshape(scores.shape)[:, permutation].ravel()
    for key in ("selected", "within_block_selected"):
        np.testing.assert_array_equal(np.sort(original_ids[permuted[key]]), original[key])


def test_model_free_report_sketch(tmp_path: Path) -> None:
    for directory in ("configs", "comparison", "provenance"):
        (tmp_path / directory).mkdir()
    (tmp_path / "provenance/plan.md").write_text("saved plan")
    (tmp_path / "configs/frozen_study.json").write_text(json.dumps({"steps": 1300, "sigmas": [0.9] * 10}))
    sections = [{"design": "Planned figure: x = steps; y = mean gate."} for _ in range(5)]
    (tmp_path / "configs/narrative.json").write_text(json.dumps({"sections": sections}))
    (tmp_path / "comparison/masks.json").write_text("{}")
    build_report(tmp_path)
    text = (tmp_path / "REPORT.md").read_text()
    assert "Not available yet" in text
    assert "0 / 1300 steps" in text
    assert not list(tmp_path.rglob("*.png"))
    assert json.loads((tmp_path / "monitoring/report_validation.json").read_text())["passed"]
    (tmp_path / "comparison/masks.json").unlink()
    with pytest.raises(ValueError, match="missing saved evidence"):
        build_report(tmp_path)


def test_full_milestone_analysis_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    for directory in ("configs", "comparison"):
        (tmp_path / directory).mkdir()
    sigmas = list(np.linspace(0.94, 0.48, 4))
    atomic_json(tmp_path / "configs/frozen_study.json", {"sigmas": sigmas})
    kept = np.ones((48, 32), dtype=int)
    kept.ravel()[:292] = 0
    atomic_json(
        tmp_path / "comparison/masks.json", {"references": [{"name": "clipped_reset_u600", "kept": kept.tolist()}]}
    )
    heads = [{"block": 0, "head": h, "drop": h / 32, "reason": "kept"} for h in range(32)]
    atomic_json(tmp_path / "comparison/events.json", [{"update": u, "heads": heads} for u in range(50, 601, 50)])
    seen = []

    def capture_figure(fig: object, directory: Path, name: str, data: dict) -> None:
        assert directory.name == "figures"
        assert isinstance(data, dict)
        seen.append(name)
        analysis.plt.close(fig)

    monkeypatch.setattr(analysis, "save_figure", capture_figure)
    mean = np.broadcast_to(np.linspace(0.1, 1.6, 1536).reshape(1, 48, 32), (4, 48, 32)).copy()
    counts = np.full(4, 40, dtype=np.int64)
    tokens = counts * 34
    hist = np.zeros((4, 48, 32, 2048), dtype=np.int64)
    locations = np.indices((4, 48, 32))
    hist[(*locations, (mean * 1024).astype(int))] = tokens[:, None, None]
    arrays = {
        "histogram": hist,
        "samples": counts,
        "tokens": tokens,
        "clean_tokens": counts * 2,
        "gate_sum": mean * tokens[:, None, None],
        "gate_square_sum": mean**2 * tokens[:, None, None],
        "threshold_counts": np.zeros((4, 48, 4, 32), dtype=np.int64),
        "frame_gate_sum": np.broadcast_to(mean[:, :, None, :], (4, 48, 18, 32)) * (counts * 2)[:, None, None, None],
        "frame_high_count": np.zeros((4, 48, 18, 32), dtype=np.int64),
    }
    for step in (100, 200):
        directory = tmp_path / f"milestones/step_{step:04d}"
        receipts = [
            {"draw_index": i, "actor": str(i % 93), "view": str(i % 256), "sigma_index": i % 10}
            for i in range(step * 4)
            if [0, 9, 1, 8, 2, 7, 3, 6, 4, 5][i % 10] in [1, 4, 7, 9]
        ]
        cumulative = {key: value.copy() * (step // 100) for key, value in arrays.items()}
        save_arrays(directory / "cumulative.npz", cumulative, {"receipts": receipts})
        save_arrays(directory / "window.npz", arrays, {"receipts": receipts[-160:]})
        atomic_json(directory / "complete.json", {"study": "fixture", "draws": len(receipts)})
        result = analysis.analyze(tmp_path, step)
        assert result["agreement"]["clipped_reset_u600"]["aggregate"]["roc_auc"] == 1
        if step == 200:
            assert result["convergence"]["distribution_change"]["maximum"] == [0] * 4
            assert result["convergence"]["stable_fraction"] == [0] * 4
    assert "E2a_mean_stability" in seen
    assert "E4d_event_drops" in seen


def test_context_byte_hash_is_not_archive_hash(tmp_path: Path) -> None:
    original = tmp_path / "context.pt"
    context = torch.arange(16, dtype=torch.bfloat16).reshape(1, 4, 4)
    torch.save(context, original)
    raw = hashlib.sha256(context.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    record = {"path": str(original), "bytes_sha256": raw, "tensor_sha256": tensor_hash(context)}
    frozen = freeze_context(record, tmp_path / "new_run")
    assert frozen["file_sha256"] != raw
    assert frozen["bytes_sha256"] == raw
    assert frozen["path"] != str(original)
    torch.save(context + 1, original)
    with pytest.raises(ValueError, match="context tensor changed"):
        freeze_context(record, tmp_path / "bad_run")


def test_resume_finishes_missing_shard_receipt_without_replaying_draws(tmp_path: Path) -> None:
    path = tmp_path / "step_0100.npz"
    acc = Accumulator(2, "frozen")
    acc.add(1, 0, sample(), receipt(1))
    acc.save(path)
    original = path.read_bytes()
    publish_shard_receipt(path, "frozen", [1], 4, 100)
    pin_path = path.with_suffix(".json")
    pin = pin_path.read_bytes()
    publish_shard_receipt(path, "frozen", [1], 4, 100)
    assert pin_path.read_bytes() == pin
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="draw identities differ"):
        publish_shard_receipt(path, "frozen", [1, 2], 4, 100)
    pin_path.write_text("{}")
    with pytest.raises(ValueError, match="completed artifact differs"):
        publish_shard_receipt(path, "frozen", [1], 4, 100)
