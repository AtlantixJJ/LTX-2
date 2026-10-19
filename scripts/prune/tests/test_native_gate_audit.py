"""Independent saved-evidence controls for source identity and exact shard merging."""

from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.prune.score import native_gate_audit as audit
from scripts.prune.score.native_gate_accumulator import Accumulator, atomic_json, digest


def values() -> dict[str, np.ndarray]:
    """Two constant heads: native values 0.25 and 1.75 on every local frame."""
    gate = np.array([[0.25, 1.75]], dtype=np.float64)
    logit = np.log(gate / (2 - gate))
    result = {}
    for prefix, tensor, count in (
        ("gate_", gate, 34),
        ("logit_", logit, 34),
        ("clean_gate_", gate, 2),
        ("clean_logit_", logit, 2),
    ):
        result.update(
            {
                prefix + "sum": tensor * count,
                prefix + "square_sum": tensor**2 * count,
                prefix + "minimum": tensor.copy(),
                prefix + "maximum": tensor.copy(),
            }
        )
    hist = np.zeros((1, 2, 2048), dtype=np.int64)
    hist[0, 0, 256] = hist[0, 1, 1792] = 34
    threshold = np.array([[[0, 0], [34, 0], [0, 34], [0, 0]]], dtype=np.int64)
    result.update(
        histogram=hist,
        threshold_counts=threshold,
        clean_threshold_counts=threshold // 17,
        frame_gate_sum=np.broadcast_to(gate[:, None, :] * 2, (1, 18, 2)).copy(),
        frame_high_count=np.broadcast_to(np.array([[[0, 2]]], dtype=np.int64), (1, 18, 2)).copy(),
    )
    return result


@pytest.fixture
def saved_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> tuple[Path, dict]:
    monkeypatch.setattr(audit, "HEAD_SHAPE", (1, 2))
    steps = getattr(request, "param", 100)
    cycle = [0, 9, 1, 8, 2, 7, 3, 6, 4, 5]
    member = {
        "relative_dir": "training_view",
        "actor": "actor",
        "shape": [128, 28, 1, 2],
        "n_latent_frames": 28,
        "capture_latent_sha256": "capture",
    }
    draws = [
        {"microbatch": i, "sigma_index": cycle[i % 10], "view": "training_view", "noise_seed": i + 1000}
        for i in range(steps * 4)
    ]
    study = {
        "steps": steps,
        "source_sigma_indices": [1, 4, 7, 9],
        "sigmas": [0.94, 0.91, 0.77, 0.48],
        "gpus": [4, 5, 6],
        "context": {"tensor_sha256": "context", "bytes_sha256": "raw_context"},
        "source": {
            "root": str(tmp_path / "training"),
            "splits": {"learning": [member]},
            "config": {"sigmas": list(np.linspace(0.95, 0.1, 10))},
        },
    }
    study["selected_draw_indices"] = [i for i, d in enumerate(draws) if d["sigma_index"] in [1, 4, 7, 9]]
    study["skipped_draw_indices"] = [i for i in range(steps * 4) if i not in study["selected_draw_indices"]]
    atomic_json(tmp_path / "configs/draws.json", draws)
    study["files"] = {"configs/draws.json": audit.sha256(tmp_path / "configs/draws.json")}
    study["identity"] = digest(study)
    atomic_json(tmp_path / "configs/frozen_study.json", study)
    cumulative = Accumulator(4, study["identity"])
    for endpoint in range(100, steps + 1, 100):
        shards = [Accumulator(4, study["identity"]) for _ in range(3)]
        for index in (i for i in study["selected_draw_indices"] if (endpoint - 100) * 4 <= i < endpoint * 4):
            draw = draws[index]
            group = study["source_sigma_indices"].index(draw["sigma_index"])
            manifest = tmp_path / f"inputs/{index}/manifest.json"
            atomic_json(
                manifest,
                {
                    "seed": draw["noise_seed"],
                    "sigmas": study["source"]["config"]["sigmas"],
                    "text_context": {"sha256": "raw_context"},
                    "videos": [{"artifacts": {"epsilon_sha256": "epsilon", "capture_sha256": "capture"}}],
                },
            )
            start = random.Random(f"uniform_f18:{draw['noise_seed']}").randrange(11)
            receipt = {
                "draw_index": index,
                "step": index // 4 + 1,
                "logical_rank": index % 4,
                "source_sigma_index": draw["sigma_index"],
                "sigma_index": group,
                "sigma": study["sigmas"][group],
                "view": draw["view"],
                "actor": "actor",
                "noise_seed": draw["noise_seed"],
                "capture_sha256": "capture",
                "context_sha256": "context",
                "epsilon_sha256": "epsilon",
                "frames": 18,
                "tokens_per_frame": 2,
                "input_manifest": str(manifest),
                "input_sha256": audit.sha256(manifest),
                "historical_training_match": False,
                "training_window": {
                    "latent_start": start,
                    "latent_stop": start + 18,
                    "position_policy": "window_local_first_keyframe",
                    "noise_seed": draw["noise_seed"],
                    "source_latent_frames": 28,
                    "tokens_per_latent_frame": 2,
                },
            }
            atomic_json(tmp_path / f"draws/{index:05d}.json", receipt)
            shards[index % 3].add(index, group, values(), receipt)
        merged = Accumulator(4, study["identity"])
        for slot, gpu in enumerate(study["gpus"]):
            path = tmp_path / f"workers/gpu_{gpu}/step_{endpoint:04d}.npz"
            shards[slot].save(path)
            indices = [r["draw_index"] for r in shards[slot].receipts]
            atomic_json(
                path.with_suffix(".json"),
                {
                    "study": study["identity"],
                    "sha256": audit.sha256(path),
                    "indices": indices,
                    "gpu": gpu,
                    "step": endpoint,
                },
            )
            merged.merge(shards[slot])
        cumulative.merge(merged)
        directory = tmp_path / f"milestones/step_{endpoint:04d}"
        cumulative.save(directory / "cumulative.npz")
        merged.save(directory / "window.npz")
        atomic_json(
            directory / "complete.json",
            {
                "study": study["identity"],
                "step": endpoint,
                "draws": endpoint * 8 // 5,
                "scheduled_draws": endpoint * 4,
                "skipped_draws": endpoint * 12 // 5,
                "per_sigma": [endpoint * 2 // 5] * 4,
                "receipt_chain_sha256": digest(cumulative.receipts),
                "cumulative_sha256": audit.sha256(directory / "cumulative.npz"),
                "window_sha256": audit.sha256(directory / "window.npz"),
            },
        )
    return tmp_path, study


def test_complete_prefix_audit(saved_prefix: tuple[Path, dict]) -> None:
    root, _ = saved_prefix
    result = audit.audit_milestone(root, 100)
    assert result["passed"]
    assert result["exact_worker_merge"]
    assert result["per_sigma"] == [40] * 4
    assert result["fresh_input_receipts"] == 160
    assert len(result["window_worker_provenance"]) == 160


@pytest.mark.parametrize("field", ["sigma_index", "latent_start", "epsilon_sha256"])
def test_receipt_changes_rejected(saved_prefix: tuple[Path, dict], field: str) -> None:
    root, study = saved_prefix
    draw = json.loads((root / "configs/draws.json").read_text())[1]
    receipt = audit.read_json(root / "draws/00001.json")
    if field == "latent_start":
        receipt["training_window"][field] += 1
    elif field == "sigma_index":
        receipt[field] = 0
    else:
        receipt[field] = "changed"
    members = {m["relative_dir"]: m for m in study["source"]["splits"]["learning"]}
    with pytest.raises(ValueError, match=r"differs|differ"):
        audit.verify_draw(receipt, draw, study, members)


def test_rejects_altered_worker_ownership(saved_prefix: tuple[Path, dict]) -> None:
    root, _ = saved_prefix
    path = root / "workers/gpu_4/step_0100.json"
    row = audit.read_json(path)
    row["indices"] = row["indices"][1:]
    atomic_json(path, row)
    with pytest.raises(ValueError, match="ownership"):
        audit.audit_milestone(root, 100)


def test_histogram_moment_disagreement_and_missing_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "HEAD_SHAPE", (1, 2))
    acc = Accumulator(4, "fixture")
    for j in range(4):
        acc.add(j, j, values(), {"draw_index": j, "sigma_index": j, "frames": 18, "tokens_per_frame": 2})
    original = copy.deepcopy(acc)
    acc.arrays["gate_sum"] += 1
    with pytest.raises(ValueError, match="histogram and exact"):
        audit.verify_statistics(acc)
    original.arrays.pop("logit_sum")
    with pytest.raises(ValueError, match="inventory"):
        audit.verify_statistics(original)


def test_reader_imports_no_model_stack() -> None:
    code = (
        "import sys; from scripts.prune.score import native_gate_audit; "
        "assert not any(k == 'torch' or k.startswith(('ltx_core', 'ltx_pipelines', 'scripts.onestep_avatar')) "
        "for k in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("saved_prefix", [200], indirect=True)
def test_exact_predecessor_merge(saved_prefix: tuple[Path, dict]) -> None:
    root, study = saved_prefix
    result = audit.audit_milestone(root, 200)
    assert result["exact_predecessor_merge"]
    assert result["per_sigma"] == [80] * 4
    assert result["observed_draws"] == 320
    path = root / "milestones/step_0100/cumulative.npz"
    prior = Accumulator.load(path, study["identity"])
    prior.arrays["logit_sum"] += 1
    prior.save(path)
    pin_path = path.with_name("complete.json")
    pin = audit.read_json(pin_path)
    pin["cumulative_sha256"] = audit.sha256(path)
    atomic_json(pin_path, pin)
    with pytest.raises(ValueError, match="rebuilt statistic differs: logit_sum"):
        audit.audit_milestone(root, 200)
