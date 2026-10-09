"""Original producer identities and saved native-slice artifacts remain bound."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from safetensors.torch import save_file

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import hashing
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import training_slice_check as slice_check
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import adapters, common
from scripts.onestep_avatar.tests.test_causal_core import _model
from scripts.onestep_avatar.tests.test_training_runtime import CPUAccelerator
from scripts.onestep_avatar.training import checkpoints, config, resources, runtime

Context = tuple[SimpleNamespace, dict, torch.Tensor]
OriginalArtifacts = tuple[dict, config.RunSettings, dict, SimpleNamespace, dict, dict]
SavedPair = tuple[Path, Path, Path]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def original_context(tmp_path: Path) -> Context:
    root = tmp_path / "original"
    path = root / "update_states/text.pt"
    path.parent.mkdir(parents=True)
    tensor = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    torch.save(tensor, path)
    record = {"path": str(path.resolve()), "sha256": sha256(path),
              "tensor_sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
              "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    return SimpleNamespace(output=root), {"update_text": record}, tensor


def test_original_context_reads_exact_saved_tensor(original_context: Context) -> None:
    settings, saved, expected = original_context
    assert torch.equal(slice_check._original_context(settings, saved), expected)


def test_complete_parameter_inventory_selects_installed_accelerator_fsdp_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Prepared(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.adapter = torch.nn.Parameter(torch.ones(1))

        def clip_grad_norm_(self, maximum: float, norm_type: float) -> torch.Tensor:
            calls.append((maximum, norm_type))
            return torch.tensor(3.0)

    model = Prepared()
    observer = SimpleNamespace(distributed_type=DistributedType.FSDP, unscale_gradients=lambda: None,
                               _models=[model], is_fsdp2=False)

    def local_fallback(*_args, **_kwargs) -> torch.Tensor:
        pytest.fail("installed Accelerator bypassed the native model clipping path")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", local_fallback)
    norm = Accelerator.clip_grad_norm_(observer, model.parameters(), 1.0)
    assert norm == 3.0
    assert calls == [(1.0, 2)]


@pytest.mark.parametrize("change", ["file", "values", "shape", "dtype", "path", "nonfinite"])
def test_original_context_refuses_changed_producer_facts(original_context: Context, change: str) -> None:
    settings, saved, tensor = original_context
    record = saved["update_text"]
    path = settings.output / "update_states/text.pt"
    if change == "path":
        record["path"] = str(path.parent / "other.pt")
    else:
        changed = tensor + 1 if change in {"file", "values"} else tensor
        if change == "shape":
            changed = tensor.reshape(1, 4, 3)
        if change == "dtype":
            changed = tensor.view(torch.uint8)
        if change == "nonfinite":
            changed = tensor.clone()
            changed[0, 0, 0] = float("inf")
        torch.save(changed, path)
        # Keep the file-byte check valid to isolate the tensor/shape/dtype gates.
        if change != "file":
            record["sha256"] = sha256(path)
    with pytest.raises(ValueError, match="original saved text"):
        slice_check._original_context(settings, saved)


@pytest.fixture
def original_artifacts(tmp_path: Path) -> OriginalArtifacts:
    root = tmp_path / "original"
    settings = config.parse_settings([
        "--mode", "bidirectional", "--subset", str(tmp_path / "membership.json"),
        "--output", str(root), "--guide-mode", "d0", "--objective", "white",
        "--variant", "dev", "--span-latent-frames", "7", "--lora-rank", "2",
        "--chains-per-rank", "2", "--steps", "1", "--save-initial", "--save-update-state",
    ])
    settings.world_size = 4
    settings.base_identity = {"model_key": "2.5", "base_variant": "dev", "base_transformer_file": "base.safetensors",
                              "base_transformer_fingerprint": "b" * 16, "base_transformer_sha256": "b" * 64}
    membership = {"schema_version": 2, "kind": subset.KIND, "objective": "white", "splits": {"train": ["actor"]},
                  "sources": [{"relative_dir": "actor/view", "actor": "actor", "split": "train",
                  "shape": [8, 7, 2, 2], "n_latent_frames": 7}]}
    membership["sha256"] = subset.membership_hash(membership)
    plan = config.select_frame_plan(settings, membership, SpatioTemporalScaleFactors(8, 32, 32))
    corpus = tmp_path / "corpus"
    capture = corpus / "actor/view" / dataset.capture_bundle_name("white")
    capture.parent.mkdir(parents=True)
    torch.save({"master": torch.ones(8, 7, 2, 2)}, capture)
    store = SimpleNamespace(root=corpus, membership=membership)
    accelerator_config = tmp_path / "fsdp.yaml"
    accelerator_config.write_text("mixed_precision: bf16\n")
    original = {"sha256": "d" * 64, "processes": 4, "accelerate_config": str(accelerator_config)}
    profile = software.capture("training", "bidirectional")
    budget_path = tmp_path / "budget.json"
    _write_json(budget_path, {"wall_seconds_per_phase": 1800, "memory_limit_allocated_bytes": 48000000000,
                             "tolerance": slice_check.training_update_check.TOLERANCE})
    budget = resources.read_budget(budget_path)
    saved = {"base_identity": copy.deepcopy(settings.base_identity), "membership_sha256": membership["sha256"],
             "frame_plan_sha256": plan["sha256"], "software": profile,
             "producer_source_sha256": profile["sources"]["scripts/onestep_avatar/training/engine.py"],
             "queue_launch": {"original": "already validated by prepare"}, "runtime": {"original": "four ranks"}}
    _write_json(root / "config.json", saved)
    _write_json(root / "frame_plan.json", plan)
    model = adapters.attach(_model().bfloat16(), rank=2, alpha=settings.lora_alpha, target="attn", init_seed=11)
    for step in (0, 1):
        if step == 1:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if ".lora_B." in name:
                        parameter.fill_(0.0001)
        metadata = {checkpoints.CONTRACT_KEY: json.dumps(checkpoints.make_contract(settings, membership, plan, step))}
        path = checkpoints.save_lora(model, CPUAccelerator(), root / "checkpoints", step, metadata,
                                     verify_noop=step == 0)
        marker = {"schema_version": 2, "step": step, "path": str(path.resolve()), "sha256": sha256(path),
                  "state": "complete", "queue_job_sha256": original["sha256"],
                  **{key: saved[key] for key in ("producer_source_sha256", "software", "queue_launch", "runtime")},
                  "resource_budget": budget, "resource_evidence": {}, "consumer_trace_evidence": {},
                  "training_record": {"config_sha256": sha256(root / "config.json"),
                                      "frame_plan_sha256": sha256(root / "frame_plan.json")}}
        _write_json(path.with_suffix(".complete.json"), marker)
    visits = slice_check.training_update_check.first_update_visits(settings, plan["samples"], 4)
    for rank in range(4):
        selected = [visit for visit in visits if visit["rank"] == rank]
        row = {"rank": rank, "step": 1, "sigma0": selected[0]["sigma"],
               "samples": [{"source": visit["sample"]["source"], "ranges": visit["sample"]["ranges"],
                            "noise_seed": visit["noise_seed"]} for visit in selected],
               "call_counts": {"prime": 0, "denoise": 2, "backward": 2, "refresh": 0}}
        (root / f"metrics_rank{rank}.jsonl").write_text(json.dumps(row) + "\n")
    return original, settings, saved, store, plan, budget


def test_original_artifact_binding_keeps_all_checkpoint_and_visit_evidence(
    original_artifacts: OriginalArtifacts,
) -> None:
    original, settings, *_ = original_artifacts
    paths = slice_check._original_artifacts(*original_artifacts)
    assert Path(original["accelerate_config"]) in paths
    assert all(settings.output / f"metrics_rank{rank}.jsonl" in paths for rank in range(4))
    assert all(settings.output / "checkpoints" / f"lora_weights_step_{step:05d}.complete.json" in paths
               for step in (0, 1))


@pytest.mark.parametrize("change", ["base", "plan", "producer", "marker", "visit"])
def test_original_artifacts_refuse_changed_original_evidence(
    original_artifacts: OriginalArtifacts, change: str,
) -> None:
    original, settings, saved, store, plan, budget = original_artifacts
    if change == "base":
        settings.base_identity["base_transformer_sha256"] = "c" * 64
    if change == "plan":
        plan["samples"][0]["ranges"] = [[0, 6]]
    if change == "producer":
        saved["producer_source_sha256"] = "c" * 64
    if change == "marker":
        path = settings.output / "checkpoints/lora_weights_step_00001.complete.json"
        marker = json.loads(path.read_text())
        marker["training_record"]["config_sha256"] = "c" * 64
        _write_json(path, marker)
    if change == "visit":
        path = settings.output / "metrics_rank3.jsonl"
        row = json.loads(path.read_text())
        row["samples"][0]["noise_seed"] += 1
        path.write_text(json.dumps(row) + "\n")
    expected = {"base": "original frame plan", "plan": "original frame plan", "producer": "training producer",
                "marker": "producer binding", "visit": "rank source/ranges/noise/sigma"}
    with pytest.raises(ValueError, match=expected[change]):
        slice_check._original_artifacts(original, settings, saved, store, plan, budget)


@pytest.fixture
def saved_pair(tmp_path: Path) -> SavedPair:
    budget_path = tmp_path / "budget.json"
    _write_json(budget_path, {"wall_seconds_per_phase": 1800, "memory_limit_allocated_bytes": 48000000000,
                             "tolerance": slice_check.training_update_check.TOLERANCE})
    budget = resources.read_budget(budget_path)
    generator = torch.Generator().manual_seed(11)
    names = ["diffusion_model.q.lora_A.weight", "diffusion_model.q.lora_B.weight"]
    matrices = [torch.nn.Parameter(torch.randn(2, 4, generator=generator)), torch.nn.Parameter(torch.zeros(4, 2))]
    initial = {name: hashing.tensor_sha256(value) for name, value in zip(names, matrices, strict=True)}
    zero = {name: value.detach().bfloat16().clone() for name, value in zip(names, matrices, strict=True)}
    optimizer = torch.optim.AdamW(matrices, lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    x = torch.randn(3, 4, generator=generator)
    loss = ((x @ matrices[0].T @ matrices[1].T) - torch.ones(3, 4)).square().mean()
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(matrices, 1.0))
    optimizer.step()
    states = {name: {"step": 1, "exp_avg": optimizer.state[p]["exp_avg"],
                     "exp_avg_sq": optimizer.state[p]["exp_avg_sq"]} for name, p in zip(names, matrices, strict=True)}
    final = {name: value.detach().bfloat16() for name, value in zip(names, matrices, strict=True)}
    model = adapters.attach(_model().bfloat16(), rank=2, alpha=2, target="attn", init_seed=11)
    applied = runtime.gather(runtime.capture(model, SimpleNamespace(process_index=0, num_processes=1,
        mixed_precision="bf16", distributed_type=DistributedType.NO), common.SIGMA_PRECISION),
        SimpleNamespace(num_processes=1, mixed_precision="bf16"))
    protocol = {"execution": "serial", "original_job": {"id": "fixed original"}, "original_rank_key": 0,
                "selected_slots": 1, "visits": [{"rank": 0, "slot": 0}], "world_size": 1,
                "averaging_denominator": 1, "budget": budget, "tolerance": slice_check.training_update_check.TOLERANCE,
                "input_sha256": {"original": "a" * 64}, "kernel_control": {"deterministic_algorithms": True},
                "software": software.capture("training", "bidirectional", extra_sources=slice_check.EXTRA_SOURCES)}
    measurements = [{"schema_version": 1, "phase": phase, "rank": 0, "device": "cuda:0", "elapsed_s": 1,
                     "peak_allocated_bytes": 1000, "peak_reserved_bytes": 2000,
                     "budget_sha256": budget["sha256"], "state": "passed", "error": None}
                    for phase in ("load", "update", "export")]
    paths = []
    for name in ("left", "right"):
        root = tmp_path / name
        _write_json(root / "protocol.json", protocol)
        _write_json(root / "initial_fp32.json", initial)
        (root / "resources_rank0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in measurements))
        torch.save(states, root / "adam.pt")
        checkpoint = root / "checkpoints/lora_weights_step_00001.safetensors"
        checkpoint.parent.mkdir()
        save_file(zero, str(checkpoint.with_name("lora_weights_step_00000.safetensors")))
        save_file(final, str(checkpoint))
        trace_path = root / "consumer_trace.json"
        _write_json(trace_path, {"events": [], "complete": True,
                                "binding": {"launch_sha256": sha256(root / "protocol.json")}})
        record = {"state": "complete", "execution": "serial", "protocol": protocol, "initial_fp32": initial,
                  "resources": measurements, "runtime": applied, "checkpoint": str(checkpoint.resolve()),
                  "checkpoint_sha256": sha256(checkpoint), "grad_norm": norm, "loss": float(loss.detach()),
                  "consumer_trace_evidence": {"path": str(trace_path.resolve()), "sha256": sha256(trace_path),
                                              "events": 0, "complete": True},
                  "output_files": {str(path.relative_to(root)): sha256(path) for path in root.rglob("*")
                                   if path.is_file()}}
        _write_json(root / "result.json", record)
        paths.append(root)
    return *paths, tmp_path / "comparison.json"


def test_saved_pair_reaches_real_numerical_comparison(saved_pair: SavedPair) -> None:
    result = slice_check.compare(*saved_pair)
    assert result["passed"]
    assert all(row["rms"] == 0 for row in result["clipped_gradients"].values())
    assert json.loads(saved_pair[-1].read_text())["inputs"] == {
        str(path.resolve()): sha256(path / "result.json") for path in saved_pair[:2]}


@pytest.mark.parametrize("change", ["checkpoint_path", "checkpoint_hash", "inventory", "protocol", "initial",
                                    "resources", "trace", "escape", "bytes", "kernel", "software"])
def test_saved_comparison_refuses_unbound_or_changed_artifacts(saved_pair: SavedPair, change: str) -> None:
    left, _right, output = saved_pair
    path = left / "result.json"
    record = json.loads(path.read_text())
    if change == "checkpoint_path":
        record["checkpoint"] = str(saved_pair[1] / "checkpoints/lora_weights_step_00001.safetensors")
    if change == "checkpoint_hash":
        record["checkpoint_sha256"] = "f" * 64
    if change == "inventory":
        record["output_files"].pop("adam.pt")
    if change == "protocol":
        record["protocol"]["selected_slots"] = 2
    if change == "initial":
        record["initial_fp32"][next(iter(record["initial_fp32"]))] = "f" * 64
    if change == "resources":
        record["resources"][0]["peak_allocated_bytes"] = 0
    if change == "trace":
        record["consumer_trace_evidence"]["path"] = str(saved_pair[1] / "consumer_trace.json")
    if change == "escape":
        target = left.parent / "escaped.json"
        _write_json(target, {"external": "bytes"})
        (left / "escaped.json").symlink_to(target)
        record["output_files"]["escaped.json"] = sha256(target)
    if change == "bytes":
        (left / "adam.pt").write_bytes(b"changed actual tensor file")
    if change in {"kernel", "software"}:
        if change == "kernel":
            record["protocol"]["kernel_control"]["deterministic_algorithms"] = False
        else:
            record["protocol"]["software"] = software.capture("training", "bidirectional")
        _write_json(left / "protocol.json", record["protocol"])
        record["output_files"]["protocol.json"] = sha256(left / "protocol.json")
        trace_path = left / "consumer_trace.json"
        trace = json.loads(trace_path.read_text())
        trace["binding"]["launch_sha256"] = record["output_files"]["protocol.json"]
        _write_json(trace_path, trace)
        record["output_files"]["consumer_trace.json"] = sha256(trace_path)
        record["consumer_trace_evidence"]["sha256"] = record["output_files"]["consumer_trace.json"]
    _write_json(path, record)
    with pytest.raises(ValueError, match="saved slice"):
        slice_check.compare(*saved_pair)
    assert not output.exists()
