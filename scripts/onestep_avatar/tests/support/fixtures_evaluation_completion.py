"""Shared saved-input builders and fixtures; never a collected test module."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import scripts.prune.core.session as sessions
import scripts.prune.data.prompt_cache as prompts
from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar.execution import completion as queue_completion
from scripts.onestep_avatar.execution import jobs as queue_jobs
from scripts.onestep_avatar.hashing import sha256
from scripts.prune.core import preflight


def controlled_scientific_receipt(job: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Isolate saved science from the separately tested native supervision proof.

    These CPU producers do not launch a queue child. Supply an explicit fake
    execution observation without removing the prepared job's effective budget.
    Queue lifecycle tests retain the real ownership and proof-negative checks.
    """
    attempt = {"command": ["controlled CPU scientific fixture"], "gpus": [0], "attempts": []}
    monkeypatch.setattr(queue_completion, "verify_attempt_evidence", lambda _job, _attempt: [])
    receipt = queue_completion.completion_receipt(job, attempt=attempt)
    receipt["execution"]["binding"]["completed_at"] = 1.0
    return receipt


@pytest.fixture
def completed(checked_settings: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple:
    del tmp_path
    settings, membership = checked_settings
    specification = evaluate.backbone.resolve("2.5", "dev")
    Path(specification.paths.transformer()).write_bytes(b"controlled base weights")
    monkeypatch.setattr(
        evaluate.backbone, "identity", lambda path, *_a, **_k: {"base_transformer_sha256": sha256(Path(path))}
    )

    calls = []

    class Session:
        def __init__(self, *_args: object) -> None:
            calls.append("session")
            self.device = torch.device("cpu")

        def transformer(self, **_kwargs: object) -> nullcontext:
            calls.append("transformer")
            model = LTXModel(
                model_type=LTXModelType.VideoOnly,
                num_attention_heads=2,
                attention_head_dim=4,
                in_channels=2,
                out_channels=2,
                num_layers=2,
                cross_attention_dim=8,
                caption_projection=torch.nn.Linear(4, 8),
            )
            generator = torch.Generator().manual_seed(0)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
            return nullcontext(X0Model(model.to(dtype=torch.bfloat16).eval()))

    monkeypatch.setattr(sessions, "Session", Session)
    # These synthetic adapter matrices pin completion identity, not a real LTX
    # target inventory. Actual unmerged effects/loading are tested in test_adapters.
    monkeypatch.setattr(
        evaluate.adapter_loader, "inference_transformer", lambda session, *_a, **_kw: session.transformer()
    )
    monkeypatch.setattr(preflight, "check", lambda *_a, **_k: calls.append("preflight"))
    monkeypatch.setattr(prompts, "get_or_build", lambda *_a, **_k: torch.ones(1, 3, 4, dtype=torch.bfloat16))
    # Keep only the evaluator's device selection on CPU; shared Torch stays intact.
    monkeypatch.setattr(
        evaluate, "torch", SimpleNamespace(**{**vars(torch), "device": lambda *_a: torch.device("cpu")})
    )
    arguments = [
        "--mode",
        "bidirectional",
        "--subset",
        str(settings.subset),
        "--output",
        str(settings.output),
        "--variant",
        "dev",
        "--guide-mode",
        "d0",
        "--schedule",
        "0.725",
        "0",
        "--seed",
        "42",
    ]

    def execute(
        extra: tuple[str, ...] = (),
        *,
        mode: str | None = None,
        guide_mode: str | None = None,
        execution_owner: ModuleType | None = None,
    ) -> tuple:
        command = arguments + list(extra)
        if mode is not None:
            command[command.index("--mode") + 1] = mode
        if guide_mode is not None:
            command[command.index("--guide-mode") + 1] = guide_mode
        if execution_owner is None:
            args = evaluate.parse_args(command)
            evaluate.execute_evaluation(args)
        else:
            scientific = list(command)
            index = scientific.index("--output")
            del scientific[index : index + 2]
            spec = settings.output.parent / "causality_spec.json"
            spec.write_text(json.dumps({"schema_version": 1, "protocol": "future_noise", "arguments": scientific}))
            execution_owner.main(["--spec", str(spec), "--output", str(settings.output)])
        paths = sorted(settings.output.glob("case_*/variant_*/result.json"))
        if not paths:
            paths = sorted(settings.output.glob("case_*/variant_*/*/result.json"))
        assert paths
        job = {
            "id": "checked",
            "kind": "evaluate",
            "arguments": command,
            "output": str(settings.output),
            "completion": {"records": [str(path) for path in paths]},
        }
        if execution_owner is not None:
            job.update(
                kind="experiment",
                experiment="causality",
                spec=str(spec),
                spec_sha256=sha256(spec),
                arguments=["--spec", str(spec), "--output", str(settings.output)],
                completion={"manifest": str(settings.output / "manifest.json")},
            )
        jobs_path = settings.output.parent / "queue_jobs.json"
        jobs_path.write_text(json.dumps({"schema_version": 1, "jobs": [job]}))
        job = queue_jobs.prepare_jobs(jobs_path)[0]
        before = len(calls)
        assert queue_completion.verify_completion(job)
        assert len(calls) == before, "completion opened a native model session"
        return job, paths

    return execute, arguments, settings, membership, calls


build_completed = completed.__wrapped__
