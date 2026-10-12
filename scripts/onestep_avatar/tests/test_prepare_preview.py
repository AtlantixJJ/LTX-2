"""Prepared fixed tensors are consumed unchanged by the actual preview readers."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import pytest
import torch

import scripts.prune.core.session as sessions
import scripts.prune.data.prompt_cache as prompts
from scripts.onestep_avatar import evaluate, prepare_inputs, previews
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import common
from scripts.onestep_avatar.tests.support.fixtures_prepare_preview import (
    preview_case as preview_case,  # noqa: PLC0414 -- explicit pytest fixture re-export
)
from scripts.onestep_avatar.training import engine


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
@pytest.mark.parametrize("task", ["d0", "d1"])
def test_prepared_inputs_consumed_by_actual_training_and_tensor_checker(
    preview_case: tuple, mode: str, task: str
) -> None:
    setup, settings, _ = preview_case
    args = setup(mode, task)
    fixed = prepare_inputs.prepare_preview(args)
    settings.mode = mode
    parsed = evaluate.parse_args([*fixed["evaluation_arguments"], "--output", "/unused"])
    settings.mode_settings = parsed.mode_settings
    settings.guide_mode = task
    checked = engine.read_preview_inputs(args.output / "preview.json", settings)
    assert checked == fixed
    assert ("guide" in fixed["input_files"]) == (task == "d1")
    capture, _ = dataset.load_training_master(Path(fixed["input_files"]["capture"]["path"]))
    spec = evaluate.backbone.resolve("2.5", "dev")
    grid = common.ClipGrid.build(
        7, 64, 64, 30, spec, device=torch.device("cpu"), dtype=torch.bfloat16, latent_channels=2
    )
    tokens = grid.patchify(capture.unsqueeze(0).to(dtype=torch.bfloat16))
    tensors = {
        role: torch.load(Path(identity["path"]), weights_only=True)
        for role, identity in fixed["input_files"].items()
        if role not in ("subset", "capture", "guide")
    }
    tensors["capture"] = tokens
    if task == "d1":
        tensors["guide"] = tokens
    previews.verify_preview_tensors(fixed, tensors)
    assert torch.equal(tensors["first_image"], tokens[:, : grid.tokens_per_latent_frame])
    assert tensors["noise"].shape == (1, 28, 2)
    assert tensors["noise"].dtype == torch.bfloat16
    assert not (args.output / "unexecuted").exists()
    assert all(Path(row["path"]).is_absolute() for row in fixed["input_files"].values())


def test_guided_preview_pins_negative_text_and_rejects_missing_pin(preview_case: tuple) -> None:
    setup, settings, calls = preview_case
    args = setup(extra=["--cfg", "2", "--negative-prompt", "negative"])
    fixed = prepare_inputs.prepare_preview(args)
    negative = torch.load(Path(fixed["input_files"]["negative_text"]["path"]), weights_only=True)
    assert torch.equal(negative, torch.full_like(negative, 2))
    assert "negative" in calls
    del fixed["input_files"]["negative_text"]
    (args.output / "preview.json").write_text(json.dumps(fixed))
    with pytest.raises(ValueError, match="pinned negative text"):
        engine.read_preview_inputs(args.output / "preview.json", settings)


@pytest.mark.parametrize(
    "extra", [["--source", "other"], ["--output", "other"], ["--gpu-id", "0"], ["--checkpoint", "other"]]
)
def test_invalid_selection_before_text_or_gpu(preview_case: tuple, extra: list[str]) -> None:
    setup, _, calls = preview_case
    args = setup(extra=extra)
    with pytest.raises(ValueError):  # noqa: PT011 -- mutation controls check refusal type and preserved bytes across different validators
        prepare_inputs.prepare_preview(args)
    assert not calls
    assert not args.output.exists()


def test_wrong_reference_coverage_before_text(preview_case: tuple) -> None:
    setup, _, calls = preview_case
    args = setup(extra=["--span-latent-frames", "5"])
    with pytest.raises(ValueError, match="coverage"):
        prepare_inputs.prepare_preview(args)
    assert not calls
    assert not args.output.exists()


def test_preparation_non_tensor_inputs_remain_pinned(preview_case: tuple) -> None:
    setup, settings, _ = preview_case
    args = setup()
    prepare_inputs.prepare_preview(args)
    settings.subset.write_text(settings.subset.read_text() + "\n")
    with pytest.raises(ValueError, match="file changed"):
        engine.read_preview_inputs(args.output / "preview.json", settings)


def test_guided_execution_loads_fixed_contexts_before_transformer(
    preview_case: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup, _, _calls = preview_case
    args = setup(extra=["--cfg", "2", "--negative-prompt", "negative"])
    fixed = prepare_inputs.prepare_preview(args)
    monkeypatch.setattr(prompts, "get_or_build", lambda *_a, **_k: pytest.fail("fixed text rebuilt"))
    monkeypatch.setattr(
        evaluate, "torch", SimpleNamespace(**{**vars(torch), "device": lambda *_a: torch.device("cpu")})
    )
    monkeypatch.setattr(sessions, "Session", lambda _spec, device, *_a: SimpleNamespace(device=device))
    seen = []

    def transformer(*_a, **_k) -> Never:
        seen.append("transformer")
        raise RuntimeError("checked tensors reached transformer boundary")

    monkeypatch.setattr(evaluate.adapter_loader, "inference_transformer", transformer)
    command = evaluate.parse_args([*fixed["evaluation_arguments"], "--output", str(args.output / "raw")])
    command.preview_fixed = fixed
    with pytest.raises(RuntimeError, match="transformer boundary"):
        evaluate.execute_evaluation(command, preview_tensor_validator=previews.verify_preview_tensors)
    assert seen == ["transformer"]


def test_causal_preparation_omitted_span_remains_null(preview_case: tuple) -> None:
    setup, settings, _calls = preview_case
    args = setup("causal", "d1")
    index = args.evaluation_arguments.index("--span-latent-frames")
    del args.evaluation_arguments[index : index + 2]
    fixed = prepare_inputs.prepare_preview(args)
    assert "--span-latent-frames" not in fixed["evaluation_arguments"]
    selected = evaluate.parse_args([*fixed["evaluation_arguments"], "--output", "/unused"])
    assert selected.mode_settings.span_latent_frames is None
    assert selected.output_latent_frames == 7
    settings.mode = "causal"
    settings.mode_settings = selected.mode_settings
    settings.guide_mode = "d1"
    assert engine.read_preview_inputs(args.output / "preview.json", settings) == fixed


def test_causal_preparation_rejects_trimmed_explicit_span_before_text(preview_case: tuple) -> None:
    setup, settings, calls = preview_case
    args = setup("causal")
    membership = json.loads(settings.subset.read_text())
    for source in membership["sources"]:
        path = Path(membership["corpus_root"]) / source["relative_dir"] / dataset.capture_bundle_name("white")
        bundle = torch.load(path, weights_only=True)
        bundle["master"] = torch.zeros(2, 18, 2, 2)
        torch.save(bundle, path)
        source.update(n_latent_frames=18, shape=[2, 18, 2, 2], capture_latent_sha256=sha256(path))
    membership["sha256"] = subset.membership_hash(membership)
    settings.subset.write_text(json.dumps(membership))
    args.evaluation_arguments[args.evaluation_arguments.index("--span-latent-frames") + 1] = "8"
    with pytest.raises(SystemExit):
        prepare_inputs.prepare_preview(args)
    assert not calls
    assert not args.output.exists()
