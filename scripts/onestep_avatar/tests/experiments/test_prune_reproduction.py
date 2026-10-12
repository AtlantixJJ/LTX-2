"""CPU checks for the pruning-reproduction controls; native results remain a separate run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from scripts.onestep_avatar.execution import completion as queue_completion
from scripts.onestep_avatar.execution import jobs as queue_jobs
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import prune_reproduction as study
from scripts.onestep_avatar.experiments import spec_protocol
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.experiments.test_prune_baseline import _BoundCaps, _BoundSession
from scripts.prune.data import prompt_cache
from scripts.prune.evaluate import whole_clip_baseline as producer
from scripts.prune.tests.test_whole_clip_baseline import SCALE, _vae, _views

SIGMAS = ("0.725", "0.909375")
CONTROLS = ("sigma_bf16", "global_sigma_bf16", "dense_mask", "historical_inputs", "bf16_reduction_off")


def write_spec(path: Path, arguments: list[str]) -> Path:
    path.write_text(json.dumps({"schema_version": 1, "protocol": study.SELECTOR, "arguments": arguments}))
    return path


def outer(spec: Path, output: Path) -> list[str]:
    return ["--spec", str(spec), "--output", str(output)]


@pytest.fixture
def saved_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Two saved runs from the real tiny producer, as the reference and the earlier current run."""
    view = _views(tmp_path / "corpus")[0]
    vae = _vae(tmp_path)
    transformer, text = tmp_path / "transformer.safetensors", tmp_path / "text.safetensors"
    for path in (transformer, text):
        save_file({"weight": torch.zeros(1)}, path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fixed tiny baseline text")
    components = SimpleNamespace(
        key="2.5", version=(2, 5), scale_factors=SCALE, caps=_BoundCaps(),
        sigmas=[1.0, 0.909375, 0.725, 0.421875, 0.0],
        paths=SimpleNamespace(transformer=lambda: transformer, text_encoder=lambda: text, video_vae=lambda: vae),
    )
    monkeypatch.setattr(study.baseline.backbone, "resolve", lambda *_args: components)
    session = _BoundSession(vae, transformer)
    cache = tmp_path / "fixed_context.pt"
    torch.save(session.context, cache)
    monkeypatch.setattr(prompt_cache, "cache_path", lambda *_args: cache)
    monkeypatch.setattr(producer, "open_session", lambda *_args, **_kwargs: session)
    common = ["--view", str(view), "--sigmas", *SIGMAS, "--raw-only", "--arms", "d0", "--prompt-file", str(prompt)]
    saved = {}
    for name in study.SAVED:
        saved[name] = tmp_path / name
        assert producer.main([*common, "--output", str(saved[name])]) == 0
    return SimpleNamespace(view=view, common=common, saved=saved, session=session, tmp=tmp_path)


def spec_for(inputs: SimpleNamespace, controls: tuple[str, ...] = CONTROLS, name: str = "spec.json") -> Path:
    return write_spec(inputs.tmp / name, [*inputs.common, "--reference", str(inputs.saved["reference"]),
                                          "--earlier", str(inputs.saved["earlier"]), "--controls", *controls])


@pytest.fixture
def completed(saved_inputs: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    spec = spec_for(saved_inputs)
    output = saved_inputs.tmp / "runs" / "controls"
    assert study.main(outer(spec, output)) == 0
    monkeypatch.setattr(producer, "open_session", lambda *_args, **_kwargs: pytest.fail("completion opened a model"))
    monkeypatch.setattr(producer, "rollout", lambda *_args, **_kwargs: pytest.fail("completion ran a model"))
    results = json.loads((output / "results.json").read_text())
    return SimpleNamespace(**vars(saved_inputs), spec=spec, output=output, results=results)


def test_controls_record_their_actual_model_inputs_and_compare_every_arm(completed: SimpleNamespace) -> None:
    results = completed.results
    assert results["arms"] == [study.FIRST, *CONTROLS, study.LAST]
    assert results["noise"]["fresh_draw_equals_saved"] is True
    records = results["arm_records"]
    assert list(records) == results["arms"]
    for arm in results["arms"]:
        # One untraced call per sigma, then one traced repeat of the first sigma.
        assert len(records[arm]["calls"]) == len(SIGMAS) + 1
        assert records[arm]["profiled_equals_unprofiled"] is True
    assert any("scaled_dot_product" in name for name in records[study.FIRST]["attention_names"])
    current, bf16 = records[study.FIRST]["calls"][0], records["sigma_bf16"]["calls"][0]
    assert current["sigma"] == {"dtype": "torch.float32", "values": [pytest.approx(0.725)]}
    assert current["attention_mask"] is None
    assert bf16["sigma"] == {"dtype": "torch.bfloat16", "values": [0.7265625]}
    assert bf16["timesteps"] == {"dtype": "torch.bfloat16", "values": [0.0, 0.7265625]}
    assert records["sigma_bf16"]["calls"][1]["sigma"]["values"] == [0.91015625]
    # The old producer's pair: BF16 global sigma with float32 per-token timesteps.
    global_only = records["global_sigma_bf16"]["calls"][0]
    assert global_only["sigma"] == {"dtype": "torch.bfloat16", "values": [0.7265625]}
    assert global_only["timesteps"] == current["timesteps"]
    assert current["timesteps"]["dtype"] == "torch.float32"
    masked = records["dense_mask"]["calls"][0]
    mask = masked["attention_mask"]
    assert mask["dtype"] == "torch.float32"
    assert mask["minimum"] == mask["maximum"] == 1.0
    assert mask["shape"][0] == 1
    assert mask["shape"][1] == mask["shape"][2]
    assert records["historical_inputs"]["calls"][0]["sigma"]["dtype"] == "torch.bfloat16"
    assert records["historical_inputs"]["calls"][0]["timesteps"]["dtype"] == "torch.float32"
    assert records["historical_inputs"]["calls"][0]["attention_mask"] == mask
    # Every arm receives the same noisy input tokens; only its named change differs.
    assert masked["latent_sha256"] == bf16["latent_sha256"] == current["latent_sha256"]
    during = records["bf16_reduction_off"]["flags"]["during"]
    assert during["allow_bf16_reduced_precision_reduction"] is False
    assert records["bf16_reduction_off"]["flags"]["before"]["allow_bf16_reduced_precision_reduction"] is True
    summary = results["summary"]
    assert summary["current_repeats_earlier"] is True
    assert summary["current_first_equals_last"] is True
    assert {study.FIRST, study.LAST} <= set(summary["arms_reproducing_reference"])
    # BF16 timesteps reach the main AdaLN and the X0 conversion. The global sigma reaches only
    # the prompt AdaLN, which the tiny model lacks, so neither global-sigma arm changes it.
    assert "sigma_bf16" in summary["arms_changing_current"]
    assert {"global_sigma_bf16", "historical_inputs"}.isdisjoint(summary["arms_changing_current"])
    assert set(results["comparisons"]) == set(results["arms"])
    row = results["comparisons"]["sigma_bf16"]["0.725000"]
    assert set(row) == {"reference", "earlier", study.FIRST}
    assert row["reference"]["exact"] is False
    assert row["reference"]["rms"] > 0
    assert len(row["reference"]["per_latent_frame"]) == torch.load(
        completed.output / results["outputs"]["sigma_bf16"]["0.725000"]["path"], weights_only=True).shape[2]


def test_completion_rechecks_inputs_software_and_comparisons_without_a_model(completed: SimpleNamespace) -> None:
    args = study.read_spec(completed.spec, completed.output)
    study.verify_completion(completed.spec, completed.output)
    record = json.loads((completed.output / "acceptance.json").read_text())
    assert record["settings"] == spec_protocol.settings(args)
    assert record["software"] == software.capture("evaluation", "causal", extra_sources=study.EXTRA_SOURCES)
    saved = {str((completed.saved[name] / "manifest.json").resolve()) for name in study.SAVED}
    assert saved <= set(record["inputs"])
    evidence = set(study.evidence_paths(completed.spec, completed.output))
    assert completed.output / "results.json" in evidence
    assert len([path for path in evidence if path.suffix == ".pt"]) == len(SIGMAS) * (len(args.controls) + 2)


def test_queue_routes_the_selector_and_checks_saved_completion(completed: SimpleNamespace) -> None:
    job = queue_jobs.prepare_job({
        "id": "controls", "kind": "experiment", "experiment": study.SELECTOR, "spec": str(completed.spec),
        "spec_sha256": sha256(completed.spec), "arguments": outer(completed.spec, completed.output),
        "output": str(completed.output), "completion": {"manifest": str(completed.output / "manifest.json")},
    }, completed.spec.parent)
    command, environment = queue_jobs.job_command(job, (3,))
    assert command[2] == "scripts.onestep_avatar.experiments.prune_reproduction"
    assert command[-2:] == ["--gpu-id", "0"]
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert queue_completion.verify_completion(job)


def test_rewritten_comparison_numbers_refuse_even_with_a_matching_inventory(completed: SimpleNamespace) -> None:
    results = completed.results
    results["comparisons"]["sigma_bf16"]["0.725000"]["reference"]["rms"] = 0.0
    (completed.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    manifest = json.loads((completed.output / "manifest.json").read_text())
    manifest["artifacts"] = spec_protocol.inventory(completed.output)
    (completed.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="saved comparisons differ"):
        study.verify_completion(completed.spec, completed.output)


@pytest.mark.parametrize("change", ["output", "reference"])
def test_changed_saved_tensor_refuses_completion(completed: SimpleNamespace, change: str) -> None:
    if change == "output":
        path = completed.output / completed.results["outputs"]["dense_mask"]["0.725000"]["path"]
    else:
        path = next(completed.saved["reference"].glob("*_sigma_0.725000_*_d0.pt"))
    path.write_bytes(path.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="differ"):
        study.verify_completion(completed.spec, completed.output)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--controls", "sigma_bf16", "sigma_bf16"], "distinct controls"),
        (["--controls", "unknown"], "invalid checker"),
        ([], "invalid checker"),
    ],
)
def test_scientific_arguments_require_saved_runs_and_distinct_known_controls(
    saved_inputs: SimpleNamespace, arguments: list[str], message: str
) -> None:
    saved = ["--reference", str(saved_inputs.saved["reference"]), "--earlier", str(saved_inputs.saved["earlier"])]
    spec = write_spec(saved_inputs.tmp / "bad.json", [*saved_inputs.common, *saved, *arguments])
    with pytest.raises(ValueError, match=message):
        study.read_spec(spec, saved_inputs.tmp / "unused")


def test_d1_arm_is_refused(saved_inputs: SimpleNamespace) -> None:
    common = list(saved_inputs.common)
    index = common.index("--arms")
    common[index + 1:index + 2] = ["d0", "d1"]
    spec = write_spec(saved_inputs.tmp / "d1.json", [*common, "--reference", str(saved_inputs.saved["reference"]),
                                                     "--earlier", str(saved_inputs.saved["earlier"]),
                                                     "--controls", "dense_mask"])
    with pytest.raises(ValueError, match="D0 arm only"):
        study.read_spec(spec, saved_inputs.tmp / "unused")


@pytest.mark.parametrize("change", ["hash", "sigma", "seed"])
def test_saved_rollout_requires_its_manifest_hashes_sigmas_and_settings(
    saved_inputs: SimpleNamespace, change: str
) -> None:
    root = saved_inputs.saved["reference"]
    args = study.read_spec(spec_for(saved_inputs), saved_inputs.tmp / "unused")
    if change == "hash":
        path = next(root.glob("*_epsilon.pt"))
        path.write_bytes(path.read_bytes() + b"altered")
        match = "differs from its manifest"
    elif change == "sigma":
        args.sigmas = [0.725, 1.0]
        match = "lacks one epsilon or a requested D0 sigma"
    else:
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["seed"] += 1
        (root / "manifest.json").write_text(json.dumps(manifest))
        match = "settings differ"
    with pytest.raises(ValueError, match=match):
        study.saved_rollout(root, args)


@pytest.fixture
def deterministic_policy() -> None:
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cudnn.deterministic)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    yield
    torch.use_deterministic_algorithms(before[0], warn_only=before[1])
    torch.backends.cudnn.deterministic = before[2]


@pytest.mark.usefixtures("deterministic_policy")
@pytest.mark.parametrize(
    ("arm", "field"),
    [("deterministic_off", "deterministic_algorithms"), ("cudnn_deterministic_off", "cudnn_deterministic"),
     ("bf16_reduction_off", "allow_bf16_reduced_precision_reduction")],
)
def test_each_flag_control_changes_one_flag_and_restores_it(arm: str, field: str) -> None:
    start = study.flags()
    with study.numerical_control(arm) as state:
        changed = {key for key in state["during"] if state["during"][key] != state["before"][key]}
        assert changed == {field}
        assert state["during"][field] is False
    assert study.flags() == start


@pytest.mark.parametrize("arm", ["deterministic_off", "cudnn_deterministic_off"])
def test_determinism_controls_need_the_deterministic_policy(arm: str) -> None:
    if torch.are_deterministic_algorithms_enabled() or torch.backends.cudnn.deterministic:
        pytest.skip("this process already applies a deterministic policy")
    with pytest.raises(ValueError, match="deterministic policy"), study.numerical_control(arm):
        pass


@dataclass(frozen=True)
class _Input:
    latent: torch.Tensor
    sigma: torch.Tensor
    timesteps: torch.Tensor
    attention_mask: torch.Tensor | None = None


def test_dense_mask_control_refuses_an_already_masked_call() -> None:
    masked = _Input(torch.zeros(1, 4, 2), torch.ones(1), torch.ones(1, 4), torch.ones(1, 4, 4))
    call = study.controlled(lambda modality: modality.latent, ("dense_mask",), [])
    with pytest.raises(ValueError, match="unmasked whole-clip call"):
        call(masked)


def test_current_arm_passes_the_model_input_unchanged() -> None:
    value = _Input(torch.zeros(1, 4, 2), torch.tensor([0.725]), torch.tensor([[0.0, 0.725, 0.725, 0.725]]))
    seen = []
    calls: list[dict] = []
    study.controlled(lambda modality: seen.append(modality) or modality.latent, (), calls)(value)
    assert seen == [value]
    assert seen[0] is value
    assert calls[0]["timesteps"]["dtype"] == "torch.float32"


def test_difference_reports_exact_rms_maximum_and_frames() -> None:
    expected = torch.zeros(1, 2, 3, 1, 1, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[0, 0, 1] = 0.5
    result = study.difference(actual, expected)
    assert result["exact"] is False
    assert result["unequal_elements"] == 1
    assert result["maximum"] == 0.5
    assert result["rms"] == pytest.approx((0.25 / 6) ** 0.5)
    assert [frame["unequal_elements"] for frame in result["per_latent_frame"]] == [0, 1, 0]
    assert study.difference(expected, expected.clone())["exact"] is True
    with pytest.raises(ValueError, match="shape or dtype"):
        study.difference(actual.float(), expected)


def test_legacy_reproduction_does_not_inherit_direct_dev_support():
    with pytest.raises(ValueError, match='distilled D0'):
        study.scientific_args([
            '--reference', '/tmp/reference', '--earlier', '/tmp/earlier', '--controls', 'dense_mask',
            '--view', '/tmp/view', '--output', '/tmp/output', '--raw-only', '--arms', 'd0',
            '--variant', 'dev', '--dev-schedule', 'direct', '--sigmas', '0.9489720463752747'])
