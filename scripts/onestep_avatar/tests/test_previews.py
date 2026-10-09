"""Preview scheduling pins bytes and cannot turn an incomplete save into a job."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.onestep_avatar import evaluate, hashing, media, previews
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.test_checkpoint_contract import A, B, _contract
from scripts.onestep_avatar.training import config, engine
from scripts.onestep_avatar.training.checkpoints import CONTRACT_KEY


def _fixed_inputs(tmp_path):
    files = {}
    for role in ("subset", "capture", "guide", "first_image", "text", "noise"):
        path = tmp_path / role
        path.write_text(role)
        files[role] = {"path": str(path), "sha256": sha256(path), "tensor_sha256": sha256(path)}
    record = {
        "schema_version": 2,
        "kind": "onestep_avatar.preview_inputs",
        "mode": "bidirectional",
        "schedule": [0.725, 0],
        "input_files": files,
        "evaluation_arguments": [
            "--mode",
            "bidirectional",
            "--subset",
            files["subset"]["path"],
            "--noise-file",
            files["noise"]["path"],
            "--schedule",
            "0.725",
            "0",
        ],
    }
    path = tmp_path / "preview.json"
    path.write_text(json.dumps(record))
    settings = config.RunSettings(
        "bidirectional", Path(files["subset"]["path"]), tmp_path / "run", config.BidirectionalSettings()
    )
    return path, settings


def _completed(tmp_path):
    checkpoint = tmp_path / "checkpoint.safetensors"
    contract = _contract()
    contract["adapter"]["step"] = 100
    save_file({A: torch.ones(2, 4), B: torch.zeros(4, 2)}, checkpoint, metadata={CONTRACT_KEY: json.dumps(contract)})
    checkpoint.with_suffix(".complete.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "path": str(checkpoint),
                "sha256": sha256(checkpoint),
                "state": "complete",
                "step": 100,
            }
        )
    )
    return checkpoint


def test_fixed_inputs_checked_without_output_or_models(tmp_path):
    path, settings = _fixed_inputs(tmp_path)
    checked = engine.read_preview_inputs(path, settings)
    assert len(checked["sha256"]) == 64
    assert not settings.output.exists()
    Path(checked["input_files"]["guide"]["path"]).write_text("changed guide")
    with pytest.raises(ValueError, match="guide file changed"):
        engine.read_preview_inputs(path, settings)


def test_enqueue_is_idempotent_and_keeps_existing_state(tmp_path):
    path, settings = _fixed_inputs(tmp_path)
    record = engine.read_preview_inputs(path, settings)
    checkpoint = _completed(tmp_path)
    job_path = engine.enqueue_preview(checkpoint, record, settings.output)
    job = json.loads(job_path.read_text())
    assert job["state"] == "pending"
    job["state"] = "failed"
    job["error"] = "decoder failure"
    job_path.write_text(json.dumps(job))
    assert engine.enqueue_preview(checkpoint, record, settings.output) == job_path
    assert json.loads(job_path.read_text())["error"] == "decoder failure"
    assert sha256(checkpoint) == job["checkpoint"]["sha256"]


def test_incomplete_or_changed_checkpoint_never_enqueues(tmp_path):
    path, settings = _fixed_inputs(tmp_path)
    record = engine.read_preview_inputs(path, settings)
    checkpoint = _completed(tmp_path)
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="incomplete or changed"):
        engine.enqueue_preview(checkpoint, record, settings.output)
    assert not settings.output.exists()


def test_executor_owned_options_rejected(tmp_path):
    path, settings = _fixed_inputs(tmp_path)
    record = json.loads(path.read_text())
    record["evaluation_arguments"] += ["--checkpoint", "untracked.safetensors"]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="executor owns"):
        engine.read_preview_inputs(path, settings)


def test_generation_failure_records_reason_and_keeps_checkpoint(tmp_path, monkeypatch):
    path, settings = _fixed_inputs(tmp_path)
    fixed = engine.read_preview_inputs(path, settings)
    checkpoint = _completed(tmp_path)
    digest = sha256(checkpoint)
    marker_digest = sha256(checkpoint.with_suffix(".complete.json"))
    job_path = engine.enqueue_preview(checkpoint, fixed, settings.output)

    def fail_execution(args, **kwargs):
        assert kwargs["preview_tensor_validator"] is previews.verify_preview_tensors
        assert args.checkpoint == [checkpoint]
        assert args.preview_fixed == fixed
        assert args.output.name == "attempt_0000"
        raise RuntimeError("input replay failed before transformer")

    monkeypatch.setattr(evaluate, "execute_evaluation", fail_execution)
    with pytest.raises(RuntimeError, match="input replay failed"):
        previews.generate_preview(job_path, gpu_id=4)
    job = json.loads(job_path.read_text())
    assert job["state"] == "failed"
    assert job["error"] == "RuntimeError: input replay failed before transformer"
    assert sha256(checkpoint) == digest
    assert sha256(checkpoint.with_suffix(".complete.json")) == marker_digest


def test_preview_actual_tensor_check_rejects_changed_bytes():
    value = torch.ones(1, 4, 2, dtype=torch.bfloat16)
    fixed = {"input_files": {"capture": {"tensor_sha256": hashing.tensor_sha256(value)}}}
    previews.verify_preview_tensors(fixed, {"capture": value})
    with pytest.raises(ValueError, match="fixed capture tensor"):
        previews.verify_preview_tensors(fixed, {"capture": value + 1})
    with pytest.raises(ValueError, match="fixed capture tensor"):
        previews.verify_preview_tensors(fixed, {})


def test_generation_corrupt_input_records_failure_before_execution(tmp_path, monkeypatch):
    path, settings = _fixed_inputs(tmp_path)
    fixed = engine.read_preview_inputs(path, settings)
    checkpoint = _completed(tmp_path)
    job_path = engine.enqueue_preview(checkpoint, fixed, settings.output)
    Path(fixed["input_files"]["noise"]["path"]).write_bytes(b"changed input")

    def forbidden(args, **_kwargs):
        pytest.fail("changed pinned files must fail before generation")

    monkeypatch.setattr(evaluate, "execute_evaluation", forbidden)
    with pytest.raises(ValueError, match="noise file changed"):
        previews.generate_preview(job_path, gpu_id=4)
    failed = json.loads(job_path.read_text())
    assert failed["state"] == "failed"
    assert "noise file changed" in failed["error"]


def test_generation_records_raw_evidence_and_retry_preserves_prior_attempt(tmp_path, monkeypatch):
    path, settings = _fixed_inputs(tmp_path)
    fixed = engine.read_preview_inputs(path, settings)
    checkpoint = _completed(tmp_path)
    checkpoint_hash = sha256(checkpoint)
    marker_hash = sha256(checkpoint.with_suffix(".complete.json"))
    job_path = engine.enqueue_preview(checkpoint, fixed, settings.output)
    attempts = []

    def save_raw(args, **_kwargs):
        attempts.append(args.output)
        record = {
            "mode": fixed["mode"], "adapter": str(checkpoint), "adapter_sha256": checkpoint_hash,
            "software": software.capture("evaluation", fixed["mode"]),
            **{
                key: fixed["input_files"][role]["tensor_sha256"]
                for role, key in (
                    ("capture", "capture_sha256"), ("guide", "guide_sha256"),
                    ("first_image", "c0_sha256"), ("text", "text_sha256"), ("noise", "noise_sha256"),
                )
            },
        }
        evaluate.save_case(torch.zeros(1, 4, 3, 2, 2), record, args.output / "case_0000" / "variant_0000")
        return 0

    monkeypatch.setattr(evaluate, "execute_evaluation", save_raw)
    results = previews.generate_preview(job_path, gpu_id=4)
    job = json.loads(job_path.read_text())
    assert job["state"] == "running"
    assert job["raw_attempt"] == 0
    assert job["results"] == results
    prior_hash = sha256(Path(results[0]["path"]))
    previews.set_preview_state(job_path, "failed", error="reference rendering failed")
    retry = previews.generate_preview(job_path, gpu_id=4)
    assert attempts[0].name == "attempt_0000"
    assert attempts[1].name == "attempt_0001"
    assert retry != results
    assert sha256(Path(results[0]["path"])) == prior_hash
    assert json.loads(job_path.read_text())["state"] == "running"
    assert sha256(checkpoint) == checkpoint_hash
    assert sha256(checkpoint.with_suffix(".complete.json")) == marker_hash


def test_preview_failure_and_retry_do_not_change_checkpoint(tmp_path):
    path, settings = _fixed_inputs(tmp_path)
    fixed = engine.read_preview_inputs(path, settings)
    checkpoint = _completed(tmp_path)
    digest = sha256(checkpoint)
    marker_digest = sha256(checkpoint.with_suffix(".complete.json"))
    job = engine.enqueue_preview(checkpoint, fixed, settings.output)
    assert previews.set_preview_state(job, "running")["state"] == "running"
    with pytest.raises(ValueError, match="requires raw results"):
        previews.set_preview_state(job, "complete")
    assert json.loads(job.read_text())["state"] == "running"
    failed = previews.set_preview_state(job, "failed", error="decode failed")
    assert failed["error"] == "decode failed"
    assert previews.set_preview_state(job, "running")["state"] == "running"
    assert sha256(checkpoint) == digest
    assert sha256(checkpoint.with_suffix(".complete.json")) == marker_digest


def test_changed_inputs_fail_before_job_claim(tmp_path):
    path, settings = _fixed_inputs(tmp_path)
    fixed = engine.read_preview_inputs(path, settings)
    checkpoint = _completed(tmp_path)
    job = engine.enqueue_preview(checkpoint, fixed, settings.output)
    Path(fixed["input_files"]["noise"]["path"]).write_text("changed noise")
    with pytest.raises(ValueError, match="noise file changed"):
        previews.set_preview_state(job, "running")
    assert json.loads(job.read_text())["state"] == "pending"
    assert previews.set_preview_state(job, "failed", error="fixed noise changed")["error"] == "fixed noise changed"


def test_completion_requires_matched_raw_and_rendered_evidence(tmp_path):
    path, settings = _fixed_inputs(tmp_path)
    fixed = engine.read_preview_inputs(path, settings)
    checkpoint = _completed(tmp_path)
    job_path = engine.enqueue_preview(checkpoint, fixed, settings.output)
    job = previews.set_preview_state(job_path, "running")
    result = {
        "mode": "bidirectional",
        "software": software.capture("evaluation", "bidirectional"),
        "adapter": str(checkpoint),
        "adapter_sha256": sha256(checkpoint),
        **{
            key: fixed["input_files"][role]["tensor_sha256"]
            for role, key in (
                ("capture", "capture_sha256"),
                ("guide", "guide_sha256"),
                ("first_image", "c0_sha256"),
                ("text", "text_sha256"),
                ("noise", "noise_sha256"),
            )
        },
    }
    destination = tmp_path / "raw"
    evaluate.save_case(torch.zeros(1, 128, 7, 2, 2), result, destination)
    raw_path = destination / "result.json"
    results = [{"path": str(raw_path), "sha256": sha256(raw_path)}]
    with pytest.raises(ValueError, match="requires raw results and rendered"):
        previews.set_preview_state(job_path, "complete", results=results)
    rgb = torch.zeros(2, 3, 16, 32)
    panels = [
        media.Panel(role, title, rgb, (0, 1))
        for role, title in (
            ("recorded", "Recording"),
            ("decoded", "Decoded"),
            ("guide", "Guide"),
            ("baseline", "Base"),
            ("changed", "Trained"),
        )
    ]
    pixels, rendering = media.render_panels(
        panels,
        question="Does training help?",
        layout="training",
        fps=30,
        common_settings={
            "preview_job_id": job["id"],
            "fixed_inputs_sha256": fixed["sha256"],
            "result_records": results,
        },
    )
    rendering["software"] = software.capture("decoding")
    media.save_render(pixels, rendering, tmp_path / "rendered")
    rendered_path = tmp_path / "rendered" / "rendering.json"
    renderings = [{"path": str(rendered_path), "sha256": sha256(rendered_path)}]
    assert (
        previews.set_preview_state(job_path, "complete", results=results, renderings=renderings)["state"] == "complete"
    )


def test_reference_bundle_pins_capture_source_and_guide(tmp_path):
    path, settings = _fixed_inputs(tmp_path)
    fixed = json.loads(path.read_text())
    fixed['evaluation_arguments'] += ['--source', 'actor/view']
    guide_path = Path(fixed['input_files']['guide']['path'])
    torch.save({'input_fingerprint': 'a' * 64}, guide_path)
    fixed['input_files']['guide']['sha256'] = sha256(guide_path)
    pixels = torch.zeros(9, 3, 16, 16)
    panels = [media.Panel(role, role, pixels, tuple(range(9))) for role in ('recorded', 'decoded', 'guide')]
    producer = {
        'source': 'actor/view', 'capture_encoding_sha256': fixed['input_files']['capture']['sha256'],
        'guide_rgb_sha256': 'a' * 64, 'source_frames': list(range(9)),
        'panels': [{'role': p.role, 'pixels_sha256': media._pixel_identity(p.pixels)} for p in panels],
    }
    destination = tmp_path / 'reference_bundle'
    media.save_training_references(panels, producer, destination)
    manifest = destination / 'references.json'
    fixed['reference_bundle'] = {'path': str(manifest.resolve()), 'sha256': sha256(manifest)}
    path.write_text(json.dumps(fixed))
    checked = engine.read_preview_inputs(path, settings)
    assert previews.check_preview_reference_bundle(checked) == producer
    checked['input_files']['capture']['sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='different capture encoding'):
        previews.check_preview_reference_bundle(checked)
    checked['input_files']['capture']['sha256'] = producer['capture_encoding_sha256']
    (destination / 'guide.pt').write_bytes(b'changed RGB')
    with pytest.raises(ValueError, match='pixel file changed'):
        previews.check_preview_reference_bundle(checked)


def test_d1_preview_refuses_absent_guide_even_with_missing_fingerprint(tmp_path):
    path, settings = _fixed_inputs(tmp_path)
    fixed = json.loads(path.read_text())
    fixed['evaluation_arguments'] += ['--source', 'actor/view']
    guide = Path(fixed['input_files']['guide']['path'])
    torch.save({}, guide)
    fixed['input_files']['guide']['sha256'] = sha256(guide)
    pixels = torch.zeros(9, 3, 16, 16)
    panels = [media.Panel(role, role, pixels, tuple(range(9))) for role in ('recorded', 'decoded')]
    panels.append(media.Panel('guide', 'Guide RGB', None, tuple(range(9)), missing_reason='Guide not used'))
    producer = {
        'source': 'actor/view', 'capture_encoding_sha256': fixed['input_files']['capture']['sha256'],
        'guide_rgb_sha256': None, 'source_frames': list(range(9)),
        'panels': [{'role': p.role, 'pixels_sha256': media._pixel_identity(p.pixels)} for p in panels],
    }
    destination = tmp_path / 'references'
    media.save_training_references(panels, producer, destination)
    manifest = destination / 'references.json'
    fixed['reference_bundle'] = {'path': str(manifest.resolve()), 'sha256': sha256(manifest)}
    path.write_text(json.dumps(fixed))
    with pytest.raises(ValueError, match='D1 preview requires a checked guide reference'):
        engine.read_preview_inputs(path, settings)


@pytest.mark.parametrize("changed_runtime", [False, True])
@pytest.mark.parametrize("reference_case", ["producer", "capture_only", "no_fit"])
def test_generation_with_reference_bundle_renders_and_completes(tmp_path, monkeypatch, changed_runtime, reference_case):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from scripts.onestep_avatar.corpus import dataset, precompute
    from scripts.prune.core import session as native_session

    path, settings = _fixed_inputs(tmp_path)
    fixed = json.loads(path.read_text())
    fixed['evaluation_arguments'] += ['--source', 'actor/view']
    guide_path = Path(fixed['input_files']['guide']['path'])
    torch.save({'input_fingerprint': 'a' * 64}, guide_path)
    fixed['input_files']['guide']['sha256'] = sha256(guide_path)
    vae = tmp_path / 'vae.safetensors'
    vae.write_bytes(b'fixed decoder')
    pixels = torch.zeros(9, 3, 16, 16)
    view = tmp_path / 'actor/view'
    view.mkdir(parents=True)
    capture_path = view / dataset.capture_bundle_name('white')
    torch.save({'schema_version': 2, 'master': torch.zeros(128, 2, 1, 1), 'fps': 30}, capture_path)
    fixed['input_files']['capture'].update(path=str(capture_path), sha256=sha256(capture_path))
    capture_only = reference_case == 'capture_only'
    source = {
        'relative_dir': 'actor/view', 'capture_latent_sha256': sha256(capture_path),
        'guide_sha256': None if capture_only else 'a' * 64, 'shape': [128, 2, 1, 1], 'fps': 30,
        'capture_encode_record': {'vae_fingerprint': precompute.file_fingerprint(vae)},
    }
    monkeypatch.setattr(media, 'recorded_capture_rgb', lambda *a: pixels)
    monkeypatch.setattr(media, 'recorded_guide_rgb', lambda *a: pixels)
    monkeypatch.setattr(media, 'decode', lambda *a: pixels)
    preparation_session = SimpleNamespace(model=SimpleNamespace(paths=SimpleNamespace(video_vae=lambda: vae)))
    panels, producer = media.prepare_training_references(
        preparation_session, None, source, tmp_path, 'white', 2, 42, require_guide=not capture_only,
    )
    if capture_only:
        settings.guide_mode = 'd0'
        fixed['evaluation_arguments'] += ['--guide-mode', 'd0']
        del fixed['input_files']['guide']
    if reference_case == 'no_fit':
        from dataclasses import replace
        panels[1] = replace(panels[1], title='A reference title that cannot possibly fit inside any supported training panel')
    destination = tmp_path / 'references'
    media.save_training_references(panels, producer, destination)
    manifest = destination / 'references.json'
    fixed['reference_bundle'] = {'path': str(manifest.resolve()), 'sha256': sha256(manifest)}
    path.write_text(json.dumps(fixed))
    fixed = engine.read_preview_inputs(path, settings)
    checkpoint = _completed(tmp_path)
    job_path = engine.enqueue_preview(checkpoint, fixed, settings.output)
    original_hash = sha256(checkpoint)

    def save_raw(args, **_kwargs):
        record = {
            'mode': 'bidirectional', 'frames': 2, 'source': 'actor/view', 'fps': 30,
            'software': software.capture('evaluation', 'bidirectional'),
            'adapter': str(checkpoint), 'adapter_sha256': sha256(checkpoint),
            **{key: fixed['input_files'][role]['tensor_sha256'] if role in fixed['input_files'] else None for role, key in (
                ('capture', 'capture_sha256'), ('guide', 'guide_sha256'), ('first_image', 'c0_sha256'),
                ('text', 'text_sha256'), ('noise', 'noise_sha256'),
            )},
        }
        evaluate.save_case(torch.zeros(1, 128, 2, 1, 1), record, args.output / 'case_0000/variant_0000')

    monkeypatch.setattr(evaluate, 'execute_evaluation', save_raw)
    monkeypatch.setattr(evaluate.backbone, 'resolve', lambda *a: SimpleNamespace(paths=SimpleNamespace(video_vae=lambda: vae)))
    monkeypatch.setattr(native_session, 'Session', lambda *a: SimpleNamespace(decoder=lambda: nullcontext(None)))
    monkeypatch.setattr(media, 'decode', lambda *a: pixels)
    if changed_runtime or reference_case == 'no_fit':
        monkeypatch.setattr(media, 'native_decoder_settings', lambda: {**producer['decoder_settings'], 'torch': 'different'})
        if not changed_runtime:
            monkeypatch.setattr(media, 'native_decoder_settings', lambda: producer['decoder_settings'])
        monkeypatch.setattr(native_session, 'Session', lambda *a: pytest.fail('decoder opened before runtime validation'))
        monkeypatch.setattr(media, 'decode', lambda *a: pytest.fail('pixels decoded before layout/runtime validation'))
        with pytest.raises(ValueError, match='decoder settings differ' if changed_runtime else 'no readable compact layout'):
            previews.generate_preview(job_path, gpu_id=4)
        failed = json.loads(job_path.read_text())
        assert failed['state'] == 'failed'
        assert failed['results']
        assert sha256(checkpoint) == original_hash
        assert not list(Path(failed['output']).glob('render_attempt_*'))
        return
    results = previews.generate_preview(job_path, gpu_id=4)
    completed = json.loads(job_path.read_text())
    assert completed['state'] == 'complete' and completed['results'] == results
    assert sha256(checkpoint) == original_hash
    rendering = json.loads(Path(completed['renderings'][0]['path']).read_text())
    assert rendering['common_settings']['reference_bundle'] == fixed['reference_bundle']
    assert len(rendering['common_settings']['decoder_records']) == 1
    assert rendering['layout'] == 'compact_training'
    assert rendering['font_size'] * 480 / rendering['display_size'][0] >= 16
    guide_panel = next(row for row in rendering['panels'] if row['role'] == 'guide')
    if capture_only:
        assert guide_panel['missing_reason'] == 'Guide not used'
        assert guide_panel['pixels_sha256'] is None
