"""Scientific queue completion uses real saved masters/results and controlled CPU execution."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.execution import queue
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.test_subset import old_subset  # noqa: F401 -- fixture dependency
from scripts.onestep_avatar.tests.test_training_preflight import checked_settings  # noqa: F401


@pytest.fixture
def completed(checked_settings, monkeypatch, tmp_path):
    settings, membership = checked_settings
    specification = evaluate.backbone.resolve('2.5', 'dev')
    Path(specification.paths.transformer()).write_bytes(b'controlled base weights')
    monkeypatch.setattr(evaluate.backbone, 'identity', lambda path, *_a, **_k: {
        'base_transformer_sha256': sha256(Path(path))})
    import scripts.prune.core.preflight as preflight
    import scripts.prune.core.session as sessions
    import scripts.prune.data.prompt_cache as prompts

    calls = []

    class Session:
        def __init__(self, *_args):
            calls.append('session')
            self.device = torch.device('cpu')

        def transformer(self, **_kwargs):
            calls.append('transformer')
            model = LTXModel(
                model_type=LTXModelType.VideoOnly, num_attention_heads=2, attention_head_dim=4,
                in_channels=2, out_channels=2, num_layers=2, cross_attention_dim=8,
                caption_projection=torch.nn.Linear(4, 8),
            )
            generator = torch.Generator().manual_seed(0)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.copy_(torch.randn(parameter.shape, generator=generator)*0.05)
            return nullcontext(X0Model(model.to(dtype=torch.bfloat16).eval()))

    monkeypatch.setattr(sessions, 'Session', Session)
    # These synthetic adapter matrices pin completion identity, not a real LTX
    # target inventory. Actual unmerged effects/loading are tested in test_adapters.
    monkeypatch.setattr(evaluate.adapter_loader, 'inference_transformer',
                        lambda session, *_a, **_kw: session.transformer())
    monkeypatch.setattr(preflight, 'check', lambda *_a, **_k: calls.append('preflight'))
    monkeypatch.setattr(prompts, 'get_or_build', lambda *_a, **_k: torch.ones(1, 3, 4, dtype=torch.bfloat16))
    # Keep only the evaluator's device selection on CPU; shared Torch stays intact.
    monkeypatch.setattr(evaluate, 'torch', SimpleNamespace(**{**vars(torch), 'device': lambda *_a: torch.device('cpu')}))
    arguments = ['--mode', 'bidirectional', '--subset', str(settings.subset), '--output', str(settings.output),
                 '--variant', 'dev', '--guide-mode', 'd0', '--schedule', '0.725', '0', '--seed', '42']

    def execute(extra=(), *, mode=None, guide_mode=None):
        command = arguments + list(extra)
        if mode is not None:
            command[command.index('--mode')+1] = mode
        if guide_mode is not None:
            command[command.index('--guide-mode')+1] = guide_mode
        args = evaluate.parse_args(command)
        evaluate.execute_evaluation(args)
        paths = sorted(settings.output.glob('case_*/variant_*/result.json'))
        if not paths:
            paths = sorted(settings.output.glob('case_*/variant_*/*/result.json'))
        assert paths
        job = {'id': 'checked', 'kind': 'evaluate', 'arguments': command, 'output': str(settings.output),
               'completion': {'records': [str(path) for path in paths]}}
        jobs_path = settings.output.parent/'queue_jobs.json'
        jobs_path.write_text(json.dumps({'schema_version': 1, 'jobs': [job]}))
        job = queue.prepare_jobs(jobs_path)[0]
        before = len(calls)
        assert queue.verify_completion(job)
        assert len(calls) == before, 'completion opened a native model session'
        return job, paths

    return execute, arguments, settings, membership, calls


def test_completed_science_and_normal_fresh_output_gate(completed):
    execute, arguments, _, _, _ = completed
    job, paths = execute()
    assert len(paths) == 2
    with pytest.raises(ValueError, match='already used'):
        evaluate.prepare_evaluation(evaluate.parse_args(arguments))
    assert queue.verify_completion(job)
    receipt = queue.completion_receipt(job)
    assert len(receipt['evidence']) == 7
    assert queue.verify_receipt(job, receipt)


@pytest.mark.parametrize("extra,message", [
    (["--split", "train", "--source", "unknown/video"], "unknown videos"),
    (["--split", "validation"], "selection is empty"),
])
def test_split_selection_rejects_invalid_inventory_before_sessions(completed, extra, message):
    _, arguments, _, _, calls = completed
    with pytest.raises(ValueError, match=message):
        evaluate.prepare_evaluation(evaluate.parse_args(arguments + extra))
    assert not calls


def test_split_selection_limits_results_and_rejects_cross_split_source(completed):
    execute, arguments, _, membership, calls = completed
    held_out = next(source["relative_dir"] for source in membership["sources"] if source["split"] == "held_out")
    with pytest.raises(ValueError, match="differs from the requested fixed-video split"):
        evaluate.prepare_evaluation(evaluate.parse_args(arguments + ["--split", "train", "--source", held_out]))
    assert not calls
    job, paths = execute(["--split", "held_out"])
    assert len(paths) == 1
    assert json.loads(paths[0].read_text())["source"] == held_out
    assert queue.verify_completion(job)
    job["arguments"][job["arguments"].index("--split") + 1] = "train"
    with pytest.raises(ValueError, match="scientific settings or input evidence"):
        queue.verify_completion(job)


@pytest.mark.parametrize('mode,expected_calls', [('bidirectional', 1), ('causal', 6)])
def test_both_modes_publish_verifiable_real_small_transformer_outputs(completed, mode, expected_calls):
    execute, _, _, _, _ = completed
    job, paths = execute(mode=mode)
    assert queue.verify_completion(job)
    assert all(json.loads(path.read_text())['call_counts']['model_calls'] == expected_calls for path in paths)
    if mode == 'causal':
        job['arguments'].append('--teacher-forcing')
        with pytest.raises(ValueError, match='scientific settings or input evidence'):
            queue.verify_completion(job)


def test_causal_physical_prefix_roundtrip_preserves_null_span_c0_calls_and_saved_completion(completed):
    execute, _arguments, settings, membership, calls = completed
    # Extend checked continuous masters; the selected generation still covers only [0,7).
    for source in membership['sources']:
        path = Path(membership['corpus_root']) / source['relative_dir'] / dataset.capture_bundle_name('white')
        bundle = torch.load(path, weights_only=True)
        bundle['master'] = torch.arange(144).reshape(2, 18, 2, 2).float()
        torch.save(bundle, path)
        source.update(n_latent_frames=18, shape=[2, 18, 2, 2], capture_latent_sha256=sha256(path))
    membership['sha256'] = subset.membership_hash(membership)
    settings.subset.write_text(json.dumps(membership))
    job, paths = execute(['--output-latent-frames', '7'], mode='causal')
    for path in paths:
        record = json.loads(path.read_text())
        assert record['frames'] == 7
        assert record['mode_settings']['span_latent_frames'] is None
        assert record['conditions']['mode_settings']['span_latent_frames'] is None
        assert record['conditions']['shape']['frames'] == 7
        assert record['call_counts']['model_calls'] == 6
        generated = torch.load(path.parent / 'generated.pt', weights_only=True)
        assert generated.shape == (1, 2, 7, 2, 2)
        assert evaluate.tensor_sha256(generated[:, :, :1].permute(0, 2, 3, 4, 1).reshape(1, 4, 2)) == record['c0_sha256']
    before = list(calls)
    assert queue.verify_completion(job)
    assert calls == before
    job['arguments'][job['arguments'].index('--output-latent-frames') + 1] = '9'
    with pytest.raises(ValueError, match='saved noise differs from requested input'):
        queue.verify_completion(job)
    assert calls == before


@pytest.mark.parametrize('field,value', [
    ('seed', 43), ('schedule', [0.5, 0]), ('source', 'other/video'), ('fps', 24), ('frames', 5),
    ('mode_settings', {'span_latent_frames': 5}), ('conditions', {}), ('input_file_hashes', {}),
    ('membership_sha256', 'b'*64), ('adapter', '/other/adapter'), ('application_method', 'unmerged'),
    ('prompt', 'other prompt'), ('negative_prompt', 'other negative'), ('guidance', {}),
    ('text_sha256', 'b'*64), ('capture_sha256', 'b'*64), ('guide_sha256', 'b'*64),
    ('c0_sha256', 'b'*64), ('noise_sha256', 'b'*64), ('producer_source_sha256', 'b'*64),
])
def test_result_cannot_claim_other_scientific_conditions(completed, field, value):
    execute, _, _, _, _ = completed
    job, paths = execute()
    record = json.loads(paths[0].read_text())
    record[field] = value
    paths[0].write_text(json.dumps(record))
    with pytest.raises(ValueError, match='scientific settings or input evidence'):
        queue.verify_completion(job)


@pytest.mark.parametrize('option,value', [('--seed', '43'), ('--guide-mode', 'd1'), ('--variant', 'distilled')])
def test_result_cannot_complete_different_requested_job(completed, option, value):
    execute, _, _, _, _ = completed
    job, _ = execute()
    job['arguments'][job['arguments'].index(option)+1] = value
    with pytest.raises((ValueError, FileNotFoundError)):
        queue.verify_completion(job)


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'extra', 'wrong_path'])
def test_exact_case_inventory_is_required(completed, defect):
    execute, _, _, _, _ = completed
    job, paths = execute()
    if defect == 'missing':
        job['completion']['records'].pop()
    elif defect == 'duplicate':
        job['completion']['records'].append(str(paths[0]))
    else:
        alias = paths[0].parent/'unrequested.json'
        alias.write_bytes(paths[0].read_bytes())
        if defect == 'extra':
            job['completion']['records'].append(str(alias))
        else:
            job['completion']['records'][0] = str(alias)
    with pytest.raises(ValueError, match='inventory differs'):
        queue.verify_completion(job)


@pytest.mark.parametrize('asset', ['text', 'noise', 'dtype', 'first_frame', 'shape', 'base'])
def test_saved_evidence_changes_fail_even_with_updated_output_hash(completed, asset):
    execute, _, settings, _, _ = completed
    job, paths = execute()
    record = json.loads(paths[0].read_text())
    if asset == 'base':
        evaluate.backbone.resolve('2.5', 'dev').paths.transformer().write_bytes(b'changed base weights')
    elif asset in ('text', 'noise'):
        path = settings.output/'text.pt' if asset == 'text' else settings.output/'case_0000/noise.pt'
        value = torch.load(path, weights_only=True)
        torch.save(value+1, path)
    else:
        path = Path(record['output']['path'])
        value = torch.load(path, weights_only=True)
        if asset == 'dtype': value = value.float()
        if asset == 'first_frame': value[:, :, :1] += 1
        if asset == 'shape': value = value[:, :, :5].contiguous()
        torch.save(value, path)
        record['output'].update(sha256=sha256(path), shape=list(value.shape))
        paths[0].write_text(json.dumps(record))
    with pytest.raises(ValueError):
        queue.verify_completion(job)


def test_future_noise_requires_both_branches_and_original_noise_input(completed):
    execute, _, settings, membership, _ = completed
    noise = torch.randn(1, 28, 2, dtype=torch.bfloat16)
    changed = noise.clone()
    changed[:, 12:] += 1
    root = settings.output.parent
    noise_path, changed_path = root/'noise.pt', root/'changed.pt'
    torch.save(noise, noise_path)
    torch.save(changed, changed_path)
    source = membership['sources'][0]['relative_dir']
    job, paths = execute(['--source', source, '--noise-file', str(noise_path),
                          '--changed-noise-file', str(changed_path), '--future-noise-start', '3'])
    assert len(paths) == 2
    job['completion']['records'] = [str(paths[0])]
    with pytest.raises(ValueError, match='inventory differs'):
        queue.verify_completion(job)
    job['completion']['records'] = [str(path) for path in paths]
    torch.save(noise+1, noise_path)
    with pytest.raises(ValueError):
        queue.verify_completion(job)


def test_guidance_saves_actual_negative_text_and_binds_prompt(completed):
    execute, _, settings, _, _ = completed
    job, _ = execute(['--cfg', '3', '--negative-prompt', 'fixed negative text'])
    assert (settings.output/'negative_text.pt').is_file()
    job['arguments'][-1] = 'different negative text'
    with pytest.raises(ValueError, match='scientific settings or input evidence'):
        queue.verify_completion(job)


@pytest.mark.parametrize('defect', ['bytes', 'path', 'missing_variant'])
def test_checked_adapter_bytes_path_and_complete_variant_inventory(completed, defect):
    from safetensors.torch import save_file

    from scripts.onestep_avatar.tests.test_checkpoint_contract import A, B, _contract
    from scripts.onestep_avatar.training import checkpoints

    execute, _, settings, _, _ = completed
    contract = _contract()
    contract['model']['base_sha256'] = sha256(evaluate.backbone.resolve('2.5', 'dev').paths.transformer())
    contract['shape']['channels'] = 2
    contract['task']['guide_mode'] = 'd0'
    contract['training']['sigma_levels'] = [0.725]
    contract['training']['schedules'] = [[0.725, 0]]
    checkpoints.validate_contract(contract)
    adapter = settings.output.parent/'adapter.safetensors'
    metadata = {checkpoints.CONTRACT_KEY: json.dumps(contract)}
    save_file({A: torch.ones(2, 4, dtype=torch.bfloat16), B: torch.zeros(4, 2, dtype=torch.bfloat16)},
              adapter, metadata=metadata)
    job, paths = execute(['--span-latent-frames', '7', '--checkpoint', str(adapter), '--include-base'])
    assert len(paths) == 4
    if defect == 'bytes':
        save_file({A: torch.ones(2, 4, dtype=torch.bfloat16), B: torch.ones(4, 2, dtype=torch.bfloat16)},
                  adapter, metadata=metadata)
    elif defect == 'path':
        other = adapter.with_name('other.safetensors')
        other.write_bytes(adapter.read_bytes())
        job['arguments'][job['arguments'].index('--checkpoint')+1] = str(other)
    else:
        job['completion']['records'].pop()
    with pytest.raises(ValueError, match='scientific settings or input evidence|inventory differs'):
        queue.verify_completion(job)


@pytest.mark.parametrize('asset', ['text', 'noise'])
def test_receipts_pin_input_file_bytes_even_when_tensor_content_is_identical(completed, asset):
    execute, _, settings, _, _ = completed
    job, _ = execute()
    receipt = queue.completion_receipt(job)
    path = settings.output/'text.pt' if asset == 'text' else settings.output/'case_0000/noise.pt'
    before = sha256(path)
    value = torch.load(path, weights_only=True)
    torch.save(value, path, _use_new_zipfile_serialization=False)
    assert sha256(path) != before
    assert queue.verify_completion(job), 'scientific tensor values changed'
    with pytest.raises(ValueError, match='receipt evidence changed'):
        queue.verify_receipt(job, receipt)


@pytest.mark.parametrize('field', ['change_start_encoded_frame', 'records', 'earlier_output_bit_identical',
                                  'earlier_output_max_abs_delta', 'later_output_max_abs_delta'])
def test_future_noise_diagnostic_is_bound_to_both_actual_outputs(completed, field):
    execute, _, settings, membership, _ = completed
    noise = torch.randn(1, 28, 2, dtype=torch.bfloat16)
    changed = noise.clone()
    changed[:, 12:] += 1
    noise_path, changed_path = settings.output.parent/'noise.pt', settings.output.parent/'changed.pt'
    torch.save(noise, noise_path)
    torch.save(changed, changed_path)
    job, _ = execute(['--source', membership['sources'][0]['relative_dir'], '--noise-file', str(noise_path),
                      '--changed-noise-file', str(changed_path), '--future-noise-start', '3'])
    receipt = queue.completion_receipt(job)
    assert len(receipt['evidence']) == 8
    path = settings.output/'case_0000/variant_000/future_noise.json'
    data = json.loads(path.read_text())
    if field == 'records': data[field].reverse()
    elif field == 'earlier_output_bit_identical': data[field] = not data[field]
    else: data[field] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='future-noise diagnostic differs'):
        queue.verify_completion(job)


@pytest.mark.parametrize('changed', [None, 'guide', 'sidecar'])
def test_d1_checks_paired_guide_provenance_and_capture_first_frame(completed, changed):
    execute, _, settings, membership, _ = completed
    for source in membership['sources']:
        view = Path(membership['corpus_root'])/source['relative_dir']
        render = view/dataset.render_name('white')
        render.write_bytes(b'controlled guide render bytes')
        sidecar = view/dataset.render_metadata_name('white')
        sidecar.write_text(json.dumps({'objective': 'white',
                                     'compositing_version': dataset.GUIDE_COMPOSITING_VERSION}))
        capture = torch.load(view/dataset.capture_bundle_name('white'), weights_only=True)
        encoding = {**source['capture_encode_record'], 'input_fingerprint': sha256(render)}
        guide_path = view/dataset.guide_bundle_name('white')
        torch.save({**capture, **encoding, 'master': capture['master']+1}, guide_path)
        source.update(guide_latent_sha256=sha256(guide_path), guide_encode_record=encoding,
                      guide_sidecar_sha256=sha256(sidecar))
    membership['sha256'] = subset.membership_hash(membership)
    settings.subset.write_text(json.dumps(membership))
    job, paths = execute(guide_mode='d1')
    assert all(json.loads(path.read_text())['guide_sha256'] for path in paths)
    if changed:
        source = membership['sources'][0]
        view = Path(membership['corpus_root'])/source['relative_dir']
        path = view/(dataset.guide_bundle_name('white') if changed == 'guide'
                     else dataset.render_metadata_name('white'))
        path.write_bytes(b'changed evidence')
        with pytest.raises(ValueError, match='content changed'):
            queue.verify_completion(job)


@pytest.mark.parametrize('history,kv,field', [('recompute', 'refresh', 'history_mode'),
                                             ('joint', 'refresh', 'history_mode'),
                                             ('cache', 'denoise', 'kv_source')])
def test_causal_diagnostic_adapter_preflight_before_sessions(checked_settings, monkeypatch, tmp_path, history, kv, field):
    from safetensors.torch import save_file

    from scripts.onestep_avatar.tests.test_checkpoint_contract import A, B, _contract
    from scripts.onestep_avatar.training.checkpoints import CONTRACT_KEY
    from scripts.prune.core import preflight, session
    from scripts.prune.data import prompt_cache

    settings, membership = checked_settings
    contract = _contract('causal')
    contract['shape']['channels'] = 2
    contract['task']['guide_mode'] = 'd0'
    adapter = tmp_path / 'adapter.safetensors'
    save_file({A: torch.ones(2, 4), B: torch.zeros(4, 2)}, adapter,
              metadata={CONTRACT_KEY: json.dumps(contract)})
    arguments = ['--mode', 'causal', '--subset', str(settings.subset), '--output', str(settings.output),
                 '--variant', 'dev', '--guide-mode', 'd0', '--schedule', '0.725', '0',
                 '--history-mode', history, '--kv-source', kv, '--checkpoint', str(adapter)]
    forbidden = lambda *a, **kw: pytest.fail('off-condition adapter reached a model/text session')
    monkeypatch.setattr(preflight, 'check', forbidden)
    monkeypatch.setattr(session, 'Session', forbidden)
    monkeypatch.setattr(prompt_cache, 'get_or_build', forbidden)
    with pytest.raises(ValueError, match='incompatible.*' + field):
        evaluate.execute_evaluation(evaluate.parse_args(arguments))
    assert not settings.output.exists()
    _, _, cases, _ = evaluate.prepare_evaluation(evaluate.parse_args(arguments + ['--research-override']))
    for _, _, requested, adapters in cases:
        assert requested['history_mode'] == history and requested['kv_source'] == kv
        assert len(adapters[0]['overrides']) == 1
        assert field in adapters[0]['overrides'][0]
        assert 'cached_refresh_global_sigma0' in adapters[0]['overrides'][0]
    # Base-only diagnostics retain their explicit computation without calibration overrides.
    base_arguments = arguments[:-2]
    _, _, cases, _ = evaluate.prepare_evaluation(evaluate.parse_args(base_arguments))
    assert all(adapters[0]['overrides'] == [] for _, _, _, adapters in cases)
    assert all(requested['history_mode'] == history and requested['kv_source'] == kv
               for _, _, requested, _ in cases)
    assert not settings.output.exists()


def test_evaluation_releases_each_resident_model_before_the_next_case(completed, monkeypatch):
    import weakref
    from contextlib import contextmanager
    execute, _, _, _, _ = completed
    references = []

    @contextmanager
    def resident(session, *_args, **_kwargs):
        assert all(reference() is None for reference in references), 'prior full model remains resident'
        with session.transformer() as model:
            references.append(weakref.ref(model))
            yield model

    monkeypatch.setattr(evaluate.adapter_loader, 'inference_transformer', resident)
    execute()
    assert len(references) == 2
    assert all(reference() is None for reference in references)


def test_model_owner_change_blocks_current_completion_without_entry_change(completed, monkeypatch):
    from scripts.onestep_avatar.execution import software
    execute, _, _, _, _ = completed
    job, paths = execute()
    record = json.loads(paths[0].read_text())
    digest = record['producer_source_sha256']
    original = software.sha256
    monkeypatch.setattr(software, 'sha256', lambda p: 'f'*64 if p.name == 'common.py' else original(p))
    assert sha256(Path(evaluate.__file__)) == digest
    software.validate(record['software'])
    with pytest.raises(ValueError, match='software.*changed since preflight'):
        queue.verify_completion(job)


def test_changed_model_owner_during_sampling_prevents_publication(completed, monkeypatch):
    from scripts.onestep_avatar.execution import software
    execute, _, settings, _, _ = completed
    original_hash = software.sha256
    original_sample = evaluate.sample_case

    def sample(*a, **kw):
        result = original_sample(*a, **kw)
        monkeypatch.setattr(software, 'sha256', lambda p: 'f'*64 if p.name == 'common.py' else original_hash(p))
        return result

    monkeypatch.setattr(evaluate, 'sample_case', sample)
    with pytest.raises(ValueError, match='software.*changed since preflight'):
        execute()
    assert not list(settings.output.glob('case_*/variant_*/result.json'))
