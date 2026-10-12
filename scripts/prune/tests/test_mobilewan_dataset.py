"""CPU controls for full-view assignment and shared score-input reconstruction."""
import torch

from scripts.prune.core.provenance import file_sha256
from scripts.prune.data import whole_clip
from scripts.prune.score.mobilewan_dataset import assigned_views
from scripts.prune.tests.test_whole_clip_baseline import _Session, _vae, _views


def test_shards_cover_every_view_once_and_preserve_splits():
    sources = [{'relative_dir': str(i), 'split': 'train'} for i in range(17)]
    sources += [{'relative_dir': 'held', 'split': 'held_out'}]
    assignments = [assigned_views(sources, 'train', i, 3) for i in range(3)]
    combined = [s['relative_dir'] for shard in assignments for s in shard]
    assert len(combined) == len(set(combined)) == 17
    assert 'held' not in combined
    assert assignments == [assigned_views(sources[::-1], 'train', i, 3) for i in range(3)]


def test_shared_score_inputs_reuse_exact_noise_and_exclude_guide(tmp_path):
    view = _views(tmp_path / 'corpus')[0]
    capture = view / 'ltx_vae_latent_white.pt'
    bundle = torch.load(capture, map_location='cpu', weights_only=True)
    current = _Session(_vae(tmp_path))
    source = {'capture': str(capture), 'capture_latent_sha256': file_sha256(capture),
              'shape': list(bundle['master'].shape), 'fps': bundle['fps'], 'resolved_dir': str(view),
              'capture_encode_record': {'schema_version': bundle['schema_version']}}
    root = tmp_path / 'inputs'
    manifest = whole_clip.prepare_score_inputs(root, source, current=current, sigmas=[.9, .4])
    original_noise = (root / 'epsilon.pt').read_bytes()
    assert whole_clip.prepare_score_inputs(root, source, current=current, sigmas=[.9, .4]) == manifest
    assert (root / 'epsilon.pt').read_bytes() == original_noise
    for sigma in [.9, .4]:
        grid, modality, c0, row = whole_clip.build_input(root, manifest, view=str(view), sigma=sigma, current=current)
        assert torch.equal(modality.latent[:, :grid.tokens_per_latent_frame], c0)
        assert (modality.timesteps[:, :grid.tokens_per_latent_frame] == 0).all()
        assert modality.sigma.dtype == torch.float32
        assert modality.attention_mask is None
        assert 'guide' not in row['artifacts']


def test_worker_saves_sigmas_then_resumes_only_missing_observation(tmp_path, monkeypatch):
    import hashlib
    import time

    from scripts.prune.data.head_score_dataset import SIGMAS
    from scripts.prune.score import mobilewan_records, token_sampling
    from scripts.prune.score.mobilewan_dataset import score_view, source_identity

    view = _views(tmp_path / 'corpus')[0]
    capture = view / 'ltx_vae_latent_white.pt'
    bundle = torch.load(capture, map_location='cpu', weights_only=True)
    current = _Session(_vae(tmp_path))
    frames, height, width = bundle['master'].shape[1:]
    indices = token_sampling.sample_indices(frames*height*width, height, width, 1, torch.device('cpu'))
    sampler = token_sampling.sampling_record(indices, tokens=frames*height*width, height=height,
                                             width=width, stride=1, sampler='stride')
    source = {'capture': str(capture), 'capture_latent_sha256': file_sha256(capture),
              'shape': list(bundle['master'].shape), 'fps': bundle['fps'], 'resolved_dir': str(view),
              'capture_encode_record': {'schema_version': 2}, 'actor': '8', 'split': 'train',
              'relative_dir': 'Part_1/0008_01/views/view00', 'sampler': sampler}
    context_hash = hashlib.sha256(current.context.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    study = {'text_context': {'tensor_sha256': context_hash}, 'components': {'transformer': {'file_sha256': 'a'*64}}}
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda _: 0)
    run = tmp_path / 'run'
    widths = {'0.attn1': 2, '1.attn1': 2}
    code_hash = source_identity()[0]
    with current.transformer() as model:
        calls = []
        handle = model.register_forward_pre_hook(lambda *_: calls.append(1))
        score_view(model, current, run, source, study, code_hash, widths, time.monotonic()+60)
        assert len(calls) == len(SIGMAS)
        completed = mobilewan_records.record_path(run / 'scores/per_view', source['relative_dir'])
        completed.unlink()
        mobilewan_records.observation_path(run / 'scores/per_sigma', source['relative_dir'], 3).unlink()
        calls.clear()
        score_view(model, current, run, source, study, code_hash, widths, time.monotonic()+60)
        assert len(calls) == 1
        assert completed.exists()
        calls.clear()
        score_view(model, current, run, source, study, code_hash, widths, time.monotonic()+60)
        assert not calls
        handle.remove()


def test_heldout_requires_valid_train_freeze(tmp_path):
    import json

    import pytest

    from scripts.prune.score.mobilewan_dataset import verify_train_freeze

    (tmp_path / 'masks').mkdir()
    (tmp_path / 'masks/frozen_train.json').write_text(json.dumps({'selection_cohort': 'held_out'}))
    dataset = {'sources': [{'relative_dir': 'train/view', 'split': 'train'}], 'membership_semantic_sha256': 'a'*64}
    study = {'components': {'transformer': {'file_sha256': 'b'*64}}, 'removal_fractions': [0, .1, .23, .33]}
    with pytest.raises(ValueError, match='train-only mask freeze'):
        verify_train_freeze(tmp_path, dataset, study)


def test_native_gate_binds_current_score_source(tmp_path):
    import json

    import pytest

    from scripts.prune.data.head_score_dataset import SIGMAS
    from scripts.prune.score import mobilewan_dataset

    (tmp_path / 'validation').mkdir()
    digest = file_sha256(mobilewan_dataset.Path(mobilewan_dataset.__file__).with_name('mobilewan_heads.py'))
    for frames in (18, 28):
        rows = [{'sigma': s, 'sampling': {'latent_frames': frames}, 'stock_parity': True,
                 'hook_transparent': True, 'all_one_equal': True, 'saved_baseline_equal': True,
                 'clean_frame_equal': True} for s in SIGMAS]
        (tmp_path / f'validation/native_check_{frames}.json').write_text(json.dumps({'runs': rows, 'score_source_sha256': digest}))
    mobilewan_dataset.verify_gates(tmp_path)
    path = tmp_path / 'validation/native_check_28.json'
    row = json.loads(path.read_text())
    row['score_source_sha256'] = '0'*64
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='fresh checks'):
        mobilewan_dataset.verify_gates(tmp_path)
