"""RGB rendering preserves evidence, source time, role placement and readable titles."""

import pytest
import torch

from scripts.onestep_avatar.media import Panel, decode_key, render_from_record, render_panels


def panels():
    pixels = torch.ones(3, 3, 24, 48)
    pixels[1] = 0
    return [
        Panel(role, title, pixels.clone(), (8, 9, 10), value=value)
        for role, title, value in (
            ("recorded", "Recording", ""),
            ("decoded", "Decoded", ""),
            ("guide", "Guide", ""),
            ("baseline", "Base", ""),
            ("changed", "Trained", "Step 100"),
        )
    ]


def test_training_roles_padding_and_source_time():
    output, record = render_panels(panels(), question="Does training help?", layout="training", fps=30)
    assert output.shape[0] == 3
    assert record["source_frames"] == [8, 9, 10]
    assert record["source_times"] == [8 / 30, 9 / 30, 10 / 30]
    assert [(p["role"], p["row"], p["column"]) for p in record["panels"]] == [
        ("recorded", 0, 0),
        ("decoded", 0, 1),
        ("guide", 0, 2),
        ("baseline", 1, 0),
        ("changed", 1, 1),
        ("unused", 1, 2),
    ]
    assert record["font_size"] * 480 / record["display_size"][0] >= 16
    # 2:1 source fits a square with vertical padding, with all image pixels retained.
    y = (record["font_size"] + 8) + 8 + 8 + 2 * (record["font_size"] + 8) + 8
    assert output[0, :, y + 10, 20].tolist() == [48, 48, 48]
    assert output[0, :, y + 100, 20].tolist() == [255, 255, 255]
    assert output[1, :, y + 100, 20].tolist() == [0, 0, 0]


def test_common_coverage_does_not_freeze_short_output():
    inputs = panels()
    inputs[-1] = Panel("changed", "Trained", inputs[-1].pixels[1:], (9, 10), value="Step 100")
    output, record = render_panels(inputs, question="Does training help?", layout="training", fps=30)
    assert len(output) == 2
    assert record["source_frames"] == [9, 10]


@pytest.mark.parametrize("mapping", [(), (8, 8, 10), (8, 10, 12)])
def test_invalid_time_mapping_fails(mapping):
    inputs = panels()
    inputs[-1] = Panel("changed", "Trained", inputs[-1].pixels, mapping)
    with pytest.raises(ValueError, match="mapping|consecutive"):
        render_panels(inputs, question="Does training help?", layout="training", fps=30)


def test_inference_still_and_absent_capture():
    guide = torch.ones(2, 3, 32, 32)
    supplied = torch.zeros(1, 3, 32, 32)
    inputs = [
        Panel("first_image", "Image (still)", supplied, still=True),
        Panel("guide", "Guide", guide, (0, 1)),
        Panel("generated", "Generated", guide, (0, 1)),
    ]
    output, record = render_panels(inputs, question="Guide and generated video", layout="compact_inference", fps=30)
    assert len(output) == 2
    assert record["panels"][0]["still"]
    assert not any(p["role"] == "recorded" for p in record["panels"])


def test_unreadable_title_rejected():
    inputs = panels()
    inputs[-1] = Panel(
        "changed", "A very long essential title that must remain readable", inputs[-1].pixels, (8, 9, 10)
    )
    with pytest.raises(ValueError, match="compact"):
        render_panels(inputs, question="Does training help?", layout="training", fps=30)
    with pytest.raises(ValueError, match="16 pixels"):
        render_panels(panels(), question="Does training help?", layout="training", fps=30, font_size=10)


def test_decode_identity_has_all_actual_settings():
    args = ["a" * 64, "b" * 64, [1, 128, 17, 8, 8], "native", 7, {"tiling": False}]
    reference = decode_key(*args)
    for index, value in enumerate(["c" * 64, "d" * 64, [1, 128, 9, 8, 8], "alternate", 8, {"tiling": True}]):
        changed = args.copy()
        changed[index] = value
        assert decode_key(*changed) != reference


def test_record_rebuild_uses_same_titles_pixels_and_poster():
    inputs = panels()
    pixels, record = render_panels(inputs, question="Does training help?", layout="compact_training", fps=30,
                                   poster_frame=1)
    rebuilt, rebuilt_record = render_from_record(inputs, record)
    assert torch.equal(pixels, rebuilt)
    assert rebuilt_record == record
    inputs[-1].pixels[0, 0, 0, 0] = 0
    with pytest.raises(ValueError, match="differ"):
        render_from_record(inputs, record)


@pytest.mark.parametrize('decoder_fails', [False, True])
def test_native_decode_activates_session_device_and_restores_it(monkeypatch, decoder_fails):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from scripts.onestep_avatar import media
    from scripts.prune.evaluate import decode as native_decode

    active = {'device': 'prior'}
    device = SimpleNamespace(type='cuda', index=4)
    session = SimpleNamespace(device=device)
    seeded = []

    @contextmanager
    def device_context(requested):
        assert requested is device
        prior = active['device']
        active['device'] = requested
        try:
            yield
        finally:
            active['device'] = prior

    class SavedEncoding:
        shape = (1, 1, 2, 1, 1)

        def to(self, requested):
            assert requested is device and active['device'] is device
            return self

    def decode_latent(requested_session, latent, decoder, *, generator):
        assert requested_session is session and active['device'] is device
        if decoder_fails:
            raise RuntimeError('native decoder failed')
        return torch.zeros(9, 3, 2, 2)

    def generator_factory(*, device):
        def seed(value):
            seeded.append(value)
            return 'fresh generator'
        return SimpleNamespace(manual_seed=seed)

    monkeypatch.setattr(torch.cuda, 'device', device_context)
    monkeypatch.setattr(torch, 'Generator', generator_factory)
    monkeypatch.setattr(native_decode, 'decode_latent', decode_latent)
    if decoder_fails:
        with pytest.raises(RuntimeError, match='native decoder failed'):
            media.decode(session, SavedEncoding(), None, 42)
    else:
        assert media.decode(session, SavedEncoding(), None, 42).shape == (9, 3, 2, 2)
    assert active['device'] == 'prior'
    assert seeded == [42]


def test_recorded_capture_rgb_reuses_producer_crop_and_rejects_changed_source(tmp_path, monkeypatch):
    import numpy as np

    from scripts.onestep_avatar import media
    from scripts.onestep_avatar.corpus import precompute  # noqa: PLC0415 -- same lazy caller scope
    from scripts.onestep_avatar.hashing import sha256

    view = tmp_path / 'actor/view'
    view.mkdir(parents=True)
    rgb, bbox = view / 'rgb.mp4', view / 'bbox.npy'
    rgb.write_bytes(b'saved raw RGB')
    bbox.write_bytes(b'saved bounding boxes')
    stat = rgb.stat()
    source = {
        'relative_dir': 'actor/view', 'n_latent_frames': 2, 'rgb_sha256': sha256(rgb), 'fps': 30,
        'box_xyxy': [0, 0, 16, 16], 'capture_encode_record': {
            'objective': 'bg', 'box_xyxy': [0, 0, 16, 16], 'edge': 16,
            'source': 'actor/view', 'fps': 30, 'pixel_frames': 9,
            'input_fingerprint': f'size={stat.st_size};mtime_ns={stat.st_mtime_ns}',
        },
    }

    def crop(capture, count, box, edge, objectives):
        assert count == 9 and box == (0, 0, 16, 16) and edge == 16 and objectives == ('bg',)
        return {'bg': np.zeros((9, 16, 16, 3), dtype=np.uint8)}

    monkeypatch.setattr(precompute, 'crop_source', crop)
    pixels = media.recorded_capture_rgb(source, tmp_path, 'bg', 2)
    assert pixels.shape == (9, 3, 16, 16) and pixels.dtype == torch.uint8
    source['capture_encode_record']['fps'] = 24
    with pytest.raises(ValueError, match='frame rate differs'):
        media.recorded_capture_rgb(source, tmp_path, 'bg', 2)
    source['capture_encode_record']['fps'] = 30
    source['capture_encode_record']['pixel_frames'] = 8
    with pytest.raises(ValueError, match='encoding coverage'):
        media.recorded_capture_rgb(source, tmp_path, 'bg', 2)
    source['capture_encode_record']['pixel_frames'] = 9
    mask = view / 'mask.mp4'
    mask.write_bytes(b'original matte')
    source['capture_encode_record']['objective'] = 'white'
    source['capture_encode_record']['input_fingerprint'] += ';mask=' + precompute.file_fingerprint(mask)
    mask.write_bytes(b'changed matte content')
    with pytest.raises(ValueError, match='matte producer fingerprint changed'):
        media.recorded_capture_rgb(source, tmp_path, 'white', 2)
    source['capture_encode_record']['objective'] = 'bg'
    rgb.write_bytes(b'changed RGB')
    with pytest.raises(ValueError, match='source content changed'):
        media.recorded_capture_rgb(source, tmp_path, 'bg', 2)


def test_recorded_guide_rgb_checks_content_and_preserves_shape(tmp_path, monkeypatch):
    import json

    import cv2
    import numpy as np

    from scripts.onestep_avatar import media
    from scripts.onestep_avatar.corpus import dataset  # noqa: PLC0415 -- same lazy caller scope
    from scripts.onestep_avatar.hashing import sha256

    view = tmp_path / 'actor/view'
    view.mkdir(parents=True)
    video = view / dataset.render_name('white')
    sidecar = view / dataset.render_metadata_name('white')
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (16, 16))
    assert writer.isOpened()
    for _ in range(9):
        writer.write(np.full((16, 16, 3), 127, dtype=np.uint8))
    writer.release()
    metadata = {'objective': 'white', 'compositing_version': 2, 'out_size': 16, 'n_frames': 9}
    sidecar.write_text(json.dumps(metadata))
    source = {
        'relative_dir': 'actor/view', 'n_latent_frames': 2, 'fps': 30,
        'box_xyxy': [0, 0, 16, 16], 'guide_sha256': sha256(video),
        'guide_sidecar_sha256': sha256(sidecar), 'guide_encode_record': {
            'objective': 'white', 'box_xyxy': [0, 0, 16, 16], 'edge': 16, 'pixel_frames': 9,
            'source': 'actor/view', 'fps': 30,
            'input_fingerprint': sha256(video),
        },
    }
    pixels = media.recorded_guide_rgb(source, tmp_path, 'white', 2)
    assert pixels.shape == (9, 3, 16, 16) and pixels.dtype == torch.uint8
    sidecar.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='sidecar changed'):
        media.recorded_guide_rgb(source, tmp_path, 'white', 2)
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *a: pytest.fail('incompatible producer reached video decode'))
    for field, value in [('objective', 'bg'), ('compositing_version', 1), ('out_size', 32), ('n_frames', 8)]:
        sidecar.write_text(json.dumps({**metadata, field: value}))
        source['guide_sidecar_sha256'] = sha256(sidecar)
        with pytest.raises(ValueError, match='producer conditions differ'):
            media.recorded_guide_rgb(source, tmp_path, 'white', 2)


def test_training_reference_roles_and_decoder_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.onestep_avatar import media
    from scripts.onestep_avatar.corpus import dataset, precompute  # noqa: PLC0415 -- same lazy caller scope
    from scripts.onestep_avatar.hashing import sha256

    view = tmp_path / 'actor/view'
    view.mkdir(parents=True)
    path = view / dataset.capture_bundle_name('white')
    torch.save({'schema_version': 2, 'master': torch.zeros(128, 2, 1, 1), 'fps': 30}, path)
    vae = tmp_path / 'vae.safetensors'
    vae.write_bytes(b'fixed VAE identity')
    source = {
        'relative_dir': 'actor/view', 'capture_latent_sha256': sha256(path), 'guide_sha256': 'a' * 64,
        'shape': [128, 2, 1, 1], 'fps': 30,
        'capture_encode_record': {'vae_fingerprint': precompute.file_fingerprint(vae)},
    }
    pixels = torch.zeros(9, 3, 32, 32, dtype=torch.uint8)
    monkeypatch.setattr(media, 'recorded_capture_rgb', lambda *a: pixels)
    monkeypatch.setattr(media, 'recorded_guide_rgb', lambda *a: pixels)
    monkeypatch.setattr(media, 'decode', lambda *a: pixels.float() / 255)
    session = SimpleNamespace(model=SimpleNamespace(paths=SimpleNamespace(video_vae=lambda: vae)))
    panels, record = media.prepare_training_references(session, None, source, tmp_path, 'white', 2, 42)
    assert [p.role for p in panels] == ['recorded', 'decoded', 'guide']
    assert all(p.source_frames == tuple(range(9)) for p in panels)
    assert record['capture_encoding_sha256'] == sha256(path)
    assert record['vae_sha256'] == sha256(vae)
    assert len(record['decode_key']) == 64
    destination = tmp_path / 'references'
    media.save_training_references(panels, record, destination)
    monkeypatch.setattr(media, 'decode', lambda *a: pytest.fail('saved reference loading ran a VAE'))
    loaded, producer = media.load_training_references(destination / 'references.json')
    assert producer == record
    assert all(torch.equal(left.pixels, right.pixels) for left, right in zip(panels, loaded))
    assert [p.source_frames for p in loaded] == [p.source_frames for p in panels]
    with pytest.raises(ValueError, match='already used'):
        media.save_training_references(panels, record, destination)
    (destination / 'guide.pt').write_bytes(b'changed pixels')
    with pytest.raises(ValueError, match='pixel file changed'):
        media.load_training_references(destination / 'references.json')
    vae.write_bytes(b'changed VAE')
    with pytest.raises(ValueError, match='decoder VAE differs'):
        media.prepare_training_references(session, None, source, tmp_path, 'white', 2, 42)


def test_saved_decoder_session_has_no_prompt_or_transformer_work(monkeypatch):
    from types import SimpleNamespace

    from scripts.onestep_avatar import media
    from scripts.prune.core import preflight, session
    from scripts.prune.data import prompt_cache

    model = SimpleNamespace(key='2.5')
    calls = []

    def check(key, *, gpu_id):
        calls.append((key, gpu_id))
        return model

    def forbidden(*_args, **_kwargs):
        pytest.fail('saved decoder requested model or text execution')

    monkeypatch.setattr(preflight, 'check', check)
    monkeypatch.setattr(prompt_cache, 'get_or_build', forbidden)
    monkeypatch.setattr(session, 'open_session', forbidden)
    monkeypatch.setattr(session.Session, 'transformer', forbidden)
    result = media.open_decoder_session('2.5', 4, script='saved acceptance')
    assert calls == [('2.5', 4)]
    assert result.model is model and result.device == torch.device('cuda:4')
    assert result.script == 'saved acceptance' and result.context is None


def test_saved_decoder_session_preserves_preflight_failure(monkeypatch):
    from scripts.onestep_avatar import media
    from scripts.prune.core import preflight, session

    def unavailable(*_args, **_kwargs):
        raise SystemExit('requested device is unavailable')

    monkeypatch.setattr(preflight, 'check', unavailable)
    monkeypatch.setattr(session, 'Session', lambda *_a, **_k: pytest.fail('constructed unchecked session'))
    with pytest.raises(SystemExit, match='unavailable'):
        media.open_decoder_session('2.5', 4, script='saved acceptance')


@pytest.fixture
def reference_cli_inputs(tmp_path, monkeypatch):
    """Real checked-master/producer flow; only pixel readers and VAE are controlled."""
    import json
    from contextlib import nullcontext
    from types import SimpleNamespace

    from scripts.onestep_avatar import media
    from scripts.onestep_avatar.corpus import dataset, precompute, subset  # noqa: PLC0415 -- same lazy caller scope
    from scripts.onestep_avatar.hashing import sha256
    from scripts.onestep_avatar.model import backbone

    view = tmp_path / 'actor/view'
    view.mkdir(parents=True)
    for name in ('rgb.mp4', 'bbox.npy', dataset.CAPTURE_MASK_NAME):
        (view / name).write_bytes(name.encode())
    vae = tmp_path / 'vae.safetensors'
    vae.write_bytes(b'controlled VAE')
    encoding = {
        'source': 'actor/view', 'objective': 'white', 'fps': 30,
        'vae_fingerprint': precompute.file_fingerprint(vae), 'edge': 32,
        'box_xyxy': [0, 0, 32, 32], 'pixel_frames': 9,
        'encode_contract_version': 1,
    }
    capture = view / dataset.capture_bundle_name('white')
    torch.save({'schema_version': 2, 'master': torch.zeros(128, 2, 1, 1), **encoding}, capture)
    source = {
        'relative_dir': 'actor/view', 'actor': 'actor', 'split': 'train',
        'shape': [128, 2, 1, 1], 'fps': 30, 'n_latent_frames': 2,
        'capture_encode_record': encoding, 'capture_latent_sha256': sha256(capture),
        'rgb_sha256': sha256(view / 'rgb.mp4'), 'guide_sha256': None,
    }
    membership = {
        'schema_version': 2, 'kind': subset.KIND, 'objective': 'white',
        'corpus_root': str(tmp_path), 'sources': [source], 'splits': {'train': ['actor']},
    }
    membership['sha256'] = subset.membership_hash(membership)
    path = tmp_path / 'membership.json'
    path.write_text(json.dumps(membership))
    model = SimpleNamespace(paths=SimpleNamespace(video_vae=lambda: vae))
    session = SimpleNamespace(model=model, decoder=lambda: nullcontext(None))
    pixels = torch.zeros(9, 3, 32, 32)
    monkeypatch.setattr(backbone, 'resolve', lambda *a: model)
    monkeypatch.setattr(media, 'recorded_capture_rgb', lambda *a: pixels)
    monkeypatch.setattr(media, 'recorded_guide_rgb', lambda *a: pytest.fail('capture-only D0 read guide RGB'))
    monkeypatch.setattr(media, 'decode', lambda *a: pixels)
    monkeypatch.setattr(media, 'open_decoder_session', lambda *a, **kw: session)
    output = tmp_path / 'references'
    arguments = ['--prepare-training-references', '--subset', str(path), '--source', 'actor/view',
                 '--encoded-frames', '2', '--guide-mode', 'd0', '--gpu-id', '4', '--output', str(output)]
    return arguments, source, session, view, path, output


def test_reference_cli_capture_only_roundtrip(reference_cli_inputs, monkeypatch):
    import json

    from scripts.onestep_avatar import media
    arguments, source, session, view, path, output = reference_cli_inputs
    assert media.main(arguments) == 0
    panels, producer = media.load_training_references(output / 'references.json')
    assert [p.title for p in panels] == ['Capture RGB', 'VAE-decoded capture', 'Guide RGB']
    assert panels[2].pixels is None and panels[2].missing_reason == 'Guide not used'
    assert producer['guide_rgb_sha256'] is None
    assert not (output / 'guide.pt').exists()
    with pytest.raises(ValueError, match='already used'):
        media.main(arguments)
    metadata = json.loads((output / 'references.json').read_text())
    metadata['panels'][2]['missing_reason'] = 'Missing'
    (output / 'references.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='only an unused D0 guide'):
        media.load_training_references(output / 'references.json')


@pytest.mark.parametrize('change', ['d1', 'range', 'unknown', 'changed_capture', 'broken_optional_guide', 'vae'])
def test_reference_cli_refusals_before_decoder(reference_cli_inputs, monkeypatch, change):
    import json

    from scripts.onestep_avatar import media
    from scripts.onestep_avatar.corpus import dataset, subset  # noqa: PLC0415 -- same lazy caller scope
    arguments, source, session, view, path, output = reference_cli_inputs
    if change in ('d1', 'range', 'unknown'):
        flag, value = {'d1': ('--guide-mode', 'd1'), 'range': ('--encoded-frames', '3'),
                       'unknown': ('--source', 'missing/view')}[change]
        arguments[arguments.index(flag) + 1] = value
    elif change == 'changed_capture':
        (view / dataset.capture_bundle_name('white')).write_bytes(b'changed')
    elif change == 'broken_optional_guide':
        membership = json.loads(path.read_text())
        membership['sources'][0]['guide_sha256'] = 'a' * 64
        membership['sha256'] = subset.membership_hash(membership)
        path.write_text(json.dumps(membership))
    else:
        session.model.paths.video_vae().write_bytes(b'changed VAE')
    monkeypatch.setattr(media, 'open_decoder_session', lambda *a, **kw: pytest.fail('invalid inputs opened a decoder'))
    with pytest.raises(ValueError):
        media.main(arguments)
    assert not output.exists()


@pytest.mark.parametrize('changed_input', ['membership', 'matte'])
def test_reference_cli_refuses_changed_inputs_before_publication(reference_cli_inputs, monkeypatch, changed_input):
    from scripts.onestep_avatar import media
    from scripts.onestep_avatar.corpus import dataset  # noqa: PLC0415 -- same lazy caller scope
    arguments, source, session, view, path, output = reference_cli_inputs

    def decode(*_args):
        selected = path if changed_input == 'membership' else view / dataset.CAPTURE_MASK_NAME
        selected.write_bytes(b'changed during decode')
        return torch.zeros(9, 3, 32, 32)

    monkeypatch.setattr(media, 'decode', decode)
    with pytest.raises(ValueError, match='inputs changed during preparation'):
        media.main(arguments)
    assert not output.exists()


def test_reference_helper_requires_d1_guide_before_decode(reference_cli_inputs, monkeypatch):
    from scripts.onestep_avatar import media
    arguments, source, session, view, path, output = reference_cli_inputs
    monkeypatch.setattr(media, 'decode', lambda *a: pytest.fail('missing required guide reached VAE'))
    with pytest.raises(ValueError, match='D1 preview requires a checked guide'):
        media.prepare_training_references(session, None, source, view.parents[1], 'white', 2, 42)
