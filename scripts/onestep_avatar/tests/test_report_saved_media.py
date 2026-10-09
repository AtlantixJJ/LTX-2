"""Report refresh consumes exact saved media and cannot launch missing renders."""

import copy
import hashlib
import importlib.util
import json
import sys
import types

import pytest

from scripts.onestep_avatar import WORKSPACE_ROOT


@pytest.fixture
def reader(tmp_path, monkeypatch):
    for name in ('aggregate', 'plots', 'report'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    path = WORKSPACE_ROOT / 'expr/onestep_avatar/dev_training_20261001/code/stage2_update.py'
    spec = importlib.util.spec_from_file_location('saved_stage2_report_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    return module


def saved(reader, *, current=False):
    root = reader.ROOT
    spec_path = root / 'configs/render/d0w_corpus_r16.json'
    folder = root / 'media/videos/d0w_corpus_r16'
    spec_path.parent.mkdir(parents=True)
    folder.mkdir(parents=True)
    panels = []
    for index in range(4):
        latent = root / f'panel_{index}.pt'
        latent.write_bytes(f'original latent {index}'.encode())
        panels.append({'title': f'panel {index}', 'latent': ('capture:' if index == 0 else '') + str(latent)})
    comparisons = [{'name': 'case', 'view': str(root/'source_view'), 'caption': 'exact original caption',
                    'panels': panels}]
    spec_path.write_text(json.dumps({'comparisons': comparisons}))
    rows = copy.deepcopy(comparisons)
    row = rows[0]
    row.update(frames=129, video='case.mp4', poster='case_poster.png')
    for field in ('video', 'poster'):
        (folder/row[field]).write_bytes(f'completed {field}'.encode())
    if current:
        row['rendering'] = {'fps': 30, 'source_frames': list(range(129)),
                            'panels': [{'row': i//3, 'column': i%3,
                                        'role': f'panel_{i}' if i<4 else 'unused',
                                        'title': f'panel {i}' if i<4 else 'Unused'} for i in range(6)],
                            'outputs': {field: {'path': str(folder/row[field]),
                                               'sha256': reader.file_hash(folder/row[field])}
                                        for field in ('video', 'poster')}}
        row['inputs'] = [{'path': str(root/f'panel_{i}.pt'), 'sha256': reader.file_hash(root/f'panel_{i}.pt')}
                         for i in range(4)]
    manifest = {'spec': str(spec_path), 'results': rows}
    path = folder/'render_manifest.json'
    path.write_text(json.dumps(manifest))
    if not current:
        audit = {'schema_version':1, 'manifest_sha256':reader.file_hash(path),
                 'comparisons':{'case':{'layout':{'rows':2,'columns':3,
                                                'panel_titles':[p['title'] for p in panels]},
                                        'video':{'sha256':reader.file_hash(folder/row['video']), 'frames':129,'fps':30},
                                        'poster':{'sha256':reader.file_hash(folder/row['poster'])}}}}
        (root/'configs/stage2_media_evidence.json').write_text(json.dumps(audit))
    return spec_path, path


@pytest.mark.parametrize('current', [False, True])
def test_saved_reader_uses_spec_names_and_actual_layout(reader, current):
    spec, manifest = saved(reader, current=current)
    before = (spec.read_bytes(), manifest.read_bytes())
    # Unrelated media is not silently promoted into the comparison.
    (manifest.parent/'other.mp4').write_bytes(b'unrelated')
    videos = reader.read_saved_comparisons(reader.RUNS[0])
    assert len(videos) == 1 and videos[0]['video'].endswith('/case.mp4')
    assert '2 by 3 panels' in videos[0]['caption']
    assert ('hashes verified' if current else 'no media/input hashes') in videos[0]['caption']
    assert before == (spec.read_bytes(), manifest.read_bytes())


@pytest.mark.parametrize('defect', ['manifest', 'video', 'poster', 'inventory', 'panels', 'view', 'caption',
                                  'frames', 'video_hash', 'input_hash', 'fps', 'source_frames', 'aliases', 'render_titles'])
def test_missing_or_changed_results_fail_before_report_writes(reader, monkeypatch, defect):
    _, manifest_path = saved(reader, current=True)
    manifest = json.loads(manifest_path.read_text())
    row = manifest['results'][0]
    if defect in ('video', 'poster'):
        (manifest_path.parent/row[defect]).unlink()
    elif defect == 'inventory':
        manifest['results'] = []
    elif defect == 'panels':
        row['panels'].reverse()
    elif defect in ('view', 'caption'):
        row[defect] = 'changed'
    elif defect == 'frames':
        row['frames'] = 128
    elif defect == 'video_hash':
        (manifest_path.parent/row['video']).write_bytes(b'changed')
    elif defect == 'input_hash':
        (reader.ROOT/'panel_0.pt').write_bytes(b'changed')
    elif defect in ('fps', 'source_frames'):
        row['rendering'][defect] = 24 if defect == 'fps' else list(range(128))
    elif defect == 'aliases':
        manifest['comparisons'] = []
    elif defect == 'render_titles':
        row['rendering']['panels'][0]['title'] = 'changed'
    manifest_path.write_text(json.dumps(manifest))
    if defect == 'manifest':
        manifest_path.unlink()

    def forbidden():
        raise AssertionError('report mutated state before checking saved results')

    monkeypatch.setattr(reader.aggregate, 'main', forbidden, raising=False)
    monkeypatch.setattr(reader, 'read_metrics', lambda *args: {'completed_final_metrics': True})
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in reader.ROOT.rglob('*') if path.is_file()}
    with pytest.raises((ValueError, FileNotFoundError)):
        reader.refresh()
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in reader.ROOT.rglob('*') if path.is_file()}
    assert before == after


@pytest.mark.parametrize('defect',[None,'missing','font','pixels','video','times',
                                  'hash_missing','hash_null','hash_number','hash_short','hash_upper','hash_nonhex'])
def test_schema_three_reader_exposes_checked_compact_format(reader,defect):
    _,path=saved(reader,current=True)
    manifest=json.loads(path.read_text());manifest['schema_version']=3
    row=manifest['results'][0]
    for panel in row['rendering']['panels']:
        panel.update(value='',source_frames=list(range(129)),pixels_sha256='a'*64)
    row['rendering']['poster_frame']=96
    record=copy.deepcopy(row['rendering']);record.update(viewing_width=480,font_size=28,display_size=[824,1028],
                                                    source_times=[i/30 for i in range(129)])
    record['panels']=[{**p,'row':i//2,'column':i%2} for i,p in enumerate(record['panels'][:4])]
    compact={'video':'case_compact.mp4','poster':'case_compact_poster.png','rendering':record}
    for field in ('video','poster'):
        asset=path.parent/compact[field];asset.write_bytes(b'saved compact media')
        record['outputs'][field]={'path':str(asset),'sha256':reader.file_hash(asset)}
    row['compact']=compact
    if defect=='missing': del row['compact']
    if defect=='font': record['font_size']=1
    if defect=='pixels': record['panels'][0]['pixels_sha256']='b'*64
    if defect=='video': (path.parent/compact['video']).write_bytes(b'changed')
    if defect=='times': record['source_times'][0]=9
    if defect and defect.startswith('hash_'):
        invalid = {'hash_null': None, 'hash_number': 123, 'hash_short': 'a'*63,
                   'hash_upper': 'A'*64, 'hash_nonhex': 'g'*64}
        for rendering in (row['rendering'], record):
            if defect == 'hash_missing':
                del rendering['panels'][0]['pixels_sha256']
            else:
                rendering['panels'][0]['pixels_sha256'] = invalid[defect]
    path.write_text(json.dumps(manifest));original=path.read_bytes()
    if defect:
        with pytest.raises(ValueError,match='compact|pixel hash'): reader.read_saved_comparisons(reader.RUNS[0])
    else:
        videos=reader.read_saved_comparisons(reader.RUNS[0])
        assert len(videos)==2 and videos[1]['video'].endswith('case_compact.mp4')
        assert '480-pixel' in videos[1]['caption']
    assert path.read_bytes()==original
