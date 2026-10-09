"""Saved study specs retain master coverage, report metrics and shared media ownership."""

import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import evaluate, media


def test_saved_study_spec_renders_master_and_output_with_report_metrics(tmp_path, monkeypatch):
    capture = torch.full((3, 4, 2, 2), 0.2)
    output = capture[:, :2].unsqueeze(0) + 0.1
    torch.save({'master': capture}, tmp_path / 'capture.pt')
    torch.save(output, tmp_path / 'generated.pt')
    spec = tmp_path / 'spec.json'
    comparison = {'name': 'matched', 'view': 'view', 'span': 2, 'fps': 30,
                  'caption': 'Matched source frames', 'panel_size': [80, 80], 'viewing_width': 192,
                  'panels': [{'title': 'Capture', 'latent': 'capture:capture.pt'},
                             {'title': 'Output', 'latent': 'generated.pt'}]}
    spec.write_text(json.dumps({'comparisons': [comparison]}))
    loaded = []

    def load_master(path):
        loaded.append(path)
        return torch.load(path, weights_only=True)['master'], 30

    decoded = []

    def decode(session, latent, decoder, seed):
        decoded.append((latent.clone(), seed))
        return latent[0, :, :1].permute(1, 0, 2, 3).repeat(9, 1, 1, 1)

    class Perceptual:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, left, right):
            return (left - right).square().mean(dim=(1, 2, 3), keepdim=True)

    import scripts.prune.core.session as sessions
    from scripts.prune.core import model_registry

    vae = tmp_path / 'vae.safetensors'
    vae.write_bytes(b'controlled decoder identity')
    monkeypatch.setattr(model_registry, 'resolve', lambda *_args: SimpleNamespace(
        paths=SimpleNamespace(video_vae=lambda: vae), scale_factors=SimpleNamespace(time=8)))

    monkeypatch.setattr(evaluate.dataset, 'load_training_master', load_master)
    monkeypatch.setattr(media, 'decode', decode)
    monkeypatch.setattr(sessions, 'open_session', lambda *_a, **_k: pytest.fail('saved render prepared text'))
    monkeypatch.setattr(media, 'open_decoder_session', lambda *_args, **_kwargs: SimpleNamespace(
        device=torch.device('cpu'), decoder=lambda: nullcontext(None)))
    monkeypatch.setitem(sys.modules, 'lpips', SimpleNamespace(LPIPS=lambda **_kwargs: Perceptual()))
    result = evaluate.render_saved_comparisons(spec, tmp_path / 'rendered', gpu_id=0, seed=99)
    row = result['results'][0]
    assert loaded == [tmp_path / 'capture.pt']
    assert decoded[0][0].shape == (1, 3, 2, 2, 2)
    assert [seed for _, seed in decoded] == [99, 99]
    assert row['frames'] == 9 and row['caption'] == comparison['caption']
    assert row['metrics'][0]['psnr_full'] is None
    assert row['metrics'][0]['lpips'] == 0
    assert row['metrics'][1]['psnr_full'] == pytest.approx(20, abs=1e-5)
    assert row['metrics'][1]['lpips'] == pytest.approx(0.04, abs=1e-6)
    assert row['rendering']['poster_frame'] == 8
    assert row['rendering']['layout'] == 'comparison'
    assert len(row['inputs']) == 2 and all(record['sha256'] for record in row['inputs'])
    assert (tmp_path / 'rendered' / row['video']).is_file()
    assert (tmp_path / 'rendered' / row['poster']).is_file()
    assert evaluate.verify_saved_comparison_completion(spec, tmp_path / 'rendered', seed=99)
    from scripts.onestep_avatar.execution import software

    original = software.sha256
    decoder_owner = software.LTX_ROOT / 'packages/ltx-core/src/ltx_core/model/video_vae/conv_video_decoder.py'
    assert str(decoder_owner.relative_to(software.LTX_ROOT)) in result['software']['sources']
    with monkeypatch.context() as changed_owner:
        changed_owner.setattr(software, 'sha256', lambda p: 'f'*64 if p == decoder_owner else original(p))
        software.validate(result['software'])
        with pytest.raises(ValueError, match='completion identity differs'):
            evaluate.verify_saved_comparison_completion(spec, tmp_path / 'rendered', seed=99)


@pytest.mark.parametrize('name', ['../escape', '..', '/outside'])
def test_saved_comparison_names_cannot_escape_output(tmp_path, name):
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'comparisons': [{'name': name, 'panels': []}]}))
    with pytest.raises(ValueError, match='safe path component'):
        evaluate.render_saved_comparisons(spec, tmp_path / 'out', gpu_id=0)
    assert not (tmp_path / 'out').exists()


def test_saved_comparison_rejects_mismatched_geometry_before_session(tmp_path, monkeypatch):
    for name, height in [('a', 2), ('b', 3)]:
        torch.save(torch.zeros(1, 3, 2, height, 2), tmp_path / f'{name}.pt')
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'comparisons': [{'name': 'case', 'span': 2, 'panels': [
        {'title': name, 'latent': f'{name}.pt'} for name in ['a', 'b']]}]}))
    def forbidden(*_args, **_kwargs):
        pytest.fail('invalid geometry opened a model session')

    monkeypatch.setattr(media, 'open_decoder_session', forbidden)
    with pytest.raises(ValueError, match='identical encoded geometry'):
        evaluate.render_saved_comparisons(spec, tmp_path / 'out', gpu_id=0)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('failure', ['short', 'fps', 'nonfinite'])
def test_master_input_refuses_wrong_coverage_fps_and_values(tmp_path, monkeypatch, failure):
    path = tmp_path / 'capture.pt'
    path.write_bytes(b'fixture')
    master = torch.zeros(3, 4, 2, 2)
    if failure == 'nonfinite':
        master[0, 0, 0, 0] = float('nan')
    monkeypatch.setattr(evaluate.dataset, 'load_training_master', lambda _path: (
        master, 24 if failure == 'fps' else 30))
    span = 5 if failure == 'short' else 2
    message = {'short': 'insufficient frame coverage', 'fps': 'fps differs', 'nonfinite': 'finite floating'}[failure]
    with pytest.raises(ValueError, match=message):
        evaluate._saved_panel_input({'latent': 'capture:capture.pt'}, tmp_path, span, 30)


def test_five_panel_comparison_preserves_order_and_replays_exactly():
    panels = [media.Panel(f'p{index}', f'Arm {index}', torch.full((2, 3, 16, 8), index / 5), (0, 1))
              for index in range(5)]
    pixels, record = media.render_panels(
        panels, question='Compare arms', layout='comparison', fps=30,
        panel_size=(80, 80), viewing_width=288,
    )
    assert [(item['role'], item['row'], item['column']) for item in record['panels']] == [
        ('p0', 0, 0), ('p1', 0, 1), ('p2', 0, 2), ('p3', 1, 0), ('p4', 1, 1), ('unused', 1, 2),
    ]
    assert record['padding'] == 'aspect-preserving contain with neutral padding'
    rebuilt, rebuilt_record = media.render_from_record(panels, record)
    assert torch.equal(pixels, rebuilt)
    assert rebuilt_record == record


def test_original_corpus_labels_select_two_columns_at_narrow_width():
    titles=['Capture (VAE decode)','Frozen dev D0','One clip, rank 16, u200','Corpus, rank 16, u700']
    metadata=[media.Panel(f'p{i}',title,None,()) for i,title in enumerate(titles)]
    assert media.compact_layout(metadata,question='corpus_r16_sigma0854',layout='comparison',panel_size=(400,400))=='compact_comparison'
    geometry=media.layout_geometry(metadata,question='corpus_r16_sigma0854',layout='compact_comparison',panel_size=(400,400))
    assert geometry['font_size']==28
    assert geometry['display_size'][0]==824
    assert geometry['font_size']*480/824>=16
    panels=[media.Panel(p.role,p.title,torch.full((2,3,24,16),i/4),(0,1)) for i,p in enumerate(metadata)]
    pixels,record=media.render_panels(panels,question='corpus_r16_sigma0854',layout='compact_comparison',fps=30,
                                     panel_size=(400,400))
    assert [(p['row'],p['column'],p['title']) for p in record['panels']]==[
        (0,0,titles[0]),(0,1,titles[1]),(1,0,titles[2]),(1,1,titles[3])]
    rebuilt,rebuilt_record=media.render_from_record(panels,record)
    assert torch.equal(pixels,rebuilt) and rebuilt_record==record


def test_narrow_layout_stacks_long_exact_labels_and_refuses_unfit_text():
    titles=['A long panel title that needs the full width','Another long label for the matching comparison']
    metadata=[media.Panel(f'p{i}',title,None,()) for i,title in enumerate(titles)]
    assert media.compact_layout(metadata,question='Compare results',layout='comparison',panel_size=(400,400))=='stacked_comparison'
    metadata[0]=media.Panel('p0','x'*200,None,())
    with pytest.raises(ValueError,match='no readable compact layout'):
        media.compact_layout(metadata,question='Compare results',layout='comparison',panel_size=(400,400))


def test_unreadable_saved_labels_fail_before_any_decoder_or_output(tmp_path,monkeypatch):
    torch.save(torch.zeros(1,3,2,2,2),tmp_path/'sample.pt')
    spec=tmp_path/'spec.json'
    spec.write_text(json.dumps({'comparisons':[{'name':'case','span':2,'panels':[
        {'title':'x'*200,'latent':'sample.pt'}]}]}))
    monkeypatch.setattr(media,'open_decoder_session',lambda *_a,**_k: pytest.fail('unreadable text opened VAE'))
    with pytest.raises(ValueError,match='title does not fit|no readable compact layout'):
        evaluate.render_saved_comparisons(spec,tmp_path/'out',gpu_id=0)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('changed',['missing','layout','font','size','times','title','pixels','video','poster',
                                   'hash_missing','hash_null','hash_number','hash_short','hash_upper','hash_nonhex'])
def test_compact_completion_is_required_and_bound_to_full_pixels(tmp_path,monkeypatch,changed):
    test_saved_study_spec_renders_master_and_output_with_report_metrics(tmp_path,monkeypatch)
    spec=tmp_path/'spec.json'; destination=tmp_path/'rendered'; path=destination/'render_manifest.json'
    manifest=json.loads(path.read_text()); row=manifest['comparisons'][0]; compact=row['compact']
    if changed=='missing': del row['compact']
    if changed=='layout': compact['rendering']['layout']='stacked_comparison'
    if changed=='font': compact['rendering']['font_size']=1
    if changed=='size': compact['rendering']['display_size'][0]+=2
    if changed=='times': compact['rendering']['source_times'][1]+=1
    if changed=='title': compact['rendering']['panels'][0]['title']='Other'
    if changed=='pixels': compact['rendering']['panels'][0]['pixels_sha256']='0'*64
    if changed in ('video','poster'): (destination/compact[changed]).write_bytes(b'changed media')
    if changed.startswith('hash_'):
        invalid = {'hash_null': None, 'hash_number': 123, 'hash_short': 'a'*63,
                   'hash_upper': 'A'*64, 'hash_nonhex': 'g'*64}
        for rendering in (row['rendering'], compact['rendering']):
            if changed == 'hash_missing':
                del rendering['panels'][0]['pixels_sha256']
            else:
                rendering['panels'][0]['pixels_sha256'] = invalid[changed]
    manifest['results']=manifest['comparisons']; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='compact|settings differ|readability|titles differ|hash or path differs|pixel hash'):
        evaluate.verify_saved_comparison_completion(spec,destination,seed=99)
