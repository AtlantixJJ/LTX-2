"""Prepared fixed tensors are consumed unchanged by the actual preview readers."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import evaluate, media, prepare_inputs
from scripts.onestep_avatar.corpus import dataset, subset
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.tests.test_subset import old_subset  # noqa: F401 -- fixture dependency
from scripts.onestep_avatar.tests.test_training_preflight import checked_settings  # noqa: F401
from scripts.onestep_avatar.training import engine


@pytest.fixture
def preview_case(checked_settings, monkeypatch, tmp_path):
    settings, membership = checked_settings
    spec=evaluate.backbone.resolve('2.5','dev')
    Path(spec.paths.video_vae()).write_bytes(b'controlled VAE')
    frames=tuple(range(49))
    pixels=torch.zeros(49,3,16,16,dtype=torch.uint8)
    source=membership['sources'][0]
    calls=[]
    monkeypatch.setattr(prepare_inputs.preflight,'check',lambda *_a,**_k:calls.append('preflight'))
    import scripts.prune.data.prompt_cache as prompts
    def text(_spec,prompt,*_a):
        calls.append(prompt)
        return torch.ones(1,3,4,dtype=torch.bfloat16)*(2 if prompt=='negative' else 1)
    monkeypatch.setattr(prompts,'get_or_build',text)
    monkeypatch.setattr(prepare_inputs,'torch',SimpleNamespace(**{**vars(torch),'device':lambda *_a:torch.device('cpu')}))
    def setup(mode='bidirectional',task='d0',extra=()):
        if task=='d1':
            for row in membership['sources']:
                view=Path(membership['corpus_root'])/row['relative_dir']
                render=view/dataset.render_name('white')
                render.write_bytes(b'controlled guide'+row['relative_dir'].encode())
                sidecar=view/dataset.render_metadata_name('white')
                sidecar.write_text(json.dumps({'objective':'white','compositing_version':dataset.GUIDE_COMPOSITING_VERSION}))
                record=torch.load(view/dataset.capture_bundle_name('white'),weights_only=True)
                record['input_fingerprint']=sha256(render)
                guide=view/dataset.guide_bundle_name('white')
                torch.save(record,guide)
                row.update(guide_sha256=sha256(render),guide_latent_sha256=sha256(guide),guide_sidecar_sha256=sha256(sidecar),
                           guide_encode_record={k:v for k,v in record.items() if k not in ('schema_version','master')})
            membership['sha256']=subset.membership_hash(membership)
            settings.subset.write_text(json.dumps(membership))
        panels=[media.Panel('recorded','Capture RGB',pixels,frames),media.Panel('decoded','VAE-decoded capture',pixels,frames),
                media.Panel('guide','Guide RGB',pixels if task=='d1' else None,frames,
                            missing_reason='' if task=='d1' else 'Guide not used')]
        producer={'source':source['relative_dir'],'objective':'white','fps':30,'source_frames':list(frames),
                  'capture_encoding_sha256':source['capture_latent_sha256'],'guide_rgb_sha256':source.get('guide_sha256'),
                  'vae_sha256':sha256(Path(spec.paths.video_vae())),
                  'panels':[{'role':p.role,'pixels_sha256':media._pixel_identity(p.pixels)} for p in panels]}
        references=tmp_path/(mode+'_'+task+'_refs')
        media.save_training_references(panels,producer,references)
        command=['preview','--references',str(references/'references.json'),'--output',str(tmp_path/(mode+'_'+task+'_fixed')),
                 '--gpu-id','3','--evaluation-arguments','--mode',mode,'--subset',str(settings.subset),
                 '--source',source['relative_dir'],'--guide-mode',task,'--variant','dev','--span-latent-frames','7',
                 '--schedule','0.725','0',*extra]
        return prepare_inputs.parse_args(command)
    return setup,settings,calls


@pytest.mark.parametrize('mode',['bidirectional','causal'])
@pytest.mark.parametrize('task',['d0','d1'])
def test_prepared_inputs_consumed_by_actual_training_and_tensor_checker(preview_case,mode,task):
    setup,settings,_=preview_case
    args=setup(mode,task)
    fixed=prepare_inputs.prepare_preview(args)
    settings.mode=mode
    parsed=evaluate.parse_args([*fixed['evaluation_arguments'],'--output','/unused'])
    settings.mode_settings=parsed.mode_settings
    settings.guide_mode=task
    checked=engine.read_preview_inputs(args.output/'preview.json',settings)
    assert checked==fixed and ('guide' in fixed['input_files'])==(task=='d1')
    capture,_=dataset.load_training_master(Path(fixed['input_files']['capture']['path']))
    spec=evaluate.backbone.resolve('2.5','dev')
    from scripts.onestep_avatar.model import common
    grid=common.ClipGrid.build(7,64,64,30,spec,device=torch.device('cpu'),dtype=torch.bfloat16,latent_channels=2)
    tokens=grid.patchify(capture.unsqueeze(0).to(dtype=torch.bfloat16))
    tensors={role:torch.load(Path(identity['path']),weights_only=True) for role,identity in fixed['input_files'].items()
             if role not in ('subset','capture','guide')}
    tensors['capture']=tokens
    if task=='d1': tensors['guide']=tokens
    evaluate.verify_preview_tensors(fixed,tensors)
    assert torch.equal(tensors['first_image'],tokens[:,:grid.tokens_per_latent_frame])
    assert tensors['noise'].shape==(1,28,2) and tensors['noise'].dtype==torch.bfloat16
    assert not (args.output/'unexecuted').exists()
    assert all(Path(row['path']).is_absolute() for row in fixed['input_files'].values())


def test_guided_preview_pins_negative_text_and_rejects_missing_pin(preview_case):
    setup,settings,calls=preview_case
    args=setup(extra=['--cfg','2','--negative-prompt','negative'])
    fixed=prepare_inputs.prepare_preview(args)
    negative=torch.load(Path(fixed['input_files']['negative_text']['path']),weights_only=True)
    assert torch.equal(negative,torch.full_like(negative,2)) and 'negative' in calls
    del fixed['input_files']['negative_text']
    (args.output/'preview.json').write_text(json.dumps(fixed))
    with pytest.raises(ValueError,match='pinned negative text'):
        engine.read_preview_inputs(args.output/'preview.json',settings)


@pytest.mark.parametrize('extra',[['--source','other'],['--output','other'],['--gpu-id','0'],['--checkpoint','other']])
def test_invalid_selection_before_text_or_gpu(preview_case,extra):
    setup,_,calls=preview_case
    args=setup(extra=extra)
    with pytest.raises(ValueError): prepare_inputs.prepare_preview(args)
    assert not calls and not args.output.exists()


def test_wrong_reference_coverage_before_text(preview_case):
    setup,_,calls=preview_case
    args=setup(extra=['--span-latent-frames','5'])
    with pytest.raises(ValueError,match='coverage'):
        prepare_inputs.prepare_preview(args)
    assert not calls and not args.output.exists()


def test_preparation_non_tensor_inputs_remain_pinned(preview_case):
    setup,settings,_=preview_case
    args=setup()
    prepare_inputs.prepare_preview(args)
    settings.subset.write_text(settings.subset.read_text()+'\n')
    with pytest.raises(ValueError,match='file changed'):
        engine.read_preview_inputs(args.output/'preview.json',settings)


def test_guided_execution_loads_fixed_contexts_before_transformer(preview_case,monkeypatch):
    setup,_,calls=preview_case
    args=setup(extra=['--cfg','2','--negative-prompt','negative'])
    fixed=prepare_inputs.prepare_preview(args)
    import scripts.prune.core.session as sessions
    import scripts.prune.data.prompt_cache as prompts
    monkeypatch.setattr(prompts,'get_or_build',lambda *_a,**_k:pytest.fail('fixed text rebuilt'))
    monkeypatch.setattr(evaluate,'torch',SimpleNamespace(**{**vars(torch),'device':lambda *_a:torch.device('cpu')}))
    monkeypatch.setattr(sessions,'Session',lambda spec,device,*_a:SimpleNamespace(device=device))
    seen=[]
    def transformer(*_a,**_k):
        seen.append('transformer')
        raise RuntimeError('checked tensors reached transformer boundary')
    monkeypatch.setattr(evaluate.adapter_loader,'inference_transformer',transformer)
    command=evaluate.parse_args([*fixed['evaluation_arguments'],'--output',str(args.output/'raw')])
    command.preview_fixed=fixed
    with pytest.raises(RuntimeError,match='transformer boundary'):
        evaluate.execute_evaluation(command)
    assert seen==['transformer']


def test_causal_preparation_omitted_span_remains_null(preview_case):
    setup, settings, _calls = preview_case
    args = setup('causal', 'd1')
    index = args.evaluation_arguments.index('--span-latent-frames')
    del args.evaluation_arguments[index:index + 2]
    fixed = prepare_inputs.prepare_preview(args)
    assert '--span-latent-frames' not in fixed['evaluation_arguments']
    selected = evaluate.parse_args([*fixed['evaluation_arguments'], '--output', '/unused'])
    assert selected.mode_settings.span_latent_frames is None
    assert selected.output_latent_frames == 7
    settings.mode = 'causal'
    settings.mode_settings = selected.mode_settings
    settings.guide_mode = 'd1'
    assert engine.read_preview_inputs(args.output / 'preview.json', settings) == fixed


def test_causal_preparation_rejects_trimmed_explicit_span_before_text(preview_case):
    setup, settings, calls = preview_case
    args = setup('causal')
    membership = json.loads(settings.subset.read_text())
    for source in membership['sources']:
        path = Path(membership['corpus_root']) / source['relative_dir'] / dataset.capture_bundle_name('white')
        bundle = torch.load(path, weights_only=True)
        bundle['master'] = torch.zeros(2, 18, 2, 2)
        torch.save(bundle, path)
        source.update(n_latent_frames=18, shape=[2, 18, 2, 2], capture_latent_sha256=sha256(path))
    membership['sha256'] = subset.membership_hash(membership)
    settings.subset.write_text(json.dumps(membership))
    args.evaluation_arguments[args.evaluation_arguments.index('--span-latent-frames') + 1] = '8'
    with pytest.raises(SystemExit):
        prepare_inputs.prepare_preview(args)
    assert not calls
    assert not args.output.exists()
