"""Saved-comparison jobs bind commands, inputs, decoder identity and actual media."""

import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar.execution import queue
from scripts.onestep_avatar.tests.test_saved_comparisons import (
    test_saved_study_spec_renders_master_and_output_with_report_metrics as render_fixture,
)


@pytest.fixture
def rendered(tmp_path, monkeypatch):
    render_fixture(tmp_path, monkeypatch)
    job = {'id':'saved_render', 'kind':'render', 'arguments':['--render-saved-comparisons','spec.json',
           '--output','rendered','--seed','99'], 'output':'rendered','dependencies':[],
           'completion':{'manifest':'rendered/render_manifest.json'}}
    path=tmp_path/'queue_jobs.json'
    path.write_text(json.dumps({'schema_version':1,'jobs':[job]}))
    return queue.prepare_jobs(path)[0]


def test_render_command_and_receipt_use_package_and_verified_manifest(rendered, tmp_path):
    command, environment=queue.job_command(rendered,(4,))
    assert command[1:3]==['-m','scripts.onestep_avatar.comparisons']
    assert command[-2:]==['--gpu-id','0'] and environment=={'CUDA_VISIBLE_DEVICES':'4'}
    assert queue.verify_completion(rendered)
    receipt=queue.completion_receipt(rendered)
    assert receipt['evidence'][0]['path']==str(tmp_path/'rendered/render_manifest.json')
    assert queue.verify_receipt(rendered,receipt)


@pytest.mark.parametrize('defect',['spec','seed','decoder','inventory','request','input','video','poster',
                                  'frames','fps','source_frames','titles','layout','media_path','aliases',
                                  'software','question','poster_frame','role','value','panel_frames','vae_bytes',
                                  'positions','panel_size','viewing_width'])
def test_render_completion_rejects_changed_evidence(rendered,tmp_path,defect):
    path=tmp_path/'rendered/render_manifest.json'
    original=json.loads(path.read_text())
    row=original['comparisons'][0]
    if defect=='spec':
        (tmp_path/'spec.json').write_text((tmp_path/'spec.json').read_text()+'\n')
        with pytest.raises(ValueError,match='specification changed'):
            queue.job_command(rendered,(4,))
    elif defect=='seed':
        original['seed']=42
    elif defect=='decoder':
        original['decoder']['vae_sha256']='0'*64
    elif defect=='inventory':
        original['comparisons']=[]
    elif defect=='request':
        row['caption']='changed'
    elif defect=='input':
        # Keep the input readable so this proves changed tensor identity rather
        # than depending on the current Torch unpickler's corrupt-byte exception.
        value=torch.load(tmp_path/'generated.pt',weights_only=True)
        torch.save(value+1,tmp_path/'generated.pt')
    elif defect in ('video','poster'):
        (tmp_path/'rendered'/row[defect]).write_bytes(b'changed')
    elif defect=='frames':
        row['frames']=8
    elif defect=='fps':
        row['rendering']['fps']=24
    elif defect=='source_frames':
        row['rendering']['source_frames']=list(range(8))
    elif defect=='titles':
        row['rendering']['panels'][0]['title']='changed'
    elif defect=='layout':
        row['rendering']['layout']='inference'
    elif defect=='media_path':
        row['rendering']['outputs']['video']['path']=str(tmp_path/'outside.mp4')
    elif defect=='software':
        original['source_code_sha256']['evaluate']='0'*64
    elif defect in ('question','poster_frame'):
        row['rendering'][defect]='changed' if defect=='question' else 0
    elif defect in ('role','value','panel_frames'):
        row['rendering']['panels'][0]['source_frames' if defect=='panel_frames' else defect]=[] if defect=='panel_frames' else 'changed'
    elif defect=='vae_bytes':
        (tmp_path/'vae.safetensors').write_bytes(b'changed decoder')
    elif defect=='positions':
        row['rendering']['panels'][0]['column']=1
    elif defect in ('panel_size','viewing_width'):
        row['rendering'][defect]=[100,100] if defect=='panel_size' else 480
    else:
        original['results']=[]
    path.write_text(json.dumps(original))
    with pytest.raises((ValueError, EOFError, IndexError)):
        queue.verify_completion(rendered)


@pytest.mark.parametrize('field',['manifest','video','poster','input'])
def test_render_is_incomplete_when_artifacts_are_missing(rendered,tmp_path,field):
    manifest=tmp_path/'rendered/render_manifest.json'
    row=json.loads(manifest.read_text())['comparisons'][0]
    path=manifest if field=='manifest' else tmp_path/'generated.pt' if field=='input' else tmp_path/'rendered'/row[field]
    path.unlink()
    assert not queue.verify_completion(rendered)


def test_missing_render_inputs_wait_without_gpu_queries_or_claims(rendered,tmp_path,monkeypatch):
    (tmp_path/'generated.pt').unlink()
    state=queue.read_queue_state(tmp_path/'state.json',[rendered])
    monkeypatch.setattr(subprocess,'run',lambda *_args,**_kwargs:pytest.fail('waiting render queried GPUs'))
    assert not queue.dispatch_ready([rendered],state,tmp_path/'state.json',tmp_path/'claims')
    assert not (tmp_path/'claims').exists() and not (tmp_path/'state.json').exists()


def test_render_spec_bytes_are_in_canonical_job_identity(rendered,tmp_path):
    (tmp_path/'spec.json').write_text((tmp_path/'spec.json').read_text()+'\n')
    changed=queue.prepare_jobs(tmp_path/'queue_jobs.json')[0]
    assert changed['sha256']!=rendered['sha256']


def test_only_one_render_spec_and_no_gpu_override_are_accepted(rendered,tmp_path):
    jobs_path=tmp_path/'queue_jobs.json'
    original=json.loads(jobs_path.read_text())
    original['jobs'][0]['arguments'] += ['--render-saved-comparisons','spec.json']
    jobs_path.write_text(json.dumps(original))
    with pytest.raises(ValueError,match='one explicit specification'):
        queue.prepare_jobs(jobs_path)
    original['jobs'][0]['arguments']=rendered['arguments']+['--gpu-id','0']
    original['jobs'][0]['output']=rendered['output']
    jobs_path.write_text(json.dumps(original))
    with pytest.raises(ValueError,match='forbidden execution override'):
        queue.prepare_jobs(jobs_path)


def test_render_dispatch_claims_one_device_and_publishes_verified_receipt(rendered,tmp_path,monkeypatch, controlled_queue_launch):
    from scripts.onestep_avatar.execution import process_registry
    output=tmp_path/'rendered'
    prepared=tmp_path/'held_render'
    output.rename(prepared)
    claims=tmp_path/'claims'
    claims.mkdir()
    (claims/'5').write_text('legacy reservation')
    ledger=claims/'processes.json'
    original_identity=process_registry._identity
    monkeypatch.setattr(process_registry,'_identity',lambda pid:(
        {'pid':pid,'start_ticks':1,'command':['controlled child'],'terminal':True}
        if pid==99999999 else original_identity(pid)))
    state_path=tmp_path/'state.json'
    state=queue.read_queue_state(state_path,[rendered])
    monkeypatch.setattr(subprocess,'run',lambda *_args,**_kwargs:SimpleNamespace(
        stdout='\n'.join(f'{gpu}, 0' for gpu in range(8))))

    def launch(command,**kwargs):
        assert command[1:3]==['-m','scripts.onestep_avatar.comparisons']
        assert command[-2:]==['--gpu-id','0'] and kwargs['env']['CUDA_VISIBLE_DEVICES']=='5'
        own=json.loads(ledger.read_text())['attempts'][kwargs['env']['ONESTEP_AVATAR_QUEUE_TOKEN']]
        assert own['job']=='saved_render' and own['gpus']==[5]
        assert (claims/'5').read_text()=='legacy reservation'
        prepared.rename(output)
        return SimpleNamespace(pid=99999999,poll=lambda:0,wait=lambda:0)

    monkeypatch.setattr(subprocess,'Popen',launch)
    assert queue.dispatch_ready([rendered],state,state_path,ledger)
    completed=json.loads(state_path.read_text())['jobs']['saved_render']
    assert completed['state']=='complete' and queue.verify_receipt(rendered,completed['receipt'])
    assert (claims/'5').read_text()=='legacy reservation' and not (claims/'4').exists()
