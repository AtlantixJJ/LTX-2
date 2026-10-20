"""Decode A, selection, native and r3 on 36 prespecified held-out cases.

Run in ltx with --study and --gpu-id 4, 5 or 6. Waits for both fixed endpoint
inference result sets and a free assigned GPU. No training changes or retries.
See doc/evaluate/joint_media.md for metrics, source frames and limits.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import torch

from scripts.onestep_avatar import media
from scripts.onestep_avatar.corpus.dataset import load_training_master
from scripts.prune.core import provenance
from scripts.prune.data import foreground
from scripts.prune.evaluate import ffn, joint_final
from scripts.prune.evaluate.learned_head_media import pixel_metrics, region_lpips
from scripts.prune.evaluate.mobilewan_media import saved_video, source_pixels, verify_comparison
from scripts.prune.learn import checkpoints, heads
from scripts.prune.learn.joint import input_case


def read(path):
    return json.loads(Path(path).read_text())


def run(study, gpu, shard_index=0, shard_count=1):
    if not 0 <= shard_index < shard_count:
        raise ValueError('Invalid disjoint media shard')
    root = Path(study['root'])
    while not all((root/f'evaluation/test/{arm}/results.json').exists() for arm in ('selection','final')):
        time.sleep(20)
    while True:
        info = subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        if info not in active:
            break
        time.sleep(20)
    current = ffn.current_for(study,gpu)
    parent = Path(study['parent']['input_root'])
    old = read(parent/'configs/frozen_study.json')
    assert all(study[k]==old[k] for k in ('splits','components','text_context','software','schedule'))
    import lpips
    metric = lpips.LPIPS(net='alex',verbose=False).to(current.device).eval()
    summaries=[]
    expected=sum(i % shard_count==shard_index for i in range(len(study['splits']['test'])))*3
    with current.decoder() as decoder:
        for source_index,source in enumerate(study['splits']['test']):
            if source_index % shard_count != shard_index:
                continue
            capture,_=load_training_master(Path(source['capture']))
            vae=media.decode(current,capture.unsqueeze(0),decoder,42).mul(255).round().to(torch.uint8)
            rgb=source_pixels(source)
            matte,_,matte_identity=foreground.person_coverage(Path(source['resolved_dir']),source['capture_encode_record'],replay=True)
            weights=torch.nn.functional.interpolate(torch.from_numpy(matte).float().unsqueeze(1)/255,size=vae.shape[-2:],mode='area')[:,0]
            ceiling=region_lpips(metric,vae,rgb,weights,current.device)
            for index in study['config']['monitor_sigma_indices']:
                sigma=study['config']['sigmas'][index];seed=2042;k=ffn.key(source,seed,sigma)
                dest=root/f'media/test/{k}'
                if (dest/'results.json').exists():
                    summaries.append(read(dest/'results.json'));continue
                paths={'native':parent/f'evaluation/test/native/{k}/case.json',
                       'selection':root/f'evaluation/test/selection/{k}/case.json',
                       'final':root/f'evaluation/test/final/{k}/case.json',
                       'r3':parent/f'evaluation/test/final/{k}/case.json'}
                records={arm:read(path) for arm,path in paths.items()}
                assert len({r['input_sha256'] for r in records.values()})==1
                assert len({r['noise_sha256'] for r in records.values()})==1
                case=input_case(study,current,source,seed,sigma)
                pixels={}
                for arm,row in records.items():
                    assert provenance.file_sha256(row['prediction'])==row['prediction_sha256']
                    tokens=torch.load(row['prediction'],map_location='cpu',weights_only=True).to(current.device)
                    latent=case[0].unpatchify_block(tokens,case[0].latent_frames)
                    pixels[arm]=media.decode(current,latent,decoder,42).mul(255).round().to(torch.uint8)
                    del tokens,latent
                errors={arm:dict(person=region_lpips(metric,value,vae,weights,current.device),
                    frame=region_lpips(metric,value,vae,torch.ones_like(weights),current.device),
                    pixels=pixel_metrics(value,vae,weights)) for arm,value in pixels.items()}
                roles=[('capture_rgb','Capture RGB',rgb),('capture_vae','Capture VAE',vae),
                       ('native','Native',pixels['native']),('selection','A before recovery',pixels['selection']),
                       ('final','A after recovery',pixels['final']),('r3','Earlier r3 final',pixels['r3'])]
                frames=tuple(range(len(vae)))
                panels=[media.Panel(role,title,value,frames,value='RGB' if role=='capture_rgb' else 'VAE decoding') for role,title,value in roles]
                combined,rendering=media.render_panels(panels,question='Does A preserve the person better than r3?',layout='comparison',
                    fps=source['fps'],panel_size=(384,384),viewing_width=480,poster_frame=min(30,len(frames)-1),
                    common_settings=dict(schedule=[sigma,0.],seed=seed,actor=source['actor']))
                verified=verify_comparison(media.save_render(combined,rendering,dest))
                checkpoints.publish_json(dest/'rendering.verified.json',verified)
                for role,_,value in roles:
                    saved_video(value,dest/f'{role}.mp4',source['fps'])
                row=dict(study=study['identity'],actor=source['actor'],view=source['relative_dir'],sigma=sigma,seed=seed,
                    metrics=errors,vae_to_rgb_ceiling=ceiling,matte_identity=matte_identity,rendering=str(dest/'rendering.json'),
                    input_sha256=records['native']['input_sha256'],noise_sha256=records['native']['noise_sha256'],
                    source_receipts={arm:dict(path=str(path),sha256=provenance.file_sha256(path)) for arm,path in paths.items()})
                checkpoints.publish_json(dest/'results.json',row);summaries.append(row)
                print(f'A media shard={shard_index} {len(summaries)}/{expected} actor={source["actor"]} sigma={sigma}',flush=True)
                del pixels,panels,roles,combined,case
                gc.collect()
            del capture,vae,rgb,weights,matte
            gc.collect()
    result='results.json' if shard_count==1 else f'results_shard_{shard_index}.json'
    checkpoints.publish_json(root/f'media/test/{result}',dict(study=study['identity'],cases=summaries,
        shard_index=shard_index,shard_count=shard_count,
        population='Disjoint source views, seed 2042 and all three schedules',arms=['native','selection','final','r3']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--gpu-id',type=int,required=True)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1)
    a=p.parse_args();heads.numerical_policy();torch.set_num_threads(4);run(joint_final.load(a.study),a.gpu_id,a.shard_index,a.shard_count)


if __name__=='__main__':
    main()
