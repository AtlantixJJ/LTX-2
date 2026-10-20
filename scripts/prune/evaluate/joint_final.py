"""Evaluate completed V3 checkpoints with unchanged saved full-clip inputs.

Historical study identities remain immutable. This post-training owner pins its
own source identity and never resumes training. See doc/evaluate/joint_final.md.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch

from scripts.onestep_avatar.model import common
from scripts.prune.core import ltx_adapter, provenance
from scripts.prune.data.learned_head_study import code_identity
from scripts.prune.evaluate import ffn
from scripts.prune.learn import adapters, checkpoints, heads, joint_training
from scripts.prune.learn.joint import input_case
from scripts.prune.score import export_joint, export_joint_split, export_learned_ffn, export_pruned, hooks


def load(path):
    study = json.loads(Path(path).read_text())
    value = dict(study)
    identity = value.pop('identity')
    if checkpoints.digest(value) != identity or study['format'] != 'joint_study_v3':
        raise ValueError('Changed or unsupported historical V3 study')
    owner = json.loads((Path(study['root'])/'monitoring/owner.json').read_text())
    if owner['status'] != 'complete':
        raise ValueError('Post-training evaluation requires a completed study')
    return study


def state(study, arm):
    root = Path(study['root'])
    owner = json.loads((root/'monitoring/owner.json').read_text())
    final = root/f'training/checkpoints/update_{owner["update"]:04d}.pt'
    payload = export_joint.terminal(study, final)
    if arm == 'selection':
        path = root/f'training/checkpoints/update_{payload["policy"]["selection_end"]:04d}.pt'
        payload = joint_training.read(path, study=study['identity'])
    else:
        path = final
    receipt = json.loads(path.with_suffix('.json').read_text())
    if provenance.file_sha256(path) != receipt['sha256'] or not payload['frozen'] or not payload['policy']['aligned']:
        raise ValueError('Checkpoint hash or aligned fixed-mask state differs')
    return path, payload


def functional(model, study, payload):
    attached = adapters.attach(model, study['config']['seed'], study['config']['head_rank'])
    if study['config']['arm'] != 'heads_clip_reset_v3':
        attached.update(adapters.attach_ffn(model, study['config']['seed'], study['config']['ffn_rank']))
    adapters.load_state(attached, payload['adapters'])
    source = study['components']['transformer']['path']
    masks = export_learned_ffn.masks_from_state(payload, export_pruned.checkpoint_mask_widths(source))
    h = hooks.attach_head_masks(model, {k: torch.tensor(v) for k, v in masks.items()}, requires_grad=False)
    c = hooks.attach_ffn_masks(model, {k: torch.tensor(v) for k, v in masks.items()}, requires_grad=False)
    return attached, h, c


def infer(study, gpu, arm):
    if arm not in ('selection', 'final'):
        raise ValueError('Only prespecified selection and final checkpoints are supported')
    checkpoint, payload = state(study, arm)
    current = ffn.current_for(study, gpu)
    root = Path(study['root'])/f'evaluation/test/{arm}'
    identity = dict(study=study['identity'], arm=arm, checkpoint_sha256=provenance.file_sha256(checkpoint),
                    schedules=[[study['config']['sigmas'][i], 0.] for i in study['config']['monitor_sigma_indices']],
                    source_split='test', seeds=study['config']['evaluation_seeds'])
    checkpoints.publish_json(root/'identity.json', identity)
    checkpoints.publish_json(root/'executor.json', dict(code=code_identity(), task='fixed-mask inference; no optimizer'))
    rows = []
    with current.transformer() as model, torch.no_grad():
        model.eval()
        attached, h, c = functional(model, study, payload)
        del payload
        for source, seed, sigma in ffn.cases(study, 'test'):
            dest = root/ffn.key(source, seed, sigma)
            receipt = dest/'case.json'
            if receipt.exists():
                row = json.loads(receipt.read_text())
                if row['identity'] != identity or provenance.file_sha256(row['prediction']) != row['prediction_sha256']:
                    raise ValueError(f'Changed existing prediction: {receipt}')
                rows.append(row)
                continue
            grid, video, c0, capture, epsilon, inputs, manifest = input_case(study, current, source, seed, sigma)
            velocity, clean = ltx_adapter.velocity_forward(model, video)
            clean = common.with_clean_prefix(clean, c0)
            prefix = grid.tokens_per_latent_frame
            if not torch.equal(clean[:, :prefix], c0) or not torch.isfinite(clean).all():
                raise ValueError('Changed clean prefix or nonfinite output')
            native_path = Path(study['parent']['input_root'])/f'evaluation/test/native/{dest.name}/case.json'
            native = json.loads(native_path.read_text())
            if provenance.file_sha256(inputs/'manifest.json') != native['input_sha256']:
                raise ValueError('Held-out inputs differ from the saved native baseline')
            prediction = dest/'prediction.pt'
            checkpoints.publish_tensor(prediction, clean)
            row = dict(identity=identity, actor=source['actor'], view=source['relative_dir'], seed=seed, sigma=sigma,
                       schedule=[sigma, 0.], capture_mse=float((clean[:, prefix:].float()-capture[:, prefix:].float()).square().mean()),
                       velocity_mse=float((velocity[:, prefix:].float()-(epsilon.float()-capture.float())[:, prefix:]).square().mean()),
                       prediction=str(prediction), prediction_sha256=provenance.file_sha256(prediction),
                       input_manifest=str(inputs/'manifest.json'), input_sha256=native['input_sha256'],
                       noise_sha256=manifest['videos'][0]['artifacts']['epsilon_sha256'],
                       context=study['text_context'], capture_sha256=source['capture_latent_sha256'])
            checkpoints.publish_json(receipt, row)
            rows.append(row)
            print(f'V3 {arm} {len(rows)}/108 capture_mse={row["capture_mse"]:.8f}', flush=True)
            del grid, video, c0, capture, epsilon, velocity, clean
        hooks.detach_all(h)
        hooks.detach_all(c)
        del attached, h, c
    checkpoints.publish_json(root/'results.json', dict(identity=identity, rows=rows, cases=len(rows),
            mean_capture_mse=statistics.mean(v['capture_mse'] for v in rows), status='complete_numeric_only'))


def deployment(study, gpu):
    checkpoint, payload = state(study, 'final')
    root = Path(study['root'])
    source = study['components']['transformer']['path']
    envelope = root/'exports/final_split.json'
    if not envelope.exists():
        masks = export_learned_ffn.masks_from_state(payload, export_pruned.checkpoint_mask_widths(source))
        export_joint_split.bundle(source, masks, payload['adapters'], envelope,
              provenance=dict(study=study['identity'], checkpoint=str(checkpoint),
                              checkpoint_sha256=provenance.file_sha256(checkpoint), update=payload['update']))
    current = ffn.current_for(study, gpu)
    # One view from every test actor and all three schedules: full actor coverage.
    sources = []
    for src in study['splits']['test']:
        if src['actor'] not in {v['actor'] for v in sources}:
            sources.append(src)
    specs = [(src, 2042, study['config']['sigmas'][i]) for src in sources
             for i in study['config']['monitor_sigma_indices']]
    rows = []
    with export_joint_split.open_bundle(current, envelope) as model, torch.no_grad():
        model.eval()
        for src, seed, sigma in specs:
            receipt_path = root/f'evaluation/test/final/{ffn.key(src, seed, sigma)}/case.json'
            import time
            while not receipt_path.exists():
                time.sleep(10)
            case = input_case(study, current, src, seed, sigma)
            _, clean = ltx_adapter.velocity_forward(model, case[1])
            clean = common.with_clean_prefix(clean, case[2]).cpu()
            reference = json.loads(receipt_path.read_text())
            if provenance.file_sha256(reference['prediction']) != reference['prediction_sha256']:
                raise ValueError('Changed functional reference')
            expected = torch.load(reference['prediction'], map_location='cpu', weights_only=True)
            difference = (clean.float()-expected.float()).abs()
            maximum = float(difference.max())
            rows.append(dict(actor=src['actor'], view=src['relative_dir'], sigma=sigma, max_abs=maximum,
                             bit_identical=torch.equal(clean, expected), passed=bool(torch.isfinite(difference).all() and maximum <= .02),
                             input_sha256=reference['input_sha256']))
            print('Final parity', len(rows), maximum, flush=True)
            del case, clean, expected, difference
        del model
    parity = dict(study=study['identity'], envelope=str(envelope), envelope_sha256=provenance.file_sha256(envelope),
                  cases=rows, tolerance=.02, passed=len(rows)==18 and all(v['passed'] for v in rows), code=code_identity())
    checkpoints.publish_json(root/'exports/final_split.parity.json', parity)
    if not parity['passed']:
        return
    gc.collect()
    torch.cuda.empty_cache()
    case = input_case(study, current, sources[0], 2042, study['config']['sigmas'][0])
    brackets = []
    for order in (('masked', 'bundle', 'masked'), ('bundle', 'masked', 'bundle')):
        values = []
        for name in order:
            manager = current.transformer() if name=='masked' else export_joint_split.open_bundle(current, envelope)
            with manager as model, torch.no_grad():
                model.eval()
                handles = functional(model, study, payload) if name=='masked' else None
                torch.cuda.reset_peak_memory_stats(current.device)
                value = ffn.timed(model, case[1], None, current.device)
                values.append(dict(arm=name, **value))
                if handles:
                    hooks.detach_all(handles[1])
                    hooks.detach_all(handles[2])
                    del handles
                del model
            gc.collect()
            torch.cuda.empty_cache()
        outer = (values[0]['median_seconds']+values[2]['median_seconds'])/2
        ratio = outer/values[1]['median_seconds'] if order[0]=='masked' else values[1]['median_seconds']/outer
        brackets.append(dict(order=list(order), rows=values, speedup=ratio,
                             drift=abs(values[2]['median_seconds']/values[0]['median_seconds']-1)))
    checkpoints.publish_json(root/'timing/results.json', dict(study=study['identity'], orders=brackets,
            speedups=[v['speedup'] for v in brackets], parity_passed=True,
            speed_qualified=min(v['speedup'] for v in brackets)>=1.05 and max(v['drift'] for v in brackets)<.02,
            scope='one full held-out F18/F28 source at the highest sigma; transformer only; split adapter arithmetic',
            stored_bytes=sum(Path(export_joint_split.read(envelope)[k]['path']).stat().st_size for k in ('weights', 'adapters'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--gpu-id', type=int, required=True)
    parser.add_argument('--stage', choices=['infer', 'deployment'], required=True)
    parser.add_argument('--arm', choices=['selection', 'final'])
    args = parser.parse_args()
    heads.numerical_policy()
    torch.set_num_threads(4)
    study = load(args.study)
    if args.stage == 'infer':
        infer(study, args.gpu_id, args.arm)
    else:
        deployment(study, args.gpu_id)


if __name__ == '__main__':
    main()
