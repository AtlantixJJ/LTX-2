"""Decode saved MobileWan removal evidence and render 24 synchronized six-panel videos."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from scripts.onestep_avatar import media
from scripts.prune.core import provenance
from scripts.prune.data.head_score_dataset import SIGMAS, atomic_json


def read_evaluation(run, study):
    """Validate complete saved evidence on CPU before opening a decoder."""
    result = json.loads((run / 'evaluation/results.json').read_text())
    if (result.get('schema_version') != 1 or result.get('kind') != 'mobilewan_removal_evaluation'
            or result.get('status') != 'complete' or result.get('prediction_count') != 96
            or len(result.get('cases', [])) != 3):
        raise ValueError('media requires complete 96-prediction removal evidence')
    if result['identity']['study_sha256'] != provenance.file_sha256(run / 'configs/study.json'):
        raise ValueError('evaluation study changed')
    if result['identity']['frozen_train_sha256'] != provenance.file_sha256(run / 'masks/frozen_train.json'):
        raise ValueError('evaluation frozen train masks changed')
    expected = [(c['relative_dir'], c['actor']) for c in study['cases']]
    if [(c['view'], c['actor']) for c in result['cases']] != expected:
        raise ValueError('saved evaluation differs from preselected cases')
    for case in result['cases']:
        if len(case['sigmas']) != 8 or [r['sigma'] for r in case['sigmas']] != SIGMAS:
            raise ValueError('media requires every exact selected sigma')
        first_identity = case['sigmas'][0]['identity']
        for row in case['sigmas']:
            if (row['identity'] != first_identity or row['identity']['view'] != case['view']
                    or row['identity']['actor'] != case['actor']):
                raise ValueError('paired case input identity differs across sigmas')
            if row['all_one_equal'] is not True or row['schedule'] != [row['sigma'], 0.]:
                raise ValueError('saved paired controls failed')
            if [a['requested_fraction'] for a in row['arms']] != [0., .1, .23, .33]:
                raise ValueError('media requires all four matched arms')
            identity = row['identity']
            files = [(identity['gt'], identity['gt_sha256'])]
            files += [(a['prediction'], a['prediction_sha256']) for a in row['arms']]
            for relative, digest in files:
                path = run / relative
                if not path.resolve().is_relative_to(run.resolve()) or provenance.file_sha256(path) != digest:
                    raise ValueError('saved paired latent content changed or escaped run root')
    return result


def comparison_panels(recorded, decoded_gt, predictions):
    """Explicit same-frame maps; original RGB and VAE-decoded roles remain distinct."""
    values = [recorded, decoded_gt, *predictions]
    if len(values) != 6 or len({len(v) for v in values}) != 1:
        raise ValueError('six panels must share full source-frame coverage')
    titles = [('capture_rgb', 'Capture RGB', 'White matte'),
              ('capture_vae', 'Capture VAE', 'GT decoding'),
              ('dense', 'D0 / 0%', 'VAE decoding'),
              ('removal_10', 'D0 / 10%', 'VAE decoding'),
              ('removal_23', 'D0 / 23%', 'VAE decoding'),
              ('removal_33', 'D0 / 33%', 'VAE decoding')]
    frames = tuple(range(len(recorded)))
    return [media.Panel(role, title, pixels, frames, value=label)
            for (role, title, label), pixels in zip(titles, values, strict=True)]


def render_comparison(panels, fps, *, common):
    """Use the existing RGB renderer; keep six panels in two rows by three columns."""
    return media.render_panels(panels, question='How does head removal change the video?',
                               layout='comparison', fps=fps, panel_size=(384, 384),
                               viewing_width=480, poster_frame=min(30, len(panels[0].pixels)-1),
                               common_settings=common)


def saved_video(pixels, path, fps):
    from ltx_trainer.video_utils import save_video

    path.parent.mkdir(parents=True, exist_ok=True)
    save_video(pixels, path, fps=fps, video_format='FCHW')
    checked = media.verify_saved_video(path, len(pixels), fps)
    return {'path': str(path), 'sha256': provenance.file_sha256(path), 'verification': checked}


def source_pixels(source):
    """Replay the original crop and white matte through the checked public media owner."""
    resolved = Path(source['resolved_dir'])
    root = resolved.parents[len(Path(source['relative_dir']).parts)-1]
    if (root / source['relative_dir']).resolve() != resolved.resolve():
        raise ValueError('recorded RGB root differs from audited view')
    actual = {**source, 'box_xyxy': source['capture_encode_record']['box_xyxy']}
    return media.recorded_capture_rgb(actual, root, 'white', source['n_latent_frames'])


def preserve_source(source, destination, expected=None):
    """Pin complete source bytes; interrupted copies never publish a partial file."""
    source, destination = Path(source), Path(destination)
    digest = provenance.file_sha256(source)
    if expected is not None and digest != expected:
        raise ValueError('source reference differs from audited content')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        fd, name = tempfile.mkstemp(dir=destination.parent, prefix='.source-')
        try:
            with os.fdopen(fd, 'wb') as output, source.open('rb') as original:
                shutil.copyfileobj(original, output)
                output.flush()
                os.fsync(output.fileno())
            if provenance.file_sha256(name) != digest or provenance.file_sha256(source) != digest:
                raise ValueError('source reference changed while copying')
            try:
                os.link(name, destination)
            except FileExistsError:
                pass  # A concurrent identical copy is checked below.
        finally:
            Path(name).unlink(missing_ok=True)
    if provenance.file_sha256(destination) != digest:
        raise ValueError('preserved source reference differs; retain it for diagnosis')
    return {'path': str(destination), 'sha256': digest}


def verify_comparison(record):
    """Probe the actual combined MP4 and PNG before publishing their rendering record."""
    outputs = record['outputs']
    video, poster = Path(outputs['video']['path']), Path(outputs['poster']['path'])
    checked = media.verify_saved_video(video, len(record['source_frames']), record['fps'])
    width, height = record['display_size']
    if int(checked['width']) != width or int(checked['height']) != height:
        raise ValueError('comparison video dimensions differ from recorded layout')
    media.verify_saved_png(poster, width, height)
    if any(provenance.file_sha256(item['path']) != item['sha256'] for item in outputs.values()):
        raise ValueError('comparison output changed during verification')
    outputs['video']['verification'] = checked
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--gpu-id', type=int, default=4)
    args = parser.parse_args()
    run = args.study.resolve().parent.parent
    study = json.loads(args.study.read_text())
    evaluation = read_evaluation(run, study)
    vae = Path(study['components']['video_vae']['path'])
    if provenance.file_sha256(vae) != study['components']['video_vae']['file_sha256']:
        raise ValueError('pinned decoder content changed')
    current = media.open_decoder_session('2.5', args.gpu_id, script='prune.evaluate.mobilewan_media')
    if Path(current.model.paths.video_vae()).resolve() != vae.resolve():
        raise ValueError('actual decoder differs from pinned VAE')
    records = []
    with current.decoder() as decoder:
        for source, case in zip(study['cases'], evaluation['cases'], strict=True):
            key = hashlib.sha256(case['view'].encode()).hexdigest()
            destination = run / 'media' / key
            destination.mkdir(parents=True, exist_ok=True)
            original = Path(source['resolved_dir']) / 'rgb.mp4'
            rgb_source = preserve_source(original, destination / 'source_rgb.mp4', source['rgb_sha256'])
            from scripts.onestep_avatar.corpus.dataset import CAPTURE_MASK_NAME

            mask_path = original.with_name(CAPTURE_MASK_NAME)
            mask_source = preserve_source(mask_path, destination / 'source_mask.mp4')
            recorded = source_pixels(source)
            if (provenance.file_sha256(original) != rgb_source['sha256']
                    or provenance.file_sha256(mask_path) != mask_source['sha256']):
                raise ValueError('RGB or matte source changed during recorded capture replay')
            gt = torch.load(run / case['sigmas'][0]['identity']['gt'], map_location='cpu', weights_only=True)
            capture_bundle = torch.load(source['capture'], map_location='cpu', weights_only=True)
            from scripts.onestep_avatar.corpus.dataset import load_master

            if provenance.file_sha256(source['capture']) != source['capture_latent_sha256']:
                raise ValueError('audited capture content changed')
            if not torch.equal(gt[0], load_master(Path(source['capture']), bundle=capture_bundle)):
                raise ValueError('saved GT differs from recorded capture encoding')
            decoded_gt = media.decode(current, gt, decoder, 42).mul(255).round().to(torch.uint8)
            references = {'full_source': rgb_source['path'], 'source_sha256': rgb_source['sha256'],
                          'source_mask': mask_source,
                          'capture_encoding': source['capture_encode_record'],
                          'recorded': saved_video(recorded, destination / 'capture_rgb.mp4', source['fps']),
                          'decoded_gt': {**saved_video(decoded_gt, destination / 'capture_vae.mp4', source['fps']),
                                         'latent_sha256': case['sigmas'][0]['identity']['gt_sha256']}}
            for j, row in enumerate(case['sigmas']):
                output = destination / f"step_{study['schedule']['stock_step_indices'][j]:02d}"
                predictions, individual = [], []
                for arm in row['arms']:
                    latent = torch.load(run / arm['prediction'], map_location='cpu', weights_only=True)
                    if latent.shape != gt.shape or not torch.equal(latent[:, :, 0], gt[:, :, 0]):
                        raise ValueError('prediction geometry or clean frame differs')
                    pixels = media.decode(current, latent, decoder, 42).mul(255).round().to(torch.uint8)
                    predictions.append(pixels)
                    path = output / f"d0_removal_{round(arm['requested_fraction']*100):03d}.mp4"
                    individual.append({**saved_video(pixels, path, source['fps']),
                                       'prediction_sha256': arm['prediction_sha256'],
                                       'requested_fraction': arm['requested_fraction']})
                common = {'view': case['view'], 'actor': case['actor'], 'sigma': row['sigma'],
                          'stock_step_index': study['schedule']['stock_step_indices'][j], 'seed': 42,
                          'schedule': row['schedule'], 'references': references,
                          'input_identity': row['identity'], 'decode_vae_sha256': study['components']['video_vae']['file_sha256'],
                          'decode_seed': 42, 'individual_videos': individual}
                panels = comparison_panels(recorded, decoded_gt, predictions)
                pixels, render = render_comparison(panels, source['fps'], common=common)
                complete = verify_comparison(media.save_render(pixels, render, output))
                atomic_json(output / 'rendering.json', complete)
                records.append({'view': case['view'], 'sigma': row['sigma'], 'rendering': complete})
                print(f'rendered {case["view"]} sigma_index={j}', flush=True)
                del predictions, panels, pixels
    if len(records) != 24:
        raise ValueError('incomplete comparison coverage')
    atomic_json(run / 'media/index.json', {'schema_version': 1, 'kind': 'mobilewan_removal_media',
                'status': 'complete', 'evaluation_sha256': provenance.file_sha256(run / 'evaluation/results.json'),
                'source_sha256': provenance.file_sha256(__file__), 'comparison_count': 24, 'comparisons': records})


if __name__ == '__main__':
    main()
