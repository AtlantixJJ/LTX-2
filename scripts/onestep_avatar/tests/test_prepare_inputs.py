"""Actual RGB preparation, single-image encoding and immutable provenance."""
from contextlib import contextmanager
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from ltx_core.types import SpatioTemporalScaleFactors
from scripts.onestep_avatar import infer, prepare_inputs
from scripts.onestep_avatar.corpus import precompute
from scripts.onestep_avatar.execution import software


@pytest.fixture
def image_case(tmp_path, monkeypatch):
    image, mask, guide, vae = [tmp_path/name for name in ('source.png', 'mask.png', 'guide.pt', 'vae')]
    Image.fromarray(np.zeros((96, 96, 3), dtype=np.uint8)).save(image)
    Image.fromarray(np.full((96, 96), 128, dtype=np.uint8)).save(mask)
    vae.write_bytes(b'controlled VAE')
    record = {'schema_version': 2, 'objective': 'white', 'fps': 30, 'box_xyxy': [16, 16, 80, 80],
              'edge': 64, 'encode_contract_version': 1, 'vae_fingerprint': precompute.file_fingerprint(vae),
              'master': torch.zeros(4, 7, 2, 2, dtype=torch.bfloat16), 'pixel_frames': 49}
    torch.save(record, guide)
    model = SimpleNamespace(paths=SimpleNamespace(video_vae=lambda: str(vae)),
                            caps=SimpleNamespace(latent_channels=4), scale_factors=SpatioTemporalScaleFactors(8, 32, 32))
    monkeypatch.setattr(prepare_inputs.model_registry, 'resolve', lambda *_a, **_k: model)
    args = prepare_inputs.parse_args(['supplied-image', '--image', str(image), '--mask', str(mask),
                                     '--guide', str(guide), '--output', str(tmp_path/'prepared'), '--gpu-id', '3'])
    return args, record


def test_pixel_preparation_replays_capture_formula_and_crop(image_case):
    args, record = image_case
    rgb = np.arange(96*96*3, dtype=np.uint8).reshape(96, 96, 3)
    Image.fromarray(rgb).save(args.image)
    model, actual, guide, pixels, hashes, manifest = prepare_inputs.prepare_image(args)
    alpha = 128/255
    expected = np.rint(rgb[16:80, 16:80].astype(np.float32)*alpha+255*(1-alpha)).clip(0,255).astype(np.uint8)
    expected = cv2.resize(expected, (64,64), interpolation=cv2.INTER_AREA)
    assert np.array_equal(pixels, expected)
    assert actual['box_xyxy'] == record['box_xyxy'] and guide.shape[1] == 7
    software.check_current(manifest)
    assert len(hashes) == 4 and not args.output.exists()


def test_bg_keeps_original_colors_without_matte(image_case):
    args, record = image_case
    args.mask = None
    record.update(objective='bg', edge=32, master=torch.zeros(4,7,1,1,dtype=torch.bfloat16))
    torch.save(record, args.guide)
    pixels=np.zeros((96,96,3),dtype=np.uint8)
    pixels[16:48,16:48]=[255,0,0]
    pixels[16:48,48:80]=[0,255,0]
    pixels[48:80,16:48]=[0,0,255]
    pixels[48:80,48:80]=255
    Image.fromarray(pixels).save(args.image)
    _,_,_,prepared,hashes,_=prepare_inputs.prepare_image(args)
    assert prepared.shape==(32,32,3) and len(hashes)==3
    assert prepared[0,0].tolist()==[255,0,0]
    assert prepared[0,-1].tolist()==[0,255,0]
    assert prepared[-1,0].tolist()==[0,0,255]
    assert prepared[-1,-1].tolist()==[255,255,255]


@pytest.mark.parametrize('defect', ['missing_mask', 'bg_mask', 'mask_shape', 'crop', 'edge', 'shape', 'vae', 'animated', 'rgba'])
def test_invalid_image_preparation_before_gpu(image_case, monkeypatch, defect):
    args, record = image_case
    if defect == 'missing_mask': args.mask = None
    elif defect == 'bg_mask': record['objective'] = 'bg'
    elif defect == 'mask_shape': Image.new('L', (95,96)).save(args.mask)
    elif defect == 'crop': record['box_xyxy'] = [-1,0,63,64]
    elif defect == 'edge': record['edge'] = 63
    elif defect == 'shape': record['master'] = torch.zeros(4,7,3,2)
    elif defect == 'vae': record['vae_fingerprint'] = 'other VAE'
    elif defect == 'rgba': Image.new('RGBA', (96,96)).save(args.image)
    else:
        args.image = args.image.with_suffix('.gif')
        Image.new('RGB', (96,96)).save(args.image, save_all=True, append_images=[Image.new('RGB',(96,96),'white')])
    torch.save(record, args.guide)
    monkeypatch.setattr(prepare_inputs.preflight, 'check', lambda *_a, **_k: pytest.fail('bad input opened GPU'))
    with pytest.raises((ValueError, RuntimeError)):
        prepare_inputs.encode_image(args)
    assert not args.output.exists()


def controlled_encoder(monkeypatch, run):
    monkeypatch.setattr(prepare_inputs.preflight, 'check', lambda *_a, **_k: None)
    original_to = torch.Tensor.to
    def cpu_to(self, *args, **kwargs):
        kwargs.pop('device', None)
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    for name in ('reset_peak_memory_stats', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, lambda *_a, **_k: None)
    for name in ('max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda *_a, **_k: 0)
    @contextmanager
    def encoder(*_a, **_k):
        yield SimpleNamespace(tiled_encode=run)
    monkeypatch.setattr(prepare_inputs.ltx_adapter, 'video_encoder', encoder)


def test_one_actual_image_encode_and_product_bundle_compatibility(image_case, monkeypatch):
    args, record = image_case
    calls=[]
    def run(pixels, tiling):
        calls.append((pixels.clone(),tiling))
        return torch.zeros(1,4,1,2,2,dtype=torch.bfloat16)
    controlled_encoder(monkeypatch, run)
    result=prepare_inputs.encode_image(args)
    assert len(calls)==1 and calls[0][0].shape==(1,3,1,64,64) and calls[0][1] is None
    assert calls[0][0].dtype==torch.bfloat16
    bundle=torch.load(args.output/'image.pt',weights_only=True)
    infer.check_inputs(record['master'].unsqueeze(0), bundle['master'].unsqueeze(0), record, bundle)
    assert bundle['input_role']=='supplied_image' and bundle['pixel_frames']==1
    assert bundle['source']==str(args.image.resolve())
    assert bundle['preparation']['matte']=='continuous_grayscale_255'
    assert result['image_latent_shape']==[4,1,2,2] and args.output.joinpath('preparation.json').is_file()
    assert np.array(Image.open(args.output/'prepared_image.png')).min()==127


@pytest.mark.parametrize('defect', ['changed_image','changed_software','bad_output'])
def test_failed_encoder_or_changed_inputs_cannot_publish(image_case, monkeypatch, defect):
    args,_=image_case
    original=software.sha256
    def run(_pixels,_tiling):
        if defect=='changed_image': Image.new('RGB',(96,96),'white').save(args.image)
        elif defect=='changed_software':
            monkeypatch.setattr(software,'sha256',lambda p:'f'*64 if p.name=='prepare_inputs.py' else original(p))
        return torch.zeros(1,4,2 if defect=='bad_output' else 1,2,2,dtype=torch.bfloat16)
    controlled_encoder(monkeypatch,run)
    with pytest.raises(ValueError): prepare_inputs.encode_image(args)
    assert not args.output.exists()
