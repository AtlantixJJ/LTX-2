"""Prepare actual supplied-image inputs; see doc/prepare_inputs.md."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from scripts.onestep_avatar import evaluate, hashing, media
from scripts.onestep_avatar.corpus import dataset, geometry, precompute
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import common
from scripts.prune.core import ltx_adapter, model_registry, preflight
from scripts.prune.core.session import DTYPE


def single_image(path: Path, *, matte: bool = False) -> np.ndarray:
    """Read actual unrotated pixels, rejecting animation and implicit alpha removal."""
    with Image.open(path) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('supplied input must contain exactly one RGB image or matte')
        if image.mode not in (('L', '1') if matte else ('RGB',)):
            raise ValueError('supplied image must be RGB; matte must be grayscale')
        return np.array(image.convert('L') if matte else image, dtype=np.uint8)


def prepare_image(args: argparse.Namespace) -> tuple:
    """Check all input bytes and pixel operations before any GPU model is opened."""
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('supplied-image preparation requires a fresh output directory')
    producer_software = software.capture('preparation')
    model = model_registry.resolve(args.model)
    vae = Path(model.paths.video_vae())
    paths = [args.image, args.guide, vae] + ([] if args.mask is None else [args.mask])
    identities = {str(path.resolve()): sha256(path) for path in paths}
    guide_record = torch.load(args.guide, map_location='cpu', weights_only=True)
    guide, _ = dataset.load_training_master(args.guide, bundle=guide_record)
    if (guide_record.get('vae_fingerprint') != precompute.file_fingerprint(vae)
            or guide_record.get('encode_contract_version') != precompute.ENCODE_CONTRACT_VERSION):
        raise ValueError('guide VAE or encoding contract differs from the current producer')
    objective = guide_record.get('objective')
    if objective not in dataset.OBJECTIVES:
        raise ValueError('unsupported supplied-image background objective')
    if (objective == 'white') != (args.mask is not None):
        raise ValueError('white requires an explicit matte; bg rejects a matte')
    box, edge = guide_record.get('box_xyxy'), guide_record.get('edge')
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)
            or box[2] <= box[0] or box[3] <= box[1] or box[2]-box[0] != box[3]-box[1]
            or type(edge) is not int or edge < 1
            or edge % model.scale_factors.height or edge % model.scale_factors.width):
        raise ValueError('guide crop must be finite, square and scale aligned')
    expected = (model.caps.latent_channels, edge // model.scale_factors.height, edge // model.scale_factors.width)
    if (guide.shape[0], *guide.shape[2:]) != expected:
        raise ValueError('guide channels or spatial shape differ from its recorded edge')
    image = single_image(args.image)
    cropped = geometry.crop_from_canvas(image, tuple(box))
    if args.mask is not None:
        mask = single_image(args.mask, matte=True)
        if mask.shape != image.shape[:2]:
            raise ValueError('supplied matte and RGB original canvas dimensions differ')
        alpha = geometry.crop_from_canvas(mask, tuple(box)).astype(np.float32)[..., None] / 255
        cropped = (cropped.astype(np.float32)*alpha + 255*(1-alpha)).round().clip(0, 255).astype(np.uint8)
    pixels = cv2.resize(cropped, (edge, edge), interpolation=cv2.INTER_AREA)
    check_inputs_unchanged(identities, producer_software)
    return model, guide_record, guide, pixels, identities, producer_software


def check_inputs_unchanged(identities: dict, producer_software: dict) -> None:
    software.check_current(producer_software)
    if any(sha256(Path(path)) != digest for path, digest in identities.items()):
        raise ValueError('supplied-image preparation inputs changed during execution')


def encode_image(args: argparse.Namespace) -> dict:
    """Make one native image encode and publish its checked bundle and RGB evidence."""
    model, guide_record, guide, pixels, identities, producer_software = prepare_image(args)
    preflight.check(args.model, gpu_id=args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')
    check_inputs_unchanged(identities, producer_software)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    native_pixels = torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None].to(device=device, dtype=DTYPE) / 127.5 - 1
    with ltx_adapter.video_encoder(model.paths.video_vae(), DTYPE, device) as encoder, torch.inference_mode():
        master = encoder.tiled_encode(native_pixels, None)
    del native_pixels, encoder
    expected = (1, guide.shape[0], 1, *guide.shape[2:])
    if (not isinstance(master, torch.Tensor) or tuple(master.shape) != expected
            or not master.is_floating_point() or not torch.isfinite(master).all()):
        raise ValueError('supplied-image VAE must return one finite image latent matching the guide')
    bundle = precompute.master_record(
        master, source=str(args.image.resolve()), fps=guide_record['fps'], pixel_frames=1,
        box_xyxy=tuple(guide_record['box_xyxy']), edge=guide_record['edge'], objective=guide_record['objective'],
        input_fingerprint=identities[str(args.image.resolve())], vae_fingerprint=guide_record['vae_fingerprint'],
    )
    bundle.update(input_role='supplied_image', software=producer_software,
                  preparation={'input_sha256': identities, 'original_canvas_hw': list(single_image(args.image).shape[:2]),
                               'prepared_pixels_sha256': hashing.tensor_sha256(torch.from_numpy(pixels)),
                               'resize': 'opencv_INTER_AREA', 'matte': 'continuous_grayscale_255' if args.mask else None,
                               'encoder': {'dtype': 'bfloat16', 'method': 'tiled_encode', 'tiling': None}})
    decoded = None
    if args.review:
        check_inputs_unchanged(identities, producer_software)
        session = media.open_decoder_session(args.model, args.gpu_id, script='onestep_avatar.supplied_image')
        with session.decoder() as decoder:
            decoded = media.decode(session, master, decoder, args.seed)
        del decoder
    del master
    torch.cuda.synchronize(device)
    seconds = time.perf_counter()-started
    memory = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    check_inputs_unchanged(identities, producer_software)
    args.output.mkdir(parents=True, exist_ok=False)
    image_path, bundle_path = args.output/'prepared_image.png', args.output/'image.pt'
    Image.fromarray(pixels).save(image_path)
    dataset.atomic_write(bundle_path, lambda temporary: torch.save(bundle, temporary))
    outputs = {role: {'path': str(path.resolve()), 'sha256': sha256(path)}
               for role, path in (('pixels', image_path), ('bundle', bundle_path))}
    if decoded is not None:
        decoded_path = args.output/'decoded_image.png'
        media.frame(decoded, 0).save(decoded_path)
        outputs['decoded'] = {'path': str(decoded_path.resolve()), 'sha256': sha256(decoded_path)}
    check_inputs_unchanged(identities, producer_software)
    result = {'schema_version': 1, 'kind': 'onestep_avatar.supplied_image_preparation',
              'software': producer_software, 'inputs': identities, 'outputs': outputs,
              'seconds_encode_and_optional_decode': seconds, 'memory': memory,
              'image_latent_shape': list(bundle['master'].shape), 'decoder_seed': args.seed if args.review else None,
              'capture_reference': None}
    dataset.atomic_write(args.output/'preparation.json',
                         lambda temporary: temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n'))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    image = commands.add_parser('supplied-image', help='encode exactly one actual RGB image')
    for name in ('image', 'guide', 'output'):
        image.add_argument('--'+name, type=Path, required=True)
    image.add_argument('--mask', type=Path)
    image.add_argument('--model', choices=model_registry.SUPPORTED_MODELS, default='2.5')
    image.add_argument('--gpu-id', type=int, required=True)
    image.add_argument('--review', action='store_true')
    image.add_argument('--seed', type=int, default=42)
    preview = commands.add_parser('preview', help='assemble fixed training-preview inputs without generation')
    preview.add_argument('--references', type=Path, required=True)
    preview.add_argument('--output', type=Path, required=True)
    preview.add_argument('--gpu-id', type=int, required=True)
    preview.add_argument('--evaluation-arguments', nargs=argparse.REMAINDER, required=True)
    return parser.parse_args(argv)


def preview_arguments(arguments: list[str], args: argparse.Namespace, noise: Path, frames: int) -> list[str]:
    """Keep effective settings while replacing path/length spellings with pinned ones."""
    path_flags = {'--subset': args.subset, '--frame-plan': args.frame_plan, '--corpus-root': args.corpus_root}
    stripped, index = [], 0
    while index < len(arguments):
        token = arguments[index]
        flag = token.split('=', 1)[0]
        if flag in {*path_flags, '--noise-file', '--span-latent-frames', '--output-latent-frames'}:
            index += 1 if '=' in token else 2
        else:
            stripped.append(token)
            index += 1
    for flag, path in path_flags.items():
        if path is not None:
            stripped.extend((flag, str(path.resolve())))
    stripped.extend(('--noise-file', str(noise.resolve())))
    if args.mode == 'causal':
        if args.span_latent_frames is not None:
            stripped.extend(('--span-latent-frames', str(args.span_latent_frames)))
        output_frames = getattr(args, 'output_latent_frames', None)
        stripped.extend(('--output-latent-frames', str(frames if output_frames is None else output_frames)))
    else:
        stripped.extend(('--span-latent-frames', str(frames)))
    return stripped


def prepare_preview(args: argparse.Namespace) -> dict:
    """Freeze selected native inputs and references; never execute a transformer."""
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT  # noqa: PLC0415
    from scripts.onestep_avatar.training import config, engine  # noqa: PLC0415 -- actual consumer validation
    from scripts.prune.core.session import DEFAULT_PROMPT  # noqa: PLC0415
    from scripts.prune.data import prompt_cache  # noqa: PLC0415 -- text preparation only

    if args.output.exists() or args.output.is_symlink():
        raise ValueError('preview preparation requires a fresh output directory')
    forbidden = {'--output', '--checkpoint', '--gpu-id', '--dry-run'}
    if any(token.split('=', 1)[0] in forbidden for token in args.evaluation_arguments):
        raise ValueError('preview preparation rejects execution-owned options')
    evaluation = evaluate.parse_args([*args.evaluation_arguments, '--output', str(args.output/'unexecuted')])
    if len(evaluation.source) != 1:
        raise ValueError('preview preparation requires exactly one explicit source')
    producer_software = software.capture('preparation', evaluation.mode)
    initial_paths = [evaluation.subset, args.references]
    initial_paths += [path for path in (evaluation.frame_plan, evaluation.noise_file) if path is not None]
    identities = {str(path.resolve()): sha256(path) for path in initial_paths}
    specification, _, cases, membership = evaluate.prepare_evaluation(evaluation)
    video, frames, _, _ = cases[0]
    evaluation_arguments = preview_arguments(args.evaluation_arguments, evaluation, args.output/'noise.pt', frames)
    evaluate.parse_args([*evaluation_arguments, '--output', str(args.output/'unexecuted')])
    panels, references = media.load_training_references(args.references)
    del panels
    if (references['source'] != video.source or references['fps'] != video.fps
            or references['objective'] != membership['objective']
            or references['capture_encoding_sha256'] != video.hashes['capture']
            or references['source_frames'] != list(range((frames-1)*specification.scale_factors.time+1))
            or references['vae_sha256'] != sha256(Path(specification.paths.video_vae()))):
        raise ValueError('preview reference source/crop/task/coverage/VAE differs from selected inputs')
    store = dataset.ClipStore(membership, evaluation.corpus_root)
    view = store.root/video.source
    for role in ('capture', 'guide') if evaluation.guide_mode == 'd1' else ('capture',):
        path = view/(dataset.capture_bundle_name(store.objective) if role == 'capture'
                     else dataset.guide_bundle_name(store.objective))
        identities[str(path.resolve())] = video.hashes[role]
    if evaluation.guide_mode == 'd1' and references.get('guide_rgb_sha256') != video.hashes.get('render'):
        raise ValueError('preview references use a different guide render')
    check_inputs_unchanged(identities, producer_software)
    preflight.check(evaluation.model, gpu_id=args.gpu_id, transformer_path=specification.paths.transformer())
    check_inputs_unchanged(identities, producer_software)
    device = torch.device(f'cuda:{args.gpu_id}')
    grid = common.ClipGrid.build(frames, video.z_y.shape[2]*specification.scale_factors.height,
                                video.z_y.shape[3]*specification.scale_factors.width, video.fps,
                                specification, device=device, dtype=DTYPE,
                                latent_channels=specification.caps.latent_channels)
    capture = grid.patchify(video.z_y[:, :frames].unsqueeze(0).to(device=device, dtype=DTYPE))
    guide = None if video.z_g is None else grid.patchify(video.z_g[:, :frames].unsqueeze(0).to(device=device, dtype=DTYPE))
    noise = (common.epsilon_block(capture, evaluation.seed) if evaluation.saved_noise is None
             else evaluation.saved_noise.to(device=device))
    context = prompt_cache.get_or_build(specification, DEFAULT_PROMPT if evaluation.prompt is None else evaluation.prompt,
                                        DTYPE, device)
    tensors = {'capture': capture, 'first_image': capture[:, :grid.tokens_per_latent_frame],
               'text': context, 'noise': noise}
    if guide is not None:
        tensors['guide'] = guide
    if evaluation.cfg != 1.0:
        tensors['negative_text'] = prompt_cache.get_or_build(
            specification, DEFAULT_NEGATIVE_PROMPT if evaluation.negative_prompt is None else evaluation.negative_prompt,
            DTYPE, device)
    if any(not isinstance(value, torch.Tensor) or value.dtype != DTYPE or not torch.isfinite(value).all()
           for value in tensors.values()):
        raise ValueError('preview preparation requires finite native-bf16 tensors')
    check_inputs_unchanged(identities, producer_software)
    args.output.mkdir(parents=True, exist_ok=False)
    files = {'subset': {'path': str(evaluation.subset.resolve()), 'sha256': identities[str(evaluation.subset.resolve())]}}
    for role, tensor in tensors.items():
        if role in ('capture', 'guide'):
            path = view/(dataset.capture_bundle_name(store.objective) if role == 'capture'
                         else dataset.guide_bundle_name(store.objective))
        else:
            path = args.output/(role+'.pt')
            cpu = tensor.detach().cpu()
            dataset.atomic_write(path, lambda temporary, cpu=cpu: torch.save(cpu, temporary))
        files[role] = {'path': str(path.resolve()), 'sha256': sha256(path),
                       'tensor_sha256': hashing.tensor_sha256(tensor)}
    record = {'schema_version': 2, 'kind': 'onestep_avatar.preview_inputs', 'mode': evaluation.mode,
              'schedule': evaluation.schedule, 'input_files': files, 'software': producer_software,
              'producer_inputs': {path: {'path': path, 'sha256': digest} for path, digest in identities.items()},
              'reference_bundle': {'path': str(args.references.resolve()), 'sha256': identities[str(args.references.resolve())]},
              'evaluation_arguments': evaluation_arguments}
    temporary = args.output/'preview.pending.json'
    dataset.atomic_write(temporary, lambda path: path.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n'))
    settings = config.RunSettings(evaluation.mode, evaluation.subset, args.output/'unexecuted_training',
                                  evaluation.mode_settings, guide_mode=evaluation.guide_mode)
    record = engine.read_preview_inputs(temporary, settings)
    check_inputs_unchanged(identities, producer_software)
    dataset.atomic_write(args.output/'preview.json',
                         lambda path: path.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n'))
    temporary.unlink()
    return record


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == 'supplied-image':
        encode_image(args)
    else:
        prepare_preview(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
