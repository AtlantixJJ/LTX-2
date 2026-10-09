"""Matched stock video sampling diagnostic; see doc/experiments/stock_parity.md."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
from ltx_pipelines.utils.helpers import create_noised_state
from ltx_pipelines.utils.samplers import euler_denoising_loop
from scripts.onestep_avatar import evaluate, media
from scripts.onestep_avatar.corpus import dataset, precompute, subset
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import backbone, bidirectional, common
from scripts.prune.core.session import DEFAULT_PROMPT, DTYPE, Session

ENTRY = "scripts/onestep_avatar/experiments/stock_parity.py"
EXTRA_SOURCES = (ENTRY, 'scripts/onestep_avatar/experiments/__init__.py')

PROTOCOL = {
    'scope': 'stock public video sampling path, audio absent; no outer RGB/text or joint audio-video parity',
    'initialization': 'pure noise with independently encoded supplied image; no D1 guide mixing',
    'guidance': {'cfg': 1.0, 'stg': 0.0, 'rescale': 0.0, 'modality_scale': 1.0},
    'global_sigma_dtype': 'float32',
    'repeated_stock_tolerance': 0.0,
    'before_terminal_tolerance': 0.0,
    'terminal_rule': 'Report stock reconstruction versus exact prediction separately; do not fit a tolerance',
    'timing': 'resident model execution including trace copies; not a deployment benchmark',
}


def differences(left: torch.Tensor, right: torch.Tensor) -> dict:
    if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError('comparison requires finite equal-shaped tensors')
    delta = left.float() - right.float()
    return {'unequal_elements': int(torch.count_nonzero(delta)), 'rms': float(delta.square().mean().sqrt()),
            'maximum': float(delta.abs().max())}


class Trace(torch.nn.Module):
    """Record actual native transformer inputs and outputs without altering calls."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.calls = []

    @property
    def num_blocks(self):
        return self.model.num_blocks

    def forward(self, video, audio=None, perturbations=None):
        if audio is not None:
            raise ValueError('stock video check requires audio absent')
        result = self.model(video=video, audio=audio, perturbations=perturbations)
        call = {name: getattr(video, name).detach().cpu().clone()
                for name in ('latent', 'sigma', 'timesteps', 'positions', 'keyframes_mask')}
        call['prediction'] = result[0].detach().cpu().clone()
        self.calls.append(call)
        return result


def stock_state(grid, image, seed):
    device = image.device
    generator = torch.Generator(device=device).manual_seed(seed)
    return create_noised_state(grid.tools, [VideoConditionByLatentIndex(image, 1.0, 0)],
                               GaussianNoiser(generator), DTYPE, device)


def sample_paths(model, context, grid, image, levels, seed):
    """Run actual stock components twice and the ordinary custom sampler once."""
    source = grid.tools.create_initial_state(image.device, DTYPE).latent
    generator = torch.Generator(device=image.device).manual_seed(seed)
    noise = GaussianNoiser(generator)(grid.tools.create_initial_state(image.device, DTYPE)).latent
    c0 = grid.patchify(image)
    expected_initial = common.with_clean_prefix(noise, c0)
    outputs, traces, timings = {}, {}, {}
    for name in ('stock', 'stock_repeat', 'bidirectional', 'bidirectional_default'):
        if image.device.type == 'cuda':
            torch.cuda.synchronize(image.device)
            torch.cuda.reset_peak_memory_stats(image.device)
        started = time.perf_counter()
        trace = Trace(model)
        if name in ('stock', 'stock_repeat'):
            state = stock_state(grid, image, seed)
            if not torch.equal(state.latent, expected_initial):
                raise ValueError('actual stock initial noise/image differs from fixed tokens')
            params = MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0, rescale_scale=0.0, modality_scale=1.0)
            denoiser = FactoryGuidedDenoiser(context, None, create_multimodal_guider_factory(params))
            state, absent = euler_denoising_loop(
                levels, state, None, EulerDiffusionStep(), BatchSplitAdapter(trace, 1), denoiser)
            if absent is not None:
                raise ValueError('stock loop created audio')
            output = state.latent
        else:
            output, _ = bidirectional.sample(common.denoised_from_x0_model(trace), context, grid, source, c0,
                                             schedule=levels.tolist(), seed=seed, epsilon=noise,
                                             sigma_dtype=torch.float32 if name == 'bidirectional' else None)
        outputs[name] = output.detach().cpu()
        traces[name] = trace.calls
        if image.device.type == 'cuda':
            torch.cuda.synchronize(image.device)
        timings[name] = {'seconds_including_traces': time.perf_counter()-started, 'forward_calls': len(trace.calls),
                         'peak_allocated_bytes': torch.cuda.max_memory_allocated(image.device)
                         if image.device.type == 'cuda' else None,
                         'peak_reserved_bytes': torch.cuda.max_memory_reserved(image.device)
                         if image.device.type == 'cuda' else None}
        del trace
    return outputs, traces, noise.cpu(), timings


def compare_paths(outputs, traces):
    repeated = differences(outputs['stock'], outputs['stock_repeat'])
    calls = []
    for index, (stock, custom) in enumerate(zip(traces['stock'], traces['bidirectional'], strict=True)):
        calls.append({'index': index, 'fields': {name: differences(stock[name], custom[name]) for name in stock}})
    repeat_calls = all(torch.equal(left[key], right[key])
                       for left, right in zip(traces['stock'], traces['stock_repeat'], strict=True) for key in left)
    matched = all(field['unequal_elements'] == 0 for call in calls for field in call['fields'].values())
    terminal = differences(outputs['bidirectional'], traces['stock'][-1]['prediction'])
    result = {'stock_repeat': repeated, 'stock_repeat_calls_exact': repeat_calls, 'calls': calls,
            'all_call_inputs_and_predictions_exact': matched,
            'custom_vs_stock_terminal_prediction': terminal,
            'final_outputs': differences(outputs['stock'], outputs['bidirectional']),
            'terminal_difference_only': repeated['unequal_elements'] == 0 and repeat_calls and matched
                                       and terminal['unequal_elements'] == 0}
    if 'bidirectional_default' in outputs:
        result['global_sigma_precision'] = {
            'baseline': 'float32 global sigma', 'changed': f'ordinary {common.SIGMA_PRECISION} global sigma',
            'outputs': differences(outputs['bidirectional'], outputs['bidirectional_default']),
            'calls': [{'index': index, 'fields': {
                name: dict(differences(left[name], right[name]), baseline_dtype=str(left[name].dtype),
                           changed_dtype=str(right[name].dtype)) for name in left}}
                for index, (left, right) in enumerate(zip(traces['bidirectional'],
                                                         traces['bidirectional_default'], strict=True))]}
    return result


def load_reference(path, protocol, image, context, grid):
    """Reuse saved controls only when their inputs and computation owners match."""
    result_path = path/'result.json'
    record = json.loads(result_path.read_text())
    original = record['protocol']
    software.validate(original['software'])
    old_sources = {key: value for key, value in original['software']['sources'].items() if key != ENTRY}
    current_sources = {key: value for key, value in protocol['software']['sources'].items() if key != ENTRY}
    if old_sources != current_sources or original['software']['runtime'] != protocol['software']['runtime']:
        raise ValueError('stock reference computation owners or runtime differ')
    for key in ('schedule', 'seed', 'frames', 'fps', 'prompt', 'input_files', 'guidance', 'initialization'):
        if original[key] != protocol[key]:
            raise ValueError(f'stock reference {key} differs')
    if original['global_sigma_dtype'] != 'float32':
        raise ValueError('stock reference must use float32 global sigma')
    if (record['raw']['stock_repeat']['unequal_elements'] != 0
            or not record['raw']['stock_repeat_calls_exact'] or not record['raw']['terminal_difference_only']
            or record['decoded']['stock_repeat']['unequal_elements'] != 0):
        raise ValueError('stock reference controls did not pass')
    identities = {str(result_path.resolve()): sha256(result_path)}
    for name, digest in record['output_files'].items():
        if Path(name).name != name or sha256(path/name) != digest:
            raise ValueError('stock reference raw file differs')
        identities[str((path/name).resolve())] = digest
    outputs = {name: grid.patchify(torch.load(path/(name+'.pt'), map_location='cpu', weights_only=True))
               for name in ('stock', 'stock_repeat', 'bidirectional')}
    traces = torch.load(path/'traces.pt', map_location='cpu', weights_only=True)
    noise = torch.load(path/'noise.pt', map_location='cpu', weights_only=True)
    expected = grid.tools.create_initial_state('cpu', DTYPE).latent.shape
    if noise.shape != expected or noise.dtype != DTYPE or not torch.isfinite(noise).all():
        raise ValueError('stock reference saved noise differs from the grid')
    for name, value in (('noise', noise), ('image', image), ('text', context)):
        if evaluate.tensor_sha256(value) != record[name+'_tensor_sha256']:
            raise ValueError(f'stock reference {name} tensor differs')
    actual = compare_paths(outputs, traces)
    if not actual['terminal_difference_only']:
        raise ValueError('stock reference actual raw controls differ')
    return outputs, traces, noise, identities, {'path': str(result_path.resolve()), 'sha256': sha256(result_path),
                                              'original_software': original['software'],
                                              'controls': 'reused historical raw/RGB controls; no fresh stock calls'}


def prepare(args):
    if args.output.exists():
        raise ValueError('stock check requires a fresh output directory')
    if args.steps < 2 or args.frames < 2:
        raise ValueError('stock check requires at least two steps and encoded frames')
    producer = software.capture('evaluation', 'bidirectional', decoder=True, extra_sources=EXTRA_SOURCES)
    paths = (args.first_image, args.text_record)
    identities = {str(path.resolve()): sha256(path) for path in paths}
    image_record = torch.load(args.first_image, map_location='cpu', weights_only=True)
    image, fps = dataset.load_training_master(args.first_image, bundle=image_record)
    if (image_record.get('input_role') != 'supplied_image' or image_record.get('pixel_frames') != 1
            or image.shape[1] != 1):
        raise ValueError('stock check requires an actual supplied-image encode')
    fixed = json.loads(args.text_record.read_text())
    if fixed.get('kind') != 'onestep_avatar.preview_inputs' or fixed.get('schema_version') != 2:
        raise ValueError('stock check requires a version-two fixed text record')
    if fixed.get('sha256') != subset.record_hash(fixed):
        raise ValueError('fixed text record hash differs')
    software.validate(fixed['software'])
    text_file = fixed['input_files']['text']
    text_path = Path(text_file['path'])
    if not text_path.is_absolute() or sha256(text_path) != text_file['sha256']:
        raise ValueError('fixed text file differs')
    identities[str(text_path.resolve())] = text_file['sha256']
    context = torch.load(text_path, map_location='cpu', weights_only=True)
    if (not isinstance(context, torch.Tensor) or context.dtype != DTYPE or not torch.isfinite(context).all()
            or evaluate.tensor_sha256(context) != text_file['tensor_sha256']):
        raise ValueError('fixed text tensor differs or is not native bf16')
    evaluation_args = evaluate.parse_args([*fixed['evaluation_arguments'], '--output', str(args.output)])
    prompt = DEFAULT_PROMPT if evaluation_args.prompt is None else evaluation_args.prompt
    model = backbone.resolve(args.model, 'dev')
    if image.shape[0] != model.caps.latent_channels or image_record['vae_fingerprint'] != precompute.file_fingerprint(
            Path(model.paths.video_vae())):
        raise ValueError('supplied image channels or VAE differs from native model')
    levels = LTX2Scheduler().execute(steps=args.steps).float()
    common.ClipGrid.build(args.frames, image.shape[2]*model.scale_factors.height,
                         image.shape[3]*model.scale_factors.width, fps, model,
                         device=torch.device('cpu'), dtype=DTYPE, latent_channels=model.caps.latent_channels)
    for path in (Path(model.paths.transformer()), Path(model.paths.video_vae())):
        identities[str(path.resolve())] = sha256(path)
    check_current(identities, producer)
    return model, image.unsqueeze(0).to(DTYPE), context, fps, levels, prompt, identities, producer


def check_current(identities, producer):
    software.check_current(producer)
    if any(sha256(Path(path)) != digest for path, digest in identities.items()):
        raise ValueError('stock check inputs or weights changed')


def execute(args):
    model, image, context, fps, levels, prompt, identities, producer = prepare(args)
    protocol = dict(PROTOCOL, schedule=levels.tolist(), seed=args.seed, frames=args.frames, fps=fps,
                    prompt=prompt, input_files=identities, software=producer)
    reference = None
    if args.reference_run is not None:
        cpu_grid = common.ClipGrid.build(args.frames, image.shape[3]*model.scale_factors.height,
                                        image.shape[4]*model.scale_factors.width, fps, model,
                                        device=torch.device('cpu'), dtype=DTYPE)
        outputs, traces, noise, reference_files, reference = load_reference(
            args.reference_run, protocol, image, context, cpu_grid)
        identities.update(reference_files)
        protocol.update(reference=reference, changed_global_sigma_dtype=common.SIGMA_PRECISION,
                        generated_paths=['bidirectional_default'])
    if args.dry_run:
        return protocol
    from scripts.prune.core import preflight
    preflight.check(args.model, gpu_id=args.gpu_id, transformer_path=model.paths.transformer())
    check_current(identities, producer)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    device = torch.device(f'cuda:{args.gpu_id}')
    image, context, levels = image.to(device), context.to(device), levels.to(device)
    grid = common.ClipGrid.build(args.frames, image.shape[3]*model.scale_factors.height,
                                image.shape[4]*model.scale_factors.width, fps, model, device=device, dtype=DTYPE)
    session = Session(model, device, 'onestep_avatar.stock_parity', context)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode(), session.transformer(video_tools=grid.tools) as transformer:
        torch.cuda.synchronize(device)
        loaded_seconds = time.perf_counter()-started
        load_peak = torch.cuda.max_memory_allocated(device)
        if reference is None:
            outputs, traces, noise, timings = sample_paths(transformer, context, grid, image, levels, args.seed)
        else:
            trace = Trace(transformer)
            torch.cuda.reset_peak_memory_stats(device)
            sample_started = time.perf_counter()
            source = grid.tools.create_initial_state(device, DTYPE).latent
            tokens, _ = bidirectional.sample(common.denoised_from_x0_model(trace), context, grid, source,
                                             grid.patchify(image), schedule=levels.tolist(), seed=args.seed,
                                             epsilon=noise.to(device))
            outputs['bidirectional_default'] = tokens.cpu()
            traces['bidirectional_default'] = trace.calls
            torch.cuda.synchronize(device)
            timings = {'bidirectional_default': {'seconds_including_traces': time.perf_counter()-sample_started,
                       'forward_calls': len(trace.calls), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                       'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}}
            del trace, tokens, source
    del transformer
    torch.cuda.synchronize(device)
    execution = {'seconds_including_model_load': time.perf_counter()-started,
                 'model_load_seconds': loaded_seconds, 'paths': timings,
                 'peak_allocated_bytes': max(load_peak, *(item['peak_allocated_bytes'] for item in timings.values())),
                 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    check_current(identities, producer)
    raw = compare_paths(outputs, traces)
    latents = {name: grid.unpatchify_block(value.to(device), args.frames).cpu() for name, value in outputs.items()}
    for name, value in dict(latents, noise=noise, traces=traces).items():
        dataset.atomic_write(args.output/(name+'.pt'), lambda path, value=value: torch.save(value, path))
    pixels = {}
    with torch.inference_mode(), session.decoder() as decoder:
        for name, value in latents.items():
            if reference is not None and name in ('stock', 'stock_repeat'):
                continue
            check_current(identities, producer)
            pixels[name] = media.decode(session, value, decoder, args.seed)
    del decoder
    decoded_baseline = 'stock' if reference is None else 'bidirectional'
    decoded = {name: differences(pixels[decoded_baseline], value)
               for name, value in pixels.items() if name != decoded_baseline}
    titles = ([('stock', 'Stock video; RGB decoded'), ('stock_repeat', 'Stock repeat; RGB decoded'),
               ('bidirectional', 'Explicit float32; RGB decoded'),
               ('bidirectional_default', 'Ordinary float32; RGB decoded')] if reference is None else
              [('bidirectional', 'RGB; float32 global sigma'), ('bidirectional_default', f'RGB; ordinary {common.SIGMA_PRECISION} global sigma')])
    panels = [media.Panel(name, title, pixels[name], tuple(range(len(pixels[name]))))
              for name, title in titles]
    question = 'Does video sampling match stock?' if reference is None else 'Does global sigma precision change the output?'
    panel_size = (448, 448)
    layout = media.compact_layout(panels, question=question, layout='comparison', panel_size=panel_size)
    rendered, rendering = media.render_panels(panels, question=question, layout=layout, fps=fps, panel_size=panel_size)
    rendering['software'] = producer
    media.save_render(rendered, rendering, args.output/'comparison')
    check_current(identities, producer)
    record = {'protocol': protocol, 'execution': execution, 'raw': raw, 'decoded': decoded,
              'noise_tensor_sha256': evaluate.tensor_sha256(noise), 'text_tensor_sha256': evaluate.tensor_sha256(context),
              'image_tensor_sha256': evaluate.tensor_sha256(image),
              'output_files': {path.name: sha256(path) for path in args.output.glob('*.pt')},
              'acceptance': 'native evidence requires review; this record alone does not close E1'}
    dataset.atomic_write(args.output/'result.json', lambda path: path.write_text(json.dumps(record, indent=2)+'\n'))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('first-image', 'text-record', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--model', default='2.5')
    parser.add_argument('--frames', type=int, default=17)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--reference-run', type=Path,
                        help='reuse a checked stock comparison and generate only the ordinary-sigma arm')
    record = execute(parser.parse_args(argv))
    print(json.dumps(record if record.get('scope') else record['raw'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
