"""Explicit current source/runtime identity; see doc/software.md."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import torch

from scripts.onestep_avatar.hashing import sha256

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = 'scripts/onestep_avatar/'
PROFILES = ('training', 'evaluation', 'inference', 'decoding', 'preparation')
DISTRIBUTIONS = ('torch', 'peft', 'safetensors', 'accelerate', 'transformers',
                 'ltx-core', 'ltx-pipelines', 'ltx-trainer', 'triton', 'flash-attn', 'natten',
                 'numpy', 'Pillow', 'opencv-python', 'imageio', 'imageio-ffmpeg', 'lpips')
COMMON = ('software.py', 'hashing.py', 'dataset.py', 'subset.py', 'precompute.py',
          'geometry.py', 'mask_video.py', 'training/config.py',
          'training/checkpoints.py', 'model/common.py', 'model/sampling.py', 'model/backbone.py',
          'model/adapters.py')
ENTRIES = {'training': ('train.py', 'training/engine.py', 'training/startup.py', 'training/update_state.py',
                        'training/resources.py', 'training/runtime.py', 'training/consumer_trace.py',
                        'training/numerics.py',
                        'supervision.py', 'process_registry.py', 'queue.py', 'queue_protocol.py'),
           'evaluation': ('evaluate.py', 'stock_parity.py'), 'inference': ('infer.py', 'evaluate.py'),
           'decoding': ('media.py', 'evaluate.py', 'decode_saved.py', 'sigma_sweep.py', 'sigma_sweep_results.py'),
           'preparation': ('prepare_inputs.py', 'media.py', 'evaluate.py', 'training/engine.py')}
GROUPS = ('scripts/prune/core', 'packages/ltx-core/src/ltx_core/model/transformer',
          'packages/ltx-core/src/ltx_core/loader', 'packages/ltx-core/src/ltx_core/guidance',
          'packages/ltx-core/src/ltx_core/components', 'packages/ltx-pipelines/src/ltx_pipelines/utils',
          'packages/ltx-core/src/ltx_core/conditioning',
          'packages/ltx-core/src/ltx_core/text_encoders')
DEPENDENCIES = ('scripts/prune/data/prompt_cache.py', 'packages/ltx-trainer/src/ltx_trainer/model_loader.py',
                'packages/ltx-core/src/ltx_core/model/disposable.py',
                'packages/ltx-core/src/ltx_core/model/model_protocol.py',
                'packages/ltx-core/src/ltx_core/batch_split.py',
                'packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py',
                'packages/ltx-core/src/ltx_core/tools.py', 'packages/ltx-core/src/ltx_core/types.py',
                'packages/ltx-core/src/ltx_core/utils.py')


def runtime_versions() -> dict:
    versions = {}
    for name in DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'torch_cuda_build': torch.version.cuda,
            'distributions': versions}


def _digest(record: dict) -> str:
    payload = {key: value for key, value in record.items() if key != 'sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_files(profile: str, mode: str | None, *, decoder: bool = False) -> list[str]:
    if profile not in PROFILES or mode not in (None, 'bidirectional', 'causal'):
        raise ValueError('unsupported software profile or mode')
    if profile not in ('decoding', 'preparation') and mode is None:
        raise ValueError('model software manifests require an explicit mode')
    files = {PACKAGE + name for name in (*COMMON, *ENTRIES[profile])}
    if mode is not None:
        files.add(PACKAGE + 'model/' + mode + '.py')
    groups = list(GROUPS)
    files.update(DEPENDENCIES)
    if decoder or profile in ('decoding', 'preparation'):
        files.update((PACKAGE + 'media.py', 'scripts/prune/evaluate/decode.py',
                      'packages/ltx-trainer/src/ltx_trainer/video_utils.py'))
        groups.append('packages/ltx-core/src/ltx_core/model/video_vae')
    for group in groups:
        directory = ROOT / group
        members = sorted(directory.rglob('*.py'))
        if not members:
            raise ValueError(f'software source group is absent: {group}')
        files.update(str(path.relative_to(ROOT)) for path in members)
    if any(not (ROOT / name).is_file() for name in files):
        raise ValueError('a declared software source owner is absent')
    return sorted(files)


def capture(profile: str, mode: str | None = None, *, decoder: bool = False,
            extra_sources: tuple[str, ...] = ()) -> dict:
    extras = sorted(set(extra_sources))
    if any(not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts
           or Path(name).suffix != '.py' or not (ROOT/name).is_file()
           or not (ROOT/name).resolve().is_relative_to(ROOT) for name in extras):
        raise ValueError('extra software owners must be contained Python source files')
    record = {'schema_version': 1, 'kind': 'onestep_avatar.software', 'profile': profile,
              'mode': mode, 'decoder': decoder,
              'sources': {name: sha256(ROOT / name) for name in source_files(profile, mode, decoder=decoder)},
              'runtime': runtime_versions()}
    if extras:
        record['extra_sources'] = extras
        record['sources'].update({name: sha256(ROOT/name) for name in extras})
    record['sha256'] = _digest(record)
    return record


def validate(record: dict) -> None:
    """Check saved integrity without requiring today's source or installed runtime."""
    if (not isinstance(record, dict) or record.get('schema_version') != 1
            or record.get('kind') != 'onestep_avatar.software'):
        raise ValueError('software manifest schema is invalid')
    if (record.get('profile') not in PROFILES or record.get('mode') not in (None, 'bidirectional', 'causal')
            or type(record.get('decoder')) is not bool
            or (record['profile'] not in ('decoding', 'preparation') and record['mode'] is None)):
        raise ValueError('software manifest profile/mode is invalid')
    sources = record.get('sources')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('software manifest sources are missing')
    extras = record.get('extra_sources', [])
    if (not isinstance(extras, list) or any(not isinstance(name, str) for name in extras)
            or extras != sorted(set(extras)) or any(name not in sources for name in extras)):
        raise ValueError('software manifest extra owners are malformed')
    for name, digest in sources.items():
        if (not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts
                or not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None):
            raise ValueError('software manifest source identity is malformed')
    runtime = record.get('runtime')
    if (not isinstance(runtime, dict) or set(runtime) != {'python', 'torch_cuda_build', 'distributions'}
            or not isinstance(runtime.get('python'), str)
            or (runtime.get('torch_cuda_build') is not None and not isinstance(runtime['torch_cuda_build'], str))
            or not isinstance(runtime.get('distributions'), dict)
            or set(runtime['distributions']) != set(DISTRIBUTIONS)
            or any(value is not None and not isinstance(value, str) for value in runtime['distributions'].values())):
        raise ValueError('software manifest runtime is malformed')
    if record.get('sha256') != _digest(record):
        raise ValueError('software manifest runtime or content hash is invalid')


def check_current(record: dict) -> None:
    """Refuse changed owners or runtime before current launch/publication/completion."""
    validate(record)
    actual = capture(record['profile'], record['mode'], decoder=record['decoder'],
                     extra_sources=tuple(record.get('extra_sources', [])))
    if actual != record:
        raise ValueError('software source owners or runtime changed since preflight')
