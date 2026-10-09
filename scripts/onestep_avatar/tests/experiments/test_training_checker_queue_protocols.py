"""Pinned selectors retain numerical checks and refuse changed saved science."""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest
import torch

from scripts.onestep_avatar import LTX_ROOT
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.experiments import stats, training_slice_check, training_update_check
from scripts.onestep_avatar.hashing import sha256


def write_spec(path: Path, protocol: str, arguments: list[str], operation: str = 'run') -> Path:
    path.write_text(json.dumps({'schema_version': 1, 'protocol': protocol,
                               'arguments': arguments, 'operation': operation}))
    return path


@pytest.mark.parametrize(('module', 'selector', 'operation', 'arguments'), [
    (training_update_check, 'training_update_check', 'reference', ['--job', '/fixed/job.json']),
    (training_update_check, 'training_update_check', 'compare',
     ['--job', '/fixed/job.json', '--reference', '/reference']),
    (training_slice_check, 'training_slice_check', 'run', ['--job', '/fixed/job.json', '--execution', 'fsdp']),
    (training_slice_check, 'training_slice_check', 'run', ['--job', '/fixed/job.json', '--execution', 'serial']),
    (training_slice_check, 'training_slice_check', 'compare', ['--compare', '/left', '/right']),
    (stats, 'stats', 'run', ['--pairs', '/corpus', '--no-gpu']),
])
def test_pinned_settings_and_launch_topology(tmp_path: Path, module: ModuleType, selector: str, operation: str,
                                             arguments: list[str]) -> None:
    spec = write_spec(tmp_path / 'spec.json', selector, arguments, operation)
    root = tmp_path / 'output'
    args = module.read_spec(spec, root)
    assert args.spec_sha256 == sha256(spec)
    assert args.output == root
    assert not root.exists()
    if module == training_slice_check:
        launch = module.launch_requirements(spec)
        assert launch['processes'] == 1
        if '--execution' in arguments and 'fsdp' in arguments:
            assert launch == {'processes': 1, 'launcher': 'accelerate', 'mixed_precision': 'bf16',
                              'accelerate_config': str(LTX_ROOT / 'packages/ltx-trainer/configs/accelerate/fsdp.yaml')}
        else:
            assert launch['launcher'] == 'python'
    if operation == 'compare':
        assert module.resource_phases(spec) == ['runtime']
        assert module.dependency_outputs(spec) == ([Path('/reference')] if module == training_update_check
                                                  else [Path('/left'), Path('/right')])


@pytest.mark.parametrize('arguments', [
    ['--pairs', '/corpus', '--output', '/elsewhere'], ['--pairs', '/corpus', '--gpu-id=7'],
    ['--pairs', '/corpus', '--dry-run'], ['--pairs', '/corpus', '--out', '/elsewhere'],
    ['--pairs', '/corpus', '--pairs', '/other'], ['--pairs', 'relative/corpus'],
    ['--pairs', '/corpus', '--sigma0', 'nan'], ['--pairs', '/corpus', '--eps-samples', '0'],
    ['--pairs', '/corpus', '--supervise'], ['--pairs', '/corpus', '--proces', '/ledger'],
])
def test_execution_overrides_and_invalid_science_fail_without_writes(tmp_path: Path, arguments: list[str]) -> None:
    spec = write_spec(tmp_path / 'spec.json', 'stats', arguments)
    root = tmp_path / 'output'
    with pytest.raises(ValueError, match=r"specification|scientific"):
        stats.read_spec(spec, root)
    assert not root.exists()


def test_duplicate_json_fields_and_abbreviated_outer_flags_fail(tmp_path: Path) -> None:
    path = tmp_path / 'spec.json'
    path.write_text('{"schema_version":1,"protocol":"stats","protocol":"stats","arguments":["--pairs","/corpus"]}')
    with pytest.raises(ValueError, match='repeats'):
        stats.read_spec(path, tmp_path / 'output')
    with pytest.raises(SystemExit):
        stats.parse_args(['--spe', str(path), '--output', '/output'])


@pytest.fixture
def pair_completion(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / 'corpus' / 'actor' / 'view'
    corpus.mkdir(parents=True)
    guide = torch.arange(32, dtype=torch.float32).reshape(2, 4, 2, 2)
    for name, tensor in [(dataset.guide_bundle_name('white'), guide),
                         (dataset.capture_bundle_name('white'), guide + 2)]:
        torch.save({'schema_version': 2, 'master': tensor}, corpus / name)
    spec = write_spec(tmp_path / 'spec.json', 'stats', ['--pairs', str(corpus.parent.parent), '--no-gpu',
                                                     '--objective', 'white'])
    output = tmp_path / 'output'
    assert stats.main(['--spec', str(spec), '--output', str(output)]) == 0
    return spec, output


def test_cpu_stats_completion_uses_real_saved_pair_arithmetic(pair_completion: tuple[Path, Path],
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    spec, output = pair_completion
    monkeypatch.setattr(stats, 'measure_map', lambda *_args, **_kwargs: pytest.fail('native map loader'))
    stats.verify_completion(spec, output)
    report = json.loads((output / 'analysis_summary.json').read_text())
    assert report['pairs']['r_full']['mean'] == 2.0
    assert stats.evidence_paths(spec, output) == [output / 'manifest.json', output / 'analysis_summary.json',
                                               output / 'inputs.json']


@pytest.mark.parametrize('change', ['rehashed_metric', 'source', 'missing', 'escaped', 'spec'])
def test_stats_completion_refuses_changed_science(pair_completion: tuple[Path, Path], change: str) -> None:
    spec, root = pair_completion
    if change == 'rehashed_metric':
        path = root / 'analysis_summary.json'
        record = json.loads(path.read_text())
        record['pairs']['r_full']['mean'] = 0
        path.write_text(json.dumps(record))
        manifest = json.loads((root / 'manifest.json').read_text())
        manifest['artifacts']['analysis_summary.json'] = sha256(path)
        (root / 'manifest.json').write_text(json.dumps(manifest))
    elif change == 'source':
        original = Path(next(iter(json.loads((root / 'inputs.json').read_text())['input_files'])))
        torch.save({'schema_version': 2, 'master': torch.zeros(2, 4, 2, 2)}, original)
    elif change == 'missing':
        (root / 'inputs.json').unlink()
    elif change == 'escaped':
        (root / 'escaped.json').symlink_to(spec)
    else:
        spec.write_text(spec.read_text() + '\n')
    with pytest.raises(ValueError, match=r"checker|stats"):
        stats.verify_completion(spec, root)


def test_saved_map_retains_original_metrics_and_clean_first_frame(tmp_path: Path) -> None:
    source = tmp_path / 'guide.mp4'
    source.write_bytes(b'source bytes; decoder forbidden')
    spec = write_spec(tmp_path / 'spec.json', 'stats', ['--renders', str(tmp_path), '--eps-samples', '2'])
    args = stats.read_spec(spec, tmp_path / 'output')
    guide = torch.ones(1, 128, 2, 1, 1)
    outputs = [guide.clone(), guide.clone()]
    outputs[0][:, :, 1:] = 0
    outputs[1][:, :, 1:] = 2
    raw = {'guide.mp4': {'source': str(source), 'source_sha256': sha256(source), 'guide': guide, 'outputs': outputs}}
    measured = stats._saved_map(args, raw)
    assert measured['by_video']['guide.mp4']['excursion_a'] == pytest.approx(2 ** -0.5)
    assert measured['by_video']['guide.mp4']['eps_spread'] == pytest.approx(2 ** 0.5)
    outputs[0][:, :, :1] = 0
    with pytest.raises(ValueError, match='clean first frame'):
        stats._saved_map(args, raw)


def test_native_replay_budget_matches_original_job(tmp_path: Path) -> None:
    budget = tmp_path / 'budget.json'
    budget.write_text('{}')
    job = tmp_path / 'job.json'
    job.write_text(json.dumps({'arguments': ['--mode', 'bidirectional', '--subset', '/membership.json',
                                            '--output', '/original', '--resource-budget', str(budget)]}))
    spec = write_spec(tmp_path / 'spec.json', 'training_update_check', ['--job', str(job)], 'reference')
    assert training_update_check.resource_budget(spec) == budget
