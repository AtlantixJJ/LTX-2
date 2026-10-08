"""Actual product preflight separates guide coverage from training selection."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.onestep_avatar import infer
from scripts.onestep_avatar.tests import test_infer
from scripts.onestep_avatar.tests.test_checkpoint_contract import A, B, _contract
from scripts.onestep_avatar.training.checkpoints import CONTRACT_KEY


@pytest.fixture
def product_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    return test_infer.product_files.__wrapped__(tmp_path, monkeypatch)


def _case(files: tuple, tmp_path: Path, *, explicit: bool = False, teacher: bool = False) -> list[str]:
    args, guide, _image = files
    args = list(args)
    args[args.index('--mode') + 1] = 'causal'
    bundle = torch.load(guide, weights_only=True)
    bundle.update(master=torch.zeros(128, 17, 2, 2), pixel_frames=129)
    torch.save(bundle, guide)
    contract = _contract('causal', teacher=teacher)
    if explicit:
        contract['mode_settings']['span_latent_frames'] = 7
    checkpoint = tmp_path / 'adapter.safetensors'
    save_file({A: torch.ones(2, 4), B: torch.zeros(4, 2)}, checkpoint,
              metadata={CONTRACT_KEY: json.dumps(contract)})
    return [*args, '--checkpoint', str(checkpoint), '--span-latent-frames', '7']


@pytest.mark.parametrize('explicit', [False, True])
def test_complete_coverage_matches_default_and_explicit_calibration(
    product_files: tuple, tmp_path: Path, explicit: bool,
) -> None:
    args = infer.parse_args(_case(product_files, tmp_path, explicit=explicit))
    assert args.span_latent_frames == 7
    assert args.mode_settings.span_latent_frames is None
    _spec, guide, image, _fps, requested, checked = infer.prepare_product(args)
    assert guide.shape[2] == 7
    assert image.shape[2] == 1
    assert requested['shape']['frames'] == 7
    assert requested['mode_settings']['span_latent_frames'] == (7 if explicit else None)
    assert requested['mode_settings']['blocks_per_sample'] == 3
    assert checked['overrides'] == []
    assert not args.output.exists()


@pytest.mark.parametrize('changed', ['frames', 'context', 'block', 'schedule', 'teacher'])
def test_incompatible_coverage_or_computation_still_fails_before_execution(
    product_files: tuple, tmp_path: Path, changed: str,
) -> None:
    options = _case(product_files, tmp_path, teacher=changed == 'teacher')
    if changed == 'frames':
        options[options.index('--span-latent-frames') + 1] = '9'
    elif changed == 'context':
        options += ['--context-latent-frames', '16']
    elif changed == 'block':
        options += ['--block-latent-frames', '1']
    elif changed == 'schedule':
        options[options.index('--schedule') + 1] = '0.7'
    args = infer.parse_args(options)
    with pytest.raises(ValueError, match='incompatible adapter conditions'):
        infer.prepare_product(args)
    assert not args.output.exists()
