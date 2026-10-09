"""Owner moves preserve hash bytes, fixed-preview gates and direct CLI routing."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.onestep_avatar import LTX_ROOT, PACKAGE_ROOT, evaluate, hashing


def test_tensor_identity_preserves_shape_dtype_and_original_contiguous_bytes() -> None:
    for dtype in (torch.float32, torch.bfloat16, torch.int64):
        value = torch.arange(24).reshape(2, 3, 4).to(dtype=dtype).transpose(1, 2)
        original = value.detach().cpu().contiguous()
        digest = sha256(json.dumps({"shape": list(original.shape), "dtype": str(original.dtype)}).encode())
        digest.update(original.view(torch.uint8).numpy().tobytes())
        expected = {
            'torch.float32': '5c271c218e4a794da54061cd6990b601fce95ca45b35059f347e23a7625dd54d',
            'torch.bfloat16': '9001928007e5819cb74126c887ca9fd2d29d26417d64d928d8fb6a387b224cd5',
            'torch.int64': '625e3a75724acc1be4e6ad9d86efa5ce5a5c4023f21495b97f5bca7af79ad7db',
        }
        assert hashing.tensor_sha256(value) == digest.hexdigest()
        assert hashing.tensor_sha256(value) == expected[str(dtype)]
        assert torch.equal(value, torch.arange(24).reshape(2, 3, 4).to(dtype=dtype).transpose(1, 2))


def test_tensor_hash_module_import_remains_standard_library_only() -> None:
    command = """import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname in ('torch', 'numpy', 'peft'):
   raise RuntimeError('heavy dependency at hash import: ' + fullname)
sys.meta_path.insert(0, Block())
from scripts.onestep_avatar import hashing
assert hashing.sha256 and hashing.tensor_sha256
"""
    result = subprocess.run([sys.executable, "-c", command], cwd=LTX_ROOT,
                            capture_output=True, text=True, check=False, timeout=15)
    assert result.returncode == 0, result.stderr


def test_fixed_preview_without_validator_fails_before_inputs_handles_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(preview_fixed={"input_files": {}}, output=tmp_path / "output")
    monkeypatch.setattr(evaluate.software, "capture", lambda *_args, **_kwargs: pytest.fail("preflight reached"))
    monkeypatch.setattr(evaluate, "prepare_evaluation", lambda *_args, **_kwargs: pytest.fail("inputs reached"))
    with pytest.raises(ValueError, match="explicit preview tensor validator"):
        evaluate.execute_evaluation(args)
    assert not args.output.exists()


def test_preview_validation_dependency_has_one_direction() -> None:
    tree = ast.parse((PACKAGE_ROOT / "evaluate.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "scripts.onestep_avatar.previews" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "scripts.onestep_avatar.previews"
            if node.module == "scripts.onestep_avatar":
                assert all(alias.name != "previews" for alias in node.names)
    assert "preview_tensor_validator=verify_preview_tensors" in (PACKAGE_ROOT / "previews.py").read_text()


@pytest.mark.parametrize("owner", ["previews", "comparisons"])
def test_moved_output_clis_help_without_loading_models(owner: str) -> None:
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run([sys.executable, "-m", "scripts.onestep_avatar." + owner, "--help"],
                               cwd=LTX_ROOT, env=environment, capture_output=True, text=True, check=False, timeout=30)
    assert completed.returncode == 0, completed.stderr
    expected = "--preview-job" if owner == "previews" else "--render-saved-comparisons"
    assert expected in completed.stdout


@pytest.mark.parametrize("flag", ["--preview-job", "--render-saved-comparisons"])
def test_evaluation_cli_refuses_moved_routes(
    flag: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        evaluate.main(["--mode", "bidirectional", "--subset", str(tmp_path / "subset"),
                       "--output", str(output), "--schedule", "1", "0", flag, str(tmp_path / "job")])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not output.exists()
