"""Representative ordinary entry paths load only declared software owners."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.onestep_avatar import LTX_ROOT, PACKAGE_ROOT
from scripts.onestep_avatar.execution import software


@pytest.mark.parametrize(
    ("profile", "mode"), [("evaluation", "causal"), ("evaluation", "bidirectional"), ("decoding", None)]
)
def test_profile_excludes_experiments_and_covers_loaded_entry_owners(
    tmp_path: Path, profile: str, mode: str | None
) -> None:
    declared = software.source_files(profile, mode)
    assert all("/experiments/" not in name and not name.startswith("expr/") for name in declared)
    script = r"""
import json,sys
from pathlib import Path
from scripts.onestep_avatar import LTX_ROOT
from scripts.onestep_avatar.execution import software
profile=sys.argv[1]
mode=None if sys.argv[4]=="none" else sys.argv[4]
if profile=="evaluation":
    from scripts.onestep_avatar import evaluate
    args=evaluate.parse_args(["--mode",mode,"--subset",sys.argv[2],"--output",sys.argv[3],"--schedule","1","0"])
    from scripts.onestep_avatar.model import causal
    from ltx_core.types import SpatioTemporalScaleFactors
    geometry=causal.CausalGeometry(SpatioTemporalScaleFactors(time=8,height=32,width=32),block_latent_frames=2,context_latent_frames=8)
    assert geometry.block_latent_frames==2
    evaluate.encoded_metrics(__import__("torch").zeros(1,128,3,1,1),__import__("torch").zeros(1,128,3,1,1))
else:
    from scripts.onestep_avatar import decode_saved,media
    decode_saved.parse_args(["--jobs",sys.argv[2],"--output",sys.argv[3]])
    media.native_decoder_settings()
    from scripts.onestep_avatar import evaluate
    evaluate.parse_saved_comparison_args(["--render-saved-comparisons",sys.argv[2],"--output",sys.argv[3]])
loaded=[]
for name,module in list(sys.modules.items()):
    if name.startswith("scripts.onestep_avatar") and getattr(module,"__file__",None) and ".tests" not in name:
        loaded.append(str(Path(module.__file__).resolve().relative_to(LTX_ROOT)))
entry_paths=(["evaluate.parse_args","evaluate.encoded_metrics","model.causal.CausalGeometry"]
    if profile=="evaluation" else
    ["decode_saved.parse_args","media.native_decoder_settings","evaluate.parse_saved_comparison_args"])
print(json.dumps({"profile":profile,"mode":mode,"entry_paths":entry_paths,"loaded":sorted(loaded),
    "declared":software.source_files(profile,mode)}))
"""
    hook = PACKAGE_ROOT / "tests/blocked_imports"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", ONESTEP_AVATAR_BLOCK_EXPERIMENTS="1")
    env["PYTHONPATH"] = str(hook) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    run = subprocess.run(
        [sys.executable, "-c", script, profile, str(tmp_path / "input.json"), str(tmp_path / "output"), mode or "none"],
        cwd=LTX_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(run.stdout.splitlines()[-1])
    assert result["loaded"]
    assert set(result["loaded"]).issubset(result["declared"]), sorted(set(result["loaded"]) - set(result["declared"]))
    assert not (tmp_path / "output").exists()
