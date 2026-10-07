"""Converted historical training jobs preserve settings while using package ownership."""

import json
import shlex
from pathlib import Path

import pytest

from scripts.onestep_avatar import convert_progress_jobs, queue

ROOT = Path(__file__).resolve().parents[4]
EXPR = ROOT / "expr/onestep_avatar/dev_training_20261001"


def test_converted_training_jobs_cover_each_active_legacy_row_exactly_once() -> None:
    jobs = json.loads((EXPR / "configs/package_train_jobs.json").read_text())["jobs"]
    rows = [shlex.split(line, comments=True) for line in (EXPR / "configs/train_jobs.txt").read_text().splitlines()]
    rows = [row for row in rows if row]
    assert len(jobs) == len({row[0] for row in rows})
    assert {job["id"].removeprefix("legacy_train_") for job in jobs} == {
        row[0].replace("/", "_") for row in rows
    }
    for row in rows:
        name, subset, port, *legacy = row
        job = next(item for item in jobs if item["id"] == "legacy_train_" + name.replace("/", "_"))
        args = job["arguments"]
        assert args[args.index("--mode") + 1] == "causal"
        assert args[args.index("--subset") + 1].endswith(f"converted/{subset}.membership.json")
        assert args[args.index("--frame-plan") + 1].endswith(f"converted/{subset}.frame_plan.json")
        assert args[args.index("--output") + 1].endswith(f"runs/{name}")
        assert job["port"] == int(port)
        assert job["processes"] == 4
        for flag in ("--guide-mode", "--block-latent-frames", "--sigma-levels", "--noise-policy",
                     "--warmup-steps", "--steps", "--save-every"):
            if flag in legacy:
                assert flag in args
        assert args[args.index("--seed") + 1] == "42"
        assert args[args.index("--lora-rank") + 1] == "16"
        assert args[args.index("--lora-alpha") + 1] == "16"


def test_training_execution_has_no_expr_launcher_or_forwarding_wrapper() -> None:
    assert not (EXPR / "code/train_queue.py").exists()
    assert not (EXPR / "code/train_job.sh").exists()
    assert (ROOT / "LTX-2/scripts/onestep_avatar/queue.py").is_file()
    assert (ROOT / "LTX-2/scripts/onestep_avatar/configs/fsdp_forward_prefetch.yaml").is_file()


def test_full_evaluation_jobs_use_package_split_and_explicit_saved_outputs() -> None:
    jobs = json.loads((EXPR / "configs/package_eval_jobs.json").read_text())["jobs"]
    assert len(jobs) == 417
    assert len({job["id"] for job in jobs}) == len(jobs)
    assert len({job["output"] for job in jobs}) == len(jobs)
    for job in jobs:
        args = job["arguments"]
        assert args[args.index("--mode") + 1] == "causal"
        assert args[args.index("--split") + 1] in {"train", "validation"}
        assert "--frame-plan" in args
        assert "--span-latent-frames" in args
        assert args[args.index("--output") + 1] == job["output"]
        assert "/runs/package_eval/" in job["output"]
        assert job["completion"]["records"]


def test_evaluation_execution_has_no_expr_launcher_and_archives_progress_rows() -> None:
    for name in ("eval_queue.py", "eval_ckpt.sh", "views.py"):
        assert not (EXPR / "code" / name).exists()
    archived = json.loads((EXPR / "configs/historical_visualization_jobs.json").read_text())
    assert archived["status"] == "archived_input_only"
    assert archived["owner"] == "scripts.onestep_avatar"
    assert archived["jobs"]
    assert all(job["split"].startswith("vis_") for job in archived["jobs"])
    assert (ROOT / "LTX-2/scripts/onestep_avatar/evaluate.py").is_file()


def test_progress_jobs_preserve_fixed_sources_and_exact_schedules() -> None:
    record = convert_progress_jobs.convert(EXPR)
    saved = json.loads((EXPR / "configs/package_progress_jobs.json").read_text())
    assert record == saved
    assert len(queue.validate_job_list(record)) == 52
    for job in record["jobs"]:
        args = job["arguments"]
        sources = [args[index + 1] for index, value in enumerate(args) if value == "--source"]
        assert sources == list(convert_progress_jobs.SOURCES)
        assert args[args.index("--seed") + 1] == "42"
        assert "--research-override" in args
        assert "--split" not in args
        start = args.index("--schedule") + 1
        end = args.index("--source")
        levels = list(map(float, args[start:end]))
        expected = ([0.8977352380752563, 0.0] if job["id"].endswith("vis_k1") else
                    [0.8977352380752563, 0.8355841040611267, 0.7364180088043213,
                     0.48016369342803955, 0.0])
        assert levels == expected
        assert len(job["completion"]["records"]) == 2
        if "frozen" not in job["id"]:
            assert "--checkpoint" in args


@pytest.mark.parametrize("defect", ["original", "archived"])
def test_progress_conversion_refuses_changed_historical_inventory(tmp_path: Path, defect: str) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    for name in ("eval_jobs.txt", "historical_visualization_jobs.json"):
        (configs / name).write_bytes((EXPR / "configs" / name).read_bytes())
    if defect == "original":
        with (configs / "eval_jobs.txt").open("a") as stream:
            stream.write("\n# changed inventory\n")
    else:
        path = configs / "historical_visualization_jobs.json"
        archived = json.loads(path.read_text())
        archived["jobs"].pop()
        path.write_text(json.dumps(archived))
    with pytest.raises(ValueError, match="inventory"):
        convert_progress_jobs.convert(tmp_path)
