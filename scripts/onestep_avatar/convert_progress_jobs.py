"""Convert saved historical progress rows to package evaluation job data.

Read an archived row inventory and checked converted membership. Preserve the
two original sources in order, seed, exact scheduler levels, arm and causal
geometry. Trained rows read saved configuration and name their checkpoint;
frozen rows use the base. Explicit research overrides identify off-condition
diagnostics. Write a fresh job list only after every row passes validation.
No model session, GPU selection, queue dispatch or report execution occurs.
"""

import argparse
import json
import shlex
from pathlib import Path

from scripts.onestep_avatar import subset
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model.sampling import thinned_truncated_schedule

SOURCES = ("Part_1/0008_01/views/view00_cam51", "Part_2/0013_07/views/view01_cam57")
SIGMA = 0.8977352380752563


def convert(study: Path) -> dict:
    """Return exact progress jobs; existing checkpoints need not be complete yet."""
    study = study.resolve()
    archive = json.loads((study / "configs/historical_visualization_jobs.json").read_text())
    original = study / "configs/eval_jobs.txt"
    if sha256(original) != archive["source_sha256"]:
        raise ValueError("original progress job inventory changed")
    rows = [shlex.split(line, comments=True) for line in original.read_text().splitlines()]
    expected = [{"run": row[0], "step": int(row[1]), "split": row[2],
                 "subset": row[3] if len(row) > 3 else "pilot_white_k8"}
                for row in rows if row and row[2].startswith("vis_")]
    if archive["jobs"] != expected:
        raise ValueError("archived progress rows differ from the original inventory")
    jobs = []
    for row in archive["jobs"]:
        run, step, split = row["run"], row["step"], row["split"]
        if split not in ("vis_k1", "vis_k4"):
            raise ValueError("unsupported historical progress schedule")
        membership_path = study / "converted" / f"{row['subset']}.membership.json"
        membership = json.loads(membership_path.read_text())
        subset.validate_membership(membership)
        if not set(SOURCES).issubset({source["relative_dir"] for source in membership["sources"]}):
            raise ValueError("historical progress sources are absent from membership")
        frozen = run.startswith("frozen_dev_sanity_")
        config = ({"guide_mode": "d1" if run.endswith("d1") else "d0",
                   "block_latent_frames": 16, "context_latent_frames": 8, "teacher_forcing": False}
                  if frozen else json.loads((study / "runs" / run / "config.json").read_text()))
        if config["block_latent_frames"] != 16:
            raise ValueError("historical progress conversion requires one 17-frame block")
        output = study / "runs/package_progress" / run / f"step{step}" / split
        arguments = ["--mode", "causal", "--subset", str(membership_path),
                     "--frame-plan", str(membership_path.with_name(f"{row['subset']}.frame_plan.json")),
                     "--output", str(output), "--model", "2.5", "--variant", "dev",
                     "--guide-mode", config["guide_mode"], "--span-latent-frames", "17",
                     "--block-latent-frames", "16", "--blocks-per-sample", "1",
                     "--context-latent-frames", str(config["context_latent_frames"]),
                     "--seed", "42", "--cfg", "1", "--stg", "0", "--rescale", "0",
                     "--research-override", "--schedule",
                     *map(str, thinned_truncated_schedule(SIGMA, 30, int(split[-1])))]
        for source in SOURCES:
            arguments.extend(["--source", source])
        if config["teacher_forcing"]:
            arguments.append("--teacher-forcing")
        if not frozen:
            arguments.extend(["--checkpoint", str(study / "runs" / run / "checkpoints" /
                                                 f"lora_weights_step_{step:05d}.safetensors")])
        jobs.append({"id": f"legacy_progress_{run.replace('/', '_')}_{step}_{split}",
                     "kind": "evaluate", "dependencies": [], "arguments": arguments,
                     "output": str(output), "completion": {"records": [
                         str(output / f"case_{index:04d}/variant_000/result.json")
                         for index in range(len(SOURCES))]}})
    return {"schema_version": 1, "jobs": jobs}


def main() -> None:
    """Publish converted data exclusively, so old evidence cannot be overwritten."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = convert(args.study)
    from scripts.onestep_avatar.execution.queue import validate_job_list  # noqa: PLC0415 -- data validation only

    validate_job_list(record)
    with args.output.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
