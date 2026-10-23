"""Run the frozen GT head recipe through bounded child phases and CPU logging.

Check the configured devices and matching DDP acceptance, run selection, export
the compact base, recover adapters and merge final weights. Resume only bounded
TimeoutError segments with increased published progress; other failures stop.
Publish launch/status records and stop owned loggers after the final receipt.
The default arm list includes energy, which requires a prepared ranking; use
--arms learned_joint for the gate learner alone. See doc/usage.md.

CLI: python -m scripts.prune.learn.run_full --help.
Workflows and launch settings: doc/usage.md."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from scripts.prune.core import records
from scripts.prune.data.learned_head_study import load_study
from scripts.prune.learn import protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["learned_joint", "energy"])
    parser.add_argument("--replicate-seed", type=int, default=42)
    args = parser.parse_args()
    study = load_study(args.study)
    if study["config"].get("monitor_reference") != "gt_capture" or not protocol.is_ddp(study["config"]):
        raise ValueError("direct full run requires a frozen GT uniform-F18 DDP preset")
    protocol.validate(study["config"])
    devices = ",".join(str(gpu) for gpu in study["config"]["gpu_ids"])
    world = len(study["config"]["gpu_ids"])
    method = "uniform_f18_ddp"
    if args.replicate_seed not in study["config"]["replicate_seeds"]:
        raise ValueError("replicate seed is not frozen")
    root = Path(study["root"])
    (root / "logs").mkdir(exist_ok=True)
    (root / "monitoring").mkdir(exist_ok=True)
    (root / "monitoring/owner.pid").write_text(str(os.getpid()) + "\n")
    serial = len(list((root / "logs").glob("full_*_*.log")))
    child = None
    loggers = []
    state = {
        "study": study["identity"],
        "owner_pid": os.getpid(),
        "status": "starting",
        "arms": args.arms,
        "replicate_seed": args.replicate_seed,
    }

    def status():
        state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["gpu"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        state["training"] = {}
        for arm in args.arms:
            folder = root / f"training/{arm}/replicate_{args.replicate_seed}"
            updates = sorted(p for p in (folder / "updates").glob("*.json") if p.stem.isdecimal())
            saved = sorted((folder / "checkpoints").glob("update_*.json"))
            state["training"][arm] = {
                "last_update": json.loads(updates[-1].read_text()) if updates else None,
                "checkpoint": str(saved[-1]) if saved else None,
            }
        temporary = root / "monitoring/.status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "monitoring/status.json")

    def stop(*_):
        raise KeyboardInterrupt("direct full run stopped")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def run(label, module, flags, devices="", four=False):
        nonlocal serial, child
        serial += 1
        log = root / f"logs/full_{serial:03d}_{label}.log"
        command = (
            [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}", "-m", module]
            if four
            else [sys.executable, "-m", module]
        )
        command += ["--study", str(args.study.resolve()), *flags]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
        wrapped = ["timeout", "--signal=TERM", "--kill-after=600s", "13h", *command]
        with log.open("x") as stream:
            child = subprocess.Popen(
                wrapped,
                cwd=Path(__file__).resolve().parents[3],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        state.update(status=label, child_pid=child.pid, log=str(log))
        records.publish_json(
            root / f"provenance/full_launches/{serial:03d}.json",
            {"command": command, "devices": devices, "pid": child.pid, "log": str(log)},
        )
        print(json.dumps(state), flush=True)
        while child.poll() is None:
            status()
            time.sleep(30)
        code = child.returncode
        child = None
        status()
        return code

    def train(phase, arm):
        folder = root / f"training/{arm}/replicate_{args.replicate_seed}"
        _, selection_end, compact_end = protocol.phase_budget(study["config"])
        boundary = selection_end if phase == "selection" else compact_end
        if (folder / f"checkpoints/update_{boundary:04d}.pt").exists():
            return
        for segment in range(12):
            before = len(list((folder / "updates").glob("*.json")))
            code = run(
                f"{arm}_{phase}_{segment}",
                "scripts.prune.learn.heads",
                [
                    "--phase",
                    phase,
                    "--arm",
                    arm,
                    "--replicate-seed",
                    str(args.replicate_seed),
                    "--gpu-id",
                    "0",
                    "--removed-heads",
                    "152",
                ],
                devices,
                True,
            )
            if code == 0:
                return
            failures = sorted((root / "logs").glob(f"{phase}_*.failure.json"), key=lambda p: p.stat().st_mtime)
            failure = json.loads(failures[-1].read_text()) if failures else {}
            if failure.get("error") != "TimeoutError" or len(list((folder / "updates").glob("*.json"))) <= before:
                raise RuntimeError(f"{arm} {phase} failed: {failure}; see {state['log']}")
        raise RuntimeError(f"{arm} {phase} exceeded twelve bounded segments")

    try:
        if "energy" in args.arms and not (root / "calibration/ranking.json").exists():
            raise RuntimeError(
                "the energy arm requires calibration/ranking.json from learn.heads --phase energy_calibration"
            )
        gate = root / "checks/feasibility.json"
        if not gate.exists():
            free = subprocess.check_output(
                ["nvidia-smi", "-i", devices, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
            )
            if any(int(v) < 45056 for v in free.splitlines()):
                raise RuntimeError("GPUs 0–3 must be free for native model initialization")
            if run("ddp_native", "scripts.prune.checks.ddp_native", [], devices, True):
                raise RuntimeError("native distributed acceptance failed")
        acceptance = json.loads(gate.read_text())
        if (
            acceptance.get("study") != study["identity"]
            or not acceptance.get("passed")
            or acceptance.get("method") != method
            or acceptance.get("world_size") != world
        ):
            raise ValueError("native distributed acceptance does not match the frozen study")
        if study["config"].get("wandb", {}).get("enabled", False):
            status()
            for arm in args.arms:
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.prune.learn.wandb_logger",
                    "--study",
                    str(args.study.resolve()),
                    "--arm",
                    arm,
                    "--replicate-seed",
                    str(args.replicate_seed),
                ]
                with (root / f"logs/wandb_{arm}.log").open("a") as stream:
                    logger = subprocess.Popen(
                        command,
                        cwd=Path(__file__).resolve().parents[3],
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                loggers.append(logger)
        for arm in args.arms:
            train("selection", arm)
            flags = ["--arm", arm, "--replicate-seed", str(args.replicate_seed)]
            if run(
                f"{arm}_compact_export",
                "scripts.prune.score.export_learned_heads",
                [*flags, "--stage", "compact_base", "--input-gpu-id", "0"],
                "0",
            ):
                raise RuntimeError("compact export failed")
            train("compact_recovery", arm)
            if run(f"{arm}_merge", "scripts.prune.score.export_learned_heads", [*flags, "--stage", "final_merged"]):
                raise RuntimeError("merged export failed")
        state["status"] = "paired_training_complete_GT_validation_pending"
    except BaseException as error:
        state.update(
            status="stopped" if isinstance(error, KeyboardInterrupt) else "failed",
            error={"type": type(error).__name__, "message": str(error)},
        )
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise
    finally:
        status()
        for logger in loggers:
            try:
                logger.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.terminate()
                logger.wait(timeout=30)


if __name__ == "__main__":
    main()
