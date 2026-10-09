"""Validate and run pinned training previews; see doc/previews.md."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from scripts.onestep_avatar import evaluate, hashing
from scripts.onestep_avatar.corpus import subset
from scripts.onestep_avatar.corpus.dataset import atomic_write
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import backbone
from scripts.onestep_avatar.training import checkpoints


def check_preview_reference_bundle(fixed: dict) -> dict | None:
    """Bind checked prepared RGB to the fixed capture and guide producers."""
    producer_inputs = fixed.get('producer_inputs', {})
    if not isinstance(producer_inputs, dict):
        raise ValueError('preview preparation input identities must be a mapping')
    for source in producer_inputs.values():
        if (not isinstance(source, dict) or not isinstance(source.get('path'), str)
                or not Path(source['path']).is_absolute() or sha256(Path(source['path'])) != source.get('sha256')):
            raise ValueError('preview preparation input changed or lacks an absolute identity')
    identity = fixed.get("reference_bundle")
    if identity is None:
        return None
    from scripts.onestep_avatar.media import load_training_references  # noqa: PLC0415 -- saved RGB only

    path = Path(identity["path"])
    if not path.is_absolute() or sha256(path) != identity["sha256"]:
        raise ValueError("preview reference bundle manifest changed")
    _, producer = load_training_references(path)
    if producer["capture_encoding_sha256"] != fixed["input_files"]["capture"]["sha256"]:
        raise ValueError("preview reference bundle uses a different capture encoding")
    args = evaluate.parse_args([*fixed["evaluation_arguments"], "--output", "/unused-preview-reference-check"])
    if args.source != [producer["source"]]:
        raise ValueError("preview reference bundle requires its one explicit source selection")
    if args.guide_mode == "d1":
        if not producer.get("guide_rgb_sha256"):
            raise ValueError("D1 preview requires a checked guide reference")
        guide = torch.load(Path(fixed["input_files"]["guide"]["path"]), map_location="cpu", weights_only=True)
        if not isinstance(guide, dict) or guide.get("input_fingerprint") != producer["guide_rgb_sha256"]:
            raise ValueError("preview reference bundle uses a different guide render")
    return producer



def verify_preview_job(path: Path, *, verify_files: bool = True) -> dict:
    """Read only complete, unchanged checkpoints and pinned preview inputs."""
    job = json.loads(path.read_text())
    if job.get("schema_version") != 2 or job.get("kind") != "onestep_avatar.preview_job":
        raise ValueError("preview requires a version-two job record")
    fixed, checkpoint = job["fixed_inputs"], job["checkpoint"]
    if subset.record_hash(fixed) != fixed.get("sha256"):
        raise ValueError("fixed preview record changed")
    identity = hashlib.sha256((checkpoint["sha256"] + fixed["sha256"]).encode()).hexdigest()
    if job.get("id") != identity:
        raise ValueError("preview job identity changed")
    if not verify_files:
        return job
    for role, source in fixed["input_files"].items():
        if sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"preview {role} file changed")
    check_preview_reference_bundle(fixed)
    adapter = Path(checkpoint["path"])
    marker = json.loads(adapter.with_suffix(".complete.json").read_text())
    if marker.get("state") != "complete" or sha256(adapter) != checkpoint["sha256"]:
        raise ValueError("preview checkpoint is incomplete or changed")
    if marker.get("sha256") != checkpoint["sha256"] or marker.get("step") != checkpoint["step"]:
        raise ValueError("preview completion marker differs from its pinned checkpoint")
    contract = checkpoints.read_contract(adapter)
    checkpoints.validate_adapter_tensors(adapter, contract)
    if contract["adapter"]["step"] != checkpoint["step"] or contract["mode"] != fixed["mode"]:
        raise ValueError("preview adapter step or mode differs from its job")
    return job



def _verify_preview_outputs(records: list[dict], job: dict, *, rendered: bool) -> None:  # noqa: PLR0912 -- all completion evidence gates
    if not records:
        raise ValueError("preview completion requires raw results and rendered outputs")
    for identity in records:
        path = Path(identity["path"])
        if sha256(path) != identity["sha256"]:
            raise ValueError("preview output record changed")
        record = json.loads(path.read_text())
        software.check_current(record.get("software"))
        if rendered:
            common_settings = record.get("common_settings", {})
            if (
                common_settings.get("preview_job_id") != job["id"]
                or common_settings.get("fixed_inputs_sha256") != job["fixed_inputs"]["sha256"]
            ):
                raise ValueError("preview rendering belongs to different fixed inputs")
            if common_settings.get("result_records") != job["results"]:
                raise ValueError("preview rendering does not identify its generated result records")
            outputs = record.get("outputs", {})
            if set(outputs) != {"video", "poster"}:
                raise ValueError("preview rendering lacks video or poster")
        else:
            if record.get("state") != "complete":
                raise ValueError("preview encoding is not complete")
            if record.get("mode") != job["fixed_inputs"]["mode"]:
                raise ValueError("preview result has the wrong mode")
            for role, key in (
                ("capture", "capture_sha256"),
                ("guide", "guide_sha256"),
                ("first_image", "c0_sha256"),
                ("text", "text_sha256"),
                ("noise", "noise_sha256"),
            ):
                expected = job["fixed_inputs"]["input_files"].get(role)
                if expected is not None and record.get(key) != expected.get("tensor_sha256"):
                    raise ValueError(f"preview result changed the fixed {role} tensor")
            if record.get("adapter") is not None and record.get("adapter_sha256") != job["checkpoint"]["sha256"]:
                raise ValueError("preview result uses a different checkpoint")
            outputs = {"encoding": record["output"]}
        for output in outputs.values():
            if sha256(Path(output["path"])) != output["sha256"]:
                raise ValueError("preview encoded/rendered output changed")



def set_preview_state(
    path: Path,
    state: str,
    *,
    error: str | None = None,
    results: list[dict] | None = None,
    renderings: list[dict] | None = None,
) -> dict:
    """Serialize job transitions; they never write to training or checkpoint files."""
    transitions = {
        "pending": {"running", "failed"},
        "failed": {"running"},
        "running": {"complete", "failed"},
        "complete": set(),
    }
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        job = verify_preview_job(path, verify_files=state != "failed")
        if state not in transitions.get(job.get("state"), set()):
            raise ValueError("unsupported preview state transition")
        if job["state"] == "running" and job.get("pid") != os.getpid():
            try:
                os.kill(job["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError("preview is owned by a live process")
        if state == "complete":
            _verify_preview_outputs(results or [], job, rendered=False)
            job["results"] = results
            _verify_preview_outputs(renderings or [], job, rendered=True)
            if not any(
                json.loads(Path(r["path"]).read_text()).get("adapter_sha256") == job["checkpoint"]["sha256"]
                for r in results
            ):
                raise ValueError("preview has no generated result from its pinned checkpoint")
            job.update(results=results, renderings=renderings)
        if state == "failed" and not error:
            raise ValueError("failed preview requires a recorded reason")
        job.update(state=state, pid=os.getpid(), updated_at=time.time(), error=error)
        atomic_write(path, lambda temporary: temporary.write_text(json.dumps(job, indent=2) + "\n"))
        return job



def render_preview_outputs(path: Path, *, gpu_id: int) -> dict:  # noqa: PLR0915 -- ordered decoder/media lifecycle
    """Render owned saved preview results without another transformer call."""
    from scripts.onestep_avatar import media  # noqa: PLC0415 -- saved-output rendering
    from scripts.prune.core.session import Session  # noqa: PLC0415 -- decoder-only session

    producer_software = software.capture("decoding")
    job = verify_preview_job(path)
    if job.get("state") != "running" or job.get("pid") != os.getpid():
        raise ValueError("preview rendering requires the current running owner")
    fixed = job["fixed_inputs"]
    if fixed.get("reference_bundle") is None:
        raise ValueError("preview rendering requires pinned reference pixels")
    references, producer = media.load_training_references(Path(fixed["reference_bundle"]["path"]))
    decoder_settings = media.native_decoder_settings()
    if producer.get("decoder_settings") != decoder_settings:
        raise ValueError("preview reference decoder settings differ from the current runtime")
    results = job.get("results", [])
    _verify_preview_outputs(results, job, rendered=False)
    records = [json.loads(Path(row["path"]).read_text()) for row in results]
    changed = [record for record in records if record.get("adapter_sha256") == job["checkpoint"]["sha256"]]
    base = [record for record in records if record.get("adapter") is None]
    if len(changed) != 1 or len(base) > 1 or len(records) != len(changed) + len(base):
        raise ValueError("preview rendering requires one pinned adapter and at most one base")
    for record in records:
        if (
            record.get("source") != producer["source"]
            or record.get("fps") != producer["fps"]
            or (record["frames"] - 1) * 8 + 1 != len(producer["source_frames"])
        ):
            raise ValueError("preview output source/timebase/coverage differs from its references")
    args = evaluate.parse_args([*fixed["evaluation_arguments"], "--output", job["output"]])
    specification = backbone.resolve(args.model, args.variant)
    vae_hash = sha256(Path(specification.paths.video_vae()))
    if vae_hash != producer["vae_sha256"]:
        raise ValueError("preview rendering VAE differs from its fixed references")
    destination = Path(job["output"]) / f"render_attempt_{job['raw_attempt']:04d}"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("preview rendering output is already used")
    question = "Does the adapter change the output?"
    output_panels = [
        media.Panel(role, title, None, tuple(producer["source_frames"]),
                    value=("no adapter" if role == "baseline" else f"step {job['checkpoint']['step']}")
                    if selected else "", missing_reason="Not requested" if not selected else "")
        for role, selected, title in (("baseline", base, "Base output"), ("changed", changed, "Adapter output"))
    ]
    planned = references + output_panels
    try:
        media.layout_geometry(planned, question=question, layout="training")
        layout = "training"
    except ValueError:
        layout = media.compact_layout(planned, question=question, layout="training")
    software.check_current(producer_software)
    session = Session(specification, torch.device(f"cuda:{gpu_id}"), "onestep_avatar.preview_render", None)
    decode_records = []
    with session.decoder() as decoder:
        for role, selected, title in (("baseline", base, "Base output"), ("changed", changed, "Adapter output")):
            if not selected:
                references.append(media.Panel(role, title, None, missing_reason="Not requested"))
                continue
            record = selected[0]
            latent = torch.load(Path(record["output"]["path"]), map_location="cpu", weights_only=True)
            pixels = media.decode(session, latent, decoder, producer["decode_seed"])
            references.append(
                media.Panel(
                    role,
                    title,
                    pixels,
                    tuple(producer["source_frames"]),
                    value="no adapter" if role == "baseline" else f"step {job['checkpoint']['step']}",
                )
            )
            decode_records.append(
                {
                    "role": role,
                    "decode_key": media.decode_key(
                        record["output"]["sha256"],
                        vae_hash,
                        list(latent.shape),
                        "native_decode_video",
                        producer["decode_seed"],
                        decoder_settings,
                    ),
                }
            )
    pixels, rendering = media.render_panels(
        references,
        question=question,
        layout=layout,
        fps=producer["fps"],
        common_settings={
            "preview_job_id": job["id"],
            "fixed_inputs_sha256": fixed["sha256"],
            "result_records": results,
            "reference_bundle": fixed["reference_bundle"],
            "decoder_records": decode_records,
        },
    )
    rendering["software"] = producer_software
    media.save_render(pixels, rendering, destination)
    rendered_path = destination / "rendering.json"
    return set_preview_state(
        path,
        "complete",
        results=results,
        renderings=[{"path": str(rendered_path.resolve()), "sha256": sha256(rendered_path)}],
    )



def generate_preview(path: Path, *, gpu_id: int) -> list[dict]:
    """Run a pinned preview's raw stage; rendering is required for completion."""
    try:
        job = verify_preview_job(path)
    except Exception as error:
        identity = verify_preview_job(path, verify_files=False)
        if identity.get("state") == "pending":
            set_preview_state(path, "failed", error=f"{type(error).__name__}: {error}")
        raise
    job = set_preview_state(path, "running")
    try:
        output = Path(job["output"])
        attempt = 0
        while (output / f"attempt_{attempt:04d}").exists():
            attempt += 1
        destination = output / f"attempt_{attempt:04d}"
        args = evaluate.parse_args(
            [
                *job["fixed_inputs"]["evaluation_arguments"],
                "--checkpoint",
                job["checkpoint"]["path"],
                "--output",
                str(destination),
                "--gpu-id",
                str(gpu_id),
            ]
        )
        args.preview_fixed = job["fixed_inputs"]
        evaluate.execute_evaluation(args, preview_tensor_validator=verify_preview_tensors)
        results = [
            {"path": str(record.resolve()), "sha256": sha256(record)}
            for record in sorted(destination.glob("case_*/variant_*/result.json"))
        ]
        _verify_preview_outputs(results, job, rendered=False)
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = verify_preview_job(path)
            if current.get("state") != "running" or current.get("pid") != os.getpid():
                raise ValueError("preview generation lost its job ownership")
            current.update(results=results, raw_attempt=attempt, updated_at=time.time())
            atomic_write(path, lambda temporary: temporary.write_text(json.dumps(current, indent=2) + "\n"))
        if job["fixed_inputs"].get("reference_bundle") is not None:
            render_preview_outputs(path, gpu_id=gpu_id)
        return results
    except Exception as error:
        set_preview_state(path, "failed", error=f"{type(error).__name__}: {error}")
        raise



def verify_preview_tensors(fixed: dict, tensors: dict[str, torch.Tensor | None]) -> None:
    """Reject changed execution tensors before opening a transformer."""
    for role, identity in fixed["input_files"].items():
        if role == "subset":
            continue
        tensor = tensors.get(role)
        if tensor is None or hashing.tensor_sha256(tensor) != identity.get("tensor_sha256"):
            raise ValueError(f"preview execution changed the fixed {role} tensor")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the pinned preview job; its evaluator arguments come from the record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-job", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generate_preview(args.preview_job, gpu_id=args.gpu_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
