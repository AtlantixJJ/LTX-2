"""Resolve version-two sweep cells from scientific evaluation evidence, without execution.

Inputs are a mutable decoder spec and its input-hash map. Validate the frozen
causal cell settings and shared paired source/text/noise first. Pin all saved
evidence before calling the public queue completion verifier; recheck afterward.
Only complete verified results supply generated tensor paths/hashes. Missing
results fail without launching models or writing files. Retain embedded job
choices in the normalized spec. The caller owns tensor geometry, decoding and
final publication; schema-one historical tensor specs bypass this helper.
The ordinary parser rejects intervention flags before cell checks, so this
reader never requests experiment-only attributes from its parsed settings.
"""

import json
from pathlib import Path

from scripts.onestep_avatar import evaluate
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.execution import queue
from scripts.onestep_avatar.hashing import sha256


def resolve_cells(spec: dict, hashes: dict, levels: dict) -> None:
    """Resolve only the exact eight checked generation cells into decoder inputs."""
    shared = None
    for cell in spec["cells"]:
        job = cell.get("evaluation_job", {})
        if job.get("kind") != "evaluate" or "path" in cell or "sha256" in cell:
            raise ValueError("result-bound sweep requires evaluation jobs without guessed tensor identities")
        args = evaluate.parse_args(job["arguments"])
        _, calls = levels[cell["sigma"]]
        if (args.mode != "causal" or args.model != "2.5" or args.variant != "distilled"
                or args.guide_mode != cell["arm"] or args.schedule[0] != cell["sigma"]
                or len(args.schedule) != calls + 1 or args.span_latent_frames != 17
                or args.mode_settings.block_latent_frames != 2 or args.mode_settings.blocks_per_sample != 8
                or args.mode_settings.context_latent_frames != 8 or args.mode_settings.teacher_forcing
                or args.history_mode != "cache" or args.kv_source != "refresh"
                or args.cfg != 1 or args.stg != 0 or args.rescale != 0 or args.checkpoint
                or args.research_override
                or args.noise_file is None or not args.source or len(args.source) != 1):
            raise ValueError("result-bound sweep evaluation conditions differ")
        signature = (str(args.subset.resolve()), tuple(args.source), args.seed, args.prompt,
                     str(args.noise_file.resolve()))
        if shared is not None and signature != shared:
            raise ValueError("result-bound sweep shared inputs differ")
        shared = signature
        record = args.output / "case_0000/variant_000/result.json"
        if (Path(job["output"]).resolve() != args.output.resolve()
                or job["completion"].get("records") != [str(record)]):
            raise ValueError("result-bound sweep result inventory differs")
        membership = json.loads(args.subset.read_text())
        if membership["objective"] != "white":
            raise ValueError("result-bound sweep paired masters differ")
        source = next(row for row in membership["sources"] if row["relative_dir"] == args.source[0])
        for role in ("capture", "guide"):
            path = Path(membership["corpus_root"]) / args.source[0]
            path /= (dataset.capture_bundle_name(membership["objective"]) if role == "capture"
                     else dataset.guide_bundle_name(membership["objective"]))
            if (path.resolve() != Path(spec[role]["path"]).resolve()
                    or source[f"{role}_latent_sha256"] != spec[role]["sha256"]):
                raise ValueError("result-bound sweep paired masters differ")
        paths = [args.subset, args.noise_file, *evaluate.evaluation_evidence_paths(job["arguments"], [record])]
        before = {str(path.resolve()): sha256(path) for path in paths}
        if not queue.verify_completion(job):
            raise ValueError("result-bound sweep evaluation is incomplete")
        output = json.loads(record.read_text())["output"]
        if any(sha256(Path(path)) != digest for path, digest in before.items()):
            raise ValueError("result-bound sweep evidence changed during verification")
        hashes.update(before)
        cell.update(path=output["path"], sha256=output["sha256"])
