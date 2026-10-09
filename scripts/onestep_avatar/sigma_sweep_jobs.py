"""Prepare package generation jobs from historical sigma-sweep evidence; see doc/sigma_sweep_jobs.md."""

import argparse
import json
from hashlib import sha256 as bytes_sha256
from pathlib import Path

import torch

from scripts.onestep_avatar import evaluate, sigma_sweep
from scripts.onestep_avatar.corpus import dataset, precompute, subset
from scripts.onestep_avatar.execution import queue
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import backbone
from scripts.prune.core.provenance import checkpoint_fingerprint


def prepare(cases_path: Path, output: Path) -> dict:  # noqa: PLR0912, PLR0915 -- ordered original-evidence gates and publication
    """Validate every source/noise before publishing data; never execute a model."""
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("generation preparation requires a fresh output directory")
    cases = json.loads(cases_path.read_text())
    if cases.get("schema_version") != 1 or not cases.get("cases"):
        raise ValueError("generation preparation requires a nonempty version-one case list")
    model = backbone.resolve("2.5", "distilled")
    transformer = checkpoint_fingerprint(model.paths.transformer())
    vae = checkpoint_fingerprint(model.paths.video_vae())
    original_hashes = {str(cases_path.resolve()): sha256(cases_path)}
    vae_path = Path(model.paths.video_vae()).resolve()
    original_hashes[str(vae_path)] = sha256(vae_path)
    prepared, jobs, tags = [], [], set()
    for case in cases["cases"]:
        tag = case["tag"]
        if not isinstance(tag, str) or Path(tag).name != tag or tag in tags or tag in (".", ".."):
            raise ValueError("generation case tags must be safe unique names")
        tags.add(tag)
        manifests = {}
        for kind in ("official", "one_step"):
            path = Path(case[f"{kind}_manifest"])
            original_hashes[str(path.resolve())] = sha256(path)
            if original_hashes[str(path.resolve())] != case[f"{kind}_sha256"]:
                raise ValueError("historical generation manifest bytes changed")
            manifests[kind] = json.loads(path.read_text())
        official = manifests["official"]
        membership = subset.from_saved_probe(Path(case["official_manifest"]))
        store = dataset.ClipStore(membership)
        store.verify(require_guide=True)
        source = membership["sources"][0]
        if source["shape"][1] < 17 or source["fps"] != 30:
            raise ValueError("historical generation master geometry or fps differs")
        if source["capture_encode_record"]["vae_fingerprint"] != precompute.file_fingerprint(
            Path(model.paths.video_vae())
        ):
            raise ValueError("historical generation encoding VAE differs")
        art = official["videos"][0]["artifacts"]
        prompt = official["text_context"]["prompt"]
        for kind, manifest in manifests.items():
            expected = sorted(sigma for sigma, (owner, _) in sigma_sweep.LEVELS.items() if owner == kind)
            if sorted(row["sigma"] for row in manifest["videos"]) != expected:
                raise ValueError("historical generation exact schedule inventory differs")
            if (manifest.get("checkpoint") is not None or manifest.get("objective") != "white"
                    or manifest.get("teacher_forcing") or manifest.get("history_mode") != "cache"
                    or manifest.get("seed") != official["seed"] or manifest["text_context"] != official["text_context"]
                    or manifest["model"]["transformer_fingerprint"] != transformer
                    or manifest["model"]["video_vae_fingerprint"] != vae
                    or any(manifest["geometry"].get(key) != value for key, value in
                           (("block_latent_frames", 2), ("context_latent_frames", 8), ("sink_latent_frames", 1)))):
                raise ValueError("historical generation conditions differ from the frozen causal sweep")
            guidance = manifest.get("guidance", case.get("legacy_guidance"))
            if not isinstance(guidance, dict) or any(guidance.get(key) != value for key, value in
                                                   (("cfg", 1), ("stg", 0), ("rescale", 0))):
                raise ValueError("historical generation guidance differs")
            if "guidance" not in manifest:
                if not guidance.get("source") or not guidance.get("source_sha256"):
                    raise ValueError("historical guidance declaration requires checked source evidence")
                evidence_path = Path(guidance["source"]).resolve()
                original_hashes[str(evidence_path)] = sha256(evidence_path)
                if original_hashes[str(evidence_path)] != guidance["source_sha256"]:
                    raise ValueError("historical guidance source bytes changed")
            for row in manifest["videos"]:
                if any(row["artifacts"].get(key) != art.get(key) for key in
                       ("view", "capture_sha256", "guide_sha256", "epsilon_sha256", "fps")):
                    raise ValueError("historical generation source or shared noise differs")
        noise_path = Path(case["official_manifest"]).parent / art["epsilon"]
        noise_hash = sha256(noise_path)
        if noise_hash != art["epsilon_sha256"]:
            raise ValueError("historical generation saved noise bytes changed")
        original_hashes[str(noise_path.resolve())] = noise_hash
        other_art = manifests["one_step"]["videos"][0]["artifacts"]
        other_noise = Path(case["one_step_manifest"]).parent / other_art["epsilon"]
        if sha256(other_noise) != noise_hash:
            raise ValueError("historical generation invocations used different saved noise")
        original_hashes[str(other_noise.resolve())] = noise_hash
        noise = torch.load(noise_path, map_location="cpu", weights_only=True)
        channels, frames, height, width = source["shape"]
        if (not isinstance(noise, torch.Tensor) or noise.dtype != torch.bfloat16
                or tuple(noise.shape) != (1, frames * height * width, channels) or not torch.isfinite(noise).all()):
            raise ValueError("historical generation noise geometry or dtype differs")
        prefix = noise[:, :17 * height * width].contiguous()
        prefix_path = output / f"{tag}_noise.pt"
        membership_path = output / f"{tag}_membership.json"
        for sigma, (kind, calls) in sigma_sweep.LEVELS.items():
            rows = [row for row in manifests[kind]["videos"] if row["sigma"] == sigma]
            if len(rows) != 1 or len(rows[0]["schedule"]) != calls + 1 or rows[0]["schedule"][0] != sigma:
                raise ValueError("historical generation exact schedule inventory differs")
            for arm in ("d0", "d1"):
                destination = output / "runs" / tag / f"sigma{sigma:.6f}_{arm}"
                arguments = ["--mode", "causal", "--subset", str(membership_path), "--source", source["relative_dir"],
                             "--output", str(destination), "--model", "2.5", "--variant", "distilled",
                             "--guide-mode", arm, "--span-latent-frames", "17", "--block-latent-frames", "2",
                             "--blocks-per-sample", "8", "--context-latent-frames", "8", "--history-mode", "cache",
                             "--kv-source", "refresh", "--seed", str(official["seed"]), "--prompt", prompt,
                             "--noise-file", str(prefix_path), "--cfg", "1", "--stg", "0", "--rescale", "0",
                             "--schedule", *map(str, rows[0]["schedule"])]
                jobs.append({"id": f"sweep_{tag}_{sigma:.6f}_{arm}", "kind": "evaluate", "arguments": arguments,
                             "output": str(destination), "completion": {"records": [
                                 str(destination / "case_0000/variant_000/result.json")]}})
        case_jobs = jobs[-8:]
        spec_path = output / f"{tag}_decode.json"
        spec = {"schema_version": 2, "tag": tag, "model": "2.5", "fps": 30, "decode_seed": 42,
                "vae": {"path": str(vae_path), "sha256": original_hashes[str(vae_path)]},
                "capture": {"path": art["capture"], "sha256": art["capture_sha256"]},
                "guide": {"path": art["guide"], "sha256": art["guide_sha256"]},
                "cells": [{"sigma": sigma, "arm": arm, "evaluation_job": job}
                          for (sigma, arm), job in zip(
                              [(sigma, arm) for sigma in sigma_sweep.LEVELS for arm in ("d0", "d1")],
                              case_jobs, strict=True)]}
        media_output = output / "decoded" / tag
        jobs.append({"id": f"decode_{tag}", "kind": "sigma_sweep",
                     "arguments": ["--spec", str(spec_path), "--output", str(media_output)],
                     "dependencies": [job["id"] for job in case_jobs], "output": str(media_output),
                     "completion": {"manifest": str(media_output / "manifest.json")}})
        prepared.append((membership_path, membership, prefix_path, prefix, spec_path, spec))
    record = {"schema_version": 1, "jobs": jobs}
    queue.validate_job_list(record)
    for job in jobs:
        (evaluate.parse_args if job["kind"] == "evaluate" else sigma_sweep.parse_args)(job["arguments"])
    if any(sha256(Path(path)) != digest for path, digest in original_hashes.items()):
        raise ValueError("historical generation inputs changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    for membership_path, membership, prefix_path, prefix, spec_path, spec in prepared:
        membership_path.write_text(json.dumps(membership, indent=2) + "\n")
        torch.save(prefix, prefix_path)
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    jobs_text = json.dumps(record, indent=2) + "\n"
    derived_hashes = {path.name: sha256(path) for row in prepared for path in (row[0], row[2], row[4])}
    derived_hashes["jobs.json"] = bytes_sha256(jobs_text.encode()).hexdigest()
    if any(sha256(Path(path)) != digest for path, digest in original_hashes.items()):
        raise ValueError("historical generation inputs changed during publication")
    provenance = {"schema_version": 2, "input_file_hashes": original_hashes,
                  "derived_file_hashes": derived_hashes,
                  "source_hashes": {Path(module.__file__).name: sha256(Path(module.__file__))
                                    for module in (subset, evaluate)},
                  "producer_sha256": sha256(Path(__file__)),
                  "status": "prepared_only_native_generation_pending",
                  "model_sampled_fingerprints": {"transformer": transformer, "vae": vae}}
    dataset.atomic_write(output / "preparation.json",
                         lambda path: path.write_text(json.dumps(provenance, indent=2) + "\n"))
    dataset.atomic_write(output / "jobs.json", lambda path: path.write_text(jobs_text))
    return record


def main() -> None:
    """Write checked job data only; the package queue owns actual execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.cases, args.output)


if __name__ == "__main__":
    main()
