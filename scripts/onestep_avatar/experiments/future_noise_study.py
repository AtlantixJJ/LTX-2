"""Prepare preserved future-noise study inputs and package jobs; see doc/experiments/future_noise_study.md."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.onestep_avatar.corpus import subset
from scripts.onestep_avatar.corpus.dataset import atomic_write
from scripts.onestep_avatar.execution.queue import validate_job_list
from scripts.onestep_avatar.hashing import sha256, tensor_sha256
from scripts.onestep_avatar.model.sampling import validate_schedule


def full_noise(manifest: dict, noises: dict, blocks: dict) -> dict[str, torch.Tensor]:
    """Keep the saved c0 noise slot and prove exact generated-frame slicing."""
    if manifest["latent_frames"] != 9 or set(noises) != {"A", "B", "B2", "B3"}:
        raise ValueError("historical study requires nine frames and A/B/B2/B3")
    tpf = manifest["tokens_per_latent_frame"]
    if type(tpf) is not int or tpf < 1:
        raise ValueError("invalid historical tokens per frame")
    eps = blocks["epsilons"][:4]
    if blocks["seed"] != manifest["seed"]:
        raise ValueError("saved block noise seed differs from the historical study")
    expected = [(1, (3 if index == 0 else 2) * tpf, 128) for index in range(4)]
    if len(eps) != 4 or any(not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                           for value, shape in zip(eps, expected, strict=True)):
        raise ValueError("saved block noise geometry differs from the historical study")
    if any(value.dtype != torch.bfloat16 or not torch.isfinite(value).all() for value in eps):
        raise ValueError("saved block noise must be finite bf16")
    original = torch.cat([eps[0][:, tpf:], *eps[1:]], dim=1)
    result = {}
    for name, value in noises.items():
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, 8 * tpf, 128)
                or value.dtype != torch.bfloat16 or not torch.isfinite(value).all()):
            raise ValueError("saved generated noise must have exact finite bf16 geometry")
        payload_hash = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        if payload_hash[:16] != manifest["noise"][name]:
            raise ValueError("historical saved noise payload changed")
        if not torch.equal(value[:, :2 * tpf], original[:, :2 * tpf]):
            raise ValueError("intervention changed earlier generated noise")
        if name == "A" and not torch.equal(value, original):
            raise ValueError("saved A does not reproduce original block noise")
        if name != "A" and torch.equal(value[:, 2 * tpf:], original[:, 2 * tpf:]):
            raise ValueError("intervention must change later generated noise")
        result[name] = torch.cat([eps[0][:, :tpf], value], dim=1)
    return result


def job_data(manifest: dict, destination: Path, *, specifications: dict | None = None) -> tuple[dict, dict]:
    """Specify all intervention and repeat controls with the public evaluator CLI."""
    schedule = list(validate_schedule(manifest["schedule"]))
    if schedule[0] != 1.0 or manifest["geometry"]["block_latent_frames"] != 2:
        raise ValueError("historical study requires sigma one and two-frame causal blocks")
    jobs, outputs = [], {}
    cases = [("J", "bidirectional", name) for name in ("B", "B2", "B3")]
    cases += [("J-repeat", "bidirectional", None), ("C", "causal", "B")]
    for label, mode, changed in cases:
        output = destination / "runs" / f"{label}_{changed or 'A'}"
        args = ["--mode", mode, "--subset", str(destination / "membership.json"),
                "--source", manifest["source"], "--output", str(output), "--model", "2.5",
                "--variant", "distilled", "--guide-mode", "d0", "--span-latent-frames", "9",
                "--seed", str(manifest["seed"]), "--noise-file", str(destination / "noise_A.pt"),
                "--schedule", *map(str, schedule)]
        if mode == "causal":
            args += ["--block-latent-frames", "2", "--blocks-per-sample", "4",
                     "--context-latent-frames", str(manifest["geometry"]["context_latent_frames"])]
        base = output / "case_0000/variant_000"
        if changed:
            args += ["--changed-noise-file", str(destination / f"noise_{changed}.pt"),
                     "--future-noise-start", "3"]
            records = [str(base / name / "result.json") for name in ("original", "changed")]
            outputs.setdefault(f"{label}-A", records[0])
            outputs[f"{label}-{changed}"] = records[1]
        else:
            records = [str(base / "result.json")]
            outputs["J-A-repeat"] = records[0]
        scientific = list(args)
        output_index = scientific.index('--output')
        del scientific[output_index:output_index + 2]
        spec = {'schema_version': 1, 'protocol': 'future_noise', 'arguments': scientific}
        spec_path = destination / f'spec_{label}_{changed or "A"}.json'
        serialized = json.dumps(spec, indent=2) + '\n'
        if specifications is not None:
            specifications[spec_path.name] = spec
        jobs.append({'id': f'future_noise_{label}_{changed or "A"}', 'kind': 'experiment',
                     'experiment': 'causality', 'spec': str(spec_path),
                     'spec_sha256': hashlib.sha256(serialized.encode()).hexdigest(),
                     'arguments': ['--spec', str(spec_path), '--output', str(output)],
                     'output': str(output), 'completion': {'manifest': str(output / 'manifest.json')}})
    record = {"schema_version": 1, "jobs": jobs}
    validate_job_list(record)
    return record, outputs


def prepare(manifest_path: Path, noise_path: Path, block_path: Path, subset_path: Path, output: Path) -> dict:
    """Validate original inputs before publishing new data, with no execution."""
    output = output.resolve()
    if output.exists():
        raise ValueError("study conversion requires a fresh output directory")
    inputs = {str(path.resolve()): sha256(path) for path in (manifest_path, noise_path, block_path, subset_path)}
    manifest = json.loads(manifest_path.read_text())
    noises = full_noise(manifest, torch.load(noise_path, map_location="cpu", weights_only=True),
                       torch.load(block_path, map_location="cpu", weights_only=True))
    membership, plan = subset.convert_legacy(json.loads(subset_path.read_text()),
                                              original_file_sha256=inputs[str(subset_path.resolve())])
    if membership["objective"] != "bg" or manifest["source"] not in {
        source["relative_dir"] for source in membership["sources"]
    }:
        raise ValueError("historical source/background differs from converted membership")
    specifications = {}
    jobs, outputs = job_data(manifest, output.resolve(), specifications=specifications)
    if any(sha256(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError("historical conversion inputs changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in noises.items():
        torch.save(value, output / f"noise_{name}.pt")
    for name, value in specifications.items():
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    for name, value in (("membership", membership), ("frame_plan", plan), ("jobs", jobs)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    artifacts = {name: sha256(output / name) for name in
                 ("noise_A.pt", "noise_B.pt", "noise_B2.pt", "noise_B3.pt",
                  "membership.json", "frame_plan.json", "jobs.json", *specifications)}
    if any(sha256(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError("historical conversion inputs changed during derived-file writes")
    record = {"schema_version": 2, "kind": "onestep_avatar.future_noise_preparation",
              "input_file_hashes": inputs, "input_manifest": str(manifest_path.resolve()),
              "input_paths": {name: str(path.resolve()) for name, path in zip(
                  ("manifest", "noise", "blocks", "subset"),
                  (manifest_path, noise_path, block_path, subset_path), strict=True)},
              "producer_source_sha256": sha256(Path(__file__)), "artifact_file_hashes": artifacts, "outputs": outputs,
              "noise_tensor_sha256": {name: tensor_sha256(value) for name, value in noises.items()},
              "status": "prepared_only_native_parity_pending"}
    atomic_write(output / "conversion.json", lambda path: path.write_text(json.dumps(record, indent=2) + "\n"))
    return record


def verify_preparation(output: Path) -> dict:
    """Check unchanged preparation bytes and job meaning without executing work."""
    record = json.loads((output / "conversion.json").read_text())
    if (record.get("schema_version") != 2 or record.get("kind") != "onestep_avatar.future_noise_preparation"
            or record.get("status") != "prepared_only_native_parity_pending"
            or record.get("producer_source_sha256") != sha256(Path(__file__))):
        raise ValueError("future-noise preparation identity changed")
    manifest = json.loads(Path(record["input_manifest"]).read_text())
    specifications = {}
    jobs, roles = job_data(manifest, output.resolve(), specifications=specifications)
    expected = {"noise_A.pt", "noise_B.pt", "noise_B2.pt", "noise_B3.pt",
                "membership.json", "frame_plan.json", "jobs.json", *specifications}
    artifacts = record.get("artifact_file_hashes", {})
    inputs = record.get("input_file_hashes", {})
    paths = record.get("input_paths", {})
    if (set(artifacts) != expected or set(paths) != {"manifest", "noise", "blocks", "subset"}
            or len(inputs) != 4 or set(paths.values()) != set(inputs)
            or record.get("input_manifest") != paths.get("manifest")):
        raise ValueError("future-noise preparation inventory changed")
    if any(sha256(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError("future-noise preparation original input changed")
    if any(sha256(output / name) != digest for name, digest in artifacts.items()):
        raise ValueError("future-noise preparation derived artifact changed")
    if json.loads((output / "jobs.json").read_text()) != jobs or record.get("outputs") != roles:
        raise ValueError("future-noise preparation job settings or roles changed")
    if any(json.loads((output / name).read_text()) != spec for name, spec in specifications.items()):
        raise ValueError("future-noise preparation specification meaning changed")
    noises = full_noise(manifest, torch.load(paths["noise"], map_location="cpu", weights_only=True),
                       torch.load(paths["blocks"], map_location="cpu", weights_only=True))
    for name, expected_noise in noises.items():
        actual = torch.load(output / f"noise_{name}.pt", map_location="cpu", weights_only=True)
        if (not isinstance(actual, torch.Tensor) or actual.dtype != expected_noise.dtype
                or not torch.equal(actual, expected_noise)
                or record.get("noise_tensor_sha256", {}).get(name) != tensor_sha256(actual)):
            raise ValueError("future-noise preparation noise differs from original input")
    membership, plan = subset.convert_legacy(json.loads(Path(paths["subset"]).read_text()),
                                              original_file_sha256=inputs[paths["subset"]])
    if (json.loads((output / "membership.json").read_text()) != membership
            or json.loads((output / "frame_plan.json").read_text()) != plan):
        raise ValueError("future-noise preparation membership or frame plan differs from original input")
    return record


def main(argv: list[str] | None = None) -> None:
    """Publish data only; the package queue is the execution owner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-preparation", type=Path)
    for name in ("manifest", "noise", "blocks", "subset", "output"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args(argv)
    inputs = [args.manifest, args.noise, args.blocks, args.subset, args.output]
    if args.verify_preparation is not None:
        if any(value is not None for value in inputs):
            parser.error("verification rejects conversion arguments")
        verify_preparation(args.verify_preparation)
        return
    if any(value is None for value in inputs):
        parser.error("conversion requires --manifest, --noise, --blocks, --subset and --output")
    prepare(args.manifest, args.noise, args.blocks, args.subset, args.output)


if __name__ == "__main__":
    main()
