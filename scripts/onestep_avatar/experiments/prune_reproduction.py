"""Test candidate causes of the pruning-baseline mismatch; see doc/experiments/prune_reproduction.md."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from scripts.onestep_avatar.execution import runtime, software
from scripts.onestep_avatar.experiments import prune_baseline as baseline
from scripts.onestep_avatar.experiments import spec_protocol
from scripts.onestep_avatar.hashing import sha256, tensor_sha256
from scripts.onestep_avatar.model import causal, common
from scripts.onestep_avatar.training import numerics
from scripts.prune.evaluate import whole_clip_baseline as producer

SELECTOR = "prune_reproduction"
ENTRY = "scripts/onestep_avatar/experiments/prune_reproduction.py"
EXTRA_SOURCES = (ENTRY, *baseline.EXTRA_SOURCES)
# Model-input changes per control. The three process-flag controls change no model input.
# The old producer made the global sigma in the BF16 token dtype, while its per-token
# timesteps came from ltx-core's float32 denoise mask; historical_inputs is that pair of
# inputs plus its all-visible mask. sigma_bf16 casts both schedule inputs.
INPUT_CHANGES = {
    "global_sigma_bf16": ("global_sigma_bf16",),
    "sigma_bf16": ("global_sigma_bf16", "timesteps_bf16"),
    "dense_mask": ("dense_mask",),
    "deterministic_off": (),
    "cudnn_deterministic_off": (),
    "bf16_reduction_off": (),
    "historical_inputs": ("global_sigma_bf16", "dense_mask"),
}
FIRST, LAST = "current_first", "current_last"
SAVED = ("reference", "earlier")
ATTENTION = re.compile(r"attention|attn|fmha|flash|sdpa|cudnn|fprop", re.IGNORECASE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return spec_protocol.outer_args(argv, __doc__)


def scientific_args(argv: list[str]) -> argparse.Namespace:
    """The pruning-baseline settings plus the two saved runs and the requested controls."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--earlier", type=Path, required=True)
    parser.add_argument("--controls", nargs="+", choices=sorted(INPUT_CHANGES), required=True)
    extra, rest = parser.parse_known_args(argv)
    args = baseline.scientific_args(rest)
    if args.arms != ["d0"] or len(set(extra.controls)) != len(extra.controls):
        raise ValueError("prune_reproduction requires the D0 arm only and distinct controls")
    args.reference, args.earlier, args.controls = extra.reference, extra.earlier, list(extra.controls)
    return args


def read_spec(spec: Path, root: Path) -> argparse.Namespace:
    return spec_protocol.read_spec(
        parse_args(["--spec", str(spec), "--output", str(root)]), SELECTOR, scientific_args
    )


def resource_phases(spec: Path) -> list[str]:
    read_spec(spec, spec.parent / "unused_output")
    return ["runtime"]


def runtime_profile(spec: Path) -> dict:
    return {"generic_phase": True, "phases": resource_phases(spec), "processes": 1}


def arms(args: argparse.Namespace) -> list[str]:
    """The current producer runs first and last, around the requested controls."""
    return [FIRST, *args.controls, LAST]


def saved_rollout(root: Path, args: argparse.Namespace) -> dict:
    """Find one view's saved epsilon and D0 outputs through the producer manifest's own hashes."""
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("whole_clip") is not True or manifest.get("seed") != args.seed
            or manifest.get("objective") != args.objective or manifest.get("model_variant") != args.variant
            or manifest["geometry"]["context_latent_frames"] != args.context_latent_frames):
        raise ValueError(f"saved rollout settings differ from the requested baseline: {root}")
    view = args.view[0].resolve()
    rows = [video["artifacts"] for video in manifest["videos"] if Path(video["view"]).resolve() == view]
    noises = {(row["epsilon"], row["epsilon_sha256"]) for row in rows}
    latents: dict[tuple[float, str], tuple[str, str]] = {}
    for row in rows:
        for item in row["latents"]:
            entry = (item["path"], item["sha256"])
            if latents.setdefault((float(item["sigma"]), item["arm"]), entry) != entry:
                raise ValueError(f"saved rollout lists two outputs for one sigma: {root}")
    if len(noises) != 1 or any((sigma, "d0") not in latents for sigma in args.sigmas):
        raise ValueError(f"saved rollout lacks one epsilon or a requested D0 sigma: {root}")
    ((noise, noise_digest),) = noises
    files = {"epsilon": (noise, noise_digest), **{f"{sigma:.6f}": latents[(sigma, "d0")] for sigma in args.sigmas}}
    for name, digest in files.values():
        path = root / name
        if path.resolve().parent != root.resolve() or sha256(path) != digest:
            raise ValueError(f"saved rollout file differs from its manifest: {path}")
    return {"manifest": root / "manifest.json", "context_sha256": manifest["text_context"]["sha256"],
            "files": {key: root / name for key, (name, _) in files.items()}}


def input_paths(args: argparse.Namespace) -> list[Path]:
    paths = set(baseline.input_paths(args))
    for root in (args.reference, args.earlier):
        saved = saved_rollout(root, args)
        paths.update(path.resolve() for path in (saved["manifest"], *saved["files"].values()))
    return sorted(paths)


def acceptance(args: argparse.Namespace) -> dict:
    return {
        "schema_version": 1,
        "protocol": SELECTOR,
        "settings": spec_protocol.settings(args),
        "inputs": {str(path): sha256(path) for path in input_paths(args)},
        "model": baseline.model_identity(args),
        "software": software.capture("evaluation", "causal", extra_sources=EXTRA_SOURCES),
    }


def flags() -> dict:
    observed = numerics.capture()
    matmul = torch.backends.cuda.matmul
    observed["allow_bf16_reduced_precision_reduction"] = matmul.allow_bf16_reduced_precision_reduction
    observed["sdp_backends"] = {
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(), "flash": torch.backends.cuda.flash_sdp_enabled(),
        "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(), "math": torch.backends.cuda.math_sdp_enabled(),
    }
    return observed


@contextmanager
def numerical_control(arm: str) -> Iterator[dict]:
    """Change at most one process flag for this arm, then prove that the starting flags returned."""
    before = flags()
    if arm == "deterministic_off":
        if not before["deterministic_algorithms"]:
            raise ValueError("deterministic_off needs a job that applies the deterministic policy")
        torch.use_deterministic_algorithms(False)
    elif arm == "cudnn_deterministic_off":
        if not before["cudnn_deterministic"]:
            raise ValueError("cudnn_deterministic_off needs a job that applies the deterministic policy")
        torch.backends.cudnn.deterministic = False
    elif arm == "bf16_reduction_off":
        if not before["allow_bf16_reduced_precision_reduction"]:
            raise ValueError("bf16_reduction_off needs reduced-precision BF16 reduction enabled at start")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        yield {"before": before, "during": flags()}
    finally:
        torch.use_deterministic_algorithms(before["deterministic_algorithms"],
                                           warn_only=before["deterministic_warn_only"])
        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            before["allow_bf16_reduced_precision_reduction"])
    if flags() != before:
        raise ValueError("a numerical control did not restore the starting flags")


def model_input(modality) -> dict:  # noqa: ANN001 -- ltx_core Modality
    mask = modality.attention_mask
    return {
        "sigma": {"dtype": str(modality.sigma.dtype), "values": modality.sigma.float().tolist()},
        "timesteps": {"dtype": str(modality.timesteps.dtype),
                      "values": torch.unique(modality.timesteps).float().tolist()},
        "attention_mask": None if mask is None else {
            "shape": list(mask.shape), "dtype": str(mask.dtype),
            "minimum": float(mask.min()), "maximum": float(mask.max())},
        "latent_sha256": tensor_sha256(modality.latent),
    }


def controlled(denoise_fn: Callable, changes: tuple[str, ...], calls: list[dict]) -> Callable:
    """Apply this arm's model-input changes at the model call, and record every actual input."""
    def call(modality):  # noqa: ANN001, ANN202 -- ltx_core Modality in, denoised tokens out
        updates = {}
        if "global_sigma_bf16" in changes:
            updates["sigma"] = modality.sigma.to(torch.bfloat16)
        if "timesteps_bf16" in changes:
            updates["timesteps"] = modality.timesteps.to(torch.bfloat16)
        if "dense_mask" in changes:
            if modality.attention_mask is not None:
                raise ValueError("the dense-mask control expects the unmasked whole-clip call")
            # One block: every key is visible to every query, as the old producer's mask.
            ids = torch.zeros(modality.latent.shape[1], dtype=torch.long, device=modality.latent.device)
            updates["attention_mask"] = causal.block_causal_mask(ids)
        if updates:
            modality = dataclasses.replace(modality, **updates)
        calls.append(model_input(modality))
        return denoise_fn(modality)
    return call


def traced(run: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, dict[str, int]]:
    """Run once under the profiler and count attention-related operator and kernel names."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profile:
        output = run()
    names: dict[str, int] = {}
    for event in profile.events():
        if ATTENTION.search(event.name):
            key = f"{event.device_type.name.lower()}:{event.name}"
            names[key] = names.get(key, 0) + 1
    return output, dict(sorted(names.items()))


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Exact equality plus float64 RMS and maximum absolute difference, overall and per latent frame."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("compared D0 outputs differ in shape or dtype")
    delta = actual.double() - expected.double()
    unequal = actual != expected

    def measured(part: torch.Tensor, changed: torch.Tensor) -> dict:
        return {"unequal_elements": int(changed.sum()), "rms": float(part.square().mean().sqrt()),
                "maximum": float(part.abs().max())}

    return {"exact": bool(torch.equal(actual, expected)), **measured(delta, unequal),
            "per_latent_frame": [{"frame": frame, **measured(delta[:, :, frame], unequal[:, :, frame])}
                                 for frame in range(actual.shape[2])]}


def comparisons(outputs: dict[str, dict[str, torch.Tensor]], saved: dict[str, dict[str, torch.Tensor]]) -> dict:
    return {arm: {sigma: {"reference": difference(tensor, saved["reference"][sigma]),
                          "earlier": difference(tensor, saved["earlier"][sigma]),
                          FIRST: difference(tensor, outputs[FIRST][sigma])}
                  for sigma, tensor in by_sigma.items()}
            for arm, by_sigma in outputs.items()}


def summary(table: dict) -> dict:
    def exact(arm: str, against: str) -> bool:
        return all(row[against]["exact"] for row in table[arm].values())

    return {
        "current_repeats_earlier": exact(FIRST, "earlier"),
        "current_first_equals_last": exact(LAST, FIRST),
        "arms_reproducing_reference": [arm for arm in table if exact(arm, "reference")],
        "arms_changing_current": [arm for arm in table if not exact(arm, FIRST)],
        "rms_to_reference": {arm: {sigma: row["reference"]["rms"] for sigma, row in table[arm].items()}
                             for arm in table},
    }


def load(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.dtype != producer.DTYPE or not torch.isfinite(value).all():
        raise ValueError(f"saved tensor is not a finite {producer.DTYPE} tensor: {path}")
    return value


def run_controls(args: argparse.Namespace, session, epsilon: torch.Tensor) -> tuple[dict, dict]:  # noqa: ANN001
    """Run every arm on the same session, input and saved noise; return records and output tensors."""
    view = args.view[0]
    z_y, z_g, fps = producer.load_masters(view, args.objective)
    sigmas = producer.checked_sigmas(args.sigmas, session.model.sigmas)
    geometry = causal.deployed_geometry(
        session.model.scale_factors, block_latent_frames=producer.whole_clip_block([z_y.shape[1]]),
        context_latent_frames=args.context_latent_frames,
    )
    records: dict = {"start_flags": flags(), "arm_records": {}}
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    with session.transformer(loras=()) as transformer:
        denoise_fn = common.denoised_from_x0_model(transformer)
        grid = producer.clip_grid(z_y, fps, geometry, device=session.device,
                                  latent_channels=session.model.caps.latent_channels)
        plan = geometry.plan(grid.latent_frames)
        if plan != [(0, grid.latent_frames)]:
            raise ValueError(f"whole-clip plan does not cover the capture once: {plan}")
        source = grid.patchify(z_y.unsqueeze(0).to(device=session.device, dtype=producer.DTYPE))
        fresh = common.epsilon_block(source, args.seed)
        noise = epsilon.to(device=session.device, dtype=fresh.dtype)
        records["noise"] = {"fresh_draw_equals_saved": bool(torch.equal(fresh, noise)),
                            "shape": list(noise.shape), "dtype": str(noise.dtype)}
        clip = producer.WholeClip(z_y, z_g, grid, geometry, plan,
                                  [noise[:, slice(*grid.token_span(*span))] for span in plan], session.device)
        for arm in arms(args):
            calls: list[dict] = []
            fn = controlled(denoise_fn, INPUT_CHANGES.get(arm, ()), calls)

            def rollout(sigma: float, fn: Callable = fn) -> torch.Tensor:
                return producer.rollout(fn, clip, session.context, sigma, arm="d0", seed=args.seed,
                                        schedule=None).cpu()

            with numerical_control(arm) as state:
                if session.device.type == "cuda":
                    # Return the earlier arms' free blocks, so arm order cannot add fragmentation
                    # to the large transient of the dense-mask preprocessing.
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(session.device)
                started = time.perf_counter()
                outputs[arm] = {f"{sigma:.6f}": rollout(sigma) for sigma in sigmas}
                seconds = time.perf_counter() - started
                profiled, kernels = traced(functools.partial(rollout, sigmas[0]))
            records["arm_records"][arm] = {
                "model_input_changes": list(INPUT_CHANGES.get(arm, ())), "flags": state, "calls": calls,
                "attention_names": kernels,
                "profiled_equals_unprofiled": bool(torch.equal(profiled, outputs[arm][f"{sigmas[0]:.6f}"])),
                "rollout_seconds": seconds,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(session.device)
                if session.device.type == "cuda" else None,
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(session.device)
                if session.device.type == "cuda" else None,
            }
    return records, outputs


def save_outputs(root: Path, outputs: dict[str, dict[str, torch.Tensor]]) -> dict:
    saved = {}
    for arm, by_sigma in outputs.items():
        (root / arm).mkdir()
        saved[arm] = {}
        for sigma, tensor in by_sigma.items():
            path = root / arm / f"sigma_{sigma}_d0.pt"
            torch.save(tensor, path)
            saved[arm][sigma] = {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
    return saved


def saved_tensors(args: argparse.Namespace) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    outputs, noise = {}, {}
    for name, root in zip(SAVED, (args.reference, args.earlier), strict=True):
        files = saved_rollout(root, args)["files"]
        noise[name] = load(files.pop("epsilon"))
        outputs[name] = {sigma: load(path) for sigma, path in files.items()}
    if not torch.equal(noise["reference"], noise["earlier"]):
        raise ValueError("the reference and earlier runs saved different noise")
    return outputs, noise


def inspect_completion(args: argparse.Namespace, root: Path) -> dict:
    """Recheck inputs, software, saved outputs and every comparison number, without a model."""
    if root.resolve() != args.output.resolve():
        raise ValueError("prune_reproduction completion root differs from requested output")
    manifest = spec_protocol.inspect(args, SELECTOR)
    record = json.loads((root / "acceptance.json").read_text())
    if record != acceptance(args):
        raise ValueError("prune_reproduction acceptance inputs, settings or software differ")
    software.check_current(record["software"])
    results = json.loads((root / "results.json").read_text())
    expected = [f"{sigma:.6f}" for sigma in args.sigmas]
    if results["arms"] != arms(args) or any(list(results["outputs"][arm]) != expected for arm in arms(args)):
        raise ValueError("prune_reproduction saved arms or sigmas differ from the specification")
    items = [item for arm in arms(args) for item in results["outputs"][arm].values()]
    if any(sha256(root / item["path"]) != item["sha256"] for item in items):
        raise ValueError("prune_reproduction saved output bytes differ from results.json")
    outputs = {arm: {sigma: load(root / item["path"]) for sigma, item in results["outputs"][arm].items()}
               for arm in arms(args)}
    references, _ = saved_tensors(args)
    table = comparisons(outputs, references)
    if results["comparisons"] != table or results["summary"] != summary(table):
        raise ValueError("prune_reproduction saved comparisons differ from the saved tensors")
    return manifest


def verify_completion(spec: Path, root: Path) -> dict:
    """Raise on invalid existing completion, as required by the experiment queue contract."""
    return inspect_completion(read_spec(spec, root), root)


def evidence_paths(spec: Path, root: Path, *, verified: bool = False) -> list[Path]:
    record = (spec_protocol.inspect(read_spec(spec, root), SELECTOR) if verified
              else verify_completion(spec, root))
    return spec_protocol.evidence(record, root)


def main(argv: list[str] | None = None) -> int:
    args = spec_protocol.read_spec(parse_args(argv), SELECTOR, scientific_args)
    if args.dry_run:
        print(json.dumps({"protocol": SELECTOR, "settings": spec_protocol.settings(args)}, indent=2))  # noqa: T201
        return 0
    if args.output.exists():
        raise ValueError("prune_reproduction output must be fresh")
    before = acceptance(args)
    references, noise = saved_tensors(args)
    session = producer.open_session(args, script=f"onestep_avatar.experiments.{SELECTOR}",
                                    prompt=producer.resolve_prompt(args), transformer_path=args.transformer)
    context = hashlib.sha256(session.context.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    if any(saved_rollout(root, args)["context_sha256"] != context.hexdigest()
           for root in (args.reference, args.earlier)):
        raise ValueError("the current prompt context differs from the saved runs")
    stamp = session.stamp(dtype=str(producer.DTYPE))
    identity = before["model"]
    if (stamp.get("transformer_fingerprint") != identity["transformer_fingerprint"]
            or str(Path(stamp.get("transformer_path", "")).resolve()) != identity["transformer_path"]):
        raise ValueError("the opened transformer differs from the pinned inputs")
    records, outputs = run_controls(args, session, noise["reference"])
    args.output.mkdir(parents=True)
    table = comparisons(outputs, references)
    results = {"schema_version": 1, "protocol": SELECTOR, "arms": arms(args), "model_stamp": stamp,
               "context_sha256": context.hexdigest(), **records,
               "outputs": save_outputs(args.output, outputs), "comparisons": table, "summary": summary(table)}
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    software.check_current(before["software"])
    if before != acceptance(args):
        raise ValueError("pruning reproduction inputs or software changed during production")
    with (args.output / "acceptance.json").open("x") as stream:
        stream.write(json.dumps(before, indent=2) + "\n")
    spec_protocol.publish(args, SELECTOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(runtime.run(main))
