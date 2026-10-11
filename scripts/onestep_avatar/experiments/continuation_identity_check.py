"""Independent causal prefix and real-image controls; see the mirrored design doc."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ltx_core.components.patchifiers import get_pixel_coords
from ltx_core.model.transformer.modality import Modality
from scripts.onestep_avatar import evaluate, hashing
from scripts.onestep_avatar.corpus import dataset, precompute, subset
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.experiments import spec_protocol
from scripts.onestep_avatar.model import adapters, causal, common

SELECTOR = "continuation_identity_check"
EXTRAS = (
    "scripts/onestep_avatar/experiments/__init__.py",
    "scripts/onestep_avatar/experiments/continuation_identity_check.py",
    spec_protocol.ENTRY,
)
ARMS = (("original", 17), ("repeat", 17), ("prefix_5", 5), ("prefix_13", 13), ("changed_c0", 17))
FIELDS = ("latent", "sigma", "timesteps", "positions", "keyframes_mask")


def write_json(path: Path, record: dict) -> None:
    dataset.atomic_write(path, lambda temporary: temporary.write_text(json.dumps(record, indent=2) + "\n"))


def file_identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashing.sha256(path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return spec_protocol.outer_args(argv, __doc__)


def scientific_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("subset", "noise-file", "donor-image", "donor-export", "donor-view", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("donor-image-sha256", "donor-export-sha256", "source"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=("train", "held_out", "validation", "test"))
    parser.add_argument("--prompt")
    parser.add_argument("--sigma", type=float, choices=(1.0, 0.909375), required=True)
    parser.add_argument("--history", choices=("cache", "recompute"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    return args


def read_spec(spec: Path, root: Path) -> argparse.Namespace:
    return spec_protocol.read_spec(parse_args(["--spec", str(spec), "--output", str(root)]),
                                   SELECTOR, scientific_args)


def evaluation_arguments(args: argparse.Namespace) -> list[str]:
    arguments = ["--mode", "causal", "--subset", str(args.subset), "--source", args.source,
                 "--model", "2.5", "--variant", "dev", "--guide-mode", "d1", "--include-base",
                 "--span-latent-frames", "17", "--block-latent-frames", "2",
                 "--context-latent-frames", "8", "--schedule", str(args.sigma), "0",
                 "--history-mode", args.history, "--kv-source", "refresh", "--seed", str(args.seed),
                 "--noise-file", str(args.noise_file), "--output", str(args.output)]
    if args.split is not None:
        arguments += ["--split", args.split]
    if args.prompt is not None:
        arguments += ["--prompt", args.prompt]
    return arguments


def check_donor(args: argparse.Namespace, video: object, specification: object) -> tuple[torch.Tensor, dict]:
    """Recheck unchanged preprocessing, original pixels and another person's metadata."""
    if (hashing.sha256(args.donor_image) != args.donor_image_sha256
            or hashing.sha256(args.donor_export) != args.donor_export_sha256):
        raise ValueError("E3 donor bundle or export bytes changed")
    bundle = torch.load(args.donor_image, map_location="cpu", weights_only=True)
    image, fps = dataset.load_training_master(args.donor_image, bundle=bundle)
    export = json.loads(args.donor_export.read_text())
    metadata_path = args.donor_view.parent.parent / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    person = str(metadata.get("actor", {}).get("id", ""))
    donor_source = "/".join(args.donor_view.parts[-4:])
    inputs = bundle.get("preparation", {}).get("input_sha256", {})
    guide_path = args.donor_view / dataset.guide_bundle_name("white")
    donor_guide = torch.load(guide_path, map_location="cpu", weights_only=True)
    if (not person or person == str(video.actor) or export.get("source") != donor_source
            or export.get("source_frame") != 0 or export.get("capture_preprocessing_pixels_bit_identical") is not True
            or bundle.get("input_role") != "supplied_image" or bundle.get("pixel_frames") != 1
            or bundle.get("objective") != "white"
            or bundle.get("encode_contract_version") != precompute.ENCODE_CONTRACT_VERSION
            or bundle.get("vae_fingerprint") != precompute.file_fingerprint(Path(specification.paths.video_vae()))
            or tuple(image.shape) != (video.z_y.shape[0], 1, *video.z_y.shape[2:]) or fps != video.fps
            or str(guide_path.resolve()) not in inputs
            or bundle.get("input_fingerprint") != inputs.get(bundle.get("source"))
            or any(bundle.get(key) != donor_guide.get(key) for key in
                   ("box_xyxy", "edge", "objective", "vae_fingerprint", "encode_contract_version", "fps"))
            or donor_guide.get("source") != donor_source
            or any(bundle.get(key) != export.get(key) for key in ("box_xyxy", "edge"))
            or export.get("prepared_pixels_tensor_sha256") !=
            bundle.get("preparation", {}).get("prepared_pixels_sha256")):
        raise ValueError("E3 donor must be a separately encoded real first image from another person")
    for identities in (inputs, export.get("inputs", {}), export.get("output_files", {})):
        if not identities or any(hashing.sha256(Path(path)) != digest for path, digest in identities.items()):
            raise ValueError("E3 donor original preprocessing inputs or outputs changed")
    if export["output_files"].get(bundle["source"]) != bundle["input_fingerprint"]:
        raise ValueError("E3 donor export does not identify its encoded RGB input")
    provenance = {
        "bundle": file_identity(args.donor_image), "export": file_identity(args.donor_export),
        "metadata": file_identity(metadata_path), "actor": person, "source": donor_source,
        "image_source": bundle["source"], "image_sha256": bundle["input_fingerprint"],
        "box_xyxy": bundle["box_xyxy"], "edge": bundle["edge"],
        "vae_fingerprint": bundle["vae_fingerprint"], "preprocessing_inputs": inputs,
        "export_inputs": export["inputs"], "export_outputs": export["output_files"],
    }
    return image, provenance


def prepare(args: argparse.Namespace, *, fresh: bool = True) -> dict:
    ordinary = evaluate.parse_args(evaluation_arguments(args))
    specification, variants, cases, membership = evaluate.prepare_evaluation(ordinary, require_fresh_output=fresh)
    subset.verify_prepared_membership(membership)
    if (variants != [None] or len(cases) != 1 or membership["objective"] != "white"
            or cases[0][1] != 17 or ordinary.saved_noise is None):
        raise ValueError("E3 identity check requires one white base-only 17-frame case with saved noise")
    video, _, requested, checks = cases[0]
    if (video.fps != 30 or tuple(video.z_y.shape[:1]) != (128,) or tuple(video.z_y.shape[2:]) != (32, 32)
            or ordinary.mode_settings.teacher_forcing):
        raise ValueError("E3 identity check requires native 1024-square, 30-fps, generated history")
    donor, donor_record = check_donor(args, video, specification)
    if torch.equal(donor.to(torch.bfloat16), video.z_y[:, :1].to(torch.bfloat16)):
        raise ValueError("E3 donor image intervention must change c0 bytes")
    return {"ordinary": ordinary, "specification": specification, "video": video,
            "membership": membership, "requested": requested, "adapter_check": checks[0],
            "donor": donor, "donor_record": donor_record}


def input_tokens(prepared: dict, frames: int, changed: bool, device: torch.device) -> tuple:
    video, specification = prepared["video"], prepared["specification"]
    grid = common.ClipGrid.build(frames, video.z_y.shape[2] * specification.scale_factors.height,
                                video.z_y.shape[3] * specification.scale_factors.width, video.fps, specification,
                                device=device, dtype=torch.bfloat16,
                                latent_channels=specification.caps.latent_channels)
    capture = grid.patchify(video.z_y[:, :frames].unsqueeze(0).to(device=device, dtype=torch.bfloat16))
    guide = grid.patchify(video.z_g[:, :frames].unsqueeze(0).to(device=device, dtype=torch.bfloat16))
    if changed:
        capture = capture.clone()
        capture[:, :grid.tokens_per_latent_frame] = grid.patchify(
            prepared["donor"].unsqueeze(0).to(device=device, dtype=torch.bfloat16))
    noise = prepared["ordinary"].saved_noise[:, :frames * grid.tokens_per_latent_frame].to(device=device)
    return grid, capture, guide, noise


class BoundaryObserver:
    """Public read-only hooks; copied Modality/cache data cannot change execution."""

    def __init__(self, transformer: torch.nn.Module, grid: common.ClipGrid, history: str) -> None:
        self.transformer, self.grid, self.history = transformer, grid, history
        self.calls: list[dict] = []
        self.handles = []

    def __enter__(self) -> BoundaryObserver:
        self.handles = [self.transformer.register_forward_pre_hook(self.before, with_kwargs=True),
                        self.transformer.register_forward_hook(self.after, with_kwargs=True)]
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self.handles:
            handle.remove()

    def sink(self, modality: Modality) -> dict | None:
        if not modality.kv_caches or modality.kv_caches[0].length < self.grid.tokens_per_latent_frame:
            return None
        cache = modality.kv_caches[0]
        return {name: getattr(cache, name)[:, :self.grid.tokens_per_latent_frame].detach().cpu().clone()
                for name in ("k", "v")}

    def before(self, _module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        modality = kwargs.get("video", args[0] if args else None)
        if not isinstance(modality, Modality) or kwargs.get("audio") is not None:
            raise ValueError("E3 observer requires the actual video-only Modality")
        call = {name: getattr(modality, name).detach().cpu().clone() for name in FIELDS}
        call.update(context_sha256=hashing.tensor_sha256(modality.context), kv_write=modality.kv_write,
                    kv_start=modality.kv_start,
                    cache_lengths_before=None if modality.kv_caches is None else
                    [cache.length for cache in modality.kv_caches],
                    sink_before=None, sink_after=None)
        block = len(self.calls) // 2 if self.history == "cache" else len(self.calls)
        if self.history == "cache" and not modality.kv_write and block in (5, 7):
            call["sink_before"] = self.sink(modality)
        self.calls.append(call)

    def after(self, _module: torch.nn.Module, args: tuple, kwargs: dict, _output: object) -> None:
        modality = kwargs.get("video", args[0] if args else None)
        call = self.calls[-1]
        call["cache_lengths_after"] = (None if modality.kv_caches is None else
                                       [cache.length for cache in modality.kv_caches])
        if self.history == "cache" and len(self.calls) == 2:
            call["sink_after"] = self.sink(modality)


def same_tensor(actual: object, expected: torch.Tensor, name: str) -> None:
    if (not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype
            or actual.shape != expected.shape or not torch.equal(actual, expected)):
        raise ValueError(f"E3 actual transformer {name} differs")


def validated_positions(value: object, grid: common.ClipGrid) -> torch.Tensor:
    """Check exact native coordinates, then keep their bytes across every control.

    CPU divides float32 timestamps by fps. CUDA uses a rounded reciprocal and
    multiplies them; one ULP can separate their results. Both use the public
    integer patch bounds. Accept these two exact constructions, never a fitted
    tolerance, and compare observed calls with the saved native construction.
    """
    integer = grid.tools.patchifier.get_patch_grid_bounds(
        output_shape=grid.tools.target_shape, device=torch.device("cpu"))
    native = get_pixel_coords(integer, grid.tools.scale_factors, causal_fix=grid.tools.causal_fix).float()
    native[:, 0] *= torch.tensor(1 / grid.fps, dtype=torch.float32)
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.shape != grid.positions.shape
            or not torch.isfinite(value).all()
            or not (torch.equal(value, grid.positions) or torch.equal(value, native))):
        raise ValueError("E3 saved native positions differ from exact patch bounds and frame rate")
    return value


def verify_boundaries(  # noqa: PLR0912, PLR0915 -- ordered checks of observed native block/cache transactions
                      calls: list[dict], grid: common.ClipGrid, capture: torch.Tensor, guide: torch.Tensor,
                      noise: torch.Tensor, context: torch.Tensor, output: torch.Tensor,
                      sigma: float, history: str) -> dict:
    """Check actual native inputs against fixed data and already saved predictions."""
    geometry = causal.CausalGeometry(grid.tools.scale_factors, 2, 8)
    plan = geometry.plan(grid.latent_frames)
    if not isinstance(calls, list) or len(calls) != len(plan) * (2 if history == "cache" else 1):
        raise ValueError("E3 actual transformer trace has the wrong call count")
    generated = grid.patchify(output)
    sink = None
    sink_checks = 0
    context_hash = hashing.tensor_sha256(context)
    for index, span in enumerate(plan):
        lo, hi = grid.token_span(*span)
        noisy = common.mix_block_noise(guide[:, lo:hi], noise[:, lo:hi], sigma)
        if index == 0:
            noisy = common.with_clean_prefix(noisy, capture[:, :grid.tokens_per_latent_frame])
        history_spans = [] if history == "cache" else causal.retained_prefix_spans(plan, geometry, span[0])
        slices = [grid.token_span(start, end) for start, end, _ in history_spans] + [(lo, hi)]
        earlier = [generated[:, start:end] for start, end in slices[:-1]]
        prefix = sum(value.shape[1] for value in earlier)
        latent = torch.cat([*earlier, noisy], dim=1)
        clean_tokens = prefix + (grid.tokens_per_latent_frame if index == 0 else 0)
        timestep = torch.full((1, latent.shape[1], 1), sigma, dtype=torch.float32)
        timestep[:, :clean_tokens] = 0
        positions = torch.cat([grid.positions[:, :, start:end] for start, end in slices], dim=2)
        keyframes = torch.cat([grid.keyframes_mask[:, start:end] for start, end in slices], dim=1)
        count = min(span[0], 9) * grid.tokens_per_latent_frame
        native = calls[index * (2 if history == "cache" else 1)]
        expectations = {"latent": latent, "sigma": torch.tensor([sigma], dtype=torch.float32),
                        "timesteps": timestep, "positions": positions, "keyframes_mask": keyframes}
        for name, expected in expectations.items():
            same_tensor(native.get(name), expected, name)
        if (native.get("context_sha256") != context_hash or native.get("kv_write") is not False
                or native.get("kv_start") != (count if history == "cache" else 0)):
            raise ValueError("E3 actual transformer text or cache read boundary differs")
        if history == "recompute":
            if any(native.get(name) is not None for name in
                   ("cache_lengths_before", "cache_lengths_after", "sink_before", "sink_after")):
                raise ValueError("E3 recompute trace unexpectedly carries a cache")
            continue
        lengths = native.get("cache_lengths_before")
        if not lengths or any(length != count for length in lengths) or native.get("cache_lengths_after") != lengths:
            raise ValueError("E3 denoising changed the cache or its retained length")
        if index in (5, 7):
            current = native.get("sink_before")
            if sink is None or not isinstance(current, dict) or set(current) != {"k", "v"}:
                raise ValueError("E3 after-eviction cache sink observation is missing")
            for name in ("k", "v"):
                same_tensor(current[name], sink[name], "retained cache sink")
            sink_checks += 1
        elif native.get("sink_before") is not None:
            raise ValueError("E3 unexpected cache sink observation")
        refresh = calls[index * 2 + 1]
        expected_refresh = {"latent": generated[:, lo:hi], "sigma": torch.zeros(1, dtype=torch.float32),
                            "timesteps": torch.zeros((1, hi - lo, 1), dtype=torch.float32),
                            "positions": grid.positions[:, :, lo:hi],
                            "keyframes_mask": grid.keyframes_mask[:, lo:hi]}
        for name, expected in expected_refresh.items():
            same_tensor(refresh.get(name), expected, "refresh " + name)
        if (refresh.get("context_sha256") != context_hash or refresh.get("kv_write") is not True
                or refresh.get("kv_start") != count or refresh.get("cache_lengths_before") != lengths
                or refresh.get("cache_lengths_after") != [count + hi - lo] * len(lengths)):
            raise ValueError("E3 actual generated-history refresh boundary differs")
        if index == 0:
            sink = refresh.get("sink_after")
            if (not isinstance(sink, dict) or set(sink) != {"k", "v"}
                    or any(not isinstance(value, torch.Tensor) or value.shape[1] != grid.tokens_per_latent_frame
                           or not torch.isfinite(value).all() for value in sink.values())):
                raise ValueError("E3 first-image cache sink observation is missing")
        elif refresh.get("sink_after") is not None:
            raise ValueError("E3 unexpected post-refresh sink observation")
    return {"actual_calls": len(calls), "clean_c0_input": True, "generated_history_refresh": history == "cache",
            "retained_sink_checks": sink_checks, "history": history}


def summarize(outputs: dict[str, torch.Tensor]) -> dict:
    if set(outputs) != {name for name, _frames in ARMS}:
        raise ValueError("E3 five-arm output inventory differs")
    baseline = outputs["original"]
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in outputs.values()):
        raise ValueError("E3 controls require finite encoded outputs")
    controls = {"repeat": torch.equal(baseline, outputs["repeat"]),
                "prefix_5": torch.equal(baseline[:, :, :5], outputs["prefix_5"]),
                "prefix_13": torch.equal(baseline[:, :, :13], outputs["prefix_13"])}
    if outputs["changed_c0"].shape != baseline.shape:
        raise ValueError("E3 changed-image output shape differs")
    delta = outputs["changed_c0"][:, :, 1:].double() - baseline[:, :, 1:].double()
    sensitivity = {"encoded_frame_range": [1, 17], "bit_identical": not bool(torch.count_nonzero(delta)),
                   "difference_rms": float(delta.square().mean().sqrt()),
                   "difference_max_abs": float(delta.abs().max()),
                   "interpretation": "input sensitivity only; no identity or action quality claim"}
    return {"state": "passed" if all(controls.values()) and not sensitivity["bit_identical"] else "failed",
            "exact_controls": controls, "changed_c0_after_frame_zero": sensitivity}


def arm_provenance(args: argparse.Namespace, prepared: dict, name: str, frames: int, producer: dict) -> dict:
    return {"experiment": SELECTOR, "arm": name, "source": prepared["video"].source,
            "actor": str(prepared["video"].actor), "subset": file_identity(args.subset),
            "membership_sha256": prepared["membership"]["sha256"],
            "input_file_hashes": prepared["video"].hashes, "noise_file": file_identity(args.noise_file),
            "donor": prepared["donor_record"] if name == "changed_c0" else None,
            "conditions": prepared["requested"], "coverage_encoded_frames": [0, frames],
            "prompt": args.prompt, "software": producer}


def execute(args: argparse.Namespace) -> dict:
    prepared = prepare(args)
    producer = software.capture("evaluation", "causal", extra_sources=EXTRAS)
    from scripts.prune.core.session import DEFAULT_PROMPT, DTYPE, Session  # noqa: PLC0415
    from scripts.prune.data import prompt_cache  # noqa: PLC0415

    device = torch.device(f"cuda:{args.gpu_id}")
    prompt = DEFAULT_PROMPT if args.prompt is None else args.prompt
    context = prompt_cache.get_or_build(prepared["specification"], prompt, DTYPE, device)
    software.check_current(producer)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(context.cpu(), args.output / "text.pt")
    torch.save(prepared["ordinary"].saved_noise, args.output / "noise.pt")
    session = Session(prepared["specification"], device, "onestep_avatar.continuation_identity_check", context)
    outputs = {}
    positions = None
    with adapters.inference_transformer(session, None, None, method=adapters.UNMERGED) as transformer:
        for name, frames in ARMS:
            software.check_current(producer)
            grid, capture, guide, noise = input_tokens(prepared, frames, name == "changed_c0", device)
            actual_positions = grid.positions.detach().cpu()
            if positions is None:
                positions = actual_positions.clone()
                torch.save(positions, args.output / "positions.pt")
            same_tensor(actual_positions, positions[:, :, :frames * grid.tokens_per_latent_frame],
                        "same-source native positions")
            with BoundaryObserver(transformer, grid, args.history) as observer:
                output, record = evaluate.sample_case(
                    transformer, context, grid, capture, guide, noise, mode="causal",
                    mode_settings=prepared["ordinary"].mode_settings, guide_mode="d1",
                    schedule=[args.sigma, 0], seed=args.seed, history_mode=args.history, kv_source="refresh")
            folder = args.output / name
            evaluate.save_case(output, {**record, **arm_provenance(args, prepared, name, frames, producer)}, folder)
            torch.save(observer.calls, folder / "trace.pt")
            outputs[name] = output
    summary = summarize(outputs)
    write_json(args.output / "result.json", summary)
    software.check_current(producer)
    verify_scientific(args)
    spec_protocol.publish(args, SELECTOR)
    return summary


def verify_scientific(args: argparse.Namespace) -> dict:
    """All completion checks read saved bytes and public CPU preflight only."""
    prepared = prepare(args, fresh=False)
    producer = software.capture("evaluation", "causal", extra_sources=EXTRAS)
    context = torch.load(args.output / "text.pt", map_location="cpu", weights_only=True)
    if (not isinstance(context, torch.Tensor) or context.ndim != 3 or context.shape[0] != 1
            or context.dtype != torch.bfloat16 or not torch.isfinite(context).all()):
        raise ValueError("E3 saved text must be finite batch-one native context")
    same_tensor(torch.load(args.output / "noise.pt", map_location="cpu", weights_only=True),
                prepared["ordinary"].saved_noise, "saved full-range noise")
    original_grid, *_ = input_tokens(prepared, 17, False, torch.device("cpu"))
    positions = validated_positions(
        torch.load(args.output / "positions.pt", map_location="cpu", weights_only=True), original_grid)
    outputs, boundary_records, sinks = {}, {}, {}
    for name, frames in ARMS:
        grid, capture, guide, noise = input_tokens(prepared, frames, name == "changed_c0", torch.device("cpu"))
        grid = replace(grid, positions=positions[:, :, :frames * grid.tokens_per_latent_frame])
        folder = args.output / name
        record = json.loads((folder / "result.json").read_text())
        software.check_current(record.get("software"))
        expected = {**arm_provenance(args, prepared, name, frames, producer), "schema_version": 2,
                    "state": "complete", "mode": "causal",
                    "mode_settings": asdict(prepared["ordinary"].mode_settings), "guide_mode": "d1",
                    "frames": frames, "schedule": [args.sigma, 0], "seed": args.seed,
                    "history_mode": args.history, "kv_source": "refresh",
                    "capture_sha256": hashing.tensor_sha256(capture), "guide_sha256": hashing.tensor_sha256(guide),
                    "noise_sha256": hashing.tensor_sha256(noise), "text_sha256": hashing.tensor_sha256(context),
                    "c0_sha256": hashing.tensor_sha256(capture[:, :grid.tokens_per_latent_frame])}
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("E3 saved arm fixed inputs or actual conditions differ")
        path = folder / "generated.pt"
        output = torch.load(path, map_location="cpu", weights_only=True)
        expected_shape = [1, capture.shape[-1], frames, *prepared["video"].z_y.shape[2:]]
        if (not isinstance(output, torch.Tensor) or output.dtype != torch.bfloat16
                or list(output.shape) != expected_shape or not torch.isfinite(output).all()):
            raise ValueError("E3 saved arm output shape, dtype or finite values differ")
        first_image = grid.unpatchify_block(capture[:, :grid.tokens_per_latent_frame], 1)
        same_tensor(output[:, :, :1], first_image, "clean first-image output")
        if record.get("output") != {"path": str(path), "sha256": hashing.sha256(path), "shape": list(output.shape)}:
            raise ValueError("E3 saved arm output bytes or identity differ")
        blocks = (frames - 1) // 2
        counts = record.get("call_counts", {})
        if any(counts.get(key) != value for key, value in {
                "denoise_calls": blocks, "refresh_calls": blocks if args.history == "cache" else 0,
                "prime_calls": 0, "model_calls": blocks * (2 if args.history == "cache" else 1)}.items()):
            raise ValueError("E3 saved arm actual native call counts differ")
        trace = torch.load(folder / "trace.pt", map_location="cpu", weights_only=True)
        boundary_records[name] = verify_boundaries(trace, grid, capture, guide, noise, context, output,
                                                  args.sigma, args.history)
        if args.history == "cache" and name in ("original", "changed_c0"):
            sinks[name] = trace[1]["sink_after"]
        outputs[name] = output
    summary = summarize(outputs)
    if args.history == "cache" and all(torch.equal(sinks["original"][name], sinks["changed_c0"][name])
                                       for name in ("k", "v")):
        raise ValueError("E3 changed first-image input did not change the observed cache sink")
    if json.loads((args.output / "result.json").read_text()) != summary:
        raise ValueError("E3 saved summary differs from actual encoded outputs")
    return {**summary, "actual_boundaries": boundary_records}


def verify_completion(spec: Path, root: Path) -> dict:
    args = read_spec(spec, root)
    spec_protocol.inspect(args, SELECTOR)
    result = verify_scientific(args)
    if result["state"] != "passed":
        raise ValueError("E3 independent prefix, repeat or first-image sensitivity control failed")
    return result


def evidence_paths(spec: Path, root: Path, *, verified: bool = False) -> list[Path]:
    if not verified:
        verify_completion(spec, root)
    return spec_protocol.evidence(spec_protocol.inspect(read_spec(spec, root), SELECTOR), root)


def main(argv: list[str] | None = None) -> int:
    outer = parse_args(argv)
    args = spec_protocol.read_spec(outer, SELECTOR, scientific_args)
    if args.dry_run:
        prepared = prepare(args)
        print(json.dumps({"arms": ARMS, "conditions": prepared["requested"],  # noqa: T201
                          "donor": prepared["donor_record"]}, indent=2))
        return 0
    return 0 if execute(args)["state"] == "passed" else 2


if __name__ == "__main__":
    from scripts.onestep_avatar.execution.runtime import run

    raise SystemExit(run(main))
