"""Observe native cached/recomputed history without another model algorithm.

See doc/continuation_check.md. Capture and reference are separate bounded attempts;
CPU snapshots prevent simultaneous resident cache/reference allocation. This
experiment owner is temporary until the native source-migration gates pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from ltx_core.model.transformer.modality import Modality
from scripts.onestep_avatar import LTX_ROOT, evaluate
from scripts.onestep_avatar.corpus import dataset
from scripts.onestep_avatar.execution import software
from scripts.onestep_avatar.hashing import sha256
from scripts.onestep_avatar.model import causal, common
from scripts.onestep_avatar.training.resources import Phase

TARGETS = (("before", (3, 5)), ("after", (11, 13)))
CHUNK_ELEMENTS = 131072
EXTRA_SOURCES = ("scripts/onestep_avatar/continuation_check.py",
                 "scripts/onestep_avatar/training/resources.py",
                 "scripts/onestep_avatar/execution/supervision.py")
IMPORTED_OWNER_SHA256 = sha256(Path(__file__))


def write_json(path: Path, value: dict) -> None:
    dataset.atomic_write(path, lambda temporary: temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n"))


def file_record(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path),
            "bytes": path.stat().st_size}


def checked_path(root: Path, record: dict) -> Path:
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError("continuation snapshot file is missing, changed or outside its attempt")
    if path.stat().st_size != record["bytes"]:
        raise ValueError("continuation snapshot byte count changed")
    return path


def tensor_record(value: torch.Tensor) -> dict:
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": evaluate.tensor_sha256(value)}


def compare_chunks(left: torch.Tensor, right: torch.Tensor) -> dict:
    """Read bounded CPU chunks, hashing all native values and reducing in float64."""
    if left.shape != right.shape or left.dtype != right.dtype or left.ndim != 3 or left.shape[0] != 1:
        raise ValueError("native K/V comparison shape or dtype differs")
    header = json.dumps({"shape": list(left.shape), "dtype": str(left.dtype)}).encode()
    digests = [hashlib.sha256(header), hashlib.sha256(header)]
    difference_squared = reference_squared = maximum = 0.0
    exact = True
    width = left.shape[-1]
    if width < 1 or width > CHUNK_ELEMENTS or left.numel() < 1:
        raise ValueError("native observation exceeds its declared chunk width")
    stride = max(1, CHUNK_ELEMENTS // width)
    for start in range(0, left.shape[1], stride):
        chunks = [value[:, start:start + stride].detach().cpu().contiguous() for value in (left, right)]
        if not all(torch.isfinite(value).all() for value in chunks):
            raise ValueError("native K/V observation contains nonfinite values")
        for digest, chunk in zip(digests, chunks, strict=True):
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
        exact = exact and torch.equal(*chunks)
        delta = chunks[0].double() - chunks[1].double()
        difference_squared += float(delta.square().sum())
        reference_squared += float(chunks[1].double().square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {"shape": list(left.shape), "dtype": str(left.dtype), "elements": left.numel(),
            "cached_sha256": digests[0].hexdigest(), "reference_sha256": digests[1].hexdigest(),
            "bit_identical": exact, "rms_delta": math.sqrt(difference_squared / left.numel()),
            "max_abs_delta": maximum,
            "relative_l2": None if reference_squared == 0 else math.sqrt(difference_squared / reference_squared),
            "reference_norm_zero": reference_squared == 0, "chunk_elements": CHUNK_ELEMENTS}


def retained_frames(grid: common.ClipGrid, geometry: causal.CausalGeometry, span: tuple[int, int]) -> list[int]:
    return [frame for lo, hi, _ in causal.retained_prefix_spans(
        geometry.plan(grid.latent_frames), geometry, span[0]) for frame in range(lo, hi)]


class CacheRecorder:
    """Delegate the native rollout; save only target inputs and actual cache values."""

    def __init__(self, predict: Callable, grid: common.ClipGrid, geometry: causal.CausalGeometry, root: Path, *,
                 sigma: float,
                 capture: torch.Tensor, guide: torch.Tensor, noise: torch.Tensor,
                 context: torch.Tensor, teacher_forcing: bool):
        self.predict, self.grid, self.geometry, self.root = predict, grid, geometry, root
        self.sigma, self.capture, self.guide, self.noise = sigma, capture, guide, noise
        self.context, self.teacher_forcing = context, teacher_forcing
        self.clean = torch.zeros_like(capture)
        self.calls = 0
        self.snapshots: list[dict] = []
        self.previous_prediction = None

    def check_inputs(self, modality: Modality, span: tuple[int, int], refresh: int) -> None:
        lo, hi = self.grid.token_span(*span)
        c0 = self.capture[:, :self.grid.tokens_per_latent_frame]
        expected = common.block_modality(
            self.grid, modality.latent, self.context, 0.0 if refresh else self.sigma, token_slices=[(lo, hi)],
            clean_prefix_tokens=c0.shape[1] if not refresh and span[0] == 0 else 0)
        if (modality.sigma.dtype != torch.float32 or not torch.equal(modality.sigma, expected.sigma)
                or modality.timesteps.dtype != torch.float32
                or not torch.equal(modality.timesteps, expected.timesteps)
                or not torch.equal(modality.positions, expected.positions)
                or not torch.equal(modality.keyframes_mask, expected.keyframes_mask)
                or not torch.equal(modality.context, self.context)):
            raise ValueError("native cached sigma, token times or positions changed")
        if refresh:
            expected_clean = (self.capture[:, lo:hi] if self.teacher_forcing else
                              common.with_clean_prefix(self.previous_prediction, c0 if span[0] == 0 else None))
            if not torch.equal(modality.latent, expected_clean):
                raise ValueError("native refresh did not receive the declared clean history")
            self.clean[:, lo:hi] = modality.latent.detach()
        else:
            expected_noisy = common.with_clean_prefix(
                causal.mix_block_noise(self.guide[:, lo:hi], self.noise[:, lo:hi], self.sigma),
                c0 if span[0] == 0 else None)
            if not torch.equal(modality.latent, expected_noisy):
                raise ValueError("native current noisy tokens differ from the exact saved guide/noise/c0")

    def __call__(self, modality: Modality) -> torch.Tensor:
        plan = self.geometry.plan(self.grid.latent_frames)
        index, refresh = divmod(self.calls, 2)
        if index >= len(plan) or bool(modality.kv_write) != bool(refresh) or modality.kv_caches is None:
            raise ValueError("native cached call inventory differs from the fixed direct-step rollout")
        span = plan[index]
        c0 = self.capture[:, :self.grid.tokens_per_latent_frame]
        self.check_inputs(modality, span, refresh)
        target = next((name for name, wanted in TARGETS if wanted == span), None)
        snapshot = None
        if target is not None and not refresh:
            frames = retained_frames(self.grid, self.geometry, span)
            prefix = len(frames) * self.grid.tokens_per_latent_frame
            if modality.kv_start != prefix or any(cache.length != prefix for cache in modality.kv_caches):
                raise ValueError("native retained cache differs from the shared frame policy")
            directory = self.root / "snapshots" / target
            directory.mkdir(parents=True)
            inputs = {"clean_history": self.clean.detach().cpu(), "noisy": modality.latent.detach().cpu(),
                      "capture": self.capture.detach().cpu(), "guide": self.guide.detach().cpu(),
                      "noise": self.noise.detach().cpu(), "c0": c0.detach().cpu(),
                      "text": self.context.detach().cpu(), "positions": modality.positions.detach().cpu(),
                      "timesteps": modality.timesteps.detach().cpu()}
            inputs_path = directory / "inputs.pt"
            dataset.atomic_write(inputs_path, lambda temporary: torch.save(inputs, temporary))
            snapshot = {"phase": target, "span": list(span), "retained_frames": frames,
                        "prefix_tokens": prefix, "inputs": file_record(self.root, inputs_path),
                        "tensors": {name: tensor_record(value) for name, value in inputs.items()}, "layers": []}
            self.snapshots.append(snapshot)
            for layer, cache in enumerate(modality.kv_caches):
                k, v = cache.read(prefix)
                if k is None or v is None:
                    raise ValueError("target history cache is unexpectedly empty")
                path = directory / f"layer_{layer:04d}.pt"
                values = {"k": k.detach().cpu(), "v": v.detach().cpu()}
                if not all(torch.isfinite(value).all() for value in values.values()):
                    raise ValueError("cached history contains nonfinite values")
                dataset.atomic_write(path, lambda temporary, values=values: torch.save(values, temporary))
                snapshot["layers"].append({"layer": layer, "file": file_record(self.root, path),
                                            "tensors": {name: tensor_record(value) for name, value in values.items()}})
                del values, k, v
            # CPU tests use a CPU rollout: detach alone must not alias mutable clean history.
            del inputs
        prediction = self.predict(modality)
        if not refresh:
            self.previous_prediction = prediction
        if snapshot is not None:
            path = self.root / "snapshots" / target / "cached_prediction.pt"
            dataset.atomic_write(path, lambda temporary: torch.save(prediction.detach().cpu(), temporary))
            snapshot["cached_prediction"] = file_record(self.root, path)
            snapshot["cached_prediction_tensor"] = tensor_record(prediction)
        self.calls += 1
        return prediction

    def finish(self, layers: int) -> None:
        if (self.calls != 2 * len(self.geometry.plan(self.grid.latent_frames))
                or [row["phase"] for row in self.snapshots] != [name for name, _ in TARGETS]
                or any([item["layer"] for item in row["layers"]] != list(range(layers))
                       for row in self.snapshots)):
            raise ValueError("cached history observation inventory is incomplete")


def load_inputs(root: Path, row: dict) -> dict[str, torch.Tensor]:
    values = torch.load(checked_path(root, row["inputs"]), map_location="cpu", weights_only=True)
    if (set(values) != set(row["tensors"]) or any(
            not isinstance(value, torch.Tensor) or tensor_record(value) != row["tensors"][name]
            for name, value in values.items())):
        raise ValueError("saved continuation input tensors changed")
    return values


def check_snapshot_files(root: Path, snapshots: list[dict]) -> None:
    if [(row["phase"], row["span"]) for row in snapshots] != [(name, list(span)) for name, span in TARGETS]:
        raise ValueError("continuation snapshot inventory differs from the two declared targets")
    for row in snapshots:
        load_inputs(root, row)
        for layer in row["layers"]:
            checked_path(root, layer["file"])
        checked_path(root, row["cached_prediction"])


class ReferenceObserver:
    """Intercept actual attention inputs and restore every per-instance operation."""

    def __init__(self, transformer: torch.nn.Module, root: Path, progress_path: Path | None = None):
        self.blocks = common.base_model(transformer).transformer_blocks
        self.root = root
        self.originals = []
        self.active = None
        self.observations: list[dict] = []
        self.progress_path = progress_path

    def __enter__(self):
        try:
            for layer, block in enumerate(self.blocks):
                for name in ("attention_function", "masked_attention_function"):
                    attention = block.attn1
                    original = getattr(attention, name)
                    self.originals.append((attention, name, original))

                    def operation(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int,
                                  *args: object, original: Callable = original, layer: int = layer,
                                  **kwargs: object) -> torch.Tensor:
                        if self.active is not None:
                            self.observe(layer, k, v, heads)
                        return original(q, k, v, heads, *args, **kwargs)

                    setattr(attention, name, operation)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_exception):
        for module, name, original in reversed(self.originals):
            setattr(module, name, original)
        self.originals.clear()
        self.active = None

    @contextmanager
    def target(self, row: dict) -> Iterator[None]:
        if self.active is not None:
            raise ValueError("native history observation is already active")
        before = len(self.observations)
        self.active = row
        try:
            yield
            found = [item["layer"] for item in self.observations[before:]]
            if found != list(range(len(self.blocks))):
                raise ValueError("native reference did not observe each layer exactly once")
        finally:
            self.active = None

    def observe(self, layer: int, k: torch.Tensor, v: torch.Tensor, heads: int) -> None:
        row = self.active
        if [item["layer"] for item in row["layers"]] != list(range(len(self.blocks))):
            raise ValueError("saved cached layer inventory differs from the native model")
        saved = row["layers"][layer]
        values = torch.load(checked_path(self.root, saved["file"]), map_location="cpu", weights_only=True)
        prefix = row["prefix_tokens"]
        if (set(values) != {"k", "v"} or k.shape[1] <= prefix or v.shape[1] != k.shape[1]
                or any(tensor_record(value) != saved["tensors"][name] for name, value in values.items())):
            raise ValueError("saved cached K/V values or native reference range differ")
        self.observations.append({"phase": row["phase"], "span": row["span"], "layer": layer,
                                  "heads": heads, "retained_frames": row["retained_frames"],
                                  "k": compare_chunks(values["k"], k[:, :prefix]),
                                  "v": compare_chunks(values["v"], v[:, :prefix]),
                                  "scope": "actual kernel history K after normalization/RoPE and native V"})
        if self.progress_path is not None:
            write_json(self.progress_path, {"state": "observing", "layerwise": self.observations})


@torch.no_grad()
def reference_blocks(predict: Callable, transformer: torch.nn.Module, grid: common.ClipGrid,
                     geometry: causal.CausalGeometry, root: Path, snapshots: list[dict], *,
                     context: torch.Tensor, sigma: float, output_root: Path | None = None
                     ) -> tuple[list[dict], list[dict]]:
    records = []
    check_snapshot_files(root, snapshots)
    progress = None if output_root is None else output_root / "reference_progress.json"
    with ReferenceObserver(transformer, root, progress) as observer:
        for expected, row in zip(TARGETS, snapshots, strict=True):
            name, span = expected
            if (row["phase"] != name or row["span"] != list(span)
                    or row["retained_frames"] != retained_frames(grid, geometry, span)):
                raise ValueError("saved continuation target span or frame coordinates differ")
            values = load_inputs(root, row)
            lo, hi = grid.token_span(*span)
            if (not torch.equal(values["text"], context.detach().cpu())
                    or not torch.equal(values["positions"], grid.positions[:, :, lo:hi].cpu())):
                raise ValueError("native reference text or global positions differ from captured inputs")
            with observer.target(row):
                output = causal.denoise_with_clean_history(
                    predict, grid, geometry, values["clean_history"].to(context.device),
                    values["noisy"].to(context.device), context, sigma, span)
            cached = torch.load(checked_path(root, row["cached_prediction"]),
                                map_location="cpu", weights_only=True)
            if tensor_record(cached) != row["cached_prediction_tensor"]:
                raise ValueError("saved cached block prediction changed")
            record = {"phase": name, "span": list(span), "inputs": row["inputs"],
                      "input_tensors": row["tensors"], "prediction": compare_chunks(cached, output),
                      "reference_prediction_tensor": tensor_record(output)}
            if output_root is not None:
                path = output_root / f"reference_{name}.pt"
                dataset.atomic_write(path, lambda temporary, output=output: torch.save(
                    output.detach().cpu(), temporary))
                record["reference_prediction"] = file_record(output_root, path)
            records.append(record)
            if progress is not None:
                write_json(progress, {"state": "observing", "layerwise": observer.observations,
                                      "reference_blocks": records})
    return observer.observations, records


def _replace_argument(arguments: list[str], name: str, value: str) -> list[str]:
    result = list(arguments)
    if name in result:
        index = result.index(name)
        result[index + 1] = value
    else:
        result.extend([name, value])
    return result


def prepare(args: argparse.Namespace) -> dict:  # noqa: PLR0912, PLR0915 -- ordered pre-weight scientific identity checks
    """Check frozen protocol, current native inputs and original saved text on CPU."""
    snapshot = None
    control, control_identity = None, None
    if args.phase == "control":
        plan = json.loads(args.control_plan.read_text())
        if plan.get("original_protocol_sha256") != sha256(args.protocol):
            raise ValueError("the data-only control plan differs from the frozen E3 protocol")
        selected = [row for row in plan["controls"] if row["id"] == args.control_id]
        if len(selected) != 1:
            raise ValueError("control ID is missing or ambiguous in the data-only plan")
        control = selected[0]
        args.sigma = control["schedule"][0]
        args.history = "capture" if "--teacher-forcing" in control["arguments"] else "generated"
        control_identity = {"path": str(args.control_plan.resolve()), "sha256": sha256(args.control_plan),
                            "id": args.control_id}
    if args.phase == "reference":
        snapshot = json.loads(args.snapshot.read_text())
        if snapshot.get("state") != "complete" or snapshot.get("phase") != "capture":
            raise ValueError("reference requires a complete capture manifest")
        software.check_current(snapshot["software"])
        args.protocol = Path(snapshot["protocol"]["path"])
        args.sigma, args.history = snapshot["sigma"], snapshot["history"]
        if sha256(args.protocol) != snapshot["protocol"]["sha256"]:
            raise ValueError("original E3 protocol bytes changed")
        check_snapshot_files(args.snapshot.parent, snapshot["snapshots"])
    protocol = json.loads(args.protocol.read_text())
    if (protocol.get("wall_seconds_per_case") != 1800
            or protocol.get("observed_gpu_memory_limit_mib") != 48800):
        raise ValueError("continuation requires the unchanged original E3 resource meanings")
    if (protocol.get("model") != "2.5" or protocol.get("variant") != "dev"
            or protocol.get("guide_mode") != "d1" or protocol.get("objective") != "white"
            or protocol.get("latent_frames") != 17 or protocol.get("adapter") is not None
            or protocol.get("geometry") != {"B": 2, "D": 8, "sink": 1, "height": 1024, "width": 1024}
            or sha256(Path(protocol["membership"])) != protocol["membership_sha256"]):
        raise ValueError("original E3 protocol cohort or membership bytes differ")
    role = ("cached" if control["history_mode"] == "cache" else "recompute") if control is not None else (
        "cached" if args.phase == "capture" else "recompute")
    job = next((row for row in protocol["jobs"] if row["sigma"] == args.sigma and row["role"] == role), None)
    if job is None:
        raise ValueError("original protocol has no requested direct-step cohort")
    original_arguments = list(job["arguments"])
    old_output = Path(original_arguments[original_arguments.index("--output") + 1])
    original_result = old_output / "case_0000/variant_000/result.json"
    baseline = json.loads(original_result.read_text())
    if baseline.get("state") != "complete":
        raise ValueError("the original E3 baseline is incomplete")
    if sha256(Path(baseline["output"]["path"])) != baseline["output"]["sha256"]:
        raise ValueError("the original E3 baseline output bytes changed")
    software.validate(baseline["software"])
    arguments = _replace_argument(original_arguments if control is None else control["arguments"],
                                  "--output", str(args.output / "results"))
    arguments = _replace_argument(arguments, "--gpu-id", str(args.gpu_id))
    arguments = _replace_argument(arguments, "--prompt", baseline["prompt"])
    if args.history == "capture" and control is None:
        arguments.append("--teacher-forcing")
    options = evaluate.parse_args(arguments)
    if (options.mode != "causal" or options.model != "2.5" or options.variant != "dev"
            or options.guide_mode != "d1" or options.schedule != [args.sigma, 0.0]
            or options.mode_settings.block_latent_frames != 2
            or options.mode_settings.context_latent_frames != 8 or options.span_latent_frames != 17
            or options.checkpoint or options.cfg != 1 or options.stg != 0 or options.rescale != 0
            or options.history_mode != ("cache" if role == "cached" else "recompute")
            or options.kv_source != "refresh" or (control is None and options.changed_noise_file is not None)
            or options.seed != protocol["noise"]["seed"]
            or options.noise_file.resolve() != Path(protocol["noise"]["path"]).resolve()
            or sha256(options.noise_file) != protocol["noise"]["sha256"]):
        raise ValueError("E3 diagnostic inputs differ from the declared one-view scientific cohort")
    if control is not None and (control["source"] != protocol["source"] or control["schedule"] != options.schedule
            or control["history_mode"] != options.history_mode or options.mode_settings.teacher_forcing != (
                args.history == "capture") or (options.changed_noise_file is None) != (args.history == "capture")
            or (options.changed_noise_file is not None and options.future_noise_start not in (3, 11))):
        raise ValueError("data-only E3 control metadata differs from its actual parsed scientific comparison")
    specification, variants, cases, membership = evaluate.prepare_evaluation(options)
    if len(cases) != 1 or variants != [None]:
        raise ValueError("continuation observations require one source and base-only weights")
    video, frames, requested, _adapters = cases[0]
    if (video.fps != 30 or frames != 17 or video.source != protocol["source"]
            or video.hashes != baseline["input_file_hashes"]
            or requested["model"] != baseline["conditions"]["model"]
            or requested["task"] != baseline["conditions"]["task"]):
        raise ValueError("current source, base weights or preprocessing differs from the original baseline")
    patch = evaluate.VideoLatentPatchifier(1)
    capture = patch.patchify(video.z_y[:, :frames].unsqueeze(0).to(torch.bfloat16))
    guide = patch.patchify(video.z_g[:, :frames].unsqueeze(0).to(torch.bfloat16))
    context_path = (old_output / "text.pt" if snapshot is None else
                    Path(snapshot["fixed_inputs"]["input_files"]["text"]["path"]))
    context = torch.load(context_path, map_location="cpu", weights_only=True)
    fixed_values = {"capture": capture, "guide": guide, "noise": options.saved_noise,
                    "first_image": capture[:, :video.z_y.shape[2] * video.z_y.shape[3]], "text": context}
    keys = {"capture": "capture_sha256", "guide": "guide_sha256", "noise": "noise_sha256",
            "first_image": "c0_sha256", "text": "text_sha256"}
    if any(evaluate.tensor_sha256(value) != baseline[keys[name]] for name, value in fixed_values.items()):
        raise ValueError("fixed E3 capture, guide, c0, noise or saved text changed")
    if evaluate.tensor_sha256(options.saved_noise) != protocol["noise"]["tensor_sha256"]:
        raise ValueError("saved E3 noise tensor differs from the frozen protocol")
    paths = {"capture": dataset.capture_bundle_name("white"), "guide": dataset.guide_bundle_name("white")}
    directory = Path(membership["corpus_root"]) / video.source
    fixed_paths = {name: directory / filename for name, filename in paths.items()}
    fixed_paths.update(first_image=fixed_paths["capture"], noise=options.noise_file, text=context_path)
    fixed = {"input_files": {name: {"path": str(path.resolve()), "sha256": sha256(path),
                                    "tensor_sha256": evaluate.tensor_sha256(fixed_values[name])}
                              for name, path in fixed_paths.items()}}
    changed_noise = None if options.changed_noise_file is None else {
        "path": str(options.changed_noise_file.resolve()), "sha256": sha256(options.changed_noise_file),
        "tensor_sha256": evaluate.tensor_sha256(options.changed_noise), "boundary": options.future_noise_start}
    # This reuses the shared fixed-input consumer; it does not fabricate a preview job.
    options.preview_fixed = fixed
    profile = software.capture("evaluation", "causal", extra_sources=EXTRA_SOURCES)
    if profile["sources"]["scripts/onestep_avatar/continuation_check.py"] != IMPORTED_OWNER_SHA256:
        raise ValueError("diagnostic owner bytes changed since this process imported them")
    if snapshot is not None and (profile != snapshot["software"] or fixed != snapshot["fixed_inputs"]):
        raise ValueError("reference source/runtime or fixed inputs differ from capture")
    geometry = causal.CausalGeometry(specification.scale_factors, 2, 8)
    tokens = video.z_y.shape[2] * video.z_y.shape[3]
    inner = specification.caps.num_heads * specification.caps.head_dim
    layers = specification.caps.num_layers
    element = torch.tensor([], dtype=torch.bfloat16).element_size()
    snapshot_bytes = 2 * layers * (3 + 9) * tokens * inner * element
    estimate = {"snapshot_tensor_bytes": snapshot_bytes,
                "cache_tensor_bytes": 2 * layers * geometry.cache_latent_frames_for(17) * tokens * inner * element,
                "largest_cpu_cached_layer_bytes": 2 * 9 * tokens * inner * element,
                "reference_dense_mask_bytes": (11 * tokens) ** 2 * 4,
                "comparison_chunk_elements": CHUNK_ELEMENTS}
    if args.phase == "capture" and shutil.disk_usage(args.output.parent).free < snapshot_bytes + (256 << 20):
        raise ValueError("insufficient disk space for the exact full-layer E3 snapshots")
    return {"options": options, "software": profile, "fixed_inputs": fixed, "snapshot": snapshot,
            "control": control_identity, "changed_noise": changed_noise, "arguments": arguments,
            "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256(args.protocol)},
            "original_result": {"path": str(original_result.resolve()), "sha256": sha256(original_result)},
            "source_snapshot": None if args.snapshot is None else {
                "path": str(args.snapshot.resolve()), "sha256": sha256(args.snapshot)},
            "budget": {"wall_seconds_per_case": 1800, "observed_gpu_memory_limit_mib": 48800,
                       "allocated_byte_threshold": None}, "estimate": estimate}


def _device_memory(gpu_id: int) -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = str(gpu_id) if visible is None else visible.split(",")[gpu_id].strip()
    output = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True,
                            text=True, timeout=5).stdout
    for line in output.splitlines():
        index, uuid, memory = [value.strip() for value in line.split(",")]
        if physical in (index, uuid):
            return {"physical_gpu": index, "uuid": uuid, "memory_used_mib": int(memory)}
    raise ValueError("logical E3 device does not map to a direct nvidia-smi inventory row")


def _check_prepared(prepared: dict) -> None:
    software.check_current(prepared["software"])
    identities = list(prepared["fixed_inputs"]["input_files"].values())
    identities.extend(prepared[name] for name in ("protocol", "original_result"))
    if prepared["source_snapshot"] is not None:
        identities.append(prepared["source_snapshot"])
    identities.extend(prepared[name] for name in ("control", "changed_noise") if prepared[name] is not None)
    if prepared.get("launch_record") is not None:
        identities.append(prepared["launch_record"])
    if any(sha256(Path(identity["path"])) != identity["sha256"] for identity in identities):
        raise ValueError("fixed E3 input, source evidence or snapshot changed after preflight")
    if prepared["source_snapshot"] is not None:
        check_snapshot_files(Path(prepared["source_snapshot"]["path"]).parent, prepared["snapshot"]["snapshots"])


def run(args: argparse.Namespace, prepared: dict) -> dict:  # noqa: PLR0915 -- bounded attempt with failed evidence retained
    args.output.mkdir(parents=True, exist_ok=False)
    observations, block_records, snapshots = [], [], []
    samples = []
    phase = Phase(torch.device(f"cuda:{args.gpu_id}"), "continuation-case", 0, None)
    started = False
    resource = None
    error = None
    controls = []
    model_calls = None

    @torch.no_grad()
    def runner(transformer: torch.nn.Module, context: torch.Tensor, grid: common.ClipGrid,
               capture: torch.Tensor, guide: torch.Tensor | None, noise: torch.Tensor,
               **settings: object) -> tuple[torch.Tensor, dict]:
        nonlocal observations, block_records, snapshots, model_calls
        samples.append(_device_memory(args.gpu_id))
        if guide is None:
            raise ValueError("E3 requires the actual D1 guide")
        geometry = causal.CausalGeometry(grid.tools.scale_factors, 2, 8)
        predict = settings.get("predict_x0") or common.denoised_from_x0_model(transformer)
        with evaluate.measure_calls(transformer) as actual:
            if args.phase == "capture":
                recorder = CacheRecorder(predict, grid, geometry, args.output, sigma=args.sigma,
                                         capture=capture, guide=guide, noise=noise, context=context,
                                         teacher_forcing=args.history == "capture")
                snapshots = recorder.snapshots
                settings["predict_x0"] = recorder
                output, record = evaluate.sample_case(transformer, context, grid, capture, guide, noise, **settings)
                recorder.finish(len(common.base_model(transformer).transformer_blocks))
                snapshots = recorder.snapshots
                ordinary, diagnostic = recorder.calls, 0
            else:
                output, record = evaluate.sample_case(transformer, context, grid, capture, guide, noise, **settings)
                ordinary = record["call_counts"]["model_calls"]
                observations, block_records = reference_blocks(
                    predict, transformer, grid, geometry, args.snapshot.parent,
                    prepared["snapshot"]["snapshots"], context=context, sigma=args.sigma,
                    output_root=args.output)
                diagnostic = 2
        if actual["model_calls"] != ordinary + diagnostic:
            raise ValueError("actual native ordinary/diagnostic forward count differs")
        model_calls = {"actual_model_calls": actual["model_calls"],
                       "ordinary_model_calls": ordinary, "diagnostic_model_calls": diagnostic}
        record["continuation_observation"] = {"phase": args.phase, "history": args.history,
                                              "diagnostic_owner_sha256": sha256(Path(__file__)),
                                              "call_counts": model_calls}
        samples.append(_device_memory(args.gpu_id))
        return output, record

    try:
        _check_prepared(prepared)
        samples.append(_device_memory(args.gpu_id))
        phase.start()
        started = True
        if args.phase == "control":
            evaluate.execute_evaluation(prepared["options"])
            controls = control_results(prepared)
            ordinary = sum(row["call_counts"]["model_calls"] for row in controls)
            model_calls = {"actual_model_calls": ordinary, "ordinary_model_calls": ordinary,
                           "diagnostic_model_calls": 0}
        else:
            evaluate.execute_evaluation(prepared["options"], sample_runner=runner)
        samples.append(_device_memory(args.gpu_id))
        resource = phase.finish()
        started = False
        if resource["state"] != "passed" or resource["elapsed_s"] > prepared["budget"]["wall_seconds_per_case"]:
            raise ValueError("E3 case failed resource measurement or its original wall-time budget")
        if any(sample["memory_used_mib"] > prepared["budget"]["observed_gpu_memory_limit_mib"] for sample in samples):
            raise ValueError("E3 case exceeded its original sampled total-device memory budget")
        _check_prepared(prepared)
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
        if started:
            resource = phase.finish(error)
        progress = args.output / "reference_progress.json"
        if progress.is_file():
            partial = json.loads(progress.read_text())
            observations, block_records = partial["layerwise"], partial.get("reference_blocks", [])
    result = {"schema_version": 1, "kind": "onestep_avatar.continuation_check",
              "phase": args.phase, "sigma": args.sigma, "history": args.history,
              "state": "failed" if error else "complete", "error": error,
              "protocol": prepared["protocol"], "original_result": prepared["original_result"],
              "software": prepared["software"], "fixed_inputs": prepared["fixed_inputs"],
              "control": prepared["control"], "changed_noise": prepared["changed_noise"],
              "budget": prepared["budget"], "allocation_estimate": prepared["estimate"],
              "resources": resource, "sampled_total_device_memory": samples,
              "snapshots": snapshots, "layerwise": observations, "reference_blocks": block_records,
              "control_results": controls,
              "call_counts": model_calls,
              "source_snapshot": prepared["source_snapshot"],
              "launch_binding": {"attempt_token": os.environ.get("ONESTEP_AVATAR_QUEUE_TOKEN"),
                                 "job_sha256": os.environ.get("ONESTEP_AVATAR_QUEUE_JOB_SHA256")},
              "supervisor_acceptance_required": True,
              "scope": "native calculation observations; no continuation-quality or E3 closure claim"}
    result["output_files"] = {path.relative_to(args.output).as_posix(): sha256(path)
                              for path in args.output.rglob("*") if path.is_file()}
    write_json(args.output / "continuation.json", result)
    if error:
        raise RuntimeError(error)
    return result


def control_results(prepared: dict) -> list[dict]:
    """Validate the ordinary owner's saved values; false invariance remains failed."""
    destination = prepared["options"].output / "case_0000/variant_000"
    future = prepared["changed_noise"] is not None
    paths = ([destination / name / "result.json" for name in ("original", "changed")] if future else
             [destination / "result.json"])
    evaluate.verify_evaluation_conditions(prepared["arguments"], paths)
    rows = [{"file": {"path": str(path.resolve()), "sha256": sha256(path)},
             "call_counts": json.loads(path.read_text())["call_counts"], "diagnostic_extra_forwards": 0}
            for path in paths]
    if future:
        path = destination / "future_noise.json"
        diagnostic = json.loads(path.read_text())
        if diagnostic["earlier_output_bit_identical"] is not True:
            raise ValueError("E3 earlier-output future-noise invariance failed")
        for row in rows:
            row["future_noise"] = {"path": str(path.resolve()), "sha256": sha256(path),
                                    "earlier_output_bit_identical": True,
                                    "boundary": prepared["changed_noise"]["boundary"]}
    return rows


def scientific_binding(prepared: dict) -> dict:
    return {name: prepared[name] for name in ("software", "fixed_inputs", "protocol", "original_result",
                                            "source_snapshot", "budget", "estimate", "control", "changed_noise")}


def check_launch_record(path: Path, prepared: dict, args: argparse.Namespace) -> None:
    record = json.loads(path.read_text())
    from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, TOKEN_ENV  # noqa: PLC0415 -- shared names

    if (sha256(path) != os.environ.get(JOB_ENV) or not os.environ.get(TOKEN_ENV)
            or record.get("binding") != scientific_binding(prepared)
            or record.get("phase") != args.phase or record.get("sigma") != args.sigma
            or record.get("history") != args.history or record.get("output") != str(args.output)):
        raise ValueError("supervised child differs from its exact frozen E3 launch")
    prepared["launch_record"] = {"path": str(path.resolve()), "sha256": sha256(path)}


def supervised_run(args: argparse.Namespace, prepared: dict) -> dict:
    """Use the existing own registry and supervisor, with the exact original bounds."""
    from scripts.onestep_avatar.execution import (  # noqa: PLC0415 -- existing model-free process owners
        queue,
        supervision,
    )
    from scripts.onestep_avatar.execution.process_registry import ProcessRegistry, gpu_memory  # noqa: PLC0415
    from scripts.onestep_avatar.execution.queue_protocol import JOB_ENV, TOKEN_ENV  # noqa: PLC0415

    registry = ProcessRegistry(args.process_ledger)
    gpus = registry.choose(gpu_memory(), training=False)
    _check_prepared(prepared)
    if gpus is None or not registry.acquire(gpus, job=f"E3:{args.phase}:{args.control_id or args.sigma}"):
        raise ValueError("no idle allowed GPU is available for the E3 diagnostic")
    evidence = args.output.parent / f"{args.output.name}.supervision"
    child = None
    try:
        evidence.mkdir(exist_ok=False)
        launch = evidence / "launch.json"
        write_json(launch, {"schema_version": 1, "binding": scientific_binding(prepared), "phase": args.phase,
                            "sigma": args.sigma, "history": args.history, "output": str(args.output)})
        command = [sys.executable, "-m", "scripts.onestep_avatar.continuation_check", "--phase", args.phase,
                   "--output", str(args.output), "--gpu-id", "0", "--launch-record", str(launch)]
        if args.phase == "reference":
            command.extend(["--snapshot", str(args.snapshot.resolve())])
        else:
            command.extend(["--protocol", str(args.protocol.resolve())])
            if args.phase == "control":
                command.extend(["--control-plan", str(args.control_plan.resolve()), "--control-id", args.control_id])
            else:
                command.extend(["--sigma", str(args.sigma), "--history", args.history])
        changes = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), TOKEN_ENV: registry.token,
                   JOB_ENV: sha256(launch), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        environment = dict(os.environ)
        for name in (supervision.SUPERVISION_ENV, supervision.SUPERVISION_SHA_ENV, "RANK", "LOCAL_RANK", "WORLD_SIZE"):
            environment.pop(name, None)
        _check_prepared(prepared)
        with (evidence / "child.log").open("x") as log:
            child = subprocess.Popen(command, cwd=LTX_ROOT,
                                     env={**environment, **changes}, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            registry.refresh(child_pid=child.pid)
            identity = queue.process_identity(child.pid)
            row = {"child_pid": child.pid, "child_identity": identity, "child_session": child.pid,
                   "environment_changes": changes, "attempt_started_ticks": registry.started_ticks,
                   "process_ledger": str(registry.path)}
            result = supervision.supervise(
                child, identity=identity, command=command, worker_record=row, claims=registry, gpus=gpus,
                evidence_path=evidence / "result.json", notifications_path=None, startup_seconds=1800,
                phase_seconds=1800, shutdown_seconds=30, overall_seconds=1800,
                sampled_total_device_limit_bytes=48800 * (1 << 20))
        if result["state"] != "passed":
            raise ValueError(f"E3 supervision failed: {result['error']}")
        record = json.loads((args.output / "continuation.json").read_text())
        if (record["state"] != "complete" or record["software"] != prepared["software"]
                or record["fixed_inputs"] != prepared["fixed_inputs"]
                or record["launch_binding"] != {"attempt_token": registry.token, "job_sha256": sha256(launch)}):
            raise ValueError("E3 scientific diagnostic is incomplete")
        return record
    finally:
        if child is None or child.poll() is not None:
            observed = registry.observe_workers()
            if observed["complete"] and not observed["workers_live"]:
                registry.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("capture", "reference", "control"), required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--control-plan", type=Path)
    parser.add_argument("--control-id")
    parser.add_argument("--sigma", type=float, choices=(1.0, 0.909375))
    parser.add_argument("--history", choices=("generated", "capture"), default="generated")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--process-ledger", type=Path)
    parser.add_argument("--launch-record", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if ((args.phase == "capture" and (args.protocol is None or args.sigma is None or args.snapshot is not None))
            or (args.phase == "reference" and
                (args.snapshot is None or args.protocol is not None or args.sigma is not None))):
        parser.error("capture requires protocol/sigma; reference requires only its capture snapshot")
    if ((args.phase == "control" and (args.control_plan is None or args.control_id is None
                                     or args.protocol is None or args.snapshot is not None or args.sigma is not None))
            or (args.phase != "control" and (args.control_plan is not None or args.control_id is not None))):
        parser.error("control requires protocol/control-plan/control-id and no snapshot/sigma")
    if args.gpu_id < 0 or args.output.exists() or not args.output.parent.is_dir():
        parser.error("use a fresh output under an existing parent and a nonnegative local GPU")
    if ((args.supervise and (args.process_ledger is None or args.dry_run or args.launch_record is not None))
            or (args.process_ledger is not None and not args.supervise)):
        parser.error("supervision requires process-ledger and execution, without an inherited launch record")
    prepared = prepare(args)
    if args.launch_record is not None:
        check_launch_record(args.launch_record, prepared, args)
    _check_prepared(prepared)
    if args.dry_run:
        print(json.dumps({key: prepared[key] for key in ("protocol", "budget", "estimate")}, indent=2))  # noqa: T201
        return 0
    if args.supervise:
        supervised_run(args, prepared)
    else:
        run(args, prepared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
