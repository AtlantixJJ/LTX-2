"""Compare matched one-step, whole-clip D0 rollouts from two transformer checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
from contextlib import ExitStack
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from ltx_core.model.transformer.modality import Modality
from scripts.onestep_avatar import causal_core
from scripts.onestep_avatar.train import _load_training_master
from scripts.prune.core import provenance, session
from scripts.prune.data import whole_clip
from scripts.prune.score import export_pruned, hooks


def _records(manifest: dict) -> dict[tuple[str, float], dict]:
    return whole_clip.records(manifest)


def _latent_path(root: Path, row: dict) -> Path:
    return whole_clip.latent_path(root, row)


def _verify_pair(base: dict, candidate: dict) -> None:
    whole_clip.verify_pair(base, candidate)


def _direction_metrics(base_latent: torch.Tensor, candidate_latent: torch.Tensor, capture: torch.Tensor,
                       epsilon: torch.Tensor, sigma: float) -> dict[str, float]:
    """Noise-facing velocity ``(x_sigma - x0_hat)/sigma`` on generated frames only.

    The first latent frame is the clean conditioning frame, not a model prediction. Epsilon
    is saved in token order with patch size 1; ``unpatchify_block`` is a transpose/reshape.
    """
    if base_latent.shape != candidate_latent.shape or base_latent.shape != capture.shape:
        raise ValueError("latent or capture shapes differ")
    b, c, t, h, w = capture.shape
    if epsilon.shape != (b, t * h * w, c):
        raise ValueError(f"epsilon shape {tuple(epsilon.shape)} does not fit {tuple(capture.shape)}")
    noise = epsilon.transpose(1, 2).reshape(b, c, t, h, w)
    x_sigma = torch.lerp(capture.float(), noise.float(), sigma).to(capture.dtype)
    ref = ((x_sigma[:, :, 1:].float() - base_latent[:, :, 1:].float()) / sigma).flatten().double()
    pruned = ((x_sigma[:, :, 1:].float() - candidate_latent[:, :, 1:].float()) / sigma).flatten().double()
    delta = pruned - ref
    ref_norm = torch.linalg.vector_norm(ref).item()
    pruned_norm = torch.linalg.vector_norm(pruned).item()
    if ref_norm == 0 or pruned_norm == 0:
        raise ValueError("zero denoising direction")
    base_error = (base_latent[:, :, 1:].float() - capture[:, :, 1:].float()).square()
    candidate_error = (candidate_latent[:, :, 1:].float() - capture[:, :, 1:].float()).square()
    between_error = (candidate_latent[:, :, 1:].float() - base_latent[:, :, 1:].float()).square()
    return {
        "direction_relative_l2": torch.linalg.vector_norm(delta).item() / ref_norm,
        "direction_cosine": torch.dot(ref, pruned).item() / (ref_norm * pruned_norm),
        "baseline_capture_mse": base_error.mean().item(),
        "candidate_capture_mse": candidate_error.mean().item(),
        "candidate_vs_baseline_latent_mse": between_error.mean().item(),
    }


def _comparison_video(base_video: Path, candidate_video: Path, output: Path) -> None:
    """Reuse the already decoded panels; crop GT and D0, then label and align them."""
    labels = ("GT capture (VAE)", "Baseline D0 (VAE)", "Pruned D0 (VAE)")
    # Both source videos have GT|D0|D1 panels at equal sizes and matching frame rates.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "json", str(base_video)], capture_output=True, text=True, check=True,
    )
    width = json.loads(probe.stdout)["streams"][0]["width"]
    panel_width = width // 3
    font_size = max(24, panel_width // 32)
    font = ImageFont.load_default(size=font_size)
    title = Image.new("RGB", (width, font_size + 12), "black")
    draw = ImageDraw.Draw(title)
    for index, label in enumerate(labels):
        draw.text((index * panel_width + 8, 6), label, fill="white", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    title_path = output.with_suffix(".title.png")
    title.save(title_path)
    filters = (
        "[0:v]crop=iw/3:ih:0:0[p0];"
        "[0:v]crop=iw/3:ih:iw/3:0[p1];"
        "[1:v]crop=iw/3:ih:iw/3:0[p2];"
        "[p0][p1][p2]hstack=inputs=3[stack];"
        "[stack][2:v]overlay=0:0:shortest=1[v]"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(base_video), "-i", str(candidate_video),
         "-loop", "1", "-i", str(title_path), "-filter_complex", filters, "-map", "[v]", "-an",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-shortest", str(output)],
        check=True,
    )


def compare(baseline: Path, candidate: Path, output: Path, *, historical_transfer: bool = False) -> dict:
    base = whole_clip.load_manifest(baseline)
    pruned = whole_clip.load_manifest(candidate)
    whole_clip.verify_candidate(base, pruned, historical_transfer=historical_transfer)
    base_rows, candidate_rows = _records(base), _records(pruned)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, b in sorted(base_rows.items()):
        p = candidate_rows[key]
        for field in ("capture_sha256", "fps", "blocks"):
            if b["artifacts"][field] != p["artifacts"][field]:
                raise ValueError(f"unmatched {field} for {key}")
        if b["schedule"] != [key[1], 0.0] or p["schedule"] != b["schedule"]:
            raise ValueError(f"not the same one-step schedule for {key}")
        capture, _ = _load_training_master(Path(b["artifacts"]["capture"]))
        if provenance.file_sha256(b["artifacts"]["capture"]) != b["artifacts"]["capture_sha256"]:
            raise ValueError(f"capture changed since baseline: {key}")
        whole_clip.verify_saved_noise(baseline, candidate, b, p)
        epsilon = whole_clip.load_epsilon(baseline, b)
        base_latent = torch.load(_latent_path(baseline, b), map_location="cpu", weights_only=True)
        candidate_latent = torch.load(_latent_path(candidate, p), map_location="cpu", weights_only=True)
        metrics = _direction_metrics(
            base_latent, candidate_latent, capture.unsqueeze(0).to(base_latent.dtype), epsilon, key[1]
        )
        video = None
        if b["output"] and p["output"]:
            video = output / f"{Path(key[0]).parent.parent.name}_sigma_{key[1]:.6f}_gt_base_pruned.mp4"
            _comparison_video(Path(b["output"]), Path(p["output"]), video)
        rows.append({"view": key[0], "sigma": key[1], **metrics,
                     "baseline_latent": str(_latent_path(baseline, b)),
                     "candidate_latent": str(_latent_path(candidate, p)),
                     "comparison_video": str(video) if video else None})
    result = {"baseline": str(baseline), "candidate": str(candidate),
              "task": "historical_k2_transfer" if historical_transfer else whole_clip.TASK,
              "baseline_fingerprint": base["model"]["transformer_fingerprint"],
              "candidate_fingerprint": pruned["model"]["transformer_fingerprint"],
              "noise": "matched saved epsilon; same capture, prompt, seed, geometry, and schedule",
              "direction": "(x_sigma - predicted_x0) / sigma; generated latent frames 1..T-1",
              "rows": rows}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def functional_ablation(baseline: Path, masks_path: Path, output: Path, *, view: str,  # noqa: PLR0915
                        sigmas: list[float], gpu_id: int) -> dict:
    """Measure head-only, FFN-only and combined native masks on saved D0 inputs."""
    base = whole_clip.load_manifest(baseline)
    if not sigmas or any((view, sigma) not in whole_clip.records(base) for sigma in sigmas):
        raise ValueError("ablation view/sigma pair is absent from the saved baseline")
    checkpoint = Path(base["model"]["transformer_path"])
    fingerprint = base["model"]["transformer_fingerprint"]
    if provenance.checkpoint_fingerprint(checkpoint) != fingerprint:
        raise ValueError("baseline checkpoint changed since saved rollout")
    masks, digest = hooks.read_mask_artifact(
        masks_path, model_key=base["model"]["model_key"], fingerprint=fingerprint,
        widths=export_pruned.checkpoint_mask_widths(checkpoint), expected_task=whole_clip.TASK, baseline=base,
    )
    hooks.require_native_heldout_scope(masks_path, view=view, sigmas=sigmas, baseline=base)
    args = argparse.Namespace(model="2.5", gpu_id=gpu_id, seed=base["seed"])
    current = session.open_session(args, script="prune.evaluate.whole_clip_d0.functional_ablation",
                                   prompt=base["text_context"]["prompt"])
    if current.stamp()["transformer_fingerprint"] != base["model"]["transformer_fingerprint"]:
        raise ValueError("baseline checkpoint changed since saved rollout")
    rows = []
    with current.transformer(checkpoint) as model:
        head_values = {name: torch.tensor(value, device=current.device) for name, value in masks.items()
                       if not name.endswith(".ff")}
        ffn_values = {name: torch.tensor(value, device=current.device) for name, value in masks.items()
                      if name.endswith(".ff")}
        for sigma in sigmas:
            grid, modality, c0, row = whole_clip.build_input(
                baseline, base, view=view, sigma=sigma, current=current,
            )
            capture, _ = _load_training_master(Path(row["artifacts"]["capture"]))
            capture = capture.unsqueeze(0)
            epsilon = whole_clip.load_epsilon(baseline, row)
            recorded = torch.load(whole_clip.latent_path(baseline, row), map_location="cpu", weights_only=True)

            def forward(grid: causal_core.ClipGrid, modality: Modality, c0: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    prediction, _ = model(video=modality, audio=None, perturbations=None)
                    return grid.unpatchify_block(causal_core.with_clean_prefix(prediction, c0),
                                                 grid.latent_frames).cpu()

            direct = forward(grid, modality, c0)
            max_abs = float((recorded.float() - direct.float()).abs().max())
            if max_abs > 0.02:
                raise ValueError(f"direct baseline differs from saved D0 latent: {max_abs}")
            result = {"view": view, "sigma": sigma, "saved_baseline_max_abs": max_abs, "arms": {}}
            with ExitStack() as stack:
                head_attached = stack.enter_context(hooks.attach_head_masks(model, requires_grad=False))
                ffn_attached = stack.enter_context(hooks.attach_ffn_masks(model, requires_grad=False))
                for arm in ("heads_only", "ffn_only", "combined"):
                    with torch.no_grad():
                        for name, value in head_attached.items():
                            if arm == "ffn_only":
                                value.fill_(1)
                            else:
                                value.copy_(head_values[name])
                        for name, value in ffn_attached.items():
                            if arm == "heads_only":
                                value.fill_(1)
                            else:
                                value.copy_(ffn_values[name])
                    measured = forward(grid, modality, c0)
                    result["arms"][arm] = _direction_metrics(recorded, measured, capture, epsilon, sigma)
                    del measured
            rows.append(result)
            del grid, modality, c0, capture, epsilon, recorded, direct
            torch.cuda.empty_cache()
    report = {"task": whole_clip.TASK, "baseline": str(baseline), "mask": str(masks_path),
              "mask_sha256": digest, "baseline_fingerprint": base["model"]["transformer_fingerprint"],
              "method": "functional mask on the baseline; diagnostic, not compact-export quality or speed",
              "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--historical-transfer", action="store_true",
                        help="Explicitly evaluate a historical k2 mask as a cross-task transfer control")
    parser.add_argument("--functional-mask", type=Path,
                        help="Run head-only, FFN-only and combined diagnostics on a native mask")
    parser.add_argument("--view", help="Capture view for --functional-mask")
    parser.add_argument("--sigmas", type=float, nargs="+", help="Sigma levels for --functional-mask")
    parser.add_argument("--gpu-id", type=int, help="GPU for --functional-mask")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.functional_mask:
        if args.candidate or args.historical_transfer or args.view is None or not args.sigmas or args.gpu_id is None:
            parser.error("--functional-mask needs --view, --sigmas and --gpu-id, without --candidate")
        result = functional_ablation(args.baseline, args.functional_mask, args.output,
                                     view=args.view, sigmas=args.sigmas, gpu_id=args.gpu_id)
        print(json.dumps({"output": str(args.output), "rows": len(result["rows"])}, indent=2))
        return
    if args.candidate is None:
        parser.error("--candidate is required for a checkpoint comparison")
    result = compare(args.baseline, args.candidate, args.output, historical_transfer=args.historical_transfer)
    print(json.dumps({"output": str(args.output / "comparison.json"), "rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
