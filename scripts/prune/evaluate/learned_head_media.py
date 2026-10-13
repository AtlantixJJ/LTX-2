"""Decode frozen learned-head evidence and render synchronized RGB/VAE comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.onestep_avatar import media
from scripts.onestep_avatar.metrics import lpips_distance
from scripts.prune.core import provenance
from scripts.prune.data import foreground
from scripts.prune.data.learned_head_study import load_study
from scripts.prune.evaluate.learned_heads import actor_bootstrap, saved_case
from scripts.prune.evaluate.mobilewan_media import preserve_source, saved_video, source_pixels, verify_comparison
from scripts.prune.learn.checkpoints import publish_json


def region_lpips(model, student, reference, weights, device, *, boxes=None, valid=None):
    """Use GT coverage and boxes only. Exclude c0 and invalid small regions."""
    if student.shape != reference.shape or weights.shape != (len(student), *student.shape[-2:]):
        raise ValueError("LPIPS fixed region geometry differs")
    left, right, frame_ids = [], [], []
    for f in range(1, len(student)):
        if valid is not None and not valid[f]:
            continue
        w = weights[f]
        if boxes is None:
            ys, xs = torch.where(w > 0)
            if not len(xs):
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        else:
            x0, y0, x1, y1 = [int(v) for v in boxes[f]]
        if min(x1 - x0, y1 - y0) < 32:
            continue
        for value, dest in ((student, left), (reference, right)):
            pixels = value[f].float() / 255
            masked = pixels * w[None] + (1 - w[None])
            crop = masked[:, y0:y1, x0:x1].unsqueeze(0)
            if crop.numel() == 0:
                raise ValueError("reviewed GT box lies outside pixel grid")
            dest.append(
                torch.nn.functional.interpolate(crop, (256, 256), mode="bilinear", align_corners=False, antialias=True)[
                    0
                ]
            )
        frame_ids.append(f)
    if not left:
        return {"lpips": None, "valid_frames": 0, "frame_ids": [], "status": "missing_coverage"}
    return {
        "lpips": lpips_distance(model, torch.stack(left), torch.stack(right), device, batch=8),
        "valid_frames": len(frame_ids),
        "frame_ids": frame_ids,
        "status": "scored",
    }


def panels(recorded, vae_gt, teacher, candidates):
    rows = [
        ("capture_rgb", "Capture RGB", recorded, "White matte"),
        ("capture_vae", "Capture VAE", vae_gt, "GT decoding"),
        ("teacher", "Teacher", teacher, "VAE decoding"),
    ]
    rows += [(arm, arm.replace("_", " "), pixels, "VAE decoding") for arm, pixels in candidates.items()]
    if len({len(pixels) for _, _, pixels, _ in rows}) != 1:
        raise ValueError("comparison panels need complete matched source frames")
    return [
        media.Panel(role, title, pixels, tuple(range(len(pixels))), value=label) for role, title, pixels, label in rows
    ]


def pixel_metrics(student, reference, weights):
    if student.shape != reference.shape or student.ndim != 4 or weights.shape != (len(student), *student.shape[-2:]):
        raise ValueError("region pixels and fixed GT coverage differ")
    if not torch.isfinite(student).all() or not torch.isfinite(reference).all() or not torch.isfinite(weights).all():
        raise ValueError("nonfinite pixel input")
    s, r, w = student[1:].float() / 255, reference[1:].float() / 255, weights[1:].float()
    mass = w.sum()
    if mass <= 0:
        raise ValueError("required decoded region coverage is empty")
    mse = float(((s - r).square() * w[:, None]).sum() / (3 * mass))
    union = torch.maximum(w[1:], w[:-1])
    motion = float(((s[1:] - s[:-1]).abs() * union[:, None]).sum() / (3 * union.sum())) if len(s) > 1 else None
    return {
        "mse": mse,
        "psnr": float(-10 * np.log10(mse)) if mse else None,
        "motion_amount": motion,
        "lpips": None,
        "lpips_status": "requires_pinned_metric_and_valid_GT_face_hand_tracks",
        "valid_frames": int((w.sum((1, 2)) > 0).sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--replicate-seed", type=int, default=42)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--gpu-id", type=int, required=True)
    args = p.parse_args()
    study = load_study(args.study)
    root = Path(study["root"])
    summaries = {
        arm: json.loads(
            (root / f"evaluation/{args.split}/{arm}/replicate_{args.replicate_seed}/metrics.json").read_text()
        )
        for arm in args.arms
    }
    if any(s["status"] != "complete_numeric_only" for s in summaries.values()):
        raise ValueError("media requires completed paired numeric evaluation")
    # Validate every saved tensor before loading a decoder. No missing-run repair.
    matched = {}
    for arm, summary in summaries.items():
        for row in summary["rows"]:
            path = Path(row["record"])
            raw = json.loads(path.read_text())
            saved_case(path, raw["identity"])
            key = (raw["identity"]["view"], raw["identity"]["noise_seed"], raw["identity"]["sigma"])
            matched.setdefault(key, {})[arm] = raw
    if any(
        set(group) != set(args.arms) or len({r["teacher_key"] for r in group.values()}) != 1
        for group in matched.values()
    ):
        raise ValueError("comparison arms do not share every teacher/input/noise case")
    from scripts.prune.learn.heads import input_case, numerical_policy, open_current

    numerical_policy()
    current = open_current(study, args.gpu_id)
    perceptual = None
    if study["perceptual_metric"]["status"] == "pinned":
        import lpips

        perceptual = lpips.LPIPS(net="alex", verbose=False).to(current.device).eval()
    records = []
    with current.decoder() as decoder:
        for key, group in matched.items():
            view, noise_seed, sigma = key
            source = next(s for s in study["splits"][args.split] if s["relative_dir"] == view)
            first = next(iter(group.values()))
            teacher_root = root / "teacher/references" / first["teacher_key"]
            teacher_record = json.loads((teacher_root / "manifest.json").read_text())
            if provenance.file_sha256(teacher_root / "clean.pt") != teacher_record["clean"]["file_sha256"]:
                raise ValueError("teacher media reference changed")
            from scripts.onestep_avatar.corpus.dataset import load_training_master

            capture, _ = load_training_master(Path(source["capture"]))
            # Input geometry remains owned by the native reconstruction path.
            grid, _, _, _, _, _, _ = input_case(study, current, source, noise_seed, sigma)
            teacher_tokens = torch.load(teacher_root / "clean.pt", map_location="cpu", weights_only=True)
            teacher_latent = grid.unpatchify_block(teacher_tokens.to(current.device), grid.latent_frames)
            decoded = {}
            for arm, record in group.items():
                latent = torch.load(record["tensors"]["latent"]["path"], map_location="cpu", weights_only=True)
                decoded[arm] = media.decode(current, latent, decoder, 42).mul(255).round().to(torch.uint8)
            recorded = source_pixels(source)
            vae_gt = media.decode(current, capture.unsqueeze(0), decoder, 42).mul(255).round().to(torch.uint8)
            teacher = media.decode(current, teacher_latent, decoder, 42).mul(255).round().to(torch.uint8)
            matte, _, _ = foreground.person_coverage(
                Path(source["resolved_dir"]), source["capture_encode_record"], replay=False
            )
            weights = torch.from_numpy(matte).float().unsqueeze(1) / 255
            weights = torch.nn.functional.interpolate(weights, size=vae_gt.shape[-2:], mode="area")[:, 0]
            errors = {
                arm: {
                    "to_vae_gt": pixel_metrics(pixels, vae_gt, weights),
                    "to_teacher": pixel_metrics(pixels, teacher, weights),
                    "to_rgb_gt": pixel_metrics(pixels, recorded, weights),
                }
                for arm, pixels in decoded.items()
            }
            region_weights = {"person": (weights, None, None)}
            if study["region_tracks"]["status"] == "verified":
                tracks = study["region_tracks"]["views"][view]
                for name in ("face", "left_hand", "right_hand"):
                    _, coordinate = foreground.region_coverage(
                        tracks[name], matte, source["capture_encode_record"], schema=tracks["schema"]
                    )
                    mask = torch.zeros_like(weights)
                    for f, box in enumerate(coordinate["crop_boxes"]):
                        if coordinate["valid"][f]:
                            x0, y0, x1, y1 = np.floor(box).astype(int)
                            mask[f, y0:y1, x0:x1] = weights[f, y0:y1, x0:x1]
                    region_weights[name] = (mask, coordinate["crop_boxes"], coordinate["valid"])
            if perceptual:
                for region, (w, boxes, valid) in region_weights.items():
                    teacher_error = region_lpips(
                        perceptual, teacher, vae_gt, w, current.device, boxes=boxes, valid=valid
                    )
                    ceiling = region_lpips(perceptual, vae_gt, recorded, w, current.device, boxes=boxes, valid=valid)
                    for arm, pixels in decoded.items():
                        scored = region_lpips(perceptual, pixels, vae_gt, w, current.device, boxes=boxes, valid=valid)
                        increase = scored["lpips"] - teacher_error["lpips"] if scored["lpips"] is not None else None
                        errors[arm][region] = {
                            **scored,
                            "teacher_to_vae_gt": teacher_error,
                            "vae_to_rgb_gt": ceiling,
                            "paired_increase": increase,
                            "case_flag": increase is not None and increase > 0.03,
                        }
                for arm in errors:
                    hands = [
                        errors[arm][k]["paired_increase"]
                        for k in ("left_hand", "right_hand")
                        if k in errors[arm] and errors[arm][k]["paired_increase"] is not None
                    ]
                    errors[arm]["hands"] = {
                        "paired_increase": float(np.mean(hands)) if hands else None,
                        "visible_hands": len(hands),
                    }
            combined, rendering = media.render_panels(
                panels(recorded, vae_gt, teacher, decoded),
                question="Do learned head choices preserve the person?",
                layout="comparison",
                fps=source["fps"],
                panel_size=(384, 384),
                viewing_width=480,
                poster_frame=min(30, len(recorded) - 1),
                common_settings={"sigma": sigma, "seed": noise_seed, "source": view, "schedule": [sigma, 0.0]},
            )
            destination = root / "media" / f"replicate_{args.replicate_seed}" / first["teacher_key"]
            sources_path = root / "media/sources" / Path(source["resolved_dir"]).parent.parent.name / Path(view).name
            preserve_source(Path(source["resolved_dir"]) / "rgb.mp4", sources_path / "rgb.mp4", source["rgb_sha256"])
            preserve_source(Path(source["resolved_dir"]) / "mask.mp4", sources_path / "mask.mp4")
            for label, pixels in {
                "capture_rgb": recorded,
                "capture_vae": vae_gt,
                "teacher_vae": teacher,
                **decoded,
            }.items():
                saved_video(pixels, destination / f"{label}.mp4", source["fps"])
            rendering = verify_comparison(media.save_render(combined, rendering, destination))
            publish_json(destination / "rendering.json", rendering)
            publish_json(destination / "pixel_metrics.json", errors)
            records.append(
                {
                    "case": list(key),
                    "actor": source["actor"],
                    "rendering": str(destination / "rendering.json"),
                    "errors": errors,
                }
            )
    uncertainty = {}
    for arm in args.arms:
        uncertainty[arm] = {}
        for region in ("person", "face", "hands"):
            rows = [
                {"actor": r["actor"], "increase": r["errors"][arm].get(region, {}).get("paired_increase")}
                for r in records
            ]
            valid_rows = [r for r in rows if r["increase"] is not None]
            uncertainty[arm][region] = (
                actor_bootstrap(valid_rows, "increase") if len({r["actor"] for r in valid_rows}) >= 2 else None
            )
    publish_json(
        root / "media" / f"{args.split}_replicate_{args.replicate_seed}.json",
        {
            "records": records,
            "quality_pass": None,
            "uncertainty": uncertainty,
            "missing": "complete region coverage and reviewed videos are required before quality acceptance",
        },
    )


if __name__ == "__main__":
    main()
