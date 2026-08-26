#!/usr/bin/env python3
"""Region-split (foreground / halo / background) rescore of the SAM3DGS VAE+refine sweep.

Why this exists: the original sweep reported whole-frame PSNR, but the GS renders are matted
subjects on a flat white plate -- the subject covers ~11% of the pixels, so whole-frame PSNR is
dominated by how well the flat background is reproduced and says almost nothing about the human.

This script re-derives every number split by region:

  fg    subject pixels                         min(RGB) < --fg-threshold
  halo  band of --halo-px around the subject   dilate(fg) & ~fg   (where GS floaters live)
  bg    everything else                        the flat matte

The VAE round-trip (k0) is *recomputed on GPU* so its numbers are exact -- the saved
``vae_roundtrip.mp4`` is h264 at ~1.5 Mbps, whose own foreground noise floor (~35-40 dB) is close
enough to the k0 foreground PSNR (~30 dB) to bias it. The refined variants (k1..k8) sit at
13-31 dB, far below that floor, so they are rescored from their saved decodes; the floor is
measured per video and reported so the residual error is visible.

Per-stage wall times are recorded for every video and aggregated into ``profiling`` in the output.

Run (from LTX-2/):
    conda run -n ltx python3 scripts/rescore_fg_bg.py --gpus 6,7
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch.cuda.init()  # must precede `import decord` or the process segfaults

import decord  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402

from ltx_core.model.video_vae import TilingConfig  # noqa: E402
from ltx_pipelines.utils.blocks import ImageConditioner, VideoDecoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rescore")

DTYPE = torch.bfloat16
SWEEP_ROOT = WORKSPACE_ROOT / "expr" / "sam3dgs_vae_refine"
DEFAULT_VAE = WORKSPACE_ROOT / "checkpoints" / "LTX-2.3" / "ltx-2.3-22b-distilled-1.1.safetensors"


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


class Profile:
    """Accumulates wall time per named stage."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.totals[name] += dt
            self.counts[name] += 1

    def merge(self, other: dict[str, Any]) -> None:
        for k, v in other.get("totals", {}).items():
            self.totals[k] += v
        for k, v in other.get("counts", {}).items():
            self.counts[k] += v

    def as_dict(self) -> dict[str, Any]:
        return {"totals": dict(self.totals), "counts": dict(self.counts)}

    def table(self) -> list[dict[str, Any]]:
        rows = []
        grand = sum(self.totals.values())
        for name, tot in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            n = self.counts[name]
            rows.append(
                {
                    "stage": name,
                    "calls": n,
                    "total_s": round(tot, 2),
                    "mean_s": round(tot / max(n, 1), 3),
                    "pct_of_measured": round(100.0 * tot / max(grand, 1e-9), 1),
                }
            )
        return rows


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def crop_to_32(frames: np.ndarray) -> np.ndarray:
    """Center-crop H/W to multiples of 32 -- identical to the sweep's load_video_tensor."""
    _, h, w, _ = frames.shape
    if h % 32:
        ch = (h // 32) * 32
        top = (h - ch) // 2
        frames = frames[:, top : top + ch, :, :]
    if w % 32:
        cw = (w // 32) * 32
        left = (w - cw) // 2
        frames = frames[:, :, left : left + cw, :]
    return frames


def read_video(path: str, max_frames: int) -> np.ndarray:
    """Read the first `max_frames` frames as float32 [F,H,W,C] in [0,1]."""
    vr = decord.VideoReader(path)
    n = min(max_frames, len(vr))
    return vr.get_batch(range(n)).asnumpy().astype(np.float32) / 255.0


def read_source(entry: dict[str, Any]) -> np.ndarray:
    """Read the original render and apply the sweep's frame budget + 32-crop."""
    vr = decord.VideoReader(entry["file_path"])
    total = len(vr)
    max_read = min(total, entry["target_frames"])
    valid_f = ((max_read - 1) // 8) * 8 + 1
    frames = vr.get_batch(range(valid_f)).asnumpy().astype(np.float32) / 255.0
    return crop_to_32(frames)


# ---------------------------------------------------------------------------
# Masks and metrics
# ---------------------------------------------------------------------------


def build_masks(source: np.ndarray, fg_threshold: float, halo_px: int) -> dict[str, np.ndarray]:
    """Per-frame subject / halo / background masks, broadcast to 3 channels."""
    fg = source.min(axis=-1) < fg_threshold  # [F,H,W]
    if halo_px > 0:
        struct = np.ones((1, 3, 3), dtype=bool)
        grown = binary_dilation(fg, structure=struct, iterations=halo_px)
    else:
        grown = fg
    halo = grown & ~fg
    bg = ~grown
    to3 = lambda m: np.repeat(m[..., None], 3, axis=-1)  # noqa: E731
    return {"fg": to3(fg), "halo": to3(halo), "bg": to3(bg), "_fg_frac": fg.mean(), "_halo_frac": halo.mean()}


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    sq = (a - b) ** 2
    mse = float(sq.mean()) if mask is None else float(sq[mask].mean())
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def region_metrics(candidate: np.ndarray, reference: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    n = min(len(candidate), len(reference))
    c, r = candidate[:n], reference[:n]
    out = {"psnr_all": psnr(c, r)}
    for region in ("fg", "halo", "bg"):
        out[f"psnr_{region}"] = psnr(c, r, masks[region][:n])
    # Temporal difference energy, subject only -- catches flicker/popping on the person.
    if n > 1:
        m = masks["fg"][: n - 1]
        out["tdiff_fg_candidate"] = float(np.abs(c[1:] - c[:-1])[m].mean())
        out["tdiff_fg_reference"] = float(np.abs(r[1:] - r[:-1])[m].mean())
    return out


# ---------------------------------------------------------------------------
# Per-video work
# ---------------------------------------------------------------------------


def process_video(
    entry: dict[str, Any],
    sample_dir: Path,
    root: Path,
    conditioner: ImageConditioner,
    decoder: VideoDecoder,
    device: torch.device,
    fg_threshold: float,
    halo_px: int,
    prof: Profile,
) -> dict[str, Any]:
    vid = entry["video_id"]

    with prof.stage("decord_read_source"):
        source = read_source(entry)
    f, h, w, _ = source.shape

    with prof.stage("build_masks"):
        masks = build_masks(source, fg_threshold, halo_px)

    # --- exact VAE round-trip on GPU -------------------------------------------------
    norm = torch.from_numpy(source).permute(3, 0, 1, 2).unsqueeze(0).to(dtype=DTYPE, device=device)
    norm = norm * 2.0 - 1.0
    tiling = TilingConfig.default()
    with torch.no_grad():
        # NOTE: ImageConditioner/VideoDecoder rebuild and free the VAE on every __call__,
        # so the model build cost is inside these two stages and cannot be amortised.
        with prof.stage("vae_encode"):
            l_enc = conditioner(lambda enc: enc.tiled_encode(norm, tiling))
        expected = (1, 128, (f - 1) // 8 + 1, h // 32, w // 32)
        assert tuple(l_enc.shape) == expected, f"{vid}: latent {tuple(l_enc.shape)} != {expected}"
        with prof.stage("vae_decode"):
            d_vae = torch.cat(list(decoder(l_enc, tiling)), dim=0).cpu().float().numpy()
    del norm, l_enc
    torch.cuda.empty_cache()

    with prof.stage("metrics_k0"):
        variants: dict[str, Any] = {"k0": region_metrics(d_vae, source, masks)}

    # --- h264 noise floor of the saved decodes ---------------------------------------
    floor: dict[str, float] | None = None
    saved_source = sample_dir / "source.mp4"
    if saved_source.exists():
        with prof.stage("h264_floor"):
            sv = read_video(str(saved_source), f)
            floor = {k: v for k, v in region_metrics(sv, source, masks).items() if k.startswith("psnr")}
            del sv

    # --- refined variants, rescored from their saved decodes -------------------------
    for k in (1, 2, 3, 4, 8):
        decode = sample_dir / f"k{k}" / "decode.mp4"
        if not decode.exists():
            continue
        with prof.stage("rescore_variant"):
            cand = read_video(str(decode), f)
            m = region_metrics(cand, source, masks)
            m["psnr_fg_vs_d_vae"] = psnr(cand[: len(d_vae)], d_vae[: len(cand)], masks["fg"][: len(cand)])
            variants[f"k{k}"] = m
            del cand

    # --- side experiments that reuse the same source geometry --------------------------
    extras: dict[str, dict[str, Any]] = {}
    side_sets = {
        "prompt_control": (root / "prompt_sensitivity_control" / vid, "refine_*.mp4"),
        "two_stage": (root / "two_stage_comparison" / vid, "*.mp4"),
    }
    skip = {"source.mp4", "two_stage_comparison.mp4", "comparison.mp4", "vae_comparison.mp4"}
    for set_name, (side_dir, pattern) in side_sets.items():
        if not side_dir.is_dir():
            continue
        bucket: dict[str, Any] = {}
        for mp4 in sorted(side_dir.glob(pattern)):
            if mp4.name in skip:
                continue
            with prof.stage(f"rescore_{set_name}"):
                cand = read_video(str(mp4), f)
                bucket[mp4.stem] = region_metrics(cand, source, masks)
                del cand
        if bucket:
            extras[set_name] = bucket

    del d_vae

    return {
        "video_id": vid,
        "extras": extras,
        "subject_id": entry["subject_id"],
        "dataset_prefix": entry["dataset_prefix"],
        "video_class": entry["video_class"],
        "geometry": [h, w, f],
        "fg_fraction": float(masks["_fg_frac"]),
        "halo_fraction": float(masks["_halo_frac"]),
        "h264_floor": floor,
        "variants": variants,
    }


def worker(gpu_id: int, entries: list[tuple[dict, str]], args_dict: dict, out_path: str) -> None:
    # Address the GPU directly. CUDA_VISIBLE_DEVICES cannot be used here: this module calls
    # torch.cuda.init() at import time (required before `import decord`), and under the spawn
    # start method the child re-imports it before this function runs.
    device = torch.device(f"cuda:{gpu_id}")
    ckpt = args_dict["vae_checkpoint"]
    conditioner = ImageConditioner(ckpt, DTYPE, device)
    decoder = VideoDecoder(ckpt, DTYPE, device)
    prof = Profile()
    results = []
    for i, (entry, sample_dir) in enumerate(entries):
        t0 = time.perf_counter()
        try:
            rec = process_video(
                entry,
                Path(sample_dir),
                Path(args_dict["root"]),
                conditioner,
                decoder,
                device,
                args_dict["fg_threshold"],
                args_dict["halo_px"],
                prof,
            )
            results.append(rec)
            k0 = rec["variants"]["k0"]
            logger.info(
                "[GPU %d] (%d/%d) %s  fg=%.1f%%  k0: all %.2f / fg %.2f / halo %.2f / bg %.2f dB  [%.1fs]",
                gpu_id, i + 1, len(entries), rec["video_id"], 100 * rec["fg_fraction"],
                k0["psnr_all"], k0["psnr_fg"], k0["psnr_halo"], k0["psnr_bg"], time.perf_counter() - t0,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("[GPU %d] FAILED %s: %s", gpu_id, entry["video_id"], e, exc_info=True)
    Path(out_path).write_text(json.dumps({"results": results, "profile": prof.as_dict()}, indent=2))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in results:
        for kname, m in rec["variants"].items():
            for metric, val in m.items():
                by_variant[kname][metric].append(val)

    summary = {}
    for kname in sorted(by_variant, key=lambda s: (len(s), s)):
        summary[kname] = {
            metric: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
            for metric, v in by_variant[kname].items()
        }

    by_dataset: dict[str, list[float]] = defaultdict(list)
    for rec in results:
        by_dataset[rec["dataset_prefix"]].append(rec["variants"]["k0"]["psnr_fg"])
    dataset_summary = {
        d: {"n": len(v), "mean_k0_psnr_fg": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
        for d, v in sorted(by_dataset.items())
    }

    by_class: dict[str, list[float]] = defaultdict(list)
    for rec in results:
        by_class[rec["video_class"]].append(rec["variants"]["k0"]["psnr_fg"])
    class_summary = {
        c: {"n": len(v), "mean_k0_psnr_fg": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
        for c, v in sorted(by_class.items())
    }

    return {"by_variant": summary, "k0_fg_by_dataset": dataset_summary, "k0_fg_by_video_class": class_summary}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=SWEEP_ROOT)
    ap.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE)
    ap.add_argument("--gpus", default="6,7")
    ap.add_argument("--fg-threshold", type=float, default=0.85, help="source min(RGB) below this is subject")
    ap.add_argument("--halo-px", type=int, default=8, help="dilation radius for the floater band")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root: Path = args.root
    manifest = {e["video_id"]: e for e in json.loads((root / "manifest.json").read_text())}

    entries = []
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        vid = sample_dir.name
        if vid not in manifest:
            continue
        if not (sample_dir / "source.mp4").exists():
            continue
        entries.append((manifest[vid], str(sample_dir)))
    logger.info("Rescoring %d videos present on disk", len(entries))

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    shards: list[list] = [[] for _ in gpus]
    for i, e in enumerate(entries):
        shards[i % len(gpus)].append(e)

    out_dir = args.out or (root / "rescore")
    out_dir.mkdir(parents=True, exist_ok=True)
    args_dict = {
        "root": str(root),
        "vae_checkpoint": str(args.vae_checkpoint),
        "fg_threshold": args.fg_threshold,
        "halo_px": args.halo_px,
    }

    t_wall = time.perf_counter()
    ctx = mp.get_context("spawn")
    procs = []
    shard_files = []
    for gpu, shard in zip(gpus, shards):
        if not shard:
            continue
        sf = out_dir / f"_shard_gpu{gpu}.json"
        shard_files.append(sf)
        p = ctx.Process(target=worker, args=(gpu, shard, args_dict, str(sf)))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    wall = time.perf_counter() - t_wall

    results: list[dict[str, Any]] = []
    prof = Profile()
    for sf in shard_files:
        if not sf.exists():
            logger.error("shard %s missing -- worker died", sf)
            continue
        payload = json.loads(sf.read_text())
        results.extend(payload["results"])
        prof.merge(payload["profile"])

    payload = {
        "config": {
            "fg_threshold": args.fg_threshold,
            "halo_px": args.halo_px,
            "gpus": gpus,
            "vae_checkpoint": str(args.vae_checkpoint),
            "note": "k0 recomputed exactly on GPU; k1..k8 rescored from saved h264 decodes",
        },
        "n_videos": len(results),
        "aggregate": aggregate(results),
        "profiling": {
            "wall_clock_s": round(wall, 1),
            "gpu_workers": len(shard_files),
            "per_stage": prof.table(),
            "per_video_wall_s": round(wall / max(len(results), 1), 2),
        },
        "per_video": results,
    }
    (out_dir / "rescore.json").write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Wrote %s  (%d videos, %.1f s wall)", out_dir / "rescore.json", len(results), wall)


if __name__ == "__main__":
    main()
