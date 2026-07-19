"""
Focused follow-up to visualize_vae.py: *why* do the deepest encoder blocks (enc_b7/b8) collapse
to ~4 effective ranks (98% variance)?

visualize_vae.py established the collapse and showed (via per-component heatmaps) that the top
PCA modes at 8x12 are dominated by extreme-magnitude BORDER/CORNER cells (zero-padding "massive
activations"), not interior scene content. This script runs the decisive controls:

  1. full vs interior-only vs border-only effective rank  -> is the low rank a border artifact?
  2. spatial-only (single frame) rank                     -> how much is temporal redundancy?
  3. rank AFTER the encoder's conv_norm_out (PixelNorm)   -> does per-location RMS norm strip the
                                                             massive-activation DC pedestal?
  4. rank / active-channel count of the final 128ch latent-> what does it mean downstream?

Run (encoder-only, picks a free GPU automatically):
    MAX_ANALYSIS_FRAMES=249 python3 scripts/vae_bottleneck_probe.py
"""
import os
import sys
import json
import subprocess

import numpy as np
import torch
from einops import rearrange
from sklearn.decomposition import PCA

sys.path.append("/home/jianjinx/data2/VideoDiffusionModels/LTX-2/packages/ltx-core/src")
sys.path.append("/home/jianjinx/data2/VideoDiffusionModels/LTX-2/packages/ltx-trainer/src")
import decord  # noqa: E402
decord.bridge.set_bridge("torch")

from ltx_trainer.model_loader import load_video_vae_encoder  # noqa: E402
from ltx_core.model.video_vae.ops import patchify  # noqa: E402

CHECKPOINT_PATH = "/home/jianjinx/data2/VideoDiffusionModels/checkpoints/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
OUTPUT_DIR = "/home/jianjinx/data2/VideoDiffusionModels/LTX-2/results"
VIDEO_PATH = "/home/jianjinx/data2/VideoDiffusionModels/scenes/03_mountain_dialog/output_seed_31415.mp4"
DTYPE = torch.bfloat16
MAX_ANALYSIS_FRAMES = int(os.environ.get("MAX_ANALYSIS_FRAMES", 249))


def select_free_gpu() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
        rows = [tuple(int(x) for x in line.split(",")) for line in out.strip().splitlines()]
        idx, _, _ = min(rows, key=lambda r: (r[1], r[2]))
        return torch.device(f"cuda:{idx}")
    except Exception:
        return torch.device("cuda:0")


def rank_metrics(X, max_samples=200_000):
    """X: (N, C). Returns effective-rank summary computed from the PCA eigen-spectrum."""
    N = X.shape[0]
    Xfit = X if N <= max_samples else X[np.random.default_rng(0).choice(N, max_samples, replace=False)]
    pca = PCA().fit(Xfit)
    ev = np.clip(pca.explained_variance_.astype(np.float64), 0, None)
    total = float(ev.sum())
    ratio = ev / (total + 1e-12)
    cum = np.cumsum(ratio)
    nz = ratio[ratio > 0]
    return {
        "n_tokens": int(N),
        "n_channels": int(X.shape[1]),
        "cum_evr_4": float(cum[min(3, len(cum) - 1)]),
        "entropy_effective_rank": float(np.exp(-np.sum(nz * np.log(nz)))),
        "participation_ratio": float((total ** 2) / (float(np.sum(ev ** 2)) + 1e-12)),
        "n_comp_90": int(min(np.searchsorted(cum, 0.90) + 1, len(cum))),
        "n_comp_98": int(min(np.searchsorted(cum, 0.98) + 1, len(cum))),
        "evr_top4": ratio[:4].tolist(),
    }


def to_rows(feat):
    """(B,C,F,H,W) tensor -> (N, C) float32 numpy of channel-vectors per spatiotemporal token."""
    return rearrange(feat, "b c f h w -> (b f h w) c").detach().cpu().to(torch.float32).numpy()


def interior_border_split(feat, ring=1):
    """Split a (B,C,F,H,W) feature into interior tokens (border ring stripped) and border tokens."""
    B, C, F, H, W = feat.shape
    hmask = torch.ones(H, dtype=torch.bool)
    wmask = torch.ones(W, dtype=torch.bool)
    hmask[:ring] = False; hmask[H - ring:] = False
    wmask[:ring] = False; wmask[W - ring:] = False
    grid = (hmask[:, None] & wmask[None, :]).to(feat.device)   # (H,W) True = interior
    interior = feat[..., grid]                             # (B,C,F, n_int)
    border = feat[..., ~grid]                              # (B,C,F, n_bord)
    to = lambda t: rearrange(t, "b c f n -> (b f n) c").detach().cpu().to(torch.float32).numpy()
    return to(interior), to(border)


@torch.no_grad()
def main():
    device = select_free_gpu()
    print(f"Device: {device}")
    encoder = load_video_vae_encoder(CHECKPOINT_PATH, device=device, dtype=DTYPE)

    vr = decord.VideoReader(VIDEO_PATH)
    frames = vr.get_batch(range(min(len(vr), MAX_ANALYSIS_FRAMES)))
    video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(DTYPE).to(device)
    video = (video / 127.5) - 1.0
    valid_f = ((video.shape[2] - 1) // 8) * 8 + 1
    video = video[:, :, :valid_f]
    print(f"Video: {tuple(video.shape)}")

    x = encoder.conv_in(patchify(video, patch_size_hw=4, patch_size_t=1))
    captured = {}
    for i, block in enumerate(encoder.down_blocks):
        x = block(x)
        if i in (6, 7, 8):
            captured[f"enc_b{i}"] = x.clone()

    # Encoder tail: conv_norm_out is PixelNorm (per-location RMS over channels), then SiLU, then
    # conv_out -> moments; the first 128 channels (normalized) are the latent means.
    post_norm = encoder.conv_norm_out(x)                  # PixelNorm output
    moments = encoder.conv_out(encoder.conv_act(post_norm))
    latent = encoder.per_channel_statistics.normalize(moments[:, :128])
    captured["enc_postPixelNorm(b8)"] = post_norm
    captured["latent_means(128)"] = latent
    print(f"post-PixelNorm shape {tuple(post_norm.shape)} | latent shape {tuple(latent.shape)}")

    report = {}
    for name, feat in captured.items():
        full = to_rows(feat)
        rec = {"shape_BCFHW": [int(v) for v in feat.shape], "full": rank_metrics(full)}
        # interior vs border only meaningful once spatially small enough to have a real border
        if feat.shape[-1] >= 4 and feat.shape[-2] >= 4:
            interior, border = interior_border_split(feat, ring=1)
            rec["interior_only"] = rank_metrics(interior)
            rec["border_only"] = rank_metrics(border)
        # spatial-only: middle single frame, removes temporal redundancy
        mid = feat.shape[2] // 2
        rec["single_frame"] = rank_metrics(to_rows(feat[:, :, mid:mid + 1]))
        # channel activity: fraction of channels with non-trivial variance
        var = full.var(axis=0)
        rec["frac_channels_active_1e-2xmax"] = float((var > 1e-2 * var.max()).mean())
        rec["global_rms"] = float(np.sqrt(np.mean(var + full.mean(axis=0) ** 2)))
        report[name] = rec

        f = rec["full"]
        line = (f"{name:24s} shape={tuple(feat.shape)} rms={rec['global_rms']:.2e} "
                f"| FULL erank={f['entropy_effective_rank']:.1f} n@98={f['n_comp_98']} cum4={f['cum_evr_4']:.3f}")
        if "interior_only" in rec:
            io, bo = rec["interior_only"], rec["border_only"]
            line += (f" | INTERIOR erank={io['entropy_effective_rank']:.1f} n@98={io['n_comp_98']} cum4={io['cum_evr_4']:.3f}"
                     f" | BORDER erank={bo['entropy_effective_rank']:.1f} n@98={bo['n_comp_98']}")
        sf = rec["single_frame"]
        line += f" | 1FRAME erank={sf['entropy_effective_rank']:.1f} n@98={sf['n_comp_98']}"
        line += f" | active_ch={rec['frac_channels_active_1e-2xmax']:.2f}"
        print(line)

    out = os.path.join(OUTPUT_DIR, "bottleneck_probe.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
