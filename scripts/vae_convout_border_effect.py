"""
Does the border/corner massive activation in enc_b8 imprint on conv_out's output (the latent)?

The encoder tail is:  b8 -> conv_norm_out(PixelNorm) -> SiLU -> conv_out(1024->129) -> means(128)
                      -> per_channel_statistics.normalize -> latent.

PixelNorm normalizes each location by its RMS *across channels*; that RMS is dominated by the ~64
massive "register" channels, so conv_out receives a per-location gain that is set by the border
structure. conv_out also zero-pads (spatial_padding_mode=zeros), so it can re-inject a border
signal of its own. Per-channel latent normalization only rescales channels globally, so it cannot
remove a *spatial* (per-location) border pattern.

This script measures, at each tail stage, the per-location energy map (8x12) and the border-vs-
interior ratio, plus the PixelNorm gain map and a few raw latent-channel maps, to see whether the
border massive activation survives into conv_out's result.

    MAX_ANALYSIS_FRAMES=249 python3 scripts/vae_convout_border_effect.py
"""
import os
import sys
import json
import subprocess

import numpy as np
import torch
from einops import rearrange
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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


def select_free_gpu():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
        rows = [tuple(int(x) for x in line.split(",")) for line in out.strip().splitlines()]
        return torch.device(f"cuda:{min(rows, key=lambda r: (r[1], r[2]))[0]}")
    except Exception:
        return torch.device("cuda:0")


def loc_norm_map(feat):
    """(1,C,F,H,W) -> per-location L2 norm over channels, averaged over frames -> (H,W) numpy."""
    n = feat.float().pow(2).sum(dim=1).sqrt()          # (1,F,H,W)
    return n.mean(dim=1)[0].detach().cpu().numpy()      # (H,W)


def border_interior_ratio(map_hw, ring=1):
    H, W = map_hw.shape
    m = np.ones((H, W), dtype=bool)
    m[ring:H - ring, ring:W - ring] = False             # True = border ring
    return float(map_hw[m].mean() / (map_hw[~m].mean() + 1e-12))


def eff_rank(feat, max_samples=200_000):
    X = rearrange(feat, "b c f h w -> (b f h w) c").detach().cpu().float().numpy()
    N = X.shape[0]
    Xf = X if N <= max_samples else X[np.random.default_rng(0).choice(N, max_samples, replace=False)]
    ev = np.clip(PCA().fit(Xf).explained_variance_.astype(np.float64), 0, None)
    ratio = ev / (ev.sum() + 1e-12)
    cum = np.cumsum(ratio)
    nz = ratio[ratio > 0]
    return {
        "erank": float(np.exp(-np.sum(nz * np.log(nz)))),
        "n_comp_98": int(min(np.searchsorted(cum, 0.98) + 1, len(cum))),
        "active_ch_frac": float((X.var(0) > 1e-2 * X.var(0).max()).mean()),
    }


def plot_pca_component_heatmaps(feat, title, save_path, ncomp=4, nframes=4):
    """Top-`ncomp` PCA components of a (1,C,F,H,W) feature, each shown as per-frame heatmaps
    (viridis + colorbars) at the feature's native (H,W) -- the same diagnostic view used for the
    encoder blocks, so the conv_out map and the normalized latent map are directly comparable."""
    B, C, F, H, W = feat.shape
    X = rearrange(feat, "b c f h w -> (b f h w) c").detach().cpu().float().numpy()
    n = min(ncomp, C)
    proj = PCA(n_components=n).fit_transform(X)
    comps = rearrange(proj, "(b f h w) c -> b f h w c", b=B, f=F, h=H, w=W)[0]
    fidx = sorted(set(np.linspace(0, F - 1, min(nframes, F)).astype(int).tolist()))
    fig, axes = plt.subplots(n, len(fidx), figsize=(len(fidx) * 3, n * 3), squeeze=False)
    for r in range(n):
        for cc, fi in enumerate(fidx):
            ax = axes[r][cc]
            im = ax.imshow(comps[fi, :, :, r], cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"Frame {fi}")
            if cc == 0:
                ax.set_ylabel(f"PCA comp {r}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_variance_curve(feat, title, save_path, max_samples=200_000):
    """Cumulative explained-variance curve (top 50 PCs), matching the per-block *_variance.png
    produced by visualize_vae.py, so conv_out and the latent sit on the same axis as the blocks."""
    B, C, F, H, W = feat.shape
    X = rearrange(feat, "b c f h w -> (b f h w) c").detach().cpu().float().numpy()
    N = X.shape[0]
    Xf = X if N <= max_samples else X[np.random.default_rng(0).choice(N, max_samples, replace=False)]
    evr = PCA().fit(Xf).explained_variance_ratio_
    plt.figure()
    plt.plot(np.cumsum(evr[:50]), marker="o")
    plt.title(f"{title} {tuple(feat.shape)}")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


@torch.no_grad()
def main():
    device = select_free_gpu()
    print(f"Device: {device}")
    enc = load_video_vae_encoder(CHECKPOINT_PATH, device=device, dtype=DTYPE)

    vr = decord.VideoReader(VIDEO_PATH)
    frames = vr.get_batch(range(min(len(vr), MAX_ANALYSIS_FRAMES)))
    video = (frames.permute(3, 0, 1, 2).unsqueeze(0).to(DTYPE).to(device) / 127.5) - 1.0
    video = video[:, :, :((video.shape[2] - 1) // 8) * 8 + 1]

    x = enc.conv_in(patchify(video, patch_size_hw=4, patch_size_t=1))
    for block in enc.down_blocks:
        x = block(x)
    b8 = x

    # Tail, captured step by step.
    pn = enc.conv_norm_out(b8)                 # PixelNorm output
    act = enc.conv_act(pn)                      # SiLU
    moments = enc.conv_out(act)                 # (1,129,F,H,W)
    means = moments[:, :128]                    # raw conv_out means (pre per-channel norm)
    latent = enc.per_channel_statistics.normalize(means)  # normalized latent the DiT sees

    # PixelNorm denominator (per-location RMS across channels of b8) = the gain conv_out inherits.
    pn_denom = b8.float().pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt()  # (1,1,F,H,W)
    pn_denom_map = pn_denom.mean(dim=2)[0, 0].detach().cpu().numpy()          # (H,W)

    stages = {
        "b8 (block8 out)": b8,
        "post-PixelNorm": pn,
        "conv_out means (raw)": means,
        "latent (normalized)": latent,
    }
    maps = {k: loc_norm_map(v) for k, v in stages.items()}
    maps["PixelNorm gain (denom)"] = pn_denom_map

    report = {}
    for k, v in stages.items():
        mp = maps[k]
        r = eff_rank(v)
        r["border_interior_energy_ratio"] = border_interior_ratio(mp)
        r["shape"] = [int(s) for s in v.shape]
        report[k] = r
        print(f"{k:24s} shape={tuple(v.shape)} | erank={r['erank']:.1f} n@98={r['n_comp_98']} "
              f"active_ch={r['active_ch_frac']:.2f} | border/interior energy={r['border_interior_energy_ratio']:.2f}x")
    report["PixelNorm gain (denom)"] = {"border_interior_ratio": border_interior_ratio(pn_denom_map)}
    print(f"PixelNorm gain (denom)   border/interior={report['PixelNorm gain (denom)']['border_interior_ratio']:.2f}x")

    # --- Figure 1: per-location energy maps down the tail (each averaged over frames) ---
    order = ["b8 (block8 out)", "PixelNorm gain (denom)", "post-PixelNorm",
             "conv_out means (raw)", "latent (normalized)"]
    fig, axes = plt.subplots(1, len(order), figsize=(4 * len(order), 4))
    for ax, k in zip(axes, order):
        im = ax.imshow(maps[k], cmap="magma")
        br = report[k].get("border_interior_energy_ratio", report[k].get("border_interior_ratio"))
        ax.set_title(f"{k}\nborder/interior={br:.2f}x", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Border massive activation -> conv_out: per-location energy (mean over frames)")
    plt.tight_layout()
    fig1 = os.path.join(OUTPUT_DIR, "convout_border_energy_maps.png")
    plt.savefig(fig1); plt.close()

    # --- Figure 2: do individual latent channels carry a spatial border artifact? ---
    lat = latent[0].float().mean(dim=1).detach().cpu().numpy()   # (128,H,W) mean over frames
    var = lat.reshape(128, -1).var(1)
    top = np.argsort(var)[::-1][:6]
    fig, axes = plt.subplots(1, 6, figsize=(4 * 6, 4))
    for ax, c in zip(axes, top):
        im = ax.imshow(lat[c], cmap="viridis")
        ax.set_title(f"latent ch {c}\nborder/int={border_interior_ratio(np.abs(lat[c])):.2f}x", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Top-variance latent channels (8x12, mean over frames) — spatial border artifact?")
    plt.tight_layout()
    fig2 = os.path.join(OUTPUT_DIR, "convout_border_latent_channels.png")
    plt.savefig(fig2); plt.close()

    # --- Figures 3 & 4: the conv_out map and the normalized (latent) map, as top-PCA-component
    # per-frame heatmaps (same view as the enc block heatmaps, for direct comparison) ---
    fig3 = os.path.join(OUTPUT_DIR, "convout_map_component_heatmaps.png")
    fig4 = os.path.join(OUTPUT_DIR, "normalized_map_component_heatmaps.png")
    plot_pca_component_heatmaps(means, "conv_out means (raw, pre-normalization) — top PCA components", fig3)
    plot_pca_component_heatmaps(latent, "latent means (normalized, what the DiT sees) — top PCA components", fig4)

    # Cumulative explained-variance curves for the two tail stages, matching the block *_variance.png.
    fig5 = os.path.join(OUTPUT_DIR, "convout_map_variance.png")
    fig6 = os.path.join(OUTPUT_DIR, "normalized_map_variance.png")
    plot_variance_curve(means, "conv_out means (raw)", fig5)
    plot_variance_curve(latent, "latent (normalized)", fig6)
    print(f"Saved {fig5}\nSaved {fig6}")

    with open(os.path.join(OUTPUT_DIR, "convout_border_effect.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {fig1}\nSaved {fig2}\nSaved {fig3}\nSaved {fig4}\nSaved convout_border_effect.json")


if __name__ == "__main__":
    main()
