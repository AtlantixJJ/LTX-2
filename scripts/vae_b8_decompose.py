"""
"What is an enc_b8 feature vector actually made of?"

Give a concrete, additive answer for the deepest bottleneck feature x = b8 (shape 1x1024x32x8x12):

    x(p) = P + δ(p),      P = per-channel mean over tokens (a single fixed 1024-vector)
    δ(p) = Σ_i c_i(p) v_i (PCA of the centered feature)

and quantify (i) the energy budget [constant pedestal | border/position modes | scene content],
(ii) the channel composition [do the pedestal and the variation live in the same channels? how many
are dead?], (iii) whether the content is static or moving, and (iv) a reconstruction ladder proving
how few terms rebuild x. Writes results/b8_composition.{json,png}.

    MAX_ANALYSIS_FRAMES=249 python3 scripts/vae_b8_decompose.py
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
N_BORDER_MODES = 3   # top PCs shown (by the heatmaps) to be border/corner-dominated


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
    _, C, F, H, W = x.shape
    X = rearrange(x, "b c f h w -> (b f h w) c").detach().cpu().double().numpy()   # (N, C)
    N = X.shape[0]
    print(f"b8 shape=(1,{C},{F},{H},{W})  tokens N={N}")

    # ---- decomposition ----------------------------------------------------------------------
    P = X.mean(axis=0)                       # pedestal (constant 1024-vector)
    Xc = X - P                               # deviation δ(p)
    var = Xc.var(axis=0)                     # per-channel AC variance
    DC = float((P ** 2).sum())               # constant-pedestal energy  (Σ_c P_c^2)
    AC = float(var.sum())                    # variation energy          (Σ_c Var_c)
    Etot = DC + AC

    pca = PCA().fit(Xc)
    lam = np.clip(pca.explained_variance_.astype(np.float64), 0, None)   # AC per-PC energy
    evr = lam / lam.sum()
    cum = np.cumsum(evr)
    n98_ac = int(np.searchsorted(cum, 0.98) + 1)
    border_frac_of_ac = float(cum[N_BORDER_MODES - 1])                   # top-3 share of AC

    # ---- channel composition ----------------------------------------------------------------
    order_mean = np.argsort(np.abs(P))[::-1]
    order_var = np.argsort(var)[::-1]
    K = 64
    overlap = len(set(order_mean[:K].tolist()) & set(order_var[:K].tolist())) / K
    dead = int(((np.abs(P) < 1e-2 * np.abs(P).max()) & (var < 1e-2 * var.max())).sum())
    active_var = int((var > 1e-2 * var.max()).sum())
    n_ped_90 = int(np.searchsorted(np.cumsum(np.sort(P ** 2)[::-1]) / DC, 0.90) + 1)

    # ---- content: static vs moving ----------------------------------------------------------
    d = rearrange(Xc, "(f hw) c -> f hw c", f=F)          # (F, H*W, C)
    spatial_template = d.mean(axis=0, keepdims=True)      # per-location mean over frames
    temporal_resid = d - spatial_template
    static_energy = float((spatial_template ** 2).sum() * F)
    moving_energy = float((temporal_resid ** 2).sum())
    static_frac = static_energy / (static_energy + moving_energy)

    # ---- reconstruction ladder (fraction of total b8 energy Σ‖x(p)‖^2) ----------------------
    tot_sq = float((X ** 2).sum())
    def frac_after(k):  # pedestal + top-k PCs
        recon_ac = float(lam[:k].sum()) * N          # variance*N ≈ energy captured by k PCs
        return (DC * N + recon_ac) / tot_sq
    ladder = {f"pedestal+{k}PC": round(frac_after(k), 5) for k in [0, 1, 2, 3, 4, 10, n98_ac]}

    report = {
        "shape": [1, C, F, H, W], "tokens": N,
        "energy_budget_fraction_of_total": {
            "constant_pedestal": round(DC / Etot, 4),
            "border_position_modes(top%d)" % N_BORDER_MODES: round(border_frac_of_ac * AC / Etot, 4),
            "scene_content(rest)": round((1 - border_frac_of_ac) * AC / Etot, 4),
        },
        "AC_only": {"top1": round(float(evr[0]), 3), "top3": round(border_frac_of_ac, 3),
                    "n_comp_98pct": n98_ac},
        "channels": {
            "pedestal_channels_90pct": n_ped_90,
            "active_variance_channels": active_var,
            "dead_channels": dead,
            "overlap_top64_mean_vs_var": round(overlap, 3),
            "pedestal_rms_top64": round(float(np.sqrt((P[order_mean[:64]] ** 2).mean())), 1),
            "deviation_std_top64": round(float(np.sqrt(var[order_var[:64]].mean())), 1),
        },
        "content_static_fraction_of_AC": round(static_frac, 3),
        "reconstruction_ladder_frac_of_total_energy": ladder,
    }
    print(json.dumps(report, indent=2))
    with open(os.path.join(OUTPUT_DIR, "b8_composition.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- figure -----------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    b = report["energy_budget_fraction_of_total"]
    labels = ["constant\npedestal", "border/\nposition", "scene\ncontent"]
    vals = [b["constant_pedestal"], b["border_position_modes(top%d)" % N_BORDER_MODES], b["scene_content(rest)"]]
    ax[0].bar(labels, vals, color=["#888", "#d1495b", "#2e8b57"])
    ax[0].set_yscale("log"); ax[0].set_ylabel("fraction of total b8 energy (log)")
    for i, v in enumerate(vals):
        ax[0].text(i, v, f"{v*100:.2f}%", ha="center", va="bottom")
    ax[0].set_title("What b8 energy is made of")

    ax[1].semilogy(np.arange(1, C + 1), np.sort(P ** 2)[::-1] + 1e-9, label="pedestal |P_c|^2")
    ax[1].semilogy(np.arange(1, C + 1), np.sort(var)[::-1] + 1e-9, label="deviation Var_c")
    ax[1].axvline(active_var, color="k", ls=":", lw=1, label=f"~{active_var} active ch")
    ax[1].set_xlabel("channel rank"); ax[1].set_ylabel("energy (log)")
    ax[1].set_title("Channel composition"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")

    ks = [0, 1, 2, 3, 4, 10, n98_ac]
    ax[2].plot(ks, [frac_after(k) for k in ks], marker="o")
    ax[2].axhline(1.0, color="k", ls=":", lw=1)
    ax[2].set_xlabel("pedestal + top-k PCs"); ax[2].set_ylabel("fraction of total b8 energy")
    ax[2].set_title("Reconstruction ladder"); ax[2].grid(alpha=0.3)
    for k in [0, 3, n98_ac]:
        ax[2].annotate(f"{frac_after(k)*100:.2f}%", (k, frac_after(k)), fontsize=8,
                       textcoords="offset points", xytext=(4, -10))
    fig.suptitle("enc_b8 = constant pedestal + border/position modes + scene content")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "b8_composition.png")
    plt.savefig(out); plt.close()
    print(f"Saved {out}\nSaved b8_composition.json")


if __name__ == "__main__":
    main()
