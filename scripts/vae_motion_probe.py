import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import decord

# decord must be imported after torch has touched CUDA
torch.cuda.init()

sys.path.append(str(Path(__file__).resolve().parent))
import visualize_vae as vvae


def load_video(path: Path, device: torch.device) -> torch.Tensor:
    vr = decord.VideoReader(str(path))
    count = len(vr)
    frames = vr.get_batch(range(count))
    video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(vvae.DTYPE).to(device)
    video = (video / 127.5) - 1.0
    valid_f = ((video.shape[2] - 1) // 8) * 8 + 1
    return video[:, :, :valid_f]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = vvae.load_video_vae_encoder(args.checkpoint, device)
    decoder = vvae.load_video_vae_decoder(args.checkpoint, device)

    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    # Store PSNR for plots
    psnr_by_motion_and_frames = defaultdict(list)
    # per latent frame
    latent_psnr_by_motion_and_frames = defaultdict(list)

    for clip in manifest:
        subject_id = clip["subject_id"]
        motion = clip["motion"]
        frame_count = clip["frame_count"]
        video_path = Path(clip["video_path"])

        print(f"Processing {subject_id} {motion} {frame_count}")

        video = load_video(video_path, device)
        stages = vvae.encode_tail_stages(encoder, video)
        latent = stages["latent_normalized"]
        decoded_video = vvae.decode_latent_to_video(decoder, latent)
        original_pixel = vvae.prepare_pixel_video(video)

        psnr_res = vvae.compute_psnr(original_pixel, decoded_video)

        clip_out_dir = out_dir / subject_id
        clip_out_dir.mkdir(parents=True, exist_ok=True)
        vvae.save_comparison_video(
            original_pixel,
            decoded_video,
            clip_out_dir / f"{motion}_f{frame_count:03d}_cmp.mp4",
            fps=clip["fps"],
            diff_gain=4.0,
        )

        summary_record = {
            "subject_id": subject_id,
            "motion": motion,
            "frame_count": frame_count,
            "overall_psnr_db": psnr_res["overall_psnr_db"],
            "per_frame_psnr_db": psnr_res["per_frame_psnr_db"],
        }
        summary.append(summary_record)

        key = (motion, frame_count)
        psnr_by_motion_and_frames[key].append(psnr_res["overall_psnr_db"])

        bucket_size = 8
        per_frame = psnr_res["per_frame_psnr_db"]
        latent_psnr = []
        for i in range(0, len(per_frame), bucket_size):
            chunk = per_frame[i : i + bucket_size]
            latent_psnr.append(sum(chunk) / len(chunk))

        latent_psnr_by_motion_and_frames[key].append(latent_psnr)

    with open(out_dir / "psnr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot 1: psnr_by_frame_count.png
    plt.figure()
    motions = ["translate", "turntable", "wave"]
    frame_counts = [41, 81, 121]

    for motion in motions:
        means = []
        for fc in frame_counts:
            vals = psnr_by_motion_and_frames.get((motion, fc), [])
            if vals:
                means.append(np.mean(vals))
                plt.scatter([fc] * len(vals), vals, alpha=0.3, label="_nolegend_")
            else:
                means.append(None)
        plt.plot(frame_counts, means, label=motion, marker="o", linewidth=2)

    plt.legend()
    plt.xlabel("Frame Count")
    plt.ylabel("PSNR (dB)")
    plt.title("Overall PSNR by Frame Count")
    plt.xticks(frame_counts)
    plt.savefig(out_dir / "psnr_by_frame_count.png")
    plt.close()

    # Plot 2: psnr_by_latent_frame.png
    plt.figure(figsize=(15, 5))
    for i, motion in enumerate(motions):
        plt.subplot(1, 3, i + 1)
        for fc in frame_counts:
            curves = latent_psnr_by_motion_and_frames.get((motion, fc), [])
            if not curves:
                continue
            mean_curve = np.mean(curves, axis=0)
            plt.plot(mean_curve, label=f"N={fc}")
        plt.title(motion)
        plt.xlabel("Latent Frame")
        plt.ylabel("PSNR (dB)")
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "psnr_by_latent_frame.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=vvae.WORKSPACE_ROOT / "checkpoints" / "LTX-2.3" / "ltx-2.3-22b-distilled-1.1.safetensors",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default="/home/jianjinx/data2/SAM3DGS/expr/ltx_vae_motion_probe/manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default="results/vae_motion_probe")
    args = parser.parse_args()
    main(args)
