import os
import torch
import torch.nn as nn
from einops import rearrange
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

import sys
import decord
decord.bridge.set_bridge('torch')
sys.path.append("/home/jianjinx/data2/VideoDiffusionModels/LTX-2/packages/ltx-core/src")
sys.path.append("/home/jianjinx/data2/VideoDiffusionModels/LTX-2/packages/ltx-trainer/src")

from ltx_trainer.model_loader import load_video_vae_encoder, load_video_vae_decoder
from ltx_core.model.video_vae.ops import patchify
from ltx_core.model.video_vae.video_vae import UNetMidBlock3D, ResnetBlock3D

CHECKPOINT_PATH = "/home/jianjinx/data2/VideoDiffusionModels/checkpoints/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
OUTPUT_DIR = "/home/jianjinx/data2/VideoDiffusionModels/LTX-2/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_encoder_input_filter(encoder, save_path="encoder_filter.png"):
    # The first layer is conv_in
    # shape: (out_channels, in_channels, kT, kH, kW)
    w = encoder.conv_in.weight.data
    out_channels, in_channels, kT, kH, kW = w.shape
    
    # We know in_channels is 48 because patch_size_hw=4 and rgb channels=3 -> 3 * 16 = 48
    # Unpatchify the weights:
    # We treat out_channels as batch size for visualization purposes
    w_unpatched = rearrange(w, "out (c p_h p_w) kt kh kw -> out c kt (kh p_h) (kw p_w)", c=3, p_h=4, p_w=4)
    
    # We can visualize the first few filters
    num_filters_to_show = min(8, out_channels)
    
    fig, axes = plt.subplots(num_filters_to_show, kT, figsize=(kT * 2, num_filters_to_show * 2))
    
    # Normalize for visualization
    for i in range(num_filters_to_show):
        for t in range(kT):
            img = w_unpatched[i, :, t].detach().cpu().to(torch.float32).numpy()
            img = np.transpose(img, (1, 2, 0))
            # Min-max scale to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            if num_filters_to_show == 1 and kT == 1:
                ax = axes
            elif num_filters_to_show == 1:
                ax = axes[t]
            elif kT == 1:
                ax = axes[i]
            else:
                ax = axes[i, t]
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"Time {t}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved encoder filter visualization to {save_path}")

def visualize_decoder_output_filter(decoder, save_path="decoder_filter.png"):
    w = decoder.conv_out.weight.data
    out_channels, in_channels, kT, kH, kW = w.shape
    # out_channels is 48
    # Unpatchify along out_channels
    w_unpatched = rearrange(w, "(c p_h p_w) in_c kt kh kw -> in_c c kt (kh p_h) (kw p_w)", c=3, p_h=4, p_w=4)
    
    num_filters_to_show = min(8, in_channels)
    
    fig, axes = plt.subplots(num_filters_to_show, kT, figsize=(kT * 2, num_filters_to_show * 2))
    
    for i in range(num_filters_to_show):
        for t in range(kT):
            img = w_unpatched[i, :, t].detach().cpu().to(torch.float32).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            if num_filters_to_show == 1 and kT == 1:
                ax = axes
            elif num_filters_to_show == 1:
                ax = axes[t]
            elif kT == 1:
                ax = axes[i]
            else:
                ax = axes[i, t]
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"Time {t}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved decoder filter visualization to {save_path}")

def get_unroll_factors(orig_h, current_h):
    """
    Determine the exact sequence of depth-to-space operations needed to perfectly
    map an intermediate feature's spatial grid back to the original image grid.
    Since LTX uses patchify(4) initially, and then compress(2) repeatedly,
    we must reverse this by uncompressing 2s, and finally unpatchifying 4.
    """
    scale = orig_h // current_h
    if scale == 1: return []
    if scale == 2: return [2]
    if scale == 4: return [4]
    if scale == 8: return [2, 4]
    if scale == 16: return [2, 2, 4]
    if scale == 32: return [2, 2, 2, 4]
    
    # Fallback for arbitrary scales
    steps = int(np.log2(scale))
    if scale > 4:
        return [2] * (steps - 2) + [4]
    return [2] * steps

def visualize_intermediate_features_with_pca(feature, p_h_list=[], p_w_list=[], save_prefix="pca"):
    """
    feature shape: (B, C, F, H, W)
    We will do depth-to-space using the sequence of p_h and p_w factors.
    """
    B, C, F, H, W = feature.shape
    
    if not p_h_list or not p_w_list:
        total_p_h, total_p_w = 1, 1
    else:
        total_p_h = np.prod(p_h_list)
        total_p_w = np.prod(p_w_list)
    
    # Check if C is divisible by total factors
    assert C % (total_p_h * total_p_w) == 0, f"Channels {C} not divisible by {total_p_h * total_p_w}"
    
    reconstructed = feature
    for ph, pw in zip(p_h_list, p_w_list):
        reconstructed = rearrange(reconstructed, "b (c p_h p_w) f h w -> b c f (h p_h) (w p_w)", p_h=ph, p_w=pw)
    
    B, c, F, H_new, W_new = reconstructed.shape
    
    # Flatten spatial dimensions to do PCA on the channels
    # Shape: (B*F*H_new*W_new, c)
    X = rearrange(reconstructed, "b c f h w -> (b f h w) c").detach().cpu().to(torch.float32).numpy()
    
    pca = PCA()
    pca.fit(X)
    
    # The explained variance ratio gives us the sparsity / importance of components
    explained_variance = pca.explained_variance_ratio_
    
    # Plot explained variance
    plt.figure()
    plt.plot(np.cumsum(explained_variance), marker='o')
    plt.title(f"{save_prefix}: Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.savefig(f"{save_prefix}_variance.png")
    plt.close()
    
    # To show how much information is concentrated in top components
    pca = PCA(n_components=min(c, 10))
    pca.fit(X)
    print(f"Explained variance ratio (first 10): {pca.explained_variance_ratio_}")
    
    if c >= 3:
        pca_3 = PCA(n_components=3)
        X_3 = pca_3.fit_transform(X)
    elif c == 2:
        pca_2 = PCA(n_components=2)
        X_2 = pca_2.fit_transform(X)
        X_3 = np.zeros((X.shape[0], 3))
        X_3[:, :2] = X_2
    else:
        # c == 1
        X_3 = np.repeat(X, 3, axis=1)
        
    X_3_img = rearrange(X_3, "(b f h w) c -> b f h w c", b=B, f=F, h=H_new, w=W_new)
    
    # Normalize top 3 to [0, 1] for RGB visualization
    X_3_img = (X_3_img - X_3_img.min(axis=(0,1,2,3), keepdims=True)) / \
              (X_3_img.max(axis=(0,1,2,3), keepdims=True) - X_3_img.min(axis=(0,1,2,3), keepdims=True) + 1e-8)
    
    # Save a frame as an example
    for f in range(F):
        plt.figure()
        plt.imshow(X_3_img[0, f])
        plt.title(f"{save_prefix} Frame {f} Top 3 PCA Components")
        plt.axis("off")
        plt.savefig(f"{save_prefix}_frame_{f}.png")
        plt.close()
        break # Just one frame is enough
        
    print(f"Saved PCA visualizations with prefix {save_prefix}")
    print(f"Explained variance ratio (first 10): {explained_variance[:10]}")
    return explained_variance

def main():
    print("Loading Encoder...")
    encoder = load_video_vae_encoder(CHECKPOINT_PATH, device="cuda:2")
    print("Visualizing Encoder Filters...")
    visualize_encoder_input_filter(encoder, save_path=os.path.join(OUTPUT_DIR, "encoder_filter.png"))
    
    print("Loading Decoder...")
    decoder = load_video_vae_decoder(CHECKPOINT_PATH, device="cuda:2")
    print("Visualizing Decoder Filters...")
    visualize_decoder_output_filter(decoder, save_path=os.path.join(OUTPUT_DIR, "decoder_filter.png"))
    
    print("Loading Real Video and Extracting Intermediate Features...")
    video_path = "/home/jianjinx/data2/VideoDiffusionModels/scenes/03_mountain_dialog/output_seed_31415.mp4"
    vr = decord.VideoReader(video_path)
    video_frames = vr.get_batch(range(len(vr)))
    # video_frames shape: (T, H, W, C) in [0, 255]
    real_video = video_frames.permute(3, 0, 1, 2).unsqueeze(0).to(torch.bfloat16).to("cuda:2") # (1, C, T, H, W)
    real_video = (real_video / 127.5) - 1.0 # Normalize to [-1, 1]
    
    # Ensure F = 1 + 8*k
    F = real_video.shape[2]
    valid_f = ((F - 1) // 8) * 8 + 1
    if valid_f < 1:
        valid_f = 1
    real_video = real_video[:, :, :valid_f]
    
    print(f"Loaded video shape: {real_video.shape}")
    
    orig_H = real_video.shape[3]
    
    # ENCODER processing
    patched = patchify(real_video, patch_size_hw=4, patch_size_t=1)
    
    x = encoder.conv_in(patched)
    print(f"Feature shape after enc conv_in: {x.shape}")
    factors = get_unroll_factors(orig_H, x.shape[3])
    visualize_intermediate_features_with_pca(x, p_h_list=factors, p_w_list=factors, save_prefix=os.path.join(OUTPUT_DIR, "pca_enc_conv_in"))
    
    for i, block in enumerate(encoder.down_blocks):
        x = block(x)
        print(f"Feature shape after enc block {i}: {x.shape}")
        
        factors = get_unroll_factors(orig_H, x.shape[3])
        visualize_intermediate_features_with_pca(x, p_h_list=factors, p_w_list=factors, save_prefix=os.path.join(OUTPUT_DIR, f"pca_enc_block_{i}"))
        
    x = encoder.conv_norm_out(x)
    x = encoder.conv_act(x)
    moments = encoder.conv_out(x)
    latents = moments[:, :128]
    
    print("Extracted latents. Now processing DECODER...")
    sample = latents
    if hasattr(decoder, 'per_channel_statistics'):
        sample = decoder.per_channel_statistics.un_normalize(sample)
        
    sample = decoder.conv_in(sample, causal=decoder.causal)
    print(f"Feature shape after dec conv_in: {sample.shape}")
    factors = get_unroll_factors(orig_H, sample.shape[3])
    visualize_intermediate_features_with_pca(sample, p_h_list=factors, p_w_list=factors, save_prefix=os.path.join(OUTPUT_DIR, "pca_dec_conv_in"))
    
    for i, up_block in enumerate(decoder.up_blocks):
        kwargs = {}
        if isinstance(up_block, UNetMidBlock3D):
            kwargs['causal'] = decoder.causal
            kwargs['timestep'] = None
            kwargs['generator'] = None
        elif isinstance(up_block, ResnetBlock3D):
            kwargs['causal'] = decoder.causal
            kwargs['generator'] = None
        else:
            kwargs['causal'] = decoder.causal
            
        sample = up_block(sample, **kwargs)
        print(f"Feature shape after dec block {i}: {sample.shape}")
        
        factors = get_unroll_factors(orig_H, sample.shape[3])
        try:
            visualize_intermediate_features_with_pca(sample, p_h_list=factors, p_w_list=factors, save_prefix=os.path.join(OUTPUT_DIR, f"pca_dec_block_{i}"))
        except Exception as e:
            print(f"Could not PCA dec block {i}: {e}")
    
if __name__ == "__main__":
    main()
