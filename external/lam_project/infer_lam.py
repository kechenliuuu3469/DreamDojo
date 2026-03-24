"""
LAM Inference Script
====================
Extracts 32-dim continuous latent actions from video pairs
using a pretrained DreamDojo LAM checkpoint.

Usage:
    cd /n/fs/geniemodel/DreamDojo/external/lam_project
    export PYTHONPATH=$(pwd):$PYTHONPATH
    
    python infer_lam.py \
        --ckpt_path /path/to/lam_checkpoint.ckpt \
        --video_dir /n/fs/geniemodel/DreamDojo/datasets/train \
        --num_videos 5 \
        --save_dir lam_inference_output
"""

import argparse
import os
from pathlib import Path

import cv2
import torch
import numpy as np
from einops import rearrange
from PIL import Image

from lam.model import LAM
from lam.dataset import VideoDataset


def load_model(ckpt_path, device="cuda"):
    """Load LAM from a Lightning checkpoint."""
    print(f"Loading checkpoint from: {ckpt_path}")
    model = LAM.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.to(device)
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model


def load_video_pair(video_path, start_frame=0):
    """Load two consecutive frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(2):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame))
        else:
            # Repeat last frame if video is too short
            frames.append(frames[-1].clone())
    cap.release()

    video = torch.stack(frames).float() / 255.0  # [2, H, W, 3]

    # Center crop and resize to 240x320 (same as training)
    target_ratio = 640 / 480
    h, w = video.shape[1], video.shape[2]
    if w / h > target_ratio:
        target_h = h
        target_w = int(h * target_ratio)
    elif w / h < target_ratio:
        target_h = int(w / target_ratio)
        target_w = w
    else:
        target_h, target_w = h, w

    h_crop = (h - target_h) // 2
    w_crop = (w - target_w) // 2
    video = video[:, h_crop:h_crop + target_h, w_crop:w_crop + target_w]
    video = rearrange(video, "t h w c -> c t h w")
    video = torch.nn.functional.interpolate(video, (240, 320), mode="bilinear")
    video = rearrange(video, "c t h w -> t h w c")

    return video, total_frames


@torch.no_grad()
def extract_latent_actions(model, video_tensor, device="cuda"):
    """
    Extract latent action from a pair of frames.
    
    Args:
        model: LAM model
        video_tensor: [2, 240, 320, 3] float tensor
    
    Returns:
        z_mu: [32] latent action vector (the mean, used at inference)
        reconstruction: [1, 240, 320, 3] reconstructed second frame
    """
    batch = {"videos": video_tensor.unsqueeze(0).to(device)}  # [1, 2, 240, 320, 3]
    outputs = model.lam(batch)

    z_mu = outputs["z_mu"].squeeze().cpu()        # [32]
    recon = outputs["recon"].squeeze().cpu()       # [1, 240, 320, 3]

    return z_mu, recon


def extract_full_trajectory(model, video_path, device="cuda", save_dir=None, video_idx=0):
    """
    Extract latent actions for ALL consecutive frame pairs in a video.
    
    Args:
        model: LAM model
        video_path: Path to video file
        device: Device to use
        save_dir: If provided, save visualizations for each frame pair
        video_idx: Video index for naming saved files
    
    Returns:
        latent_actions: [N-1, 32] tensor of latent actions
        where N is the number of frames in the video
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_latent_actions = []

    for start in range(total_frames - 1):
        video, _ = load_video_pair(video_path, start_frame=start)
        z_mu, recon = extract_latent_actions(model, video, device)
        all_latent_actions.append(z_mu)
        
        # Save visualization for each frame pair
        if save_dir is not None:
            vis_path = os.path.join(save_dir, f"video_{video_idx:03d}_frame_{start:04d}_comparison.png")
            save_visualization(video[0], video[1], recon, z_mu, vis_path)

    cap.release()

    if len(all_latent_actions) > 0:
        return torch.stack(all_latent_actions)  # [N-1, 32]
    else:
        return torch.zeros(0, 32)


def save_visualization(frame1, frame2, recon, z_mu, save_path):
    """Save a side-by-side visualization: frame1 | frame2 (GT) | frame2 (reconstructed)"""
    frame1_np = (frame1.clamp(0, 1).numpy() * 255).astype(np.uint8)
    frame2_np = (frame2.clamp(0, 1).numpy() * 255).astype(np.uint8)
    recon_np = (recon.squeeze().clamp(0, 1).numpy() * 255).astype(np.uint8)

    # Create comparison image
    comparison = np.concatenate([frame1_np, frame2_np, recon_np], axis=1)  # side by side

    img = Image.fromarray(comparison)
    img.save(save_path)


def main():
    parser = argparse.ArgumentParser(description="LAM Inference")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to LAM checkpoint (.ckpt)")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Path to directory containing MP4 videos")
    parser.add_argument("--num_videos", type=int, default=5,
                        help="Number of videos to process")
    parser.add_argument("--save_dir", type=str, default="lam_inference_output",
                        help="Where to save outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--extract_full", action="store_true",
                        help="Extract latent actions for all frame pairs (not just first)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    model = load_model(args.ckpt_path, args.device)

    # Find videos
    video_files = sorted(Path(args.video_dir).rglob("*.mp4"))[:args.num_videos]
    print(f"Found {len(video_files)} videos to process")

    all_latent_actions = []

    for i, video_path in enumerate(video_files):
        print(f"\n{'='*60}")
        print(f"Video {i+1}/{len(video_files)}: {video_path}")

        if args.extract_full:
            # Extract latent actions for entire trajectory
            latent_actions = extract_full_trajectory(model, video_path, args.device, 
                                                     save_dir=args.save_dir, video_idx=i)
            print(f"  Trajectory length: {latent_actions.shape[0]} frame pairs")
            print(f"  Latent action shape: {latent_actions.shape}")
            print(f"  Latent action stats:")
            print(f"    Mean: {latent_actions.mean(dim=0)[:5].tolist()}... (first 5 dims)")
            print(f"    Std:  {latent_actions.std(dim=0)[:5].tolist()}... (first 5 dims)")
            print(f"    Range: [{latent_actions.min():.4f}, {latent_actions.max():.4f}]")

            # Save latent actions as numpy
            np.save(
                os.path.join(args.save_dir, f"video_{i:03d}_latent_actions.npy"),
                latent_actions.numpy()
            )
            all_latent_actions.append(latent_actions)

        else:
            # Extract just the first frame pair
            video, total_frames = load_video_pair(video_path, start_frame=0)
            z_mu, recon = extract_latent_actions(model, video, args.device)

            print(f"  Total frames in video: {total_frames}")
            print(f"  Input shape: {video.shape}")
            print(f"  Latent action (z_mu): {z_mu.shape}")
            print(f"  Latent action values: {z_mu[:8].tolist()}... (first 8 of 32 dims)")
            print(f"  Latent action range: [{z_mu.min():.4f}, {z_mu.max():.4f}]")
            print(f"  Reconstruction shape: {recon.shape}")

            # Save visualization
            vis_path = os.path.join(args.save_dir, f"video_{i:03d}_comparison.png")
            save_visualization(video[0], video[1], recon, z_mu, vis_path)
            print(f"  Saved visualization to: {vis_path}")

            # Save latent action
            np.save(
                os.path.join(args.save_dir, f"video_{i:03d}_latent_action.npy"),
                z_mu.numpy()
            )
            all_latent_actions.append(z_mu)

    # Print summary statistics across all videos
    if all_latent_actions:
        if args.extract_full:
            all_z = torch.cat(all_latent_actions, dim=0)
        else:
            all_z = torch.stack(all_latent_actions)

        print(f"\n{'='*60}")
        print(f"SUMMARY ACROSS ALL VIDEOS")
        print(f"{'='*60}")
        print(f"Total latent actions extracted: {all_z.shape[0]}")
        print(f"Latent dimension: {all_z.shape[1]}")
        print(f"Per-dimension mean: {all_z.mean(dim=0)[:8].tolist()}...")
        print(f"Per-dimension std:  {all_z.std(dim=0)[:8].tolist()}...")
        print(f"Overall range: [{all_z.min():.4f}, {all_z.max():.4f}]")

        # Save all latent actions
        np.save(
            os.path.join(args.save_dir, "all_latent_actions.npy"),
            all_z.numpy()
        )
        print(f"\nSaved all latent actions to: {args.save_dir}/all_latent_actions.npy")


if __name__ == "__main__":
    main()