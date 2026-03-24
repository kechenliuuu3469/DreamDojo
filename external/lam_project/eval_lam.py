"""
LAM Evaluation Script — Compute PSNR, SSIM, LPIPS, CLIP Score
===============================================================

Evaluates LAM checkpoints on test datasets (Bridge V2, DROID).

Metrics:
    PSNR  — Peak Signal-to-Noise Ratio (higher = better)
    SSIM  — Structural Similarity Index (higher = better, max 1.0)
    LPIPS — Learned Perceptual Image Patch Similarity (lower = better)
    CLIP  — CLIP cosine similarity between GT and reconstructed frame (higher = better, max 1.0)

Prerequisites:
    pip install lpips torchmetrics pytorch_msssim transformers --break-system-packages

Usage:
    # Single checkpoint, single dataset
    python eval_lam.py \
        --ckpt /path/to/checkpoint.ckpt \
        --data /path/to/bridge/videos/test \
        --num_samples 500

    # Full evaluation matrix
    python eval_lam.py --full_eval --num_samples 500
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from tqdm import tqdm


def load_lam_model(ckpt_path: str, device: str = "cuda:0"):
    """Load a LAM model from a Lightning checkpoint."""
    sys.path.insert(0, os.environ.get("LAM_PROJECT_DIR", "."))
    from lam.model import LAM

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    hparams = ckpt.get("hyper_parameters", {})
    model = LAM(
        image_channels=hparams.get("image_channels", 3),
        lam_model_dim=hparams.get("lam_model_dim", 1024),
        lam_latent_dim=hparams.get("lam_latent_dim", 32),
        lam_patch_size=hparams.get("lam_patch_size", 16),
        lam_enc_blocks=hparams.get("lam_enc_blocks", 24),
        lam_dec_blocks=hparams.get("lam_dec_blocks", 24),
        lam_num_heads=hparams.get("lam_num_heads", 16),
        beta=hparams.get("beta", 0.000001),
    )

    state_dict = ckpt["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    print(f"  Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ============================================================
# Load frame pairs from video files
# ============================================================

def load_frame_pair_from_video(
    video_path: str, 
    start_frame: int = 0, 
    frame_skip: int = 1,
    rgb_skip: int = 1,  # preprocessing skip (e.g., 3 for DROID)
) -> Tensor:
    """
    Load a pair of frames from a video file.
    Returns tensor of shape [2, 240, 320, 3] in float [0, 1].
    Same preprocessing as dataset.py (center crop to 4:3 + resize to 240x320).
    
    Args:
        video_path: Path to video file
        start_frame: Starting frame in effective (post-rgb_skip) space
        frame_skip: Gap between the two frames in effective space
        rgb_skip: Preprocessing skip applied before sampling (e.g., 3 for DROID)
    """
    cap = cv.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    # Compute effective frames after rgb_skip
    effective_frames = total_frames // max(rgb_skip, 1)

    if effective_frames < start_frame + frame_skip + 1:
        cap.release()
        return None

    # Convert to raw frame indices
    raw_start = start_frame * rgb_skip
    cap.set(cv.CAP_PROP_POS_FRAMES, raw_start)

    frames = []
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    frames.append(torch.from_numpy(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))

    # Skip frames: need to advance by (frame_skip * rgb_skip) raw frames
    # We already read 1 frame, so skip (frame_skip * rgb_skip - 1) more
    for _ in range(frame_skip * rgb_skip - 1):
        cap.read()

    # Read second frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    frames.append(torch.from_numpy(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))

    cap.release()

    video = torch.stack(frames).float() / 255.0  # [2, H, W, 3]

    # Center crop to 4:3 ratio
    target_ratio = 640 / 480
    h, w = video.shape[1], video.shape[2]
    if w / h > target_ratio:
        target_height = h
        target_width = int(h * target_ratio)
    elif w / h < target_ratio:
        target_height = int(w / target_ratio)
        target_width = w
    else:
        target_height = h
        target_width = w
    h_crop = (h - target_height) // 2
    w_crop = (w - target_width) // 2
    video = video[:, h_crop:h_crop + target_height, w_crop:w_crop + target_width]

    # Resize to 240x320
    video = rearrange(video, "t h w c -> c t h w")
    video = F.interpolate(video, (240, 320), mode="bilinear")
    video = rearrange(video, "c t h w -> t h w c")

    return video  # [2, 240, 320, 3]


def collect_video_files(dataset_path: str) -> list:
    """Find all MP4 files in a dataset directory."""
    mp4_list = sorted(Path(dataset_path).rglob("*.mp4"))

    filtered = [
        f for f in mp4_list
        if "left" not in str(f).lower()
        and "right" not in str(f).lower()
        and "resize" not in str(f).lower()
        and "pad" not in str(f).lower()
    ]

    # For DROID, only use view 0
    if "droid" in dataset_path.lower():
        filtered = [f for f in filtered if f.name in ("0.mp4", "1.mp4", "2.mp4")]

    return filtered


class MetricModels:
    """Holds all metric models so they're initialized once."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device

        print("Initializing metric models...")

        # PSNR (from torchmetrics, same as the reference script)
        from torchmetrics.image import PeakSignalNoiseRatio
        self.psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)

        # SSIM (from pytorch_msssim, same as the reference script)
        from pytorch_msssim import SSIM
        self.ssim_fn = SSIM(data_range=1.0, size_average=False, channel=3).to(device)

        # LPIPS (from torchmetrics, same as the reference script)
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        self.lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

        # CLIP (using HuggingFace transformers — no dependency on vidwm)
        from transformers import CLIPModel, CLIPProcessor
        print("  Loading CLIP model (openai/clip-vit-base-patch32)...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        print("  All metric models loaded.")

    def compute_clip_similarity(self, gt_frame: Tensor, recon_frame: Tensor) -> float:
        """
        Compute CLIP cosine similarity between GT and reconstructed frame.
        Both inputs: [H, W, 3] float tensors in [0, 1].
        Returns: cosine similarity (float, higher = better, max 1.0).
        """
        # Convert to uint8 PIL-like format for CLIP processor
        gt_np = (gt_frame.cpu().clamp(0, 1) * 255).byte().numpy()
        recon_np = (recon_frame.cpu().clamp(0, 1) * 255).byte().numpy()

        # Process both images
        inputs = self.clip_processor(
            images=[gt_np, recon_np],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            # Normalize
            image_features = F.normalize(image_features, dim=-1)
            # Cosine similarity between GT and recon
            cos_sim = F.cosine_similarity(
                image_features[0:1], image_features[1:2], dim=-1
            ).item()

        return cos_sim


# ============================================================
# Compute all metrics for one frame pair
# ============================================================

def compute_metrics(gt_frame: Tensor, recon_frame: Tensor, metric_models: MetricModels) -> dict:
    """
    Compute PSNR, SSIM, LPIPS, CLIP between GT and reconstructed frame.
    Both inputs: [H, W, 3] float tensors in [0, 1].
    """
    device = metric_models.device

    # Reshape to [1, 3, H, W] for metric computation
    gt = rearrange(gt_frame, "h w c -> 1 c h w").clamp(0, 1).to(device)
    recon = rearrange(recon_frame, "h w c -> 1 c h w").clamp(0, 1).to(device)

    # PSNR
    psnr_val = metric_models.psnr_fn(recon, gt).item()

    # SSIM
    ssim_val = metric_models.ssim_fn(gt, recon).mean().item()

    # LPIPS
    lpips_val = metric_models.lpips_fn(recon, gt).item()

    # CLIP cosine similarity
    clip_val = metric_models.compute_clip_similarity(gt_frame, recon_frame)

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "lpips": lpips_val,
        "clip": clip_val,
    }


# ============================================================
# Evaluate one checkpoint on one dataset
# ============================================================

@torch.no_grad()
def evaluate_checkpoint_on_dataset(
    model,
    dataset_path: str,
    metric_models: MetricModels,
    num_samples: int = 500,
    frame_skip: int = 1,
    rgb_skip: int = 1,  # preprocessing skip (e.g., 3 for DROID)
    device: str = "cuda:0",
) -> dict:
    """Evaluate a single checkpoint on a single dataset."""

    video_files = collect_video_files(dataset_path)
    if len(video_files) == 0:
        raise ValueError(f"No video files found in {dataset_path}")

    num_samples = min(num_samples, len(video_files))
    indices = np.linspace(0, len(video_files) - 1, num_samples, dtype=int)

    all_metrics = {"psnr": [], "ssim": [], "lpips": [], "clip": []}
    num_failed = 0

    for i in tqdm(indices, desc=f"  Eval on {Path(dataset_path).name}"):
        video_path = video_files[i]

        video = load_frame_pair_from_video(
            str(video_path), 
            start_frame=0, 
            frame_skip=frame_skip,
            rgb_skip=rgb_skip,
        )
        if video is None:
            num_failed += 1
            continue

        # Model forward pass
        batch = {"videos": video.unsqueeze(0).to(device)}  # [1, 2, 240, 320, 3]
        outputs = model.lam(batch)
        recon = outputs["recon"][0].cpu()  # [1, 240, 320, 3]

        gt_frame2 = video[1]       # [240, 320, 3]
        recon_frame2 = recon[0]    # [240, 320, 3]

        metrics = compute_metrics(gt_frame2, recon_frame2, metric_models)

        for k in all_metrics:
            all_metrics[k].append(metrics[k])

    if len(all_metrics["psnr"]) == 0:
        return {"error": "All samples failed to load"}

    result = {}
    for k in all_metrics:
        result[f"{k}_mean"] = float(np.mean(all_metrics[k]))
        result[f"{k}_std"] = float(np.std(all_metrics[k]))
    result["num_samples"] = len(all_metrics["psnr"])
    result["num_failed"] = num_failed

    return result


# ============================================================
# Full evaluation matrix
# ============================================================

def run_full_evaluation(
    ckpt_paths: list,
    ckpt_names: list,
    dataset_paths: list,
    dataset_names: list,
    rgb_skips: list = None,  # preprocessing skip per dataset (e.g., [1, 3] for Bridge, DROID)
    num_samples: int = 500,
    frame_skip: int = 1,
    device: str = "cuda:0",
    output_dir: str = "eval_results",
):
    os.makedirs(output_dir, exist_ok=True)

    # Default rgb_skips to 1 for all datasets
    if rgb_skips is None:
        rgb_skips = [1] * len(dataset_paths)

    # Initialize all metric models once
    metric_models = MetricModels(device=device)

    results = {}

    for ckpt_path, ckpt_name in zip(ckpt_paths, ckpt_names):
        print(f"\n{'='*70}")
        print(f"Checkpoint: {ckpt_name}")
        print(f"  Path: {ckpt_path}")
        print(f"{'='*70}")

        model = load_lam_model(ckpt_path, device=device)
        results[ckpt_name] = {}

        for dataset_path, dataset_name, rgb_skip in zip(dataset_paths, dataset_names, rgb_skips):
            print(f"\n  Evaluating on: {dataset_name} (rgb_skip={rgb_skip})")

            metrics = evaluate_checkpoint_on_dataset(
                model=model,
                dataset_path=dataset_path,
                metric_models=metric_models,
                num_samples=num_samples,
                frame_skip=frame_skip,
                rgb_skip=rgb_skip,
                device=device,
            )
            results[ckpt_name][dataset_name] = metrics

            if "error" not in metrics:
                print(f"    PSNR:  {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
                print(f"    SSIM:  {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
                print(f"    LPIPS: {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")
                print(f"    CLIP:  {metrics['clip_mean']:.4f} ± {metrics['clip_std']:.4f}")
                print(f"    Samples: {metrics['num_samples']} (failed: {metrics['num_failed']})")
            else:
                print(f"    ERROR: {metrics['error']}")

        del model
        torch.cuda.empty_cache()

    # ============================================================
    # Print results table
    # ============================================================

    metric_keys = ["psnr", "ssim", "lpips", "clip"]
    metric_labels = {"psnr": "PSNR↑", "ssim": "SSIM↑", "lpips": "LPIPS↓", "clip": "CLIP↑"}

    print(f"\n\n{'='*120}")
    print("RESULTS TABLE")
    print(f"{'='*120}\n")

    # Header
    header = f"{'Checkpoint':<30}"
    for dn in dataset_names:
        for mk in metric_keys:
            header += f" | {dn} {metric_labels[mk]:>10}"
    print(header)
    print("-" * len(header))

    # Rows
    for ckpt_name in ckpt_names:
        row = f"{ckpt_name:<30}"
        for dataset_name in dataset_names:
            m = results[ckpt_name][dataset_name]
            if "error" not in m:
                for mk in metric_keys:
                    row += f" | {m[f'{mk}_mean']:>10.3f}"
            else:
                for mk in metric_keys:
                    row += f" | {'ERROR':>10}"
        print(row)

    # Save JSON
    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved to: {json_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, "eval_results.csv")
    with open(csv_path, "w") as f:
        cols = ["Checkpoint"]
        for dn in dataset_names:
            for mk in metric_keys:
                cols.append(f"{dn} {metric_labels[mk]}")
        f.write(",".join(cols) + "\n")

        for ckpt_name in ckpt_names:
            vals = [ckpt_name]
            for dataset_name in dataset_names:
                m = results[ckpt_name][dataset_name]
                if "error" not in m:
                    for mk in metric_keys:
                        vals.append(f"{m[f'{mk}_mean']:.4f}")
                else:
                    vals.extend(["ERROR"] * len(metric_keys))
            f.write(",".join(vals) + "\n")
    print(f"CSV saved to: {csv_path}")

    return results


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LAM checkpoints")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--ckpts", nargs="+", type=str, default=None)
    parser.add_argument("--ckpt_names", nargs="+", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", type=str, default=None)
    parser.add_argument("--dataset_names", nargs="+", type=str, default=None)
    parser.add_argument("--rgb_skips", nargs="+", type=int, default=None, 
                        help="Preprocessing frame skip per dataset (e.g., 1 for Bridge, 3 for DROID)")
    parser.add_argument("--full_eval", action="store_true")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--rgb_skip", type=int, default=1, 
                        help="Preprocessing frame skip for single dataset eval")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.full_eval:
      
        ckpt_paths = [
            "/n/fs/geniemodel/DreamDojo/checkpoints/LAM/LAM_400k.ckpt",
            "/n/fs/geniemodel/DreamDojo/external/lam_project/exp_ckpts_bridge_full/last.ckpt",
            "/n/fs/geniemodel/DreamDojo/external/lam_project/exp_ckpts_droid_full/last.ckpt",
            "/n/fs/geniemodel/DreamDojo/external/lam_project/exp_ckpts_bridge_droid_full/last.ckpt",
        ]
        ckpt_names = [
            "Original LAM_400k",
            "Fine-tuned Bridge",
            "Fine-tuned DROID",
            "Fine-tuned Bridge+DROID",
        ]
        dataset_paths = [
            "/n/fs/not-fmrl/Projects/wm_alignment/cosmos-predict2/datasets/bridge/videos/test",
            "/n/fs/iromdata/droid_ctrl_world/videos/val",
        ]
        dataset_names = [
            "Bridge V2",
            "DROID",
        ]
        # Bridge: rgb_skip=1 (no preprocessing skip)
        # DROID: rgb_skip=3 (skip every 3 frames for consistent sampling)
        rgb_skips = [1, 3]

        run_full_evaluation(
            ckpt_paths=ckpt_paths,
            ckpt_names=ckpt_names,
            dataset_paths=dataset_paths,
            dataset_names=dataset_names,
            rgb_skips=rgb_skips,
            num_samples=args.num_samples,
            frame_skip=args.frame_skip,
            device=args.device,
            output_dir=args.output_dir,
        )

    elif args.ckpts is not None:
        ckpt_names = args.ckpt_names or [Path(p).stem for p in args.ckpts]
        dataset_names = args.dataset_names or [Path(p).name for p in args.datasets]
        rgb_skips = args.rgb_skips or [1] * len(args.datasets)

        run_full_evaluation(
            ckpt_paths=args.ckpts,
            ckpt_names=ckpt_names,
            dataset_paths=args.datasets,
            dataset_names=dataset_names,
            rgb_skips=rgb_skips,
            num_samples=args.num_samples,
            frame_skip=args.frame_skip,
            device=args.device,
            output_dir=args.output_dir,
        )

    elif args.ckpt is not None and args.data is not None:
        metric_models = MetricModels(device=args.device)
        model = load_lam_model(args.ckpt, device=args.device)

        metrics = evaluate_checkpoint_on_dataset(
            model=model,
            dataset_path=args.data,
            metric_models=metric_models,
            num_samples=args.num_samples,
            frame_skip=args.frame_skip,
            rgb_skip=args.rgb_skip,
            device=args.device,
        )

        print(f"\nResults:")
        print(f"  PSNR:  {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
        print(f"  SSIM:  {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        print(f"  LPIPS: {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")
        print(f"  CLIP:  {metrics['clip_mean']:.4f} ± {metrics['clip_std']:.4f}")
        print(f"  Samples: {metrics['num_samples']} (failed: {metrics['num_failed']})")

    else:
        print("Usage:")
        print("  Single:  python eval_lam.py --ckpt /path/to/ckpt --data /path/to/videos --rgb_skip 1")
        print("  Full:    python eval_lam.py --full_eval --num_samples 500")
        print("  Custom:  python eval_lam.py --ckpts ckpt1 ckpt2 --datasets data1 data2")