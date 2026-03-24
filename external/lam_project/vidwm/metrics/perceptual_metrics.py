# import os
# from pathlib import Path
# from typing import Any, Dict
# import mediapy as mp
# import torch
# from tqdm import tqdm
# import numpy as np
# import json
# import einops
# from natsort import natsorted
# from omegaconf import OmegaConf
# from omegaconf.dictconfig import DictConfig
# from hydra.utils import instantiate
# import traceback

# # metrics
# from pytorch_msssim import SSIM
# from torchmetrics.image import PeakSignalNoiseRatio
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# from vidwm.encoders.clip_encoder import CLIPNetwork, CLIPNetworkConfig
# from vidwm.metrics.base_metrics import BaseMetricConfig, BaseMetric


# class PerceptualMetricsConfig(BaseMetricConfig):
#     # option to enable verbose print
#     verbose_print: bool = False
    
#     # base output path
#     base_output_path: Path | str = "outputs/metrics/"
    

# class PerceptualMetrics(BaseMetric):
#     def __init__(
#         self,
#         cfg: PerceptualMetricsConfig |  Dict[str, Any] | OmegaConf | Path | str  = PerceptualMetricsConfig(),
#     ):
#         # init super-class
#         super().__init__(cfg=cfg)
        
#         # unpack config
#         self.eval_cfg = self.config.eval_config
        
#         # set up for evaluation
#         self.setup_eval()
        
#     def setup_eval(self):
#         """
#         Initialize for evaluation
        
#         :param self: Description
#         """
#         assert self.eval_cfg.get("input_video", None) is not None, "A path to the generated videos is required!"
        
#         # input directory
#         # self.input_dir = Path(f"{self.eval_cfg.input_video}_{self.eval_cfg.split}")
#         self.input_dir = Path(f"{self.eval_cfg.input_video}")
        
#         # make output directory, if necessary
#         self.output_dir = self.eval_cfg.get("output_dir", None)
        
#         if self.output_dir is None:
#             self.output_dir = Path(__file__).parent.parent / "scripts/outputs/perceptual_metrics"
#         else:
#             self.output_dir = Path(self.output_dir)
        
#         self.output_dir = Path(f"{self.output_dir}/{self.input_dir.stem}")
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         # output path for the evaluation config file
#         cfg_output_path = self.output_dir / "config.yaml"
        
#         # save the config file
#         with open(cfg_output_path, "w") as fh:
#             OmegaConf.save(self.eval_cfg, fh)
            
#             # log
#             self.console.log(f"Saving evaluation config to {cfg_output_path}!")
            
#         if self.eval_cfg.get("datasets", None) is not None:
#             # load the dataset 
#             self.load_dataset()
        
#     def load_dataset(self):
#         """
#         Load evaluation dataset
        
#         :param self: Description
#         """
#         # set up the datasets
#         dset_cfg = self.eval_cfg.datasets
        
#         if isinstance(list(dset_cfg.values())[0], DictConfig):
#             # unpack the config params
#             dset_cfg = list(dset_cfg.values())[0]
            
#         # dataset split
#         split = dset_cfg.cfg.dset_config.get("split", "val")
            
#         # load train and validation dataset
#         self.eval_dataset = instantiate(dset_cfg, split=split)
        
#         # create evaluation dataloader
#         self.eval_dataloader = torch.utils.data.DataLoader(
#             self.eval_dataset,
#             batch_size=self.eval_cfg.get("eval_batch_size", 1),
#             shuffle=False,
#         )
        
#     def compute_video_metrics(
#         self,
#         video_embed_model, 
#         videos: list[torch.Tensor | Path | str],
#         normalize_embeds: bool = True,
#         device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#         compute_video_error: bool = False,  # compute error w.r.t ground-truth video
#         gt_video: torch.Tensor | Path | str = None,
#         clip_model: CLIPNetwork = CLIPNetwork(config=CLIPNetworkConfig()), # initialize CLIP
#     ):
#         # infer type of video input
#         if isinstance(videos[0], torch.Tensor):
#             # input video type
#             input_vid_path = False
#         else:
#             # type-checking
#             assert isinstance(videos[0], Path) or isinstance(videos[0], str), "Input videos must be tensor or Path-like object!"
            
#             # input video type
#             input_vid_path = True
            
#         with torch.no_grad():
            
#             if compute_video_error:
#                 # lazy import
#                 import torchvision.transforms as TF 
                
#                 if input_vid_path:
#                     # load the ground-truth video
#                     gt_vid = self._load_video(
#                         video_path=gt_video,
#                         map_to_float=False,
#                         device=device,
#                     ).permute(0, 2, 1, 3, 4) # video in (B, T, C, H, W)
#                 else:
#                     gt_vid = gt_video
                
#                 # compute the video embedding
#                 gt_vid_emb = video_embed_model(gt_vid) if video_embed_model is not None else None

#                 # initialize video metrics
#                 psnr = PeakSignalNoiseRatio(data_range=1.0, reduction=None).to(device)
#                 ssim = SSIM(data_range=1.0, size_average=False, channel=3).to(device)
#                 lpips = LearnedPerceptualImagePatchSimilarity(normalize=True, reduction=None).to(device)
                
#                 # initialize the torchvision transforms
#                 video_resize_transform = TF.Resize(size=gt_vid.shape[-2:], interpolation=TF.InterpolationMode.BILINEAR)
                
                
#             # video embeddings for generated videos
#             video_embeds = []
            
#             # error with respect to the ground-truth video emebeddings
#             video_error = {
#                 "s3d_embeds": [],
#                 "ssim": [],
#                 "ssim_mean": [],
#                 "psnr": [],
#                 "psnr_mean": [],
#                 "lpips": [],
#                 "lpips_mean": [],
#                 "clip": [],
#                 "mse": [],
#             }
            
#             # TODO: Refactor
#             # ground-truth CLIP embeds
#             gt_clip_embeds = None
            
#             # load and compute video embeddings
#             for vid in tqdm(videos, desc="Computing video embeddings"):
#                 if input_vid_path:
#                     # load video
#                     vid = self._load_video(
#                         video_path=vid,
#                         map_to_float=False,
#                         device=device,
#                     ).permute(0, 2, 1, 3, 4) # video in (B, T, C, H, W)
                    
#                 # compute video embeddings
#                 vid_emb = video_embed_model(vid) if video_embed_model is not None else None
                
#                 if compute_video_error:
#                     # error with respect to the semantic features
#                     video_error["s3d_embeds"].append((gt_vid_emb - vid_emb).pow(2).sum().item())
                    
#                     # smaller video length
#                     min_video_length = min(gt_vid.shape[1], vid.shape[1])
                    
#                     # reshape from [B, T, H, W, C] to [T_min, C, H, W] for SSIM, PSNR, LPIPS
#                     metric_gt_vid = gt_vid.squeeze(dim=0).to(device)
#                     metric_gen_vid = vid.squeeze(dim=0).to(device)
                    
#                     # subsample video, if necessary
#                     metric_gt_vid = metric_gt_vid[np.linspace(0, gt_vid.shape[1] - 1, min_video_length), ...]
#                     metric_gen_vid = metric_gen_vid[np.linspace(0, vid.shape[1] - 1, min_video_length), ...]
                    
#                     # resize the generated videos
#                     metric_gen_vid = torch.stack([
#                         video_resize_transform(frame)
#                         for frame in metric_gen_vid
#                     ], dim=0).to(device)
                    
#                     # TODO: Refactor
#                     if metric_gt_vid.max() > 1.05:  # with a small buffer
#                         # map to data range [0, 1]
#                         metric_gt_vid = metric_gt_vid.float() / 255.0
                        
#                         # clip the data range
#                         metric_gt_vid = torch.clamp(metric_gt_vid, min=0, max=1)
                    
#                     if metric_gen_vid.max() > 1.05:  # with a small buffer
#                         # map to data range [0, 1]
#                         metric_gen_vid = metric_gen_vid.float() / 255.0
                        
#                         # clip the data range
#                         metric_gen_vid = torch.clamp(metric_gen_vid, min=0, max=1)
                        
#                     # compute the SSIM, PSNR, LPIPS
#                     ssim_val = ssim(metric_gt_vid, metric_gen_vid).detach().cpu().numpy().tolist()
#                     psnr_val = psnr(metric_gt_vid, metric_gen_vid).detach().cpu().numpy().tolist()
#                     lpips_val = lpips(metric_gt_vid, metric_gen_vid).detach().cpu().numpy().tolist()

#                     # compute average LPIPS over the video frames
#                     ssim_mean_val = float(np.mean(ssim_val))
#                     psnr_mean_val = float(np.mean(psnr_val))
#                     lpips_mean_val = float(np.mean(lpips_val))
                    
#                     # append the results
#                     video_error["ssim"].append(ssim_val)
#                     video_error["ssim_mean"].append(ssim_mean_val)
#                     video_error["psnr"].append(psnr_val)
#                     video_error["psnr_mean"].append(psnr_mean_val)
#                     video_error["lpips"].append(lpips_val)
#                     video_error["lpips_mean"].append(lpips_mean_val)
                    
#                     # compute CLIP emebeddings
#                     if gt_clip_embeds is None:
#                         gt_clip_embeds = clip_model.encode_image(image_list=metric_gt_vid)[1]
                    
#                     # compute gen video CLIP embeds
#                     gen_vid_clip_embeds = clip_model.encode_image(image_list=metric_gen_vid)[1]
                    
#                     # compute the cosine similarity (for the image embeddings)
#                     cos_sim_clip = torch.nn.functional.cosine_similarity(
#                         gt_clip_embeds.reshape(-1, gt_clip_embeds.shape[-1]), 
#                         gen_vid_clip_embeds.reshape(-1, gen_vid_clip_embeds.shape[-1]),
#                         dim=-1
#                     )   
                    
#                     # append the results
#                     video_error["clip"].append(cos_sim_clip.detach().cpu().numpy().tolist())
                    
#                     # compute the MSE
#                     mse_val = torch.nn.functional.mse_loss(metric_gt_vid, metric_gen_vid).item()
#                     video_error["mse"].append([mse_val])
                    
#                 # append results
#                 video_embeds.append(vid_emb)
        
#             # stack video embeds
#             video_embeds = torch.concatenate(video_embeds, dim=0).to(device)
            
#             if normalize_embeds:
#                 # normalize embeddings
#                 video_embeds = torch.nn.functional.normalize(video_embeds, dim=-1)

#         # output
#         output = {
#             "video_embeds": video_embeds,
#             "video_error": video_error,
#         }

#         # print("Video embeddings shape: ", video_embeds.shape)
#         # print("Video error metrics: ", video_error)

#         return output
    
#     def evaluate(self):
#         """
#         Computes the metrics
#         """
#         self._eval_metrics()
        
#     def _eval_metrics(self):
#         """
#         Evaluates the metrics.
#         """

#         # load the video embedding model
#         from vidwm.encoders.video_s3d import VideoS3DEncoderConfig, VideoS3DEncoder

#         # all eval videos
#         eval_videos = natsorted([in_path for in_path in self.input_dir.iterdir() if in_path.is_dir()])
        
#         # get the job assignment index for initial batch
#         job_idx = os.environ.get("EVAL_JOB_IDX", None)
        
#         if job_idx is None:
#             job_idx = 0
#         else:
#             job_idx = int(job_idx)
        
#         # get the job assignment size
#         job_size = os.environ.get("EVAL_JOB_SIZE", None)
        
#         if job_size is None:
#             job_size = len(eval_videos)
#         else:
#             job_size = int(job_size)
        
#         # get indices of samples for evaluation
#         dset_eval_idx = np.arange(job_size * job_idx, job_size * (job_idx + 1))
        
#         # filter indices
#         dset_eval_idx = dset_eval_idx[dset_eval_idx < len(eval_videos)]
        
#         # evaluate selected videos
#         eval_videos = np.array(eval_videos)[dset_eval_idx]
           
#         self.console.log(f"Evaluating metrics on job index: {job_idx} for {len(eval_videos)} batches...")
    
#         # instantiate video model
#         video_embed_model = VideoS3DEncoder(
#             config=VideoS3DEncoderConfig()
#         ).to(self.device)
        
#         # initialize CLIP
#         clip_model: CLIPNetwork = CLIPNetwork(config=CLIPNetworkConfig())
        
#         if self.eval_cfg.compute_inference_time_metrics:
#             inference_times = []

#         for in_vid in tqdm(eval_videos, desc="Running eval on the dataset"):
#             try:
#                 # full path to input video
#                 in_video_path = in_vid / "videos/pred_rgb.mp4"
                
#                 if not in_video_path.exists(): 
#                     if self.eval_cfg.check_videos:
#                         # error message
#                         err_msg = f"Input video not found: {in_video_path}, terminating..."
                        
#                         # log
#                         self.console.log(err_msg)
#                         raise FileNotFoundError(err_msg)
#                     else:
#                         self.console.log(f"Input video not found: {in_video_path}, skipping...")
#                         continue
                    
#                 # load the input video
#                 input_vid = self._load_video(
#                     video_path=in_video_path,
#                     map_to_float=False,
#                     device=self.device,
#                 ).permute(0, 2, 1, 3, 4) # video in (B, T, C, H, W)
                
#                 # video shape
#                 B, T, C, H, W = input_vid.shape
                
#                 # partition input videos into ground-truth and generated videos
#                 gt_video = input_vid[:, :, :, :H // 2, :]
#                 gen_video = input_vid[:, :, :, H // 2:, :]
                
#                 # compute the metrics for each camera view
#                 num_cam_views = self.eval_cfg.get("num_cam_views", 1)
                
#                 # video width per camera view
#                 cam_width = W // num_cam_views
                
#                 for cam_idx in range(num_cam_views):
#                     # path to save the metrics
#                     save_metrics_path = self.output_dir / f"{in_vid.stem}_cam_{cam_idx}_metrics.json"
                    
#                     # ground-truth video for camera
#                     cam_gt_video = gt_video[:, :, :, :, cam_idx * cam_width: (cam_idx + 1) * cam_width]
                    
#                     # generated video for camera
#                     cam_gen_video = gen_video[:, :, :, :, cam_idx * cam_width: (cam_idx + 1) * cam_width]
                    
#                     # compute video metrics
#                     video_metrics = self.compute_video_metrics(
#                         video_embed_model= video_embed_model,
#                         videos=[cam_gen_video],
#                         normalize_embeds=True,
#                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#                         compute_video_error=True,
#                         gt_video=cam_gt_video,
#                         clip_model=clip_model,
#                     )
                
#                     # outputs to save
#                     output_metrics = {}
#                     output_metrics["video_key"] = in_vid.stem
#                     output_metrics["cam_view"] = cam_idx
#                     output_metrics["video_path"] = str(in_video_path)
#                     output_metrics["video_error"] = video_metrics["video_error"]
                    
#                     # save metrics to file
#                     self._save_metrics_to_file(
#                         metrics=output_metrics,
#                         save_path=save_metrics_path,
#                     )
#             except (torch.OutOfMemoryError, RuntimeError) as excp:
#                 # log
#                 self.console.log(f"Error raised: {excp}")
                
#                 # log error
#                 traceback.print_exc()
                
#                 # clear memory
#                 torch.cuda.empty_cache()

#             # --- compute time metrics ---
#             stats_path = in_vid / "stats.json"
#             if self.eval_cfg.compute_inference_time_metrics and stats_path.exists():
#                 with open(stats_path, "r") as f:
#                     stats_dict = json.load(f)
                
#                 inference_times.extend(stats_dict["time"])

#         # compute average inference time metrics
#         if self.eval_cfg.compute_inference_time_metrics and len(inference_times) > 0:
#             avg_inference_time = float(np.mean(inference_times))
#             std_inference_time = float(np.std(inference_times))
            
#             # save to file
#             time_metrics_path = self.output_dir / f"inference_time_metrics_job_{job_idx}.json"
#             time_metrics = {
#                 "avg_inference_time": avg_inference_time,
#                 "std_inference_time": std_inference_time,
#                 "num_samples": len(inference_times),
#             }
#             self._save_metrics_to_file(
#                 metrics=time_metrics,
#                 save_path=time_metrics_path,
#             )
#             self.console.log(f"Saved inference time metrics to {time_metrics_path}!")
#             self.console.log(f"Average inference time: {avg_inference_time:.4f} sec, Std: {std_inference_time:.4f} sec over {len(inference_times)} samples.")

#     def _save_metrics_to_file(
#         self,
#         metrics: dict,
#         save_path: Path | str,
#     ):
#         """Saves the metrics to a JSON file."""
#         save_path = Path(save_path)
#         save_path.parent.mkdir(parents=True, exist_ok=True)
        
#         with open(save_path, "w") as f:
#             json.dump(metrics, f, indent=4)
        
#         self.console.log(f"Saved metrics to {save_path}")


# # =============================================================================
# # LAM Evaluation - Standalone functions for evaluating LAM checkpoints
# # =============================================================================
# """
# LAM Evaluation 1: Frame-to-Frame Prediction with Fixed Frame Skip
# ==================================================================

# For each frame_skip in [1, 2, 3, 4]:
#     For each video (num_samples videos):
#         Load ALL consecutive frame pairs with that frame_skip:
#             frame_skip=1: (0,1), (1,2), (2,3), ...
#             frame_skip=2: (0,2), (2,4), (4,6), ...
#             frame_skip=3: (0,3), (3,6), (6,9), ...
#             frame_skip=4: (0,4), (4,8), (8,12), ...

#         For each pair:
#             Encoder: (GT frame_t, GT frame_t+skip) → latent z
#             Decoder: GT frame_t + z → reconstructed frame_t+skip
#             Metrics: PSNR, SSIM, LPIPS, CLIP between GT and reconstructed

#         Average metrics across all pairs → per-video metrics
#     Average per-video metrics across all videos → final metrics for this frame_skip

# Output: one table per checkpoint, with rows for frame_skip 1-4

# Prerequisites:
#     pip install lpips torchmetrics pytorch_msssim --break-system-packages

# Usage:
#     python perceptual_metrics.py --lam_eval --num_samples 500
#     python perceptual_metrics.py --lam_eval --num_samples 100 --frame_skips 1 3
# """

# import argparse
# import json
# import os
# import sys
# from pathlib import Path

# import cv2 as cv
# import einops
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from tqdm import tqdm

# # metrics
# from pytorch_msssim import SSIM
# from torchmetrics.image import PeakSignalNoiseRatio
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# from vidwm.encoders.clip_encoder import CLIPNetwork, CLIPNetworkConfig


# # ============================================================
# # Load LAM model (same as existing script)
# # ============================================================

# def load_lam_model(ckpt_path: str, device: str = "cuda:0"):
#     """Load a LAM model from a Lightning checkpoint."""
#     sys.path.insert(0, str(Path(__file__).parent.parent))
#     from lam.model import LAM

#     print(f"Loading LAM checkpoint: {ckpt_path}")
#     ckpt = torch.load(ckpt_path, map_location="cpu")

#     hparams = ckpt.get("hyper_parameters", {})
#     model = LAM(
#         image_channels=hparams.get("image_channels", 3),
#         lam_model_dim=hparams.get("lam_model_dim", 1024),
#         lam_latent_dim=hparams.get("lam_latent_dim", 32),
#         lam_patch_size=hparams.get("lam_patch_size", 16),
#         lam_enc_blocks=hparams.get("lam_enc_blocks", 24),
#         lam_dec_blocks=hparams.get("lam_dec_blocks", 24),
#         lam_num_heads=hparams.get("lam_num_heads", 16),
#         beta=hparams.get("beta", 0.000001),
#     )

#     state_dict = ckpt["state_dict"]
#     model.load_state_dict(state_dict, strict=True)
#     model = model.to(device).eval()
#     print(f"  Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
#     return model


# # ============================================================
# # Load ALL frames from a video (preprocessed)
# # ============================================================

# def load_all_frames_from_video(video_path: str, rgb_skip: int = 1) -> Tensor:
#     """
#     Load all frames from a video, applying rgb_skip and preprocessing.
#     Returns tensor of shape [N, 240, 320, 3] in float [0, 1], or None if failed.
#     """
#     cap = cv.VideoCapture(str(video_path))
#     total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

#     if total_frames < 2:
#         cap.release()
#         return None

#     # Read frames with rgb_skip
#     frames = []
#     for raw_idx in range(0, total_frames, rgb_skip):
#         cap.set(cv.CAP_PROP_POS_FRAMES, raw_idx)
#         ret, frame = cap.read()
#         if ret:
#             frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#             frames.append(torch.from_numpy(frame))
#         else:
#             break
#     cap.release()

#     if len(frames) < 2:
#         return None

#     video = torch.stack(frames).float() / 255.0  # [N, H, W, 3]

#     # Center crop to 4:3 ratio
#     target_ratio = 640 / 480
#     h, w = video.shape[1], video.shape[2]
#     if w / h > target_ratio:
#         target_height = h
#         target_width = int(h * target_ratio)
#     elif w / h < target_ratio:
#         target_height = int(w / target_ratio)
#         target_width = w
#     else:
#         target_height, target_width = h, w
#     h_crop = (h - target_height) // 2
#     w_crop = (w - target_width) // 2
#     video = video[:, h_crop:h_crop + target_height, w_crop:w_crop + target_width]

#     # Resize to 240x320
#     video = einops.rearrange(video, "t h w c -> c t h w")
#     video = F.interpolate(video, (240, 320), mode="bilinear")
#     video = einops.rearrange(video, "c t h w -> t h w c")

#     return video  # [N, 240, 320, 3]


# def collect_video_files(dataset_path: str) -> list:
#     """Find all MP4 files in a dataset directory."""
#     mp4_list = sorted(Path(dataset_path).rglob("*.mp4"))

#     filtered = [
#         f for f in mp4_list
#         if "left" not in str(f).lower()
#         and "right" not in str(f).lower()
#         and "resize" not in str(f).lower()
#         and "pad" not in str(f).lower()
#     ]

#     if "droid" in dataset_path.lower():
#         filtered = [f for f in filtered if f.name in ("0.mp4", "1.mp4", "2.mp4")]

#     return filtered


# # ============================================================
# # Evaluate one checkpoint on one dataset with one frame_skip
# # ============================================================

# @torch.no_grad()
# def evaluate_frame2frame(
#     model,
#     dataset_path: str,
#     frame_skip: int = 1,
#     rgb_skip: int = 1,
#     num_samples: int = 500,
#     device: str = "cuda:0",
#     psnr_fn=None,
#     ssim_fn=None,
#     lpips_fn=None,
#     clip_model=None,
# ) -> dict:
#     """
#     Evaluate frame-to-frame prediction for a fixed frame_skip.
#     Evaluates ALL frame pairs within each video, then averages.
#     """
#     video_files = collect_video_files(dataset_path)
#     if len(video_files) == 0:
#         raise ValueError(f"No video files found in {dataset_path}")

#     num_samples = min(num_samples, len(video_files))
#     indices = np.linspace(0, len(video_files) - 1, num_samples, dtype=int)

#     # Per-video averages
#     video_psnr = []
#     video_ssim = []
#     video_lpips = []
#     video_clip = []
#     num_failed = 0
#     total_pairs = 0

#     for i in tqdm(indices, desc=f"  frame_skip={frame_skip}, {Path(dataset_path).name}"):
#         video_path = video_files[i]

#         # Load ALL frames from this video
#         all_frames = load_all_frames_from_video(str(video_path), rgb_skip=rgb_skip)
#         if all_frames is None or len(all_frames) < frame_skip + 1:
#             num_failed += 1
#             continue

#         num_effective_frames = len(all_frames)

#         # Generate all pairs: (0, skip), (skip, 2*skip), (2*skip, 3*skip), ...
#         pair_psnr = []
#         pair_ssim = []
#         pair_lpips = []
#         pair_clip = []

#         for t in range(0, num_effective_frames - frame_skip, frame_skip):
#             frame1 = all_frames[t]           # [240, 320, 3]
#             frame2 = all_frames[t + frame_skip]  # [240, 320, 3]

#             # Build batch: [1, 2, 240, 320, 3]
#             pair = torch.stack([frame1, frame2]).unsqueeze(0).to(device)
#             batch = {"videos": pair}

#             # LAM forward pass
#             outputs = model.lam(batch)
#             recon = outputs["recon"][0][0].cpu()  # reconstructed frame 2: [240, 320, 3]

#             gt = einops.rearrange(frame2, "h w c -> 1 c h w").clamp(0, 1).to(device)
#             pred = einops.rearrange(recon, "h w c -> 1 c h w").clamp(0, 1).to(device)

#             # Compute metrics
#             psnr_val = psnr_fn(pred, gt).item()
#             ssim_val = ssim_fn(gt, pred).mean().item()
#             lpips_val = lpips_fn(pred, gt).item()

#             # CLIP similarity
#             gt_clip = clip_model.encode_image(image_list=gt)[1]
#             pred_clip = clip_model.encode_image(image_list=pred)[1]
#             clip_val = F.cosine_similarity(
#                 gt_clip.reshape(-1, gt_clip.shape[-1]),
#                 pred_clip.reshape(-1, pred_clip.shape[-1]),
#                 dim=-1
#             ).mean().item()

#             pair_psnr.append(psnr_val)
#             pair_ssim.append(ssim_val)
#             pair_lpips.append(lpips_val)
#             pair_clip.append(clip_val)

#         # Average across all pairs in this video
#         if len(pair_psnr) > 0:
#             video_psnr.append(float(np.mean(pair_psnr)))
#             video_ssim.append(float(np.mean(pair_ssim)))
#             video_lpips.append(float(np.mean(pair_lpips)))
#             video_clip.append(float(np.mean(pair_clip)))
#             total_pairs += len(pair_psnr)

#     if len(video_psnr) == 0:
#         return {"error": "All samples failed"}

#     return {
#         "psnr_mean": float(np.mean(video_psnr)),
#         "psnr_std": float(np.std(video_psnr)),
#         "ssim_mean": float(np.mean(video_ssim)),
#         "ssim_std": float(np.std(video_ssim)),
#         "lpips_mean": float(np.mean(video_lpips)),
#         "lpips_std": float(np.std(video_lpips)),
#         "clip_mean": float(np.mean(video_clip)),
#         "clip_std": float(np.std(video_clip)),
#         "num_videos": len(video_psnr),
#         "num_failed": num_failed,
#         "total_pairs": total_pairs,
#         "avg_pairs_per_video": total_pairs / len(video_psnr),
#     }


# # ============================================================
# # Full evaluation: all checkpoints × all datasets × all frame_skips
# # ============================================================

# def run_frame2frame_evaluation(
#     ckpt_paths: list,
#     ckpt_names: list,
#     dataset_paths: list,
#     dataset_names: list,
#     rgb_skips: list = None,
#     frame_skips: list = None,
#     num_samples: int = 500,
#     device: str = "cuda:0",
#     output_dir: str = "eval_results",
# ):
#     os.makedirs(output_dir, exist_ok=True)

#     if rgb_skips is None:
#         rgb_skips = [1] * len(dataset_paths)
#     if frame_skips is None:
#         frame_skips = [1, 2, 3, 4]

#     # Initialize metric models once
#     print("Initializing metric models...")
#     psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
#     ssim_fn = SSIM(data_range=1.0, size_average=False, channel=3).to(device)
#     lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
#     clip_model = CLIPNetwork(config=CLIPNetworkConfig())
#     print("  Done.")

#     # results[ckpt_name][dataset_name][frame_skip] = metrics
#     results = {}

#     for ckpt_path, ckpt_name in zip(ckpt_paths, ckpt_names):
#         print(f"\n{'='*70}")
#         print(f"Checkpoint: {ckpt_name}")
#         print(f"  Path: {ckpt_path}")
#         print(f"{'='*70}")

#         model = load_lam_model(ckpt_path, device=device)
#         results[ckpt_name] = {}

#         for dataset_path, dataset_name, rgb_skip in zip(dataset_paths, dataset_names, rgb_skips):
#             results[ckpt_name][dataset_name] = {}

#             for fs in frame_skips:
#                 print(f"\n  {dataset_name} | frame_skip={fs} | rgb_skip={rgb_skip}")

#                 metrics = evaluate_frame2frame(
#                     model=model,
#                     dataset_path=dataset_path,
#                     frame_skip=fs,
#                     rgb_skip=rgb_skip,
#                     num_samples=num_samples,
#                     device=device,
#                     psnr_fn=psnr_fn,
#                     ssim_fn=ssim_fn,
#                     lpips_fn=lpips_fn,
#                     clip_model=clip_model,
#                 )
#                 results[ckpt_name][dataset_name][str(fs)] = metrics

#                 if "error" not in metrics:
#                     print(f"    PSNR:  {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
#                     print(f"    SSIM:  {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
#                     print(f"    LPIPS: {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")
#                     print(f"    CLIP:  {metrics['clip_mean']:.4f} ± {metrics['clip_std']:.4f}")
#                     print(f"    Videos: {metrics['num_videos']} | Pairs: {metrics['total_pairs']} | Avg pairs/video: {metrics['avg_pairs_per_video']:.1f}")
#                 else:
#                     print(f"    ERROR: {metrics['error']}")

#         del model
#         torch.cuda.empty_cache()

#     # ============================================================
#     # Print results tables (one per dataset)
#     # ============================================================

#     metric_keys = ["psnr", "ssim", "lpips", "clip"]
#     metric_labels = {"psnr": "PSNR↑", "ssim": "SSIM↑", "lpips": "LPIPS↓", "clip": "CLIP↑"}

#     for dataset_name in dataset_names:
#         print(f"\n\n{'='*100}")
#         print(f"RESULTS: {dataset_name}")
#         print(f"{'='*100}\n")

#         # Header
#         header = f"{'Checkpoint':<30} {'Skip':<6}"
#         for mk in metric_keys:
#             header += f" | {metric_labels[mk]:>10}"
#         header += f" | {'Pairs':>8}"
#         print(header)
#         print("-" * len(header))

#         # Rows
#         for ckpt_name in ckpt_names:
#             for fs in frame_skips:
#                 m = results[ckpt_name][dataset_name][str(fs)]
#                 if "error" not in m:
#                     row = f"{ckpt_name:<30} {fs:<6}"
#                     for mk in metric_keys:
#                         row += f" | {m[f'{mk}_mean']:>10.3f}"
#                     row += f" | {m['total_pairs']:>8}"
#                     print(row)
#                 else:
#                     row = f"{ckpt_name:<30} {fs:<6}"
#                     for mk in metric_keys:
#                         row += f" | {'ERROR':>10}"
#                     print(row)
#             print()  # blank line between checkpoints

#     # Save JSON
#     json_path = os.path.join(output_dir, "frame2frame_results.json")
#     with open(json_path, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\nJSON saved to: {json_path}")

#     # Save CSV (one row per checkpoint × frame_skip)
#     csv_path = os.path.join(output_dir, "frame2frame_results.csv")
#     with open(csv_path, "w") as f:
#         cols = ["Checkpoint", "Frame Skip"]
#         for dn in dataset_names:
#             for mk in metric_keys:
#                 cols.append(f"{dn} {metric_labels[mk]}")
#         f.write(",".join(cols) + "\n")

#         for ckpt_name in ckpt_names:
#             for fs in frame_skips:
#                 vals = [ckpt_name, str(fs)]
#                 for dataset_name in dataset_names:
#                     m = results[ckpt_name][dataset_name][str(fs)]
#                     if "error" not in m:
#                         for mk in metric_keys:
#                             vals.append(f"{m[f'{mk}_mean']:.4f}")
#                     else:
#                         vals.extend(["ERROR"] * len(metric_keys))
#                 f.write(",".join(vals) + "\n")
#     print(f"CSV saved to: {csv_path}")

#     return results


# # ============================================================
# # Main
# # ============================================================

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="LAM Frame-to-Frame Evaluation")
#     parser.add_argument("--lam_eval", action="store_true", help="Run LAM evaluation")
#     parser.add_argument("--num_samples", type=int, default=500)
#     parser.add_argument("--frame_skips", nargs="+", type=int, default=[1, 2, 3, 4],
#                         help="List of frame_skip values to evaluate")
#     parser.add_argument("--device", type=str, default="cuda:0")
#     parser.add_argument("--output_dir", type=str, default="eval_results_stack_frame2frame", help="Directory to save evaluation results")
#     args = parser.parse_args()

#     if args.lam_eval:
#         # ============================================================
#         # EDIT THESE PATHS FOR YOUR SETUP
#         # ============================================================
#         ckpt_paths = [
#             "/n/fs/geniemodel/DreamDojo/checkpoints/LAM/LAM_400k.ckpt",
#             "/n/fs/geniemodel/DreamDojo/external/lam_project/exp_ckpts_bridge_full/last.ckpt",
#             "/n/fs/geniemodel/DreamDojo/external/lam_project/exp_ckpts_droid_full_dreamzero/last.ckpt",
#             "/n/fs/geniemodel/DreamDojo/external/lam_project/exp_ckpts_bridge_droid_full_dreamzero/last.ckpt",
#         ]
#         ckpt_names = [
#             "Original LAM_400k",
#             "Fine-tuned Bridge",
#             "Fine-tuned DROID",
#             "Fine-tuned Bridge+DROID",
#         ]
#         dataset_paths = [
#             "/n/fs/not-fmrl/Projects/wm_alignment/cosmos-predict2/datasets/bridge/videos/test",
#             "/n/fs/iromdata/droid_ctrl_world/videos/val",
#         ]
#         dataset_names = ["Bridge V2", "DROID"]
#         rgb_skips = [1, 3]

#         run_frame2frame_evaluation(
#             ckpt_paths=ckpt_paths,
#             ckpt_names=ckpt_names,
#             dataset_paths=dataset_paths,
#             dataset_names=dataset_names,
#             rgb_skips=rgb_skips,
#             frame_skips=args.frame_skips,
#             num_samples=args.num_samples,
#             device=args.device,
#             output_dir=args.output_dir,
#         )
#     else:
#         print("Usage: python eval_lam_frame2frame.py --lam_eval --num_samples 500")
#         print("  --frame_skips 1 2 3 4    (which frame gaps to test)")
#         print("  --device cuda:0")
#         print("  --output_dir eval_results_stack_frame2frame")


"""
LAM Evaluation: Frame-to-Frame Prediction with Fixed Frame Skip

Supports both single-view (Bridge) and stacked multi-view (DROID DreamZero) evaluation.

For stacked DROID:
  - Loads all 3 views (0.mp4, 1.mp4, 2.mp4) per episode
  - Stacks them DreamZero style (wrist on top, left+right on bottom)
  - Resizes to 240×320 (LAM input)
  - Evaluates frame-to-frame prediction on the stacked images

Usage:
    python eval_lam_frame2frame.py --lam_eval --num_samples 500
    python eval_lam_frame2frame.py --lam_eval --num_samples 100 --frame_skips 1 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2 as cv
import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

# metrics
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from vidwm.encoders.clip_encoder import CLIPNetwork, CLIPNetworkConfig


# ============================================================
# Load LAM model
# ============================================================

def load_lam_model(ckpt_path: str, device: str = "cuda:0"):
    """Load a LAM model from a Lightning checkpoint."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from lam.model import LAM

    print(f"Loading LAM checkpoint: {ckpt_path}")
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
# Load frames: single view
# ============================================================

def load_all_frames_from_video(video_path: str, rgb_skip: int = 1) -> Tensor:
    """
    Load all frames from a single video, applying rgb_skip and preprocessing.
    Returns tensor of shape [N, 240, 320, 3] in float [0, 1], or None if failed.
    """
    cap = cv.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if total_frames < 2:
        cap.release()
        return None

    frames = []
    for raw_idx in range(0, total_frames, rgb_skip):
        cap.set(cv.CAP_PROP_POS_FRAMES, raw_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame))
        else:
            break
    cap.release()

    if len(frames) < 2:
        return None

    video = torch.stack(frames).float() / 255.0  # [N, H, W, 3]

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
        target_height, target_width = h, w
    h_crop = (h - target_height) // 2
    w_crop = (w - target_width) // 2
    video = video[:, h_crop:h_crop + target_height, w_crop:w_crop + target_width]

    # Resize to 240x320
    video = einops.rearrange(video, "t h w c -> c t h w")
    video = F.interpolate(video, (240, 320), mode="bilinear")
    video = einops.rearrange(video, "c t h w -> t h w c")

    return video  # [N, 240, 320, 3]


# ============================================================
# Load frames: stacked DreamZero
# ============================================================

def load_all_frames_stacked_dreamzero(
    episode_dir: str,
    rgb_skip: int = 1,
    left_view: str = "0.mp4",
    right_view: str = "1.mp4",
    wrist_view: str = "2.mp4",
) -> Tensor:
    """
    Load all frames from 3 DROID views, stack them DreamZero style, preprocess.

    DreamZero stacking:
      ┌──────────────────────────┐
      │   View 2 (wrist)         │  H × 2W
      ├─────────────┬────────────┤
      │  View 0     │   View 1   │  H × W each
      └─────────────┴────────────┘
      384 × 640 → resize to 240 × 320

    Returns tensor of shape [N, 240, 320, 3] in float [0, 1], or None if failed.
    """
    episode_dir = Path(episode_dir)
    left_path = str(episode_dir / left_view)
    right_path = str(episode_dir / right_view)
    wrist_path = str(episode_dir / wrist_view)

    # Check all views exist
    if not all(Path(p).exists() for p in [left_path, right_path, wrist_path]):
        return None

    # Get total frames from left view
    cap = cv.VideoCapture(left_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames < 2:
        return None

    # Read frames from all 3 views with rgb_skip
    def read_frames(path):
        cap = cv.VideoCapture(path)
        frames = []
        for raw_idx in range(0, total_frames, rgb_skip):
            cap.set(cv.CAP_PROP_POS_FRAMES, raw_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        cap.release()
        return frames

    left_frames = read_frames(left_path)
    right_frames = read_frames(right_path)
    wrist_frames = read_frames(wrist_path)

    # Ensure all views have same number of frames
    min_len = min(len(left_frames), len(right_frames), len(wrist_frames))
    if min_len < 2:
        return None

    left_frames = left_frames[:min_len]
    right_frames = right_frames[:min_len]
    wrist_frames = wrist_frames[:min_len]

    # Stack DreamZero style for each frame
    stacked_frames = []
    for t in range(min_len):
        H, W = left_frames[t].shape[:2]

        # Wrist: resize to double width
        wrist_resized = cv.resize(wrist_frames[t], (2 * W, H), interpolation=cv.INTER_LINEAR)

        # Left + Right side by side
        bottom = np.concatenate([left_frames[t], right_frames[t]], axis=1)  # H × 2W × 3

        # Stack vertically: wrist on top, left+right on bottom
        stacked = np.concatenate([wrist_resized, bottom], axis=0)  # 2H × 2W × 3
        stacked_frames.append(torch.from_numpy(stacked))

    video = torch.stack(stacked_frames).float() / 255.0  # [N, 2H, 2W, 3]

    # Resize to 240×320 (LAM input size)
    video = einops.rearrange(video, "t h w c -> c t h w")
    video = F.interpolate(video, (240, 320), mode="bilinear")
    video = einops.rearrange(video, "c t h w -> t h w c")

    return video  # [N, 240, 320, 3]


# ============================================================
# Collect video files / episode dirs
# ============================================================

def collect_video_files(dataset_path: str, stacked: bool = False) -> list:
    """
    Find video files or episode directories in a dataset.

    If stacked=False: returns list of individual .mp4 file paths
    If stacked=True:  returns list of episode directory paths (each containing 0.mp4, 1.mp4, 2.mp4)
    """
    if stacked:
        # Find all episode directories that have all 3 views
        mp4_list = sorted(Path(dataset_path).rglob("0.mp4"))
        episode_dirs = []
        for f in mp4_list:
            episode_dir = f.parent
            if (episode_dir / "0.mp4").exists() and \
               (episode_dir / "1.mp4").exists() and \
               (episode_dir / "2.mp4").exists():
                episode_dirs.append(str(episode_dir))
        return sorted(set(episode_dirs))
    else:
        mp4_list = sorted(Path(dataset_path).rglob("*.mp4"))
        filtered = [
            f for f in mp4_list
            if "left" not in str(f).lower()
            and "right" not in str(f).lower()
            and "resize" not in str(f).lower()
            and "pad" not in str(f).lower()
        ]
        if "droid" in dataset_path.lower():
            filtered = [f for f in filtered if f.name in ("0.mp4", "1.mp4", "2.mp4")]
        return filtered


# ============================================================
# Evaluate one checkpoint on one dataset with one frame_skip
# ============================================================

@torch.no_grad()
def evaluate_frame2frame(
    model,
    dataset_path: str,
    frame_skip: int = 1,
    rgb_skip: int = 1,
    num_samples: int = 500,
    device: str = "cuda:0",
    stacked: bool = False,
    psnr_fn=None,
    ssim_fn=None,
    lpips_fn=None,
    clip_model=None,
) -> dict:
    """
    Evaluate frame-to-frame prediction for a fixed frame_skip.
    Evaluates ALL frame pairs within each video, then averages.

    If stacked=True, loads DROID episodes as stacked DreamZero views.
    """
    video_files = collect_video_files(dataset_path, stacked=stacked)
    if len(video_files) == 0:
        raise ValueError(f"No video files found in {dataset_path}")

    num_samples = min(num_samples, len(video_files))
    indices = np.linspace(0, len(video_files) - 1, num_samples, dtype=int)

    video_psnr = []
    video_ssim = []
    video_lpips = []
    video_clip = []
    num_failed = 0
    total_pairs = 0

    stacked_label = " [stacked DreamZero]" if stacked else ""
    for i in tqdm(indices, desc=f"  frame_skip={frame_skip}, {Path(dataset_path).name}{stacked_label}"):
        video_entry = video_files[i]

        # Load frames based on mode
        if stacked:
            all_frames = load_all_frames_stacked_dreamzero(str(video_entry), rgb_skip=rgb_skip)
        else:
            all_frames = load_all_frames_from_video(str(video_entry), rgb_skip=rgb_skip)

        if all_frames is None or len(all_frames) < frame_skip + 1:
            num_failed += 1
            continue

        num_effective_frames = len(all_frames)

        pair_psnr = []
        pair_ssim = []
        pair_lpips = []
        pair_clip = []

        for t in range(0, num_effective_frames - frame_skip, frame_skip):
            frame1 = all_frames[t]
            frame2 = all_frames[t + frame_skip]

            pair = torch.stack([frame1, frame2]).unsqueeze(0).to(device)
            batch = {"videos": pair}

            outputs = model.lam(batch)
            recon = outputs["recon"][0][0].cpu()

            gt = einops.rearrange(frame2, "h w c -> 1 c h w").clamp(0, 1).to(device)
            pred = einops.rearrange(recon, "h w c -> 1 c h w").clamp(0, 1).to(device)

            psnr_val = psnr_fn(pred, gt).item()
            ssim_val = ssim_fn(gt, pred).mean().item()
            lpips_val = lpips_fn(pred, gt).item()

            gt_clip = clip_model.encode_image(image_list=gt)[1]
            pred_clip = clip_model.encode_image(image_list=pred)[1]
            clip_val = F.cosine_similarity(
                gt_clip.reshape(-1, gt_clip.shape[-1]),
                pred_clip.reshape(-1, pred_clip.shape[-1]),
                dim=-1
            ).mean().item()

            pair_psnr.append(psnr_val)
            pair_ssim.append(ssim_val)
            pair_lpips.append(lpips_val)
            pair_clip.append(clip_val)

        if len(pair_psnr) > 0:
            video_psnr.append(float(np.mean(pair_psnr)))
            video_ssim.append(float(np.mean(pair_ssim)))
            video_lpips.append(float(np.mean(pair_lpips)))
            video_clip.append(float(np.mean(pair_clip)))
            total_pairs += len(pair_psnr)

    if len(video_psnr) == 0:
        return {"error": "All samples failed"}

    return {
        "psnr_mean": float(np.mean(video_psnr)),
        "psnr_std": float(np.std(video_psnr)),
        "ssim_mean": float(np.mean(video_ssim)),
        "ssim_std": float(np.std(video_ssim)),
        "lpips_mean": float(np.mean(video_lpips)),
        "lpips_std": float(np.std(video_lpips)),
        "clip_mean": float(np.mean(video_clip)),
        "clip_std": float(np.std(video_clip)),
        "num_videos": len(video_psnr),
        "num_failed": num_failed,
        "total_pairs": total_pairs,
        "avg_pairs_per_video": total_pairs / len(video_psnr),
        "stacked": stacked,
    }


# ============================================================
# Full evaluation: all checkpoints × all datasets × all frame_skips
# ============================================================

def run_frame2frame_evaluation(
    ckpt_paths: list,
    ckpt_names: list,
    dataset_paths: list,
    dataset_names: list,
    rgb_skips: list = None,
    stacked_flags: list = None,
    frame_skips: list = None,
    num_samples: int = 500,
    device: str = "cuda:0",
    output_dir: str = "eval_results",
):
    os.makedirs(output_dir, exist_ok=True)

    if rgb_skips is None:
        rgb_skips = [1] * len(dataset_paths)
    if stacked_flags is None:
        stacked_flags = [False] * len(dataset_paths)
    if frame_skips is None:
        frame_skips = [1, 2, 3, 4]

    # Initialize metric models once
    print("Initializing metric models...")
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = SSIM(data_range=1.0, size_average=False, channel=3).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    clip_model = CLIPNetwork(config=CLIPNetworkConfig())
    print("  Done.")

    results = {}

    for ckpt_path, ckpt_name in zip(ckpt_paths, ckpt_names):
        print(f"\n{'='*70}")
        print(f"Checkpoint: {ckpt_name}")
        print(f"  Path: {ckpt_path}")
        print(f"{'='*70}")

        model = load_lam_model(ckpt_path, device=device)
        results[ckpt_name] = {}

        for dataset_path, dataset_name, rgb_skip, stacked in zip(
            dataset_paths, dataset_names, rgb_skips, stacked_flags
        ):
            results[ckpt_name][dataset_name] = {}

            for fs in frame_skips:
                stacked_label = " [stacked]" if stacked else ""
                print(f"\n  {dataset_name}{stacked_label} | frame_skip={fs} | rgb_skip={rgb_skip}")

                metrics = evaluate_frame2frame(
                    model=model,
                    dataset_path=dataset_path,
                    frame_skip=fs,
                    rgb_skip=rgb_skip,
                    num_samples=num_samples,
                    device=device,
                    stacked=stacked,
                    psnr_fn=psnr_fn,
                    ssim_fn=ssim_fn,
                    lpips_fn=lpips_fn,
                    clip_model=clip_model,
                )
                results[ckpt_name][dataset_name][str(fs)] = metrics

                if "error" not in metrics:
                    print(f"    PSNR:  {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
                    print(f"    SSIM:  {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
                    print(f"    LPIPS: {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")
                    print(f"    CLIP:  {metrics['clip_mean']:.4f} ± {metrics['clip_std']:.4f}")
                    print(f"    Videos: {metrics['num_videos']} | Pairs: {metrics['total_pairs']} | Avg pairs/video: {metrics['avg_pairs_per_video']:.1f}")
                else:
                    print(f"    ERROR: {metrics['error']}")

        del model
        torch.cuda.empty_cache()

    # ============================================================
    # Print results tables
    # ============================================================

    metric_keys = ["psnr", "ssim", "lpips", "clip"]
    metric_labels = {"psnr": "PSNR↑", "ssim": "SSIM↑", "lpips": "LPIPS↓", "clip": "CLIP↑"}

    for dataset_name in dataset_names:
        print(f"\n\n{'='*100}")
        print(f"RESULTS: {dataset_name}")
        print(f"{'='*100}\n")

        header = f"{'Checkpoint':<30} {'Skip':<6}"
        for mk in metric_keys:
            header += f" | {metric_labels[mk]:>10}"
        header += f" | {'Pairs':>8}"
        print(header)
        print("-" * len(header))

        for ckpt_name in ckpt_names:
            for fs in frame_skips:
                m = results[ckpt_name][dataset_name][str(fs)]
                if "error" not in m:
                    row = f"{ckpt_name:<30} {fs:<6}"
                    for mk in metric_keys:
                        row += f" | {m[f'{mk}_mean']:>10.3f}"
                    row += f" | {m['total_pairs']:>8}"
                    print(row)
                else:
                    row = f"{ckpt_name:<30} {fs:<6}"
                    for mk in metric_keys:
                        row += f" | {'ERROR':>10}"
                    print(row)
            print()

    # Save JSON
    json_path = os.path.join(output_dir, "frame2frame_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved to: {json_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, "frame2frame_results.csv")
    with open(csv_path, "w") as f:
        cols = ["Checkpoint", "Frame Skip"]
        for dn in dataset_names:
            for mk in metric_keys:
                cols.append(f"{dn} {metric_labels[mk]}")
        f.write(",".join(cols) + "\n")

        for ckpt_name in ckpt_names:
            for fs in frame_skips:
                vals = [ckpt_name, str(fs)]
                for dataset_name in dataset_names:
                    m = results[ckpt_name][dataset_name][str(fs)]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAM Frame-to-Frame Evaluation")
    parser.add_argument("--lam_eval", action="store_true", help="Run LAM evaluation")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--frame_skips", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="List of frame_skip values to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="eval_results_stack_frame2frame",
                        help="Directory to save evaluation results")
    args = parser.parse_args()

    if args.lam_eval:
        # ============================================================
        # EDIT THESE PATHS FOR YOUR SETUP
        # ============================================================
        ckpt_paths = [
            "/n/fs/geniemodel/DreamDojo/checkpoints/LAM/LAM_400k.ckpt",
            "/n/fs/geniemodel/DreamDojo/external/lam_project/exp_ckpts_bridge_full/last.ckpt",
            "/n/fs/geniemodel/DreamDojo/external/lam_project/exp_ckpts_droid_full_dreamzero/last.ckpt",
            "/n/fs/geniemodel/DreamDojo/external/lam_project/exp_ckpts_bridge_droid_full_dreamzero/last.ckpt",
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
        dataset_names = ["Bridge V2", "DROID (stacked DreamZero)"]

        # Bridge: single view, rgb_skip=1, not stacked
        # DROID:  stacked DreamZero, rgb_skip=1 (already at 5 FPS)
        rgb_skips = [1, 1]
        stacked_flags = [False, True]

        run_frame2frame_evaluation(
            ckpt_paths=ckpt_paths,
            ckpt_names=ckpt_names,
            dataset_paths=dataset_paths,
            dataset_names=dataset_names,
            rgb_skips=rgb_skips,
            stacked_flags=stacked_flags,
            frame_skips=args.frame_skips,
            num_samples=args.num_samples,
            device=args.device,
            output_dir=args.output_dir,
        )
    else:
        print("Usage: python eval_lam_frame2frame.py --lam_eval --num_samples 100")
        print("  --frame_skips 1 2 3 4    (which frame gaps to test)")
        print("  --device cuda:0")
        print("  --output_dir eval_results_stack_frame2frame")