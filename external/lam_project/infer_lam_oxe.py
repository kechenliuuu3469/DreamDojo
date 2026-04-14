"""
LAM Inference across OXE datasets.

Mirrors the preprocessing convention from config/lam_joint_all.yaml:
  - rgb_skips   : frame stride per dataset
  - stacking_modes : None / "dreamzero" / "horizontal"
  - view_maps   : which .mp4 in each episode dir is which view

For each episode under <dataset_root>/<dataset>/videos/train/<ep>/, the script
stride-samples frames at the dataset's rgb_skip, composes views according to the
stacking mode, resizes to 240x320, and runs LAM to produce a [N-1, 32] latent
action trajectory. Latents are saved to
    <dataset_root>/<dataset>/latent_actions_lam/train/<ep>/latent_actions.npy

The --percent flag limits to the first X% of episodes per dataset.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from lam.model import LAM


# Per-dataset pipeline config (subset of config/lam_joint_all.yaml we care about).
# stacking_mode: None  -> single view (view_map["view"] names the mp4)
#                "dreamzero" -> wrist on top, left+right below (2H x 2W)
#                "horizontal" -> left|right side by side (H x 2W)
DATASET_CFG = {
    "bc_z":            {"rgb_skip": 3, "stacking_mode": None,         "view_map": {"view": "0.mp4"}},
    "bridge":          {"rgb_skip": 1, "stacking_mode": None,         "view_map": {"view": "rgb.mp4"}},
    "fractal":         {"rgb_skip": 3, "stacking_mode": None,         "view_map": {"view": "0.mp4"}},
    "droid":           {"rgb_skip": 1, "stacking_mode": "dreamzero",  "view_map": {"wrist": "2.mp4", "left": "0.mp4", "right": "1.mp4"}},
    "fmb":             {"rgb_skip": 3, "stacking_mode": "dreamzero",  "view_map": {"wrist": "4.mp4", "left": "0.mp4", "right": "2.mp4"}},
    "taco_play":       {"rgb_skip": 3, "stacking_mode": None,         "view_map": {"view": "3.mp4"}},
    "furniture_bench": {"rgb_skip": 3, "stacking_mode": "horizontal", "view_map": {"left": "0.mp4", "right": "1.mp4"}},
}

TARGET_H, TARGET_W = 240, 320
TARGET_RATIO = 640 / 480


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def load_model(ckpt_path, device):
    print(f"[lam] loading {ckpt_path}")
    model = LAM.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.to(device).eval()
    print(f"[lam] params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model


# ---------------------------------------------------------------------------
# video reading
# ---------------------------------------------------------------------------
def read_strided_frames_np(video_path, rgb_skip):
    """Return a list of HxWx3 uint8 numpy arrays, keeping every rgb_skip-th frame."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % rgb_skip == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def compose_single_view(episode_dir, view_map, rgb_skip):
    fname = view_map["view"]
    path = episode_dir / fname
    if not path.exists():
        return None
    return read_strided_frames_np(path, rgb_skip)


def compose_horizontal(episode_dir, view_map, rgb_skip):
    left_path  = episode_dir / view_map["left"]
    right_path = episode_dir / view_map["right"]
    if not (left_path.exists() and right_path.exists()):
        return None
    left  = read_strided_frames_np(left_path,  rgb_skip)
    right = read_strided_frames_np(right_path, rgb_skip)
    n = min(len(left), len(right))
    return [np.concatenate([left[t], right[t]], axis=1) for t in range(n)]


def compose_dreamzero(episode_dir, view_map, rgb_skip):
    wrist_path = episode_dir / view_map["wrist"]
    left_path  = episode_dir / view_map["left"]
    right_path = episode_dir / view_map["right"]
    if not (wrist_path.exists() and left_path.exists() and right_path.exists()):
        return None
    left  = read_strided_frames_np(left_path,  rgb_skip)
    right = read_strided_frames_np(right_path, rgb_skip)
    wrist = read_strided_frames_np(wrist_path, rgb_skip)
    n = min(len(left), len(right), len(wrist))
    out = []
    for t in range(n):
        H, W = left[t].shape[:2]
        wrist_r = cv2.resize(wrist[t], (2 * W, H), interpolation=cv2.INTER_LINEAR)
        bottom = np.concatenate([left[t], right[t]], axis=1)         # H x 2W
        out.append(np.concatenate([wrist_r, bottom], axis=0))        # 2H x 2W
    return out


def build_episode_tensor(episode_dir, cfg):
    """Return a [N, 240, 320, 3] float tensor, or None if the episode is unusable."""
    mode = cfg["stacking_mode"]
    if mode is None:
        frames = compose_single_view(episode_dir, cfg["view_map"], cfg["rgb_skip"])
    elif mode == "horizontal":
        frames = compose_horizontal(episode_dir, cfg["view_map"], cfg["rgb_skip"])
    elif mode == "dreamzero":
        frames = compose_dreamzero(episode_dir, cfg["view_map"], cfg["rgb_skip"])
    else:
        raise ValueError(f"unknown stacking_mode={mode}")

    if frames is None or len(frames) < 2:
        return None

    video = torch.from_numpy(np.stack(frames)).float() / 255.0  # [N, H, W, 3]

    h, w = video.shape[1], video.shape[2]
    if w / h > TARGET_RATIO:
        th, tw = h, int(h * TARGET_RATIO)
    elif w / h < TARGET_RATIO:
        th, tw = int(w / TARGET_RATIO), w
    else:
        th, tw = h, w
    hc = (h - th) // 2
    wc = (w - tw) // 2
    video = video[:, hc:hc + th, wc:wc + tw]
    video = rearrange(video, "t h w c -> c t h w")
    video = torch.nn.functional.interpolate(video, (TARGET_H, TARGET_W), mode="bilinear")
    video = rearrange(video, "c t h w -> t h w c").contiguous()
    return video  # [N, 240, 320, 3]


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_latents_for_video(model, frames, device, batch_size):
    """frames: [N, H, W, 3] -> [N-1, 32] latent actions (consecutive pairs)."""
    pairs = torch.stack([frames[:-1], frames[1:]], dim=1)  # [N-1, 2, H, W, 3]
    outs = []
    for i in range(0, pairs.shape[0], batch_size):
        batch = {"videos": pairs[i:i + batch_size].to(device, non_blocking=True)}
        z_mu = model.lam(batch)["z_mu"]  # [B, 32]
        outs.append(z_mu.float().cpu())
    return torch.cat(outs, dim=0)


def process_dataset(model, dataset_root, dataset, cfg, percent,
                    device, batch_size, out_subdir, skip_existing):
    videos_dir = Path(dataset_root) / dataset / "videos" / "train"
    out_root = Path(dataset_root) / dataset / out_subdir / "train"

    episodes = sorted(
        [p for p in videos_dir.iterdir() if p.is_dir()],
        key=lambda p: (len(p.name), p.name),
    )
    n_total = len(episodes)
    n_keep = max(1, int(round(n_total * percent / 100.0))) if n_total else 0
    episodes = episodes[:n_keep]

    print(f"\n[{dataset}] rgb_skip={cfg['rgb_skip']} "
          f"stacking={cfg['stacking_mode']} views={cfg['view_map']} "
          f"| episodes {n_keep}/{n_total} ({percent}%) | out -> {out_root}")

    n_ok = n_skip = n_fail = 0
    for ep in tqdm(episodes, desc=dataset):
        out_dir = out_root / ep.name
        out_path = out_dir / "latent_actions.npy"
        if skip_existing and out_path.exists():
            n_skip += 1
            continue
        try:
            frames = build_episode_tensor(ep, cfg)
            if frames is None:
                n_fail += 1
                continue
            latents = extract_latents_for_video(model, frames, device, batch_size)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_path, latents.numpy().astype(np.float32))
            n_ok += 1
        except Exception as e:
            print(f"  [fail] {ep.name}: {e}")
            n_fail += 1

    print(f"[{dataset}] done: ok={n_ok} skip={n_skip} fail={n_fail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--dataset_root", default="/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4")
    ap.add_argument("--datasets", nargs="+",
                    default=["bc_z", "bridge", "droid", "fmb",
                             "fractal", "furniture_bench", "taco_play"])
    ap.add_argument("--percent", type=float, default=100.0,
                    help="Process the first X%% of episodes per dataset.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_subdir", default="latent_actions_lam")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for ds in args.datasets:
        if ds not in DATASET_CFG:
            raise ValueError(f"unknown dataset {ds}; known: {list(DATASET_CFG)}")

    model = load_model(args.ckpt_path, args.device)

    for ds in args.datasets:
        process_dataset(
            model=model,
            dataset_root=args.dataset_root,
            dataset=ds,
            cfg=DATASET_CFG[ds],
            percent=args.percent,
            device=args.device,
            batch_size=args.batch_size,
            out_subdir=args.out_subdir,
            skip_existing=not args.overwrite,
        )


if __name__ == "__main__":
    main()
