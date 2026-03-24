"""
Stacked DROID VideoDataset for LAM training.

Supports two stacking modes:

Mode A: "vertical" (Ctrl-World style)
  ┌──────────────┐
  │   View 0     │  192 × 320
  ├──────────────┤
  │   View 1     │  192 × 320
  ├──────────────┤
  │   View 2     │  192 × 320
  └──────────────┘
  Stacked: 576 × 320 → resize to 240 × 320

Mode B: "dreamzero" (DreamZero style)
  ┌──────────────────────────┐
  │   View 2 (wrist)         │  192 × 640 (resized to double width)
  ├─────────────┬────────────┤
  │  View 0     │   View 1   │  192 × 320 each, side by side
  └─────────────┴────────────┘
  Stacked: 384 × 640 → resize to 240 × 320

Place this file at:
  /n/fs/geniemodel/DreamDojo/external/lam_project/lam/dataset_stacked.py

Usage in LAM config:
  data:
    dataset_paths:
      - /path/to/bridge/videos/train
      - /path/to/droid/videos/train
    rgb_skips:
      - 1
      - 1
    # Set stacking_mode via MultiSourceSamplerDataset or as env var
"""

import os
from pathlib import Path
from random import randint
from typing import Any, Dict, List

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2 ** 13)
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset


def exists(var) -> bool:
    return var is not None


class StackedDroidVideoDataset(Dataset):
    """
    DROID dataset that loads all 3 camera views and stacks them into one image.

    Args:
        stacking_mode: "vertical" for Ctrl-World style (0,1,2 top to bottom)
                       "dreamzero" for DreamZero style (wrist on top, left+right on bottom)
        wrist_view, left_view, right_view: filenames for each camera view
    """

    def __init__(
        self,
        subset_path: str,
        padding: str = "repeat",
        randomize: bool = False,
        num_frames: int = 2,
        output_format: str = "t h w c",
        color_aug: bool = True,
        rgb_skip: int = 1,
        stacking_mode: str = "vertical",  # "vertical" or "dreamzero"
        wrist_view: str = "2.mp4",
        left_view: str = "0.mp4",
        right_view: str = "1.mp4",
    ) -> None:
        super().__init__()
        self.padding = padding
        self.randomize = randomize
        self.num_frames = num_frames
        self.output_format = output_format
        self.color_aug = color_aug
        self.rgb_skip = rgb_skip
        self.stacking_mode = stacking_mode
        self.wrist_view = wrist_view
        self.left_view = left_view
        self.right_view = right_view

        assert stacking_mode in ("vertical", "dreamzero"), \
            f"stacking_mode must be 'vertical' or 'dreamzero', got '{stacking_mode}'"

        # Find all episode directories that have all 3 views
        mp4_list = list(Path(subset_path).rglob("0.mp4"))
        self.episode_dirs = []
        for f in mp4_list:
            episode_dir = f.parent
            if (episode_dir / left_view).exists() and \
               (episode_dir / right_view).exists() and \
               (episode_dir / wrist_view).exists():
                self.episode_dirs.append(str(episode_dir))

        self.episode_dirs = sorted(self.episode_dirs)
        if len(self.episode_dirs) == 0:
            raise ValueError(f"No complete DROID episodes found in {subset_path}")

        print(f"StackedDroidVideoDataset: {len(self.episode_dirs)} episodes, "
              f"stacking_mode={stacking_mode}, rgb_skip={rgb_skip}")

    def __len__(self) -> int:
        return len(self.episode_dirs)

    def __getitem__(self, idx: int) -> Dict:
        episode_dir = self.episode_dirs[idx]
        while True:
            try:
                video = self.load_stacked_video_slice(
                    episode_dir,
                    self.num_frames,
                    None if self.randomize else 0
                )
                return self.build_data_dict(video)
            except Exception:
                idx = randint(0, len(self) - 1)
                episode_dir = self.episode_dirs[idx]

    def _read_frames_sequential(self, video_path: str, raw_start: int,
                                 num_frames: int, rgb_skip: int) -> list:
        """
        Read num_frames from a video starting at raw_start, skipping rgb_skip-1
        frames between each read.
        """
        cap = cv.VideoCapture(video_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, raw_start)

        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                if len(frames) > 0:
                    if self.padding == "repeat":
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros_like(frames[-1]))
                else:
                    cap.release()
                    raise ValueError(f"Could not read frames from {video_path}")

            # Skip rgb_skip - 1 frames
            for _ in range(rgb_skip - 1):
                cap.read()

        cap.release()
        return frames

    def _stack_frames_vertical(self, view0_frames, view1_frames, view2_frames):
        """
        Ctrl-World style: stack all 3 views vertically.

        ┌──────────────┐
        │   View 0     │  H × W
        ├──────────────┤
        │   View 1     │  H × W
        ├──────────────┤
        │   View 2     │  H × W
        └──────────────┘
        Result: 3H × W
        """
        stacked = []
        for t in range(len(view0_frames)):
            frame = np.concatenate([
                view0_frames[t],
                view1_frames[t],
                view2_frames[t]
            ], axis=0)  # 3H × W × 3
            stacked.append(frame)
        return stacked

    def _stack_frames_dreamzero(self, view0_frames, view1_frames, view2_frames):
        """
        DreamZero style: wrist (view 2) on top doubled, left+right on bottom.

        ┌──────────────────────────┐
        │   View 2 (wrist)         │  H × 2W
        ├─────────────┬────────────┤
        │  View 0     │   View 1   │  H × W each
        └─────────────┴────────────┘
        Result: 2H × 2W
        """
        stacked = []
        for t in range(len(view0_frames)):
            H, W = view0_frames[t].shape[:2]

            # Wrist: resize to double width
            wrist_resized = cv.resize(
                view2_frames[t], (2 * W, H), interpolation=cv.INTER_LINEAR
            )

            # Left + Right side by side
            bottom = np.concatenate([view0_frames[t], view1_frames[t]], axis=1)

            # Stack vertically
            frame = np.concatenate([wrist_resized, bottom], axis=0)
            stacked.append(frame)
        return stacked

    def load_stacked_video_slice(
        self,
        episode_dir: str,
        num_frames: int,
        start_frame: int = None,
    ) -> Tensor:
        """Load frames from all 3 views, stack them, return as video tensor."""
        left_path = os.path.join(episode_dir, self.left_view)
        right_path = os.path.join(episode_dir, self.right_view)
        wrist_path = os.path.join(episode_dir, self.wrist_view)

        # Get total frames
        cap = cv.VideoCapture(left_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        cap.release()

        rgb_skip = self.rgb_skip
        effective_frames = total_frames // max(rgb_skip, 1)

        # Random frame_skip
        frame_skip = randint(1, 4)
        needed_frames = num_frames * frame_skip

        # Random start in effective frame space
        start_frame = start_frame if exists(start_frame) else randint(
            0, max(0, effective_frames - needed_frames)
        )
        raw_start = start_frame * rgb_skip

        # Load frames from all 3 views with same timing
        view0_frames = self._read_frames_sequential(left_path, raw_start, needed_frames, rgb_skip)
        view1_frames = self._read_frames_sequential(right_path, raw_start, needed_frames, rgb_skip)
        view2_frames = self._read_frames_sequential(wrist_path, raw_start, needed_frames, rgb_skip)

        # Stack views based on mode
        if self.stacking_mode == "vertical":
            stacked_frames = self._stack_frames_vertical(view0_frames, view1_frames, view2_frames)
        elif self.stacking_mode == "dreamzero":
            stacked_frames = self._stack_frames_dreamzero(view0_frames, view1_frames, view2_frames)

        # Subsample with frame_skip
        stacked_frames = stacked_frames[::frame_skip]
        video = torch.stack([torch.from_numpy(f) for f in stacked_frames]).float() / 255.0

        # Resize to LAM input size (240×320)
        video = rearrange(video, "t h w c -> c t h w")
        video = F.interpolate(video, (240, 320), mode="bilinear")
        video = rearrange(video, f"c t h w -> {self.output_format}")

        return video

    def build_data_dict(self, video: Tensor) -> Dict:
        if self.color_aug:
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)
        return {"videos": video}