import math
from random import choices, randint
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2 ** 13)
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data import get_worker_info
from tqdm import tqdm


def exists(var) -> bool:
    return var is not None


def default(var, val) -> Any:
    return var if exists(var) else val


def default_worker_init_fn(worker_id: int) -> None:
    torch.manual_seed(torch.initial_seed() + worker_id)
    worker_info = get_worker_info()

    if exists(worker_info):
        dataset = worker_info.dataset
        glob_start = dataset._start
        glob_end = dataset._end

        per_worker = int((glob_end - glob_start) / worker_info.num_workers)
        worker_id = worker_info.id

        dataset._start = glob_start + worker_id * per_worker
        dataset._end = min(dataset._start + per_worker, glob_end)


def filter_video_files(file_names: List, xdof: bool = False) -> List:
    if xdof:
        return [
            f for f in file_names
            if "left" not in str(f).lower() and "right" not in str(f).lower() and "resize" not in str(f).lower() and "pad" not in str(f).lower()
            and "320_240" in str(f).lower()
        ]
    else:
        return [
            f for f in file_names
            if "left" not in str(f).lower() and "right" not in str(f).lower() and "resize" not in str(f).lower() and "pad" not in str(f).lower()
        ]


class LightningDataset(LightningDataModule):
    """
    Abstract LightningDataModule that represents a dataset we can train a Lightning module on.
    """

    def __init__(
        self,
        *args,
        batch_size: int = 8,
        num_workers: int = 32,
        train_shuffle: bool = True,
        val_shuffle: bool = False,
        val_batch_size: int = None,
        worker_init_fn: Callable = None,
        collate_fn: Callable = None,
        train_sampler: Callable = None,
        test_sampler: Callable = None,
        val_sampler: Callable = None
    ) -> None:
        super(LightningDataset, self).__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        val_batch_size = default(val_batch_size, batch_size)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.val_sampler = val_sampler
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.train_dataset,
            sampler=self.val_sampler,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if isinstance(self.test_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )


class VideoDataset(Dataset):
    def __init__(
        self,
        subset_path: str,
        padding: str = "repeat",
        randomize: bool = False,
        num_frames: int = 16,
        output_format: str = "t h w c",
        color_aug: bool = True,
        rgb_skip: int = 1,
        view_filter: Optional[str] = None,
    ) -> None:
        super(VideoDataset, self).__init__()
        self.padding = padding
        self.randomize = randomize
        self.num_frames = num_frames
        self.output_format = output_format
        self.color_aug = color_aug
        self.rgb_skip = rgb_skip

        mp4_list = list(Path(subset_path).rglob("*.mp4"))
        if len(mp4_list) > 0:
            if view_filter is not None:
                # Use only files matching the specified filename (e.g. "1.mp4")
                self.file_names = [f for f in mp4_list if f.name == view_filter]
            elif "xdof" in subset_path:
                self.file_names = filter_video_files(mp4_list, xdof=True)
            elif "droid" in subset_path.lower():
                self.file_names = [f for f in mp4_list if f.name in ("0.mp4", "1.mp4", "2.mp4")]
            else:
                self.file_names = filter_video_files(mp4_list)
        else:
            raise ValueError(f"No video files found in {subset_path}")

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.file_names[idx]
        while True:
            try:
                video = self.load_video_slice(
                    video_path,
                    self.num_frames,
                    None if self.randomize else 0
                )
                return self.build_data_dict(video)
            except:
                idx = randint(0, len(self) - 1)
                video_path = self.file_names[idx]

    def load_video_slice(
        self,
        video_path: str,
        num_frames: int,
        start_frame: int = None,
        frame_skip: int = 1
    ) -> Tensor:
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        rgb_skip = self.rgb_skip
        effective_frames = total_frames // max(rgb_skip, 1)

        start_frame = start_frame if exists(start_frame) else randint(0, max(0, effective_frames - num_frames))
        raw_start = start_frame * rgb_skip
        cap.set(cv.CAP_PROP_POS_FRAMES, raw_start)

        num_frames = num_frames * frame_skip

        frames = []
        for _ in range(raw_start, raw_start + num_frames * rgb_skip, rgb_skip):
            ret, frame = cap.read()
            if ret:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frames.append(frame)
            else:
                if len(frames) == 0:
                    cap.release()
                    raise ValueError(f"Could not read any frames from {video_path}")
                if self.padding == "none":
                    pass
                elif self.padding == "repeat":
                    frames.extend([frames[-1]] * (num_frames - len(frames)))
                elif self.padding == "zero":
                    frames.extend([torch.zeros_like(frames[-1])] * (num_frames - len(frames)))
                elif self.padding == "random":
                    frames.extend([torch.rand_like(frames[-1])] * (num_frames - len(frames)))
                else:
                    raise ValueError(f"Invalid padding type: {self.padding}")
                break

            for _ in range(rgb_skip - 1):
                cap.read()

        cap.release()

        video = torch.stack(frames[::frame_skip]) / 255.0

        target_ratio = 640 / 480
        if video.shape[2] / video.shape[1] > target_ratio:
            target_height = video.shape[1]
            target_width = int(video.shape[1] * target_ratio)
        elif video.shape[2] / video.shape[1] < target_ratio:
            target_height = int(video.shape[2] / target_ratio)
            target_width = video.shape[2]
        else:
            target_height = video.shape[1]
            target_width = video.shape[2]
        h_crop = (video.shape[1] - target_height) // 2
        w_crop = (video.shape[2] - target_width) // 2
        video = video[:, h_crop:h_crop + target_height, w_crop:w_crop + target_width]
        video = rearrange(video, "t h w c -> c t h w")
        video = F.interpolate(video, (240, 320), mode="bilinear")
        video = rearrange(video, f"c t h w -> {self.output_format}")
        return video

    def build_data_dict(self, video: Tensor) -> Dict:
        if self.color_aug:
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)

        data_dict = {
            "videos": video
        }
        return data_dict


class OriginalVideoDataset(Dataset):
    def __init__(
        self,
        dataset_paths: list,
        split: str = "train",
        padding: str = "repeat",
        randomize: bool = False,
        num_frames: int = 16,
        output_format: str = "t h w c",
        color_aug: bool = True
    ) -> None:
        super(OriginalVideoDataset, self).__init__()
        self.padding = padding
        self.randomize = randomize
        self.num_frames = num_frames
        self.output_format = output_format
        self.color_aug = color_aug

        self.file_names = []
        for subset_path in dataset_paths:
            mp4_list = list(Path(subset_path).rglob("*.mp4"))
            if len(mp4_list) > 0:
                if "xdof" in subset_path:
                    self.file_names.extend(filter_video_files(mp4_list, xdof=True))
                else:
                    self.file_names.extend(filter_video_files(mp4_list))
            else:
                raise ValueError(f"No video files found in {subset_path}")

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.file_names[idx]
        while True:
            try:
                video = self.load_video_slice(
                    video_path,
                    self.num_frames,
                    None if self.randomize else 0
                )
                return self.build_data_dict(video)
            except:
                idx = randint(0, len(self) - 1)
                video_path = self.file_names[idx]

    def load_video_slice(
        self,
        video_path: str,
        num_frames: int,
        start_frame: int = None,
        frame_skip: int = 1
    ) -> Tensor:
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_skip = randint(1, 4)
        num_frames = num_frames * frame_skip

        start_frame = start_frame if exists(start_frame) else randint(0, max(0, total_frames - num_frames))

        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frames.append(frame)
            else:
                if self.padding == "none":
                    pass
                elif self.padding == "repeat":
                    frames.extend([frames[-1]] * (num_frames - len(frames)))
                elif self.padding == "zero":
                    frames.extend([torch.zeros_like(frames[-1])] * (num_frames - len(frames)))
                elif self.padding == "random":
                    frames.extend([torch.rand_like(frames[-1])] * (num_frames - len(frames)))
                else:
                    raise ValueError(f"Invalid padding type: {self.padding}")
                break
        cap.release()

        video = torch.stack(frames[::frame_skip]) / 255.0

        target_ratio = 640 / 480
        if video.shape[2] / video.shape[1] > target_ratio:
            target_height = video.shape[1]
            target_width = int(video.shape[1] * target_ratio)
        elif video.shape[2] / video.shape[1] < target_ratio:
            target_height = int(video.shape[2] / target_ratio)
            target_width = video.shape[2]
        else:
            target_height = video.shape[1]
            target_width = video.shape[2]
        h_crop = (video.shape[1] - target_height) // 2
        w_crop = (video.shape[2] - target_width) // 2
        video = video[:, h_crop:h_crop + target_height, w_crop:w_crop + target_width]
        video = rearrange(video, "t h w c -> c t h w")
        video = F.interpolate(video, (240, 320), mode="bilinear")
        video = rearrange(video, f"c t h w -> {self.output_format}")
        return video

    def build_data_dict(self, video: Tensor) -> Dict:
        if self.color_aug:
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)

        data_dict = {
            "videos": video
        }
        return data_dict


class MultiSourceSamplerDataset(Dataset):
    def __init__(
        self,
        dataset_paths: list,
        split: str = "train",
        samples_per_epoch: int = 1000000,
        sampling_strategy: str = "sample",
        sampling_weights: Optional[list] = None,
        color_aug: bool = True,
        rgb_skips: Optional[list] = None,
        stacking_mode: Optional[str] = None,
        stacking_modes: Optional[list] = None,
        view_maps: Optional[list] = None,
        **kwargs
    ) -> None:
        self.samples_per_epoch = samples_per_epoch

        if split != "train":
            dataset_paths = [p.replace("/train", f"/{split}") for p in dataset_paths]
            print(f"[{split}] Dataset paths: {dataset_paths}")

        n = len(dataset_paths)

        if rgb_skips is None:
            rgb_skips = [1] * n
        elif len(rgb_skips) != n:
            raise ValueError(
                f"rgb_skips length ({len(rgb_skips)}) must match "
                f"dataset_paths length ({n})"
            )

        # Per-dataset stacking: stacking_modes list takes priority over global stacking_mode
        if stacking_modes is None:
            if stacking_mode is not None:
                # Legacy: apply global stacking_mode only to "droid" paths
                stacking_modes = [
                    stacking_mode if "droid" in p.lower() else None
                    for p in dataset_paths
                ]
            else:
                stacking_modes = [None] * n
        elif len(stacking_modes) != n:
            raise ValueError(
                f"stacking_modes length ({len(stacking_modes)}) must match "
                f"dataset_paths length ({n})"
            )

        if view_maps is None:
            view_maps = [None] * n
        elif len(view_maps) != n:
            raise ValueError(
                f"view_maps length ({len(view_maps)}) must match "
                f"dataset_paths length ({n})"
            )

        self.subsets = []
        for subset_path, skip, smode, vmap in tqdm(
            zip(dataset_paths, rgb_skips, stacking_modes, view_maps),
            desc="Loading subsets...", total=n
        ):
            if smode is not None:
                # Use stacked multi-view dataset
                from lam.dataset_stacked import StackedDroidVideoDataset
                stacked_kwargs = dict(
                    subset_path=subset_path,
                    color_aug=color_aug,
                    rgb_skip=skip,
                    stacking_mode=smode,
                    num_frames=kwargs.get('num_frames', 16),
                    randomize=kwargs.get('randomize', False),
                    padding=kwargs.get('padding', 'repeat'),
                    output_format=kwargs.get('output_format', 't h w c'),
                )
                if vmap is not None:
                    stacked_kwargs['wrist_view'] = vmap.get('wrist', '2.mp4')
                    stacked_kwargs['left_view'] = vmap.get('left', '0.mp4')
                    stacked_kwargs['right_view'] = vmap.get('right', '1.mp4')
                self.subsets.append(StackedDroidVideoDataset(**stacked_kwargs))
                print(f"Subset: {subset_path} | Episodes: {len(self.subsets[-1])} | "
                      f"rgb_skip: {skip} | stacking: {smode} | views: {vmap}")
            else:
                # Use regular single-view dataset
                view_filter = vmap.get('view') if vmap is not None else None
                self.subsets.append(VideoDataset(
                    subset_path=subset_path,
                    color_aug=color_aug,
                    rgb_skip=skip,
                    view_filter=view_filter,
                    **kwargs
                ))
                print(f"Subset: {subset_path} | Videos: {len(self.subsets[-1])} | "
                      f"rgb_skip: {skip} | view_filter: {view_filter}")
        print("Number of subsets:", len(self.subsets))

        if sampling_strategy == "manual":
            assert sampling_weights is not None, \
                "sampling_weights must be provided when sampling_strategy='manual'"
            assert len(sampling_weights) == n, \
                f"sampling_weights length ({len(sampling_weights)}) must match dataset_paths length ({n})"
            probs = sampling_weights
        elif sampling_strategy == "sample":
            probs = [len(d) for d in self.subsets]
        elif sampling_strategy == "dataset":
            probs = [1 for _ in self.subsets]
        elif sampling_strategy == "log":
            probs = [math.log(len(d)) if len(d) else 0 for d in self.subsets]
        elif sampling_strategy == "pi":
            probs = [len(d) ** 0.43 for d in self.subsets]
        else:
            raise ValueError(f"Unavailable sampling strategy: {sampling_strategy}")
        total_prob = sum(probs)
        assert total_prob > 0, "No sample is available"
        self.sample_probs = [x / total_prob for x in probs]
        print("Sample probabilities:", self.sample_probs)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict:
        subset = choices(self.subsets, self.sample_probs)[0]
        sample_idx = randint(0, len(subset) - 1)
        sample_item = subset[sample_idx]
        return sample_item


class LightningVideoDataset(LightningDataset):
    def __init__(
        self,
        dataset_paths: list,
        padding: str = "repeat",
        randomize: bool = False,
        num_frames: int = 16,
        output_format: str = "t h w c",
        samples_per_epoch: int = 1000000,
        sampling_strategy: str = "sample",
        sampling_weights: Optional[list] = None,
        rgb_skips: Optional[list] = None,
        stacking_mode: Optional[str] = None,
        stacking_modes: Optional[list] = None,
        view_maps: Optional[list] = None,
        **kwargs
    ) -> None:
        super(LightningVideoDataset, self).__init__(**kwargs)
        self.dataset_paths = dataset_paths
        self.padding = padding
        self.randomize = randomize
        self.num_frames = num_frames
        self.output_format = output_format
        self.samples_per_epoch = samples_per_epoch
        self.sampling_strategy = sampling_strategy
        self.sampling_weights = sampling_weights
        self.rgb_skips = rgb_skips
        self.stacking_mode = stacking_mode
        self.stacking_modes = stacking_modes
        self.view_maps = view_maps

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = MultiSourceSamplerDataset(
                dataset_paths=self.dataset_paths,
                split="train",
                padding=self.padding,
                randomize=self.randomize,
                num_frames=self.num_frames,
                output_format=self.output_format,
                samples_per_epoch=self.samples_per_epoch,
                sampling_strategy=self.sampling_strategy,
                sampling_weights=self.sampling_weights,
                rgb_skips=self.rgb_skips,
                stacking_mode=self.stacking_mode,
                stacking_modes=self.stacking_modes,
                view_maps=self.view_maps,
            )

            self.val_dataset = MultiSourceSamplerDataset(
                dataset_paths=self.dataset_paths,
                split="val",
                padding=self.padding,
                randomize=self.randomize,
                num_frames=self.num_frames,
                output_format=self.output_format,
                samples_per_epoch=self.samples_per_epoch // 100,
                sampling_strategy=self.sampling_strategy,
                sampling_weights=self.sampling_weights,
                rgb_skips=self.rgb_skips,
                stacking_mode=self.stacking_mode,
                stacking_modes=self.stacking_modes,
                view_maps=self.view_maps,
                color_aug=False
            )

        elif stage == "test":
            self.test_dataset = OriginalVideoDataset(
                dataset_paths=self.dataset_paths,
                split="test",
                padding=self.padding,
                randomize=self.randomize,
                num_frames=self.num_frames,
                output_format=self.output_format,
                color_aug=False
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")