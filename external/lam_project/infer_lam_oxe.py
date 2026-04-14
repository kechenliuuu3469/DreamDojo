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
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from lam.model import LAM


# Module-level decoder state, set once in main() and read by the video readers.
# "cv2"    -> OpenCV VideoCapture (CPU, software)
# "decord" -> decord.VideoReader  (CPU or NVDEC, depending on _DECORD_CTX)
_DECODER = "cv2"
_DECORD_CTX = None  # set to decord.cpu(0) or decord.gpu(0) when decoder="decord"


def _set_decoder(decoder: str, decord_gpu: bool):
    global _DECODER, _DECORD_CTX
    _DECODER = decoder
    if decoder == "decord":
        from decord import cpu, gpu, bridge  # noqa: F401
        _DECORD_CTX = gpu(0) if decord_gpu else cpu(0)
        print(f"[decoder] decord ctx={'gpu(0)' if decord_gpu else 'cpu(0)'}")
    else:
        _DECORD_CTX = None
        print("[decoder] cv2.VideoCapture")


# Per-dataset pipeline config, mirroring config/lam_joint_all.yaml exactly.
# stacking_mode:
#   None         -> single view. view_map["view"] names the mp4 explicitly;
#                   if view_map is None we fall back to filter_video_files
#                   (the same rule used by the training VideoDataset).
#   "dreamzero"  -> wrist on top, left+right below  (2H x 2W)
#   "horizontal" -> left|right side by side         (H  x 2W)
DATASET_CFG = {
    "egodex":          {"rgb_skip": 3, "stacking_mode": None,         "view_map": None,
                        "manifest": "annotations/annotations/manifest.json"},
    "bridge":          {"rgb_skip": 1, "stacking_mode": None,         "view_map": {"view": "rgb.mp4"}},
    "fractal":         {"rgb_skip": 3, "stacking_mode": None,         "view_map": {"view": "0.mp4"}},
    "droid":           {"rgb_skip": 1, "stacking_mode": "dreamzero",  "view_map": {"wrist": "2.mp4", "left": "0.mp4", "right": "1.mp4"}},
    "bc_z":            {"rgb_skip": 3, "stacking_mode": None,         "view_map": {"view": "0.mp4"}},
    "fmb":             {"rgb_skip": 3, "stacking_mode": "dreamzero",  "view_map": {"wrist": "4.mp4", "left": "0.mp4", "right": "2.mp4"}},
    "language_table":  {"rgb_skip": 3, "stacking_mode": None,         "view_map": None},
    "taco_play":       {"rgb_skip": 3, "stacking_mode": None,         "view_map": {"view": "3.mp4"}},
    "furniture_bench": {"rgb_skip": 3, "stacking_mode": "horizontal", "view_map": {"left": "0.mp4", "right": "1.mp4"}},
    "roboturk":        {"rgb_skip": 1, "stacking_mode": None,         "view_map": None},
}


def filter_video_files(mp4_paths):
    """Mirror of lam/dataset.filter_video_files (xdof=False branch)."""
    return [
        p for p in mp4_paths
        if "left"   not in p.name.lower()
        and "right"  not in p.name.lower()
        and "resize" not in p.name.lower()
        and "pad"    not in p.name.lower()
    ]

TARGET_H, TARGET_W = 240, 320
TARGET_RATIO = 640 / 480


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def load_model(ckpt_path, device,
               lam_model_dim=1024, lam_latent_dim=32, lam_patch_size=16,
               lam_enc_blocks=24, lam_dec_blocks=24, lam_num_heads=16):
    """Load LAM with the exact architecture used for joint_all3 training."""
    print(f"[lam] loading {ckpt_path}")
    print(f"[lam] arch: dim={lam_model_dim} enc={lam_enc_blocks} "
          f"dec={lam_dec_blocks} heads={lam_num_heads} "
          f"latent={lam_latent_dim} patch={lam_patch_size}")
    model = LAM.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        lam_model_dim=lam_model_dim,
        lam_latent_dim=lam_latent_dim,
        lam_patch_size=lam_patch_size,
        lam_enc_blocks=lam_enc_blocks,
        lam_dec_blocks=lam_dec_blocks,
        lam_num_heads=lam_num_heads,
    )
    model = model.to(device).eval()
    print(f"[lam] params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model


# ---------------------------------------------------------------------------
# video reading
# ---------------------------------------------------------------------------
def _read_cv2(video_path, rgb_skip):
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


def _read_decord(video_path, rgb_skip):
    """decord reads RGB directly and can seek/batch-decode; ~3–5x faster for strided reads."""
    from decord import VideoReader
    vr = VideoReader(str(video_path), ctx=_DECORD_CTX)
    n = len(vr)
    if n <= 0:
        return []
    idxs = list(range(0, n, rgb_skip))
    if not idxs:
        return []
    batch = vr.get_batch(idxs).asnumpy()  # [N, H, W, 3] uint8, RGB
    # Return list-of-arrays to match the cv2 reader's contract used by the composers.
    return [batch[i] for i in range(batch.shape[0])]


def read_strided_frames_np(video_path, rgb_skip):
    """Return a list of HxWx3 uint8 numpy arrays, keeping every rgb_skip-th frame."""
    if _DECODER == "decord":
        return _read_decord(video_path, rgb_skip)
    return _read_cv2(video_path, rgb_skip)


def compose_single_view(episode_dir, view_map, rgb_skip):
    if view_map is not None and "view" in view_map:
        path = episode_dir / view_map["view"]
        if not path.exists():
            return None
    else:
        # Fallback: match the training VideoDataset's filter_video_files rule.
        candidates = filter_video_files(sorted(episode_dir.glob("*.mp4")))
        if not candidates:
            return None
        path = candidates[0]
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


def _frames_to_tensor(frames):
    """Common post-processing: list of HxWx3 uint8 -> [N, 240, 320, 3] float tensor."""
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
    return video


def build_episode_tensor_from_file(mp4_path, rgb_skip):
    """Single-view episode from a direct .mp4 path (e.g. egodex manifest entry)."""
    if not Path(mp4_path).exists():
        return None
    return _frames_to_tensor(read_strided_frames_np(mp4_path, rgb_skip))


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
    return _frames_to_tensor(frames)


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_latents_for_video(model, frames, device, batch_size, amp_dtype=None):
    """frames: [N, H, W, 3] -> [N-1, 32] latent actions (consecutive pairs)."""
    pairs = torch.stack([frames[:-1], frames[1:]], dim=1)  # [N-1, 2, H, W, 3]
    outs = []
    use_amp = amp_dtype is not None and device.startswith("cuda")
    for i in range(0, pairs.shape[0], batch_size):
        batch = {"videos": pairs[i:i + batch_size].to(device, non_blocking=True)}
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                z_mu = model.lam(batch)["z_mu"]
        else:
            z_mu = model.lam(batch)["z_mu"]
        outs.append(z_mu.float().cpu())
    return torch.cat(outs, dim=0)


def _enumerate_episodes(dataset_root, dataset, cfg):
    """
    Return (items, n_total) where items is a list of dicts with keys:
      - source: 'dir' or 'file'
      - path:   Path to the episode dir (dir mode) or direct mp4 (file mode)
      - out_rel: output subpath under <dataset>/<out_subdir>/, excluding filename
      - desc:    short human label for logs
    """
    root = Path(dataset_root) / dataset
    manifest = cfg.get("manifest")

    if manifest is not None:
        mpath = root / manifest
        with open(mpath) as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError(f"{mpath}: expected top-level list, got {type(entries).__name__}")
        entries = [e for e in entries if e.get("split", "train") == "train"]
        # Stable order: task, then numeric episode id if possible.
        def sort_key(e):
            ep_id = e.get("episode_id", "")
            try:
                return (e.get("task", ""), int(ep_id), ep_id)
            except (TypeError, ValueError):
                return (e.get("task", ""), 0, ep_id)
        entries.sort(key=sort_key)

        items = []
        for e in entries:
            vrel = e["video_path"]                         # e.g. videos/train/<task>/<id>.mp4
            vpath = root / vrel
            out_rel = Path(vrel.replace("videos/", "", 1)).with_suffix("")  # train/<task>/<id>
            items.append({
                "source": "file",
                "path": vpath,
                "out_rel": out_rel,
                "desc": f"{e.get('task','?')}/{e.get('episode_id','?')}",
            })
        return items, len(items)

    # Default: episode-dir layout.
    videos_dir = root / "videos" / "train"
    episodes = sorted(
        [p for p in videos_dir.iterdir() if p.is_dir()],
        key=lambda p: (len(p.name), p.name),
    )
    items = [
        {
            "source": "dir",
            "path": p,
            "out_rel": Path("train") / p.name,
            "desc": p.name,
        }
        for p in episodes
    ]
    return items, len(items)


def _decode_item(item, cfg):
    """Run the decode pipeline for one episode. Returns (frames_tensor_or_None, error_or_None)."""
    try:
        if item["source"] == "file":
            frames = build_episode_tensor_from_file(item["path"], cfg["rgb_skip"])
        else:
            frames = build_episode_tensor(item["path"], cfg)
        return frames, None
    except Exception as e:
        return None, e


def _iter_decoded(items, cfg, prefetch):
    """Yield (item, frames, err) in order.

    With prefetch=True, decode runs in a background thread so decoding of
    episode N+1 overlaps with compute/IO of episode N.
    """
    if not items:
        return
    if not prefetch:
        for item in items:
            frames, err = _decode_item(item, cfg)
            yield item, frames, err
        return

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_decode_item, items[0], cfg)
        for idx in range(len(items)):
            frames, err = fut.result()
            if idx + 1 < len(items):
                fut = ex.submit(_decode_item, items[idx + 1], cfg)
            yield items[idx], frames, err


def process_dataset(model, dataset_root, dataset, cfg, percent,
                    device, batch_size, out_subdir, skip_existing,
                    shard_idx=0, num_shards=1, amp_dtype=None,
                    prefetch=False):
    out_base = Path(dataset_root) / dataset / out_subdir

    items, n_total = _enumerate_episodes(dataset_root, dataset, cfg)
    n_keep = max(1, int(round(n_total * percent / 100.0))) if n_total else 0
    items = items[:n_keep]
    # Stride-shard so every shard sees a balanced mix of short/long episodes.
    items = items[shard_idx::num_shards]

    # Pre-filter skip_existing so the prefetch worker never decodes videos
    # whose latents are already on disk.
    n_skip = 0
    todo = []
    for item in items:
        out_dir = out_base / item["out_rel"]
        out_path = out_dir / "latent_actions.npy"
        if skip_existing and out_path.exists():
            n_skip += 1
            continue
        item["_out_dir"] = out_dir
        item["_out_path"] = out_path
        todo.append(item)

    src = "manifest" if cfg.get("manifest") else "dir"
    print(f"\n[{dataset}] src={src} rgb_skip={cfg['rgb_skip']} "
          f"stacking={cfg['stacking_mode']} views={cfg['view_map']} "
          f"| episodes {n_keep}/{n_total} ({percent}%) "
          f"| shard {shard_idx}/{num_shards} -> {len(items)} eps "
          f"({n_skip} already done, {len(todo)} to do) "
          f"| prefetch={prefetch} | out -> {out_base}")

    n_ok = n_fail = 0
    for item, frames, err in tqdm(
        _iter_decoded(todo, cfg, prefetch), total=len(todo), desc=dataset
    ):
        if err is not None:
            print(f"  [fail decode] {item['desc']}: {err}")
            n_fail += 1
            continue
        if frames is None:
            n_fail += 1
            continue
        try:
            latents = extract_latents_for_video(model, frames, device, batch_size, amp_dtype)
            item["_out_dir"].mkdir(parents=True, exist_ok=True)
            np.save(item["_out_path"], latents.numpy().astype(np.float32))
            n_ok += 1
        except Exception as e:
            print(f"  [fail compute] {item['desc']}: {e}")
            n_fail += 1

    print(f"[{dataset}] done: ok={n_ok} skip={n_skip} fail={n_fail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--dataset_root", default="/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Datasets to process. If omitted, auto-discovers every "
                         "subdir of --dataset_root that has videos/train AND a known "
                         "entry in DATASET_CFG.")
    ap.add_argument("--percent", type=float, default=100.0,
                    help="Process the first X%% of episodes per dataset.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_subdir", default="latent_actions_lam")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--shard_idx", type=int, default=0,
                    help="Shard index for multi-GPU sharding (0..num_shards-1).")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Total number of shards. Episodes are strided across shards.")
    ap.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16",
                    help="Autocast dtype for the forward pass (H100: use bf16).")
    ap.add_argument("--decoder", choices=["cv2", "decord"], default="decord",
                    help="Video decoder. 'decord' is much faster for strided reads.")
    ap.add_argument("--decord_gpu", action="store_true",
                    help="Use decord with NVDEC (gpu(0)) instead of CPU decode.")
    ap.add_argument("--prefetch", action="store_true", default=True,
                    help="Decode episode N+1 in a background thread while N runs on GPU.")
    ap.add_argument("--no_prefetch", dest="prefetch", action="store_false")
    args = ap.parse_args()

    assert 0 <= args.shard_idx < args.num_shards, \
        f"shard_idx={args.shard_idx} must be in [0, {args.num_shards})"

    if not args.datasets:
        root = Path(args.dataset_root)
        present = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and (p / "videos" / "train").is_dir()
        )
        args.datasets = [d for d in present if d in DATASET_CFG]
        missing = [d for d in present if d not in DATASET_CFG]
        print(f"[auto] discovered datasets under {root}: {present}")
        print(f"[auto] processing (known in DATASET_CFG): {args.datasets}")
        if missing:
            print(f"[auto] skipping (no DATASET_CFG entry): {missing}")
        if not args.datasets:
            raise SystemExit("[auto] no known datasets found; pass --datasets explicitly")
    else:
        for ds in args.datasets:
            if ds not in DATASET_CFG:
                raise ValueError(f"unknown dataset {ds}; known: {list(DATASET_CFG)}")

    amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    if amp_dtype is not None:
        print(f"[lam] autocast enabled: {args.precision}")

    _set_decoder(args.decoder, args.decord_gpu)

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
            shard_idx=args.shard_idx,
            num_shards=args.num_shards,
            amp_dtype=amp_dtype,
            prefetch=args.prefetch,
        )


if __name__ == "__main__":
    main()
