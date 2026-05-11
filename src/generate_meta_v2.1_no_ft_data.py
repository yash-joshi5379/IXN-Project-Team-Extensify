#!/usr/bin/env python3
"""
Generate lerobot v2.1 meta/ files for a cleaned/trimmed dataset.

Produces:
  meta/episodes_stats.jsonl
  meta/episodes.jsonl
  meta/info.json
  meta/modality.json
  meta/tasks.jsonl

Usage (run from the directory that CONTAINS dataset-v2-pro/):
  python generate_meta.py

Dependencies:
  pip install pandas pyarrow opencv-python tqdm numpy
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration – edit these if your paths or task description differ
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = Path("dataset-v2-pro")
DATA_DIR     = BASE_DIR / "data"  / "chunk-000"
VIDEO_BASE   = BASE_DIR / "videos" / "chunk-000"
META_DIR     = BASE_DIR / "meta"

NUM_EPISODES = 121
FPS          = 30
TASK         = ("Grasp the red cylinder and place it vertically into the circular slot within the white box.")

CAMERA_KEYS  = [
    "observation.images.0_femtobolt",
    "observation.images.1_gemini330",
]

# Sample 1-in-N frames for image stats (5 ≈ same density as the raw dataset).
# Increase this number to go faster at the cost of less precise image stats.
IMAGE_SAMPLE_EVERY = 2


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: stat blocks
# ─────────────────────────────────────────────────────────────────────────────

def vector_stats(arr: np.ndarray) -> dict:
    """
    Stats for a (T, D) float array.
    Produces lists of length D for min/max/mean/std, count = [T].
    """
    arr = arr.astype(np.float64)
    return {
        "min":   arr.min(axis=0).tolist(),
        "max":   arr.max(axis=0).tolist(),
        "mean":  arr.mean(axis=0).tolist(),
        "std":   arr.std(axis=0).tolist(),
        "count": [len(arr)],
    }


def scalar_stats(arr: np.ndarray) -> dict:
    """
    Stats for a (T,) array (timestamp, frame_index, etc.).
    Produces single-element lists.
    """
    arr = arr.astype(np.float64)
    return {
        "min":   [float(arr.min())],
        "max":   [float(arr.max())],
        "mean":  [float(arr.mean())],
        "std":   [float(arr.std())],
        "count": [len(arr)],
    }


def image_stats(video_path: Path, sample_every: int = IMAGE_SAMPLE_EVERY) -> dict:
    """
    Per-channel pixel stats sampled from a video file.
    Output shape matches lerobot format: lists of [[[value]]] per channel.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    channels: list = [[], [], []]
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            for c in range(3):
                channels[c].append(rgb[:, :, c].ravel())
        frame_idx += 1
    cap.release()

    n_sampled = len(channels[0])
    if n_sampled == 0:
        raise RuntimeError(f"No frames read from {video_path}")

    result: dict = {"min": [], "max": [], "mean": [], "std": [], "count": [n_sampled]}
    for c in range(3):
        px = np.concatenate(channels[c])
        result["min"].append([[float(px.min())]])
        result["max"].append([[float(px.max())]])
        result["mean"].append([[float(px.mean())]])
        result["std"].append([[float(px.std())]])

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: detect video codec via ffprobe (falls back to "av1")
# ─────────────────────────────────────────────────────────────────────────────

def detect_codec(video_path: Path) -> str:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        codec = result.stdout.strip()
        return codec if codec else "av1"
    except Exception:
        return "av1"


def detect_pix_fmt(video_path: Path) -> str:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        fmt = result.stdout.strip()
        return fmt if fmt else "yuv420p"
    except Exception:
        return "yuv420p"


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: dtype string for info.json
# ─────────────────────────────────────────────────────────────────────────────

def dtype_str(dt: np.dtype) -> str:
    if dt == np.bool_:
        return "bool"
    if np.issubdtype(dt, np.floating):
        return str(dt)   # float32, float64 …
    if np.issubdtype(dt, np.integer):
        return str(dt)   # int32, int64 …
    return str(dt)


# ─────────────────────────────────────────────────────────────────────────────
#  Friendly names for known multi-dim columns
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_NAMES: dict = {
    "action":             [f"action_{i}" for i in range(8)],
    "observation.state":  [f"state_{i}"  for i in range(8)],
    "observation.effort": [f"effort_{i}" for i in range(8)],
    "observation.qvel":   [f"qvel_{i}"   for i in range(8)],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def check_paths() -> None:
    missing = []
    for ep in range(NUM_EPISODES):
        p = DATA_DIR / f"episode_{ep:06d}.parquet"
        if not p.exists():
            missing.append(str(p))
    if missing:
        print(f"ERROR: {len(missing)} parquet file(s) missing, e.g. {missing[0]}")
        sys.exit(1)

    for cam in CAMERA_KEYS:
        for ep in range(NUM_EPISODES):
            p = VIDEO_BASE / cam / f"episode_{ep:06d}.mp4"
            if not p.exists():
                missing.append(str(p))
    if missing:
        print(f"ERROR: {len(missing)} video file(s) missing, e.g. {missing[0]}")
        sys.exit(1)

    print(f"✓ All {NUM_EPISODES} parquet files and "
          f"{NUM_EPISODES * len(CAMERA_KEYS)} video files found.")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    check_paths()

    # ── Detect codec once from episode 0 ─────────────────────────────────────
    sample_vid = VIDEO_BASE / CAMERA_KEYS[0] / "episode_000000.mp4"
    codec   = detect_codec(sample_vid)
    pix_fmt = detect_pix_fmt(sample_vid)
    print(f"  Detected codec={codec!r}, pix_fmt={pix_fmt!r} from first video.")

    # ── Inspect parquet schema from episode 0 ────────────────────────────────
    sample_df   = pd.read_parquet(DATA_DIR / "episode_000000.parquet")
    parquet_cols = list(sample_df.columns)
    print(f"  Parquet columns: {parquet_cols}")

    # Classify columns
    vector_cols: list = []   # arrays stored per row
    scalar_cols: list = []   # single scalars per row

    for col in parquet_cols:
        val = sample_df[col].iloc[0]
        if isinstance(val, (list, np.ndarray)):
            vector_cols.append(col)
        else:
            scalar_cols.append(col)

    print(f"  Vector cols : {vector_cols}")
    print(f"  Scalar cols : {scalar_cols}")

    # ── Get video resolution ──────────────────────────────────────────────────
    cam_shape: dict = {}
    for cam in CAMERA_KEYS:
        cap = cv2.VideoCapture(str(VIDEO_BASE / cam / "episode_000000.mp4"))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        cam_shape[cam] = [h, w, 3]
        print(f"  {cam}: {h}×{w}")

    # ── Process episodes ──────────────────────────────────────────────────────
    all_ep_stats: list = []
    all_ep_info:  list = []
    total_frames = 0

    print(f"\nComputing stats for {NUM_EPISODES} episodes "
          f"(image sampling: 1 in {IMAGE_SAMPLE_EVERY} frames) …")

    for ep in tqdm(range(NUM_EPISODES)):
        df = pd.read_parquet(DATA_DIR / f"episode_{ep:06d}.parquet")
        T  = len(df)
        total_frames += T
        stats: dict = {}

        # Vector features from parquet
        for col in vector_cols:
            arr = np.stack(df[col].values)
            stats[col] = vector_stats(arr)

        # Scalar features from parquet
        for col in scalar_cols:
            arr = df[col].to_numpy()
            stats[col] = scalar_stats(arr)

        # Image features from video files
        for cam in CAMERA_KEYS:
            vid = VIDEO_BASE / cam / f"episode_{ep:06d}.mp4"
            stats[cam] = image_stats(vid)

        all_ep_stats.append({"episode_index": ep, "stats": stats})
        all_ep_info.append({"episode_index": ep, "tasks": [TASK], "length": T})

    print(f"\n  Total frames across all episodes: {total_frames}")

    # ── Write episodes_stats.jsonl ────────────────────────────────────────────
    out = META_DIR / "episodes_stats.jsonl"
    with out.open("w") as f:
        for row in all_ep_stats:
            f.write(json.dumps(row) + "\n")
    print(f"✓ {out}")

    # ── Write episodes.jsonl ──────────────────────────────────────────────────
    out = META_DIR / "episodes.jsonl"
    with out.open("w") as f:
        for row in all_ep_info:
            f.write(json.dumps(row) + "\n")
    print(f"✓ {out}")

    # ── Write tasks.jsonl ─────────────────────────────────────────────────────
    out = META_DIR / "tasks.jsonl"
    with out.open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    print(f"✓ {out}")

    # ── Write modality.json ───────────────────────────────────────────────────
    modality = {
        "state": {
            "0_xarm7+xarmgripper-arm":  {"start": 0, "end": 7},
            "1_xarm7+xarmgripper-hand": {"start": 7, "end": 8},
        },
        "effort": {
            "0_xarm7+xarmgripper-arm":  {"start": 0, "end": 7},
            "1_xarm7+xarmgripper-hand": {"start": 7, "end": 8},
        },
        "action": {
            "0_xarm7+xarmgripper-arm":  {"start": 0, "end": 7},
            "1_xarm7+xarmgripper-hand": {"start": 7, "end": 8},
        },
        "video": {
            "0_femtobolt":  {"original_key": "observation.images.0_femtobolt"},
            "1_gemini330":  {"original_key": "observation.images.1_gemini330"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        },
    }
    out = META_DIR / "modality.json"
    out.write_text(json.dumps(modality, indent=4))
    print(f"✓ {out}")

    # ── Build info.json ───────────────────────────────────────────────────────
    features: dict = {}

    for col in parquet_cols:
        val = sample_df[col].iloc[0]
        if isinstance(val, (list, np.ndarray)):
            arr   = np.array(val)
            shape = list(arr.shape)
            names = KNOWN_NAMES.get(col, [f"{col}_{i}" for i in range(shape[0])])
            features[col] = {
                "dtype":  "float32",
                "shape":  shape,
                "names":  names,
            }
        else:
            dt = sample_df[col].dtype
            features[col] = {
                "dtype":  dtype_str(dt),
                "shape":  [1],
                "names":  None,
            }

    # Add camera / video entries
    for cam in CAMERA_KEYS:
        features[cam] = {
            "dtype": "video",
            "shape": cam_shape[cam],          # [H, W, 3]
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps":          float(FPS),
                "video.codec":        codec,
                "video.pix_fmt":      pix_fmt,
                "video.is_depth_map": False,
                "has_audio":          False,
            },
        }

    info = {
        "codebase_version":  "v2.1",
        "robot_type":        "stationary",
        "total_episodes":    NUM_EPISODES,
        "total_frames":      total_frames,
        "total_tasks":       1,
        "total_videos":      NUM_EPISODES * len(CAMERA_KEYS),
        "total_chunks":      1,
        "chunks_size":       1000,
        "fps":               FPS,
        "splits":            {"train": f"0:{NUM_EPISODES}"},
        "data_path":         "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path":        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "depth_images_path": None,
        "features":          features,
    }

    out = META_DIR / "info.json"
    out.write_text(json.dumps(info, indent=4))
    print(f"✓ {out}")

    print(f"\nAll 5 meta files written to {META_DIR}/")
    print(f"  Episodes : {NUM_EPISODES}")
    print(f"  Frames   : {total_frames}")
    print(f"  Videos   : {NUM_EPISODES * len(CAMERA_KEYS)}")


if __name__ == "__main__":
    main()
