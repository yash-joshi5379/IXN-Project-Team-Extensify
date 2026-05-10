from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

from dataset_v2_common import CAMERA_KEYS, PROCESSED_ROOT, data_path, video_path


def video_info(path: Path) -> tuple[int, tuple[int, int, int] | None]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    return frame_count, tuple(frame.shape) if ok else None


def check_episode(dataset_root: Path, episode: int, expected_frames: int | None) -> bool:
    ok = True
    parquet_path = data_path(dataset_root, episode)
    if not parquet_path.exists():
        print(f"FAIL: missing parquet {parquet_path}")
        return False

    df = pd.read_parquet(parquet_path)
    print(f"episode_{episode:06d} has {len(df)} rows of data")

    if expected_frames is not None and len(df) != expected_frames:
        print(f"FAIL: expected {expected_frames} rows")
        ok = False

    for column in ["episode_index", "frame_index", "index", "task_index", "next.done"]:
        if column not in df.columns:
            print(f"FAIL: missing column {column}")
            ok = False

    if "episode_index" in df.columns and set(df["episode_index"].unique()) != {episode}:
        print("FAIL: episode_index values do not match the file episode number")
        ok = False

    if "frame_index" in df.columns and df["frame_index"].tolist() != list(range(len(df))):
        print("FAIL: frame_index is not consecutive from 0")
        ok = False

    if "next.done" in df.columns:
        done_count = int(df["next.done"].sum())
        done_last = bool(df["next.done"].iloc[-1]) if len(df) else False
        print(f"next.done true count: {done_count}; final row done: {done_last}")
        if done_count != 1 or not done_last:
            print("FAIL: next.done should have exactly one spike at the final row")
            ok = False

    for camera_key in CAMERA_KEYS:
        path = video_path(dataset_root, camera_key, episode)
        if not path.exists():
            print(f"FAIL: missing video {path}")
            ok = False
            continue
        frames, shape = video_info(path)
        print(f"{camera_key}: {frames} frames; first frame shape: {shape}")
        if frames != len(df):
            print(f"FAIL: {camera_key} frame count does not match parquet rows")
            ok = False
        if shape != (256, 256, 3):
            print(f"FAIL: {camera_key} first frame is not 256x256x3")
            ok = False

    print("PASS" if ok else "FAIL")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Check one processed dataset-v2 episode.")
    parser.add_argument("episode", type=int)
    parser.add_argument("--dataset-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--expected-frames", type=int, default=None)
    args = parser.parse_args()

    if not check_episode(args.dataset_root, args.episode, args.expected_frames):
        sys.exit(1)


if __name__ == "__main__":
    main()
