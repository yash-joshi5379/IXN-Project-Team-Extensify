from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_v2_common import (
    CAMERA_KEYS,
    PROCESSED_ROOT,
    RAW_ROOT,
    data_path,
    ensure_dataset_dirs,
    normalize_parquet_indices,
    require_episode_files,
    video_path,
)
from generate_meta_v2 import rebuild_metadata


def trim_parquet(input_root: Path, output_root: Path, episode: int, end_frame: int) -> None:
    src = data_path(input_root, episode)
    dst = data_path(output_root, episode)
    df = pd.read_parquet(src)
    if end_frame <= 0:
        raise ValueError("end_frame must be greater than 0")
    if end_frame > len(df):
        raise ValueError(f"end_frame {end_frame} is beyond parquet length {len(df)}")

    trimmed = df.iloc[:end_frame].copy()
    trimmed["episode_index"] = episode
    trimmed["frame_index"] = np.arange(len(trimmed), dtype=np.int64)
    trimmed["task_index"] = 0
    trimmed["next.done"] = False
    trimmed.loc[trimmed.index[-1], "next.done"] = True
    dst.parent.mkdir(parents=True, exist_ok=True)
    trimmed.to_parquet(dst, index=False)
    print(f"[episode_{episode:06d}] Wrote trimmed parquet: {dst} ({len(trimmed)} rows)")


def process_video(input_path: Path, output_path: Path, end_frame: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-frames:v",
        str(end_frame),
        "-vf",
        "scale=256:256:force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"  -> Saved 256x256 video: {output_path}")


def process_episode(
    input_root: Path,
    output_root: Path,
    episode: int,
    end_frame: int,
    update_meta: bool,
) -> None:
    require_episode_files(input_root, episode)
    ensure_dataset_dirs(output_root)
    trim_parquet(input_root, output_root, episode, end_frame)

    for camera_key in CAMERA_KEYS:
        print(f"[episode_{episode:06d}] Processing {camera_key}")
        process_video(video_path(input_root, camera_key, episode), video_path(output_root, camera_key, episode), end_frame)

    total_frames = normalize_parquet_indices(output_root)
    print(f"Normalized processed parquet indices across {total_frames} frames.")

    if update_meta:
        source_meta = input_root / "meta"
        rebuild_metadata(output_root, source_meta=source_meta if source_meta.exists() else None)

    print(f"Success: episode_{episode:06d} processed with {end_frame} frames.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim and resize one dataset-v2 episode.")
    parser.add_argument("episode", type=int)
    parser.add_argument("end_frame", type=int)
    parser.add_argument("--input-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--no-meta", action="store_true")
    args = parser.parse_args()

    process_episode(
        input_root=args.input_root,
        output_root=args.output_root,
        episode=args.episode,
        end_frame=args.end_frame,
        update_meta=not args.no_meta,
    )


if __name__ == "__main__":
    main()
