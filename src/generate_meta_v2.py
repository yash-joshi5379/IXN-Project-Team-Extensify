from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_v2_common import (
    CAMERA_KEYS,
    FPS,
    MOTOR_NAMES,
    PROCESSED_ROOT,
    RAW_ROOT,
    SCALAR_COLUMNS,
    TASK_DESCRIPTION,
    VECTOR_COLUMNS,
    data_path,
    ensure_dataset_dirs,
    feature_stats,
    ffprobe_video,
    image_stats,
    sorted_episode_indices,
    stack_column,
    video_path,
    write_jsonl,
)


def load_source_info(source_meta: Path | None) -> dict:
    if source_meta is None:
        return {}
    path = source_meta / "info.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def infer_features(dataset_root: Path, dfs: list[pd.DataFrame], source_info: dict) -> dict:
    first = dfs[0]
    source_features = source_info.get("features", {})
    features: dict = {}

    for column in VECTOR_COLUMNS:
        if column not in first.columns:
            continue
        sample = np.asarray(first[column].iloc[0])
        names = source_features.get(column, {}).get("names")
        if names is None and sample.shape == (8,):
            names = {"motors": MOTOR_NAMES}
        features[column] = {
            "dtype": "float32",
            "shape": [int(sample.shape[0])],
            "names": names,
        }

    first_episode = int(first["episode_index"].iloc[0]) if "episode_index" in first.columns else 0
    for camera_key in CAMERA_KEYS:
        info = ffprobe_video(video_path(dataset_root, camera_key, first_episode))
        features[camera_key] = {
            "dtype": "video",
            "shape": [info["height"], info["width"], 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": info["height"],
                "video.width": info["width"],
                "video.codec": info["codec"],
                "video.pix_fmt": info["pix_fmt"],
                "video.is_depth_map": False,
                "video.fps": int(round(info["fps"])) if info["fps"] else FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    for column in SCALAR_COLUMNS:
        if column not in first.columns:
            continue
        if column == "next.done":
            dtype = "bool"
        elif str(first[column].dtype).startswith("float"):
            dtype = "float32"
        else:
            dtype = "int64"
        features[column] = {
            "dtype": dtype,
            "shape": [1],
            "names": None,
        }

    return features


def compute_stats(dfs: list[pd.DataFrame]) -> dict:
    stats: dict = {}
    for column in VECTOR_COLUMNS:
        if column in dfs[0].columns:
            stats[column] = feature_stats(stack_column(dfs, column))

    if "next.done" in dfs[0].columns:
        values = np.concatenate([df["next.done"].astype(np.float32).to_numpy() for df in dfs]).reshape(-1, 1)
        stats["next.done"] = feature_stats(values)

    for camera_key in CAMERA_KEYS:
        stats[camera_key] = image_stats()

    return stats


def compute_episode_stats(df: pd.DataFrame) -> dict:
    stats: dict = {}
    for column in VECTOR_COLUMNS:
        if column in df.columns:
            arr = np.stack([np.asarray(value, dtype=np.float32) for value in df[column].tolist()])
            stats[column] = feature_stats(arr)

    if "next.done" in df.columns:
        stats["next.done"] = feature_stats(df["next.done"].astype(np.float32).to_numpy().reshape(-1, 1))

    for camera_key in CAMERA_KEYS:
        stats[camera_key] = image_stats()

    return stats


def rebuild_metadata(dataset_root: Path, source_meta: Path | None = None) -> None:
    dataset_root = dataset_root.resolve()
    ensure_dataset_dirs(dataset_root)
    episodes = sorted_episode_indices(dataset_root)
    if not episodes:
        raise FileNotFoundError(f"No parquet episodes found in {dataset_root / 'data/chunk-000'}")

    dfs = [pd.read_parquet(data_path(dataset_root, episode)) for episode in episodes]
    source_info = load_source_info(source_meta)
    total_frames = sum(len(df) for df in dfs)
    features = infer_features(dataset_root, dfs, source_info)
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v2.1",
        "robot_type": source_info.get("robot_type", "stationary"),
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episodes) * len(CAMERA_KEYS),
        "total_chunks": 1,
        "chunks_size": source_info.get("chunks_size", 1000),
        "fps": source_info.get("fps", FPS),
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "depth_images_path": source_info.get("depth_images_path"),
        "features": features,
    }

    (meta_dir / "info.json").write_text(json.dumps(info, indent=4))
    write_jsonl(
        meta_dir / "tasks.jsonl",
        [{"task_index": 0, "task": TASK_DESCRIPTION}],
    )
    write_jsonl(
        meta_dir / "episodes.jsonl",
        [
            {
                "episode_index": int(episode),
                "tasks": [TASK_DESCRIPTION],
                "length": int(len(df)),
            }
            for episode, df in zip(episodes, dfs, strict=True)
        ],
    )

    stats = compute_stats(dfs)
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=4))

    episode_stats = [
        {
            "episode_index": int(episode),
            "stats": compute_episode_stats(df),
        }
        for episode, df in zip(episodes, dfs, strict=True)
    ]
    write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats)

    if source_meta is not None and (source_meta / "modality.json").exists():
        src_modality = (source_meta / "modality.json").resolve()
        dst_modality = (meta_dir / "modality.json").resolve()
        if src_modality != dst_modality:
            shutil.copy2(src_modality, dst_modality)

    print(f"Metadata rebuilt at {meta_dir}")
    print(f"Episodes: {len(episodes)}")
    print(f"Frames: {total_frames}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate LeRobot v2.1 metadata for dataset-v2.")
    parser.add_argument("--dataset-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--source-meta", type=Path, default=RAW_ROOT / "meta")
    args = parser.parse_args()
    source_meta = args.source_meta if args.source_meta.exists() else None
    rebuild_metadata(args.dataset_root, source_meta=source_meta)


if __name__ == "__main__":
    main()
