from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_v2_common import (
    CAMERA_KEYS,
    PROCESSED_ROOT,
    data_path,
    normalize_parquet_indices,
    sorted_episode_indices,
    video_path,
)
from generate_meta_v2 import rebuild_metadata


def load_existing_mapping(dataset_root: Path, episodes: list[int]) -> dict[int, int]:
    mapping_path = dataset_root / "meta" / "original_episode_map.json"
    if not mapping_path.exists():
        return {episode: episode for episode in episodes}
    mapping = json.loads(mapping_path.read_text())
    current_to_original = mapping.get("current_to_original", {})
    return {int(k): int(v) for k, v in current_to_original.items()}


def rename_episode_files(dataset_root: Path, kept_episodes: list[int]) -> dict[int, int]:
    old_to_new = {old: new for new, old in enumerate(kept_episodes)}
    temp_suffix = ".tmp-clean"

    for old in kept_episodes:
        data_path(dataset_root, old).rename(data_path(dataset_root, old).with_suffix(".parquet" + temp_suffix))
        for camera_key in CAMERA_KEYS:
            video_path(dataset_root, camera_key, old).rename(
                video_path(dataset_root, camera_key, old).with_suffix(".mp4" + temp_suffix)
            )

    for old, new in old_to_new.items():
        data_path(dataset_root, old).with_suffix(".parquet" + temp_suffix).rename(data_path(dataset_root, new))
        for camera_key in CAMERA_KEYS:
            video_path(dataset_root, camera_key, old).with_suffix(".mp4" + temp_suffix).rename(
                video_path(dataset_root, camera_key, new)
            )

    return old_to_new


def fix_done_columns(dataset_root: Path) -> None:
    for episode in sorted_episode_indices(dataset_root):
        path = data_path(dataset_root, episode)
        df = pd.read_parquet(path)
        if len(df) == 0:
            raise ValueError(f"{path} has no rows")
        df["next.done"] = False
        df.loc[df.index[-1], "next.done"] = True
        df["frame_index"] = np.arange(len(df), dtype=np.int64)
        df["episode_index"] = episode
        df.to_parquet(path, index=False)


def clean_dataset(dataset_root: Path, invalid_episodes: list[int]) -> None:
    dataset_root = dataset_root.resolve()
    episodes = sorted_episode_indices(dataset_root)
    if not episodes:
        raise FileNotFoundError(f"No processed episodes found in {dataset_root}")

    invalid = sorted(set(invalid_episodes))
    missing = [episode for episode in invalid if episode not in episodes]
    if missing:
        raise ValueError(f"Invalid episode(s) not found in processed dataset: {missing}")

    current_to_original_before = load_existing_mapping(dataset_root, episodes)

    for episode in sorted(invalid, reverse=True):
        data_path(dataset_root, episode).unlink(missing_ok=False)
        for camera_key in CAMERA_KEYS:
            video_path(dataset_root, camera_key, episode).unlink(missing_ok=False)
        print(f"Removed episode_{episode:06d}")

    kept = [episode for episode in episodes if episode not in invalid]
    old_to_new = rename_episode_files(dataset_root, kept)
    fix_done_columns(dataset_root)
    total_frames = normalize_parquet_indices(dataset_root)
    rebuild_metadata(dataset_root, source_meta=(dataset_root / "meta"))

    current_to_original = {
        new: current_to_original_before[old]
        for old, new in old_to_new.items()
    }
    original_to_current = {original: current for current, original in current_to_original.items()}
    mapping = {
        "removed_current_episodes": invalid,
        "removed_original_episodes": [current_to_original_before[episode] for episode in invalid],
        "current_to_original": {str(k): v for k, v in sorted(current_to_original.items())},
        "original_to_current": {str(k): v for k, v in sorted(original_to_current.items())},
        "total_episodes": len(current_to_original),
        "total_frames": total_frames,
    }
    mapping_path = dataset_root / "meta" / "original_episode_map.json"
    mapping_path.write_text(json.dumps(mapping, indent=2))

    print()
    print(f"Cleaned dataset: {len(current_to_original)} episodes, {total_frames} frames")
    print(f"Mapping written to {mapping_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove invalid processed dataset-v2 episodes and renumber.")
    parser.add_argument("invalid_episodes", type=int, nargs="+")
    parser.add_argument("--dataset-root", type=Path, default=PROCESSED_ROOT)
    args = parser.parse_args()
    clean_dataset(args.dataset_root, args.invalid_episodes)


if __name__ == "__main__":
    main()
