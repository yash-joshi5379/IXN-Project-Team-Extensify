from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dataset_v2_common import CAMERA_KEYS, DESKTOP_SOURCE, RAW_ROOT, data_path, ensure_dataset_dirs, video_path
from generate_meta_v2 import rebuild_metadata


def copy_subset(source_root: Path, dest_root: Path, start: int, end: int, overwrite: bool) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")
    if dest_root.exists() and overwrite:
        shutil.rmtree(dest_root)

    ensure_dataset_dirs(dest_root)

    for episode in range(start, end + 1):
        src_data = data_path(source_root, episode)
        dst_data = data_path(dest_root, episode)
        if not src_data.exists():
            raise FileNotFoundError(f"Missing source parquet: {src_data}")
        shutil.copy2(src_data, dst_data)

        for camera_key in CAMERA_KEYS:
            src_video = video_path(source_root, camera_key, episode)
            dst_video = video_path(dest_root, camera_key, episode)
            if not src_video.exists():
                raise FileNotFoundError(f"Missing source video: {src_video}")
            shutil.copy2(src_video, dst_video)

    source_meta = source_root / "meta"
    rebuild_metadata(dest_root, source_meta=source_meta if source_meta.exists() else None)
    print(f"Copied episodes {start}-{end} into {dest_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a local subset of Desktop dataset-v2 into the repo.")
    parser.add_argument("--source-root", type=Path, default=DESKTOP_SOURCE)
    parser.add_argument("--dest-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.start < 0 or args.end < args.start:
        raise ValueError("--start/--end must describe a valid inclusive range")

    copy_subset(args.source_root, args.dest_root, args.start, args.end, args.overwrite)


if __name__ == "__main__":
    main()
