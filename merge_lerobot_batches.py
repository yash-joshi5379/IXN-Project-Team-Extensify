"""
Merge 3 LeRobot v2.1 batch folders into one unified dataset.
=============================================================
Merges cylinderv3-LeRobot, cylinderv4-LeRobot, cylinderv5-LeRobot
into merged-LeRobot with episodes renumbered 0..146.

USAGE:
  pip install pandas pyarrow tqdm
  python src/merge_lerobot_batches.py

OUTPUT structure:
  merged-LeRobot/
    data/chunk-000/episode_000000.parquet ... episode_000146.parquet
    videos/chunk-000/observation.images.0_femtobolt/episode_000000.mp4 ...
    videos/chunk-000/observation.images.1_gemini330/episode_000000.mp4 ...
    meta/
      info.json
      episodes.jsonl
      tasks.jsonl
      episodes_stats.jsonl
      modality.json
"""

import glob
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Root directory containing all batch folders and where output will be written
BASE_DIR = Path(__file__).resolve().parent

# Input batch folders in the order they should be merged
# Episodes will be renumbered in this order:
#   cylinderv3 → episodes 0–42
#   cylinderv4 → episodes 43–98
#   cylinderv5 → episodes 99–146
BATCH_DIRS = [
    BASE_DIR / "cylinderv3-LeRobot",
    BASE_DIR / "cylinderv4-LeRobot",
    BASE_DIR / "cylinderv5-LeRobot",
]

OUTPUT_DIR = BASE_DIR / "merged-LeRobot"

# Camera keys — must match the video subfolder names in all batch folders
CAMERA_KEYS = [
    "observation.images.0_femtobolt",
    "observation.images.1_gemini330",
]

# Unified task description for the merged dataset.
# All 3 batches get mapped to task_index=0 with this description.
MERGED_TASK = "Grasp the red cylinder and place it vertically into the circular slot within the white box."

# FPS — must be the same across all batches (confirmed 30 from info.json)
FPS = 30

# ─────────────────────────────────────────────────────────────────────────────


def load_episodes_jsonl(batch_dir: Path) -> list[dict]:
    path = batch_dir / "meta" / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_episodes_stats_jsonl(batch_dir: Path) -> list[dict]:
    path = batch_dir / "meta" / "episodes_stats.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_info_json(batch_dir: Path) -> dict:
    return json.loads((batch_dir / "meta" / "info.json").read_text())


def get_parquet_files(batch_dir: Path) -> list[Path]:
    """Return sorted list of parquet files for this batch."""
    pattern = str(batch_dir / "data" / "**" / "episode_*.parquet")
    return sorted(Path(f) for f in glob.glob(pattern, recursive=True))


def merge_data(output_dir: Path) -> tuple[int, int]:
    """
    Copy and renumber all parquet files.
    Returns (total_episodes, total_frames).
    """
    out_data_dir = output_dir / "data" / "chunk-000"
    out_data_dir.mkdir(parents=True, exist_ok=True)

    new_episode_idx = 0   # global episode counter across all batches
    global_frame_idx = 0  # global frame counter across all batches

    for batch_dir in BATCH_DIRS:
        parquet_files = get_parquet_files(batch_dir)
        print(f"\n  {batch_dir.name}: {len(parquet_files)} episodes")

        for src_path in tqdm(parquet_files, desc=f"    Parquet"):
            df = pd.read_parquet(src_path)
            n_frames = len(df)

            # Remap episode_index to new global index
            df["episode_index"] = new_episode_idx

            # Remap frame_index — stays 0..N-1 within each episode
            df["frame_index"] = np.arange(n_frames, dtype=np.int64)

            # Remap global index — continuous across entire merged dataset
            df["index"] = np.arange(
                global_frame_idx, global_frame_idx + n_frames, dtype=np.int64
            )

            # Remap task_index — all episodes map to task 0 in merged dataset
            df["task_index"] = 0

            # Write to output
            out_path = out_data_dir / f"episode_{new_episode_idx:06d}.parquet"
            df.to_parquet(out_path, index=False)

            new_episode_idx += 1
            global_frame_idx += n_frames

    return new_episode_idx, global_frame_idx


def merge_videos(output_dir: Path):
    """Copy and rename all video files."""
    for cam_key in CAMERA_KEYS:
        out_cam_dir = output_dir / "videos" / "chunk-000" / cam_key
        out_cam_dir.mkdir(parents=True, exist_ok=True)

    new_episode_idx = 0

    for batch_dir in BATCH_DIRS:
        # Find all episodes in this batch from its parquet files
        parquet_files = get_parquet_files(batch_dir)

        print(f"\n  {batch_dir.name}: copying videos ...")
        for src_parquet in tqdm(parquet_files, desc=f"    Videos"):
            # Extract original episode index from filename
            orig_idx = int(re.search(r"episode_(\d+)", src_parquet.name).group(1))

            for cam_key in CAMERA_KEYS:
                src_video = (
                    batch_dir
                    / "videos"
                    / "chunk-000"
                    / cam_key
                    / f"episode_{orig_idx:06d}.mp4"
                )
                dst_video = (
                    output_dir
                    / "videos"
                    / "chunk-000"
                    / cam_key
                    / f"episode_{new_episode_idx:06d}.mp4"
                )

                if not src_video.exists():
                    print(f"\n    WARNING: video not found: {src_video}")
                    continue

                shutil.copy2(src_video, dst_video)

            new_episode_idx += 1


def write_meta(output_dir: Path, total_episodes: int, total_frames: int):
    """Write all 5 meta files for the merged dataset."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # ── tasks.jsonl ──────────────────────────────────────────────────────────
    # All 3 batches merged into a single task
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": MERGED_TASK}) + "\n")
    print("  ✓ tasks.jsonl")

    # ── episodes.jsonl ───────────────────────────────────────────────────────
    # Rebuild from all 3 batches, renumbering episodes and unifying task name
    new_episode_idx = 0
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for batch_dir in BATCH_DIRS:
            episodes = load_episodes_jsonl(batch_dir)
            for ep in episodes:
                record = {
                    "episode_index": new_episode_idx,
                    "tasks": [MERGED_TASK],   # unified task string
                    "length": ep["length"],
                }
                f.write(json.dumps(record) + "\n")
                new_episode_idx += 1
    print(f"  ✓ episodes.jsonl  ({new_episode_idx} episodes)")

    # ── episodes_stats.jsonl ─────────────────────────────────────────────────
    # Copy all per-episode stats, remapping episode_index to new global index
    new_episode_idx = 0
    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for batch_dir in BATCH_DIRS:
            stats_records = load_episodes_stats_jsonl(batch_dir)
            for rec in stats_records:
                rec["episode_index"] = new_episode_idx
                f.write(json.dumps(rec) + "\n")
                new_episode_idx += 1
    print(f"  ✓ episodes_stats.jsonl  ({new_episode_idx} episodes)")

    # ── modality.json ────────────────────────────────────────────────────────
    # Identical across all batches — copy from first batch
    src_modality = BATCH_DIRS[0] / "meta" / "modality.json"
    shutil.copy2(src_modality, meta_dir / "modality.json")
    print("  ✓ modality.json  (copied from first batch)")

    # ── info.json ────────────────────────────────────────────────────────────
    # Read features from first batch (same across all batches)
    ref_info = load_info_json(BATCH_DIRS[0])

    # Update task descriptions in features if present
    features = ref_info.get("features", {})

    info = {
        "codebase_version":  "v2.1",
        "robot_type":        ref_info.get("robot_type", "stationary"),
        "total_episodes":    total_episodes,
        "total_frames":      total_frames,
        "total_tasks":       1,
        "total_videos":      total_episodes * len(CAMERA_KEYS),
        "total_chunks":      1,
        "chunks_size":       1000,
        "fps":               FPS,
        "splits":            {"train": f"0:{total_episodes}"},
        "data_path":         ref_info["data_path"],
        "video_path":        ref_info["video_path"],
        "depth_images_path": ref_info.get("depth_images_path", None),
        "features":          features,
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    print(f"  ✓ info.json  ({total_episodes} eps, {total_frames} frames)")


def verify_output(output_dir: Path):
    """Quick sanity checks on the merged output."""
    print("\n── Verification ─────────────────────────────────────────")

    parquet_files = sorted((output_dir / "data" / "chunk-000").glob("*.parquet"))
    print(f"  Parquet files  : {len(parquet_files)}")

    for cam_key in CAMERA_KEYS:
        vid_dir = output_dir / "videos" / "chunk-000" / cam_key
        n_videos = len(list(vid_dir.glob("*.mp4")))
        print(f"  Videos ({cam_key.split('.')[-1]}): {n_videos}")

    # Check episode numbering is consecutive
    indices = [int(re.search(r"episode_(\d+)", f.name).group(1)) for f in parquet_files]
    expected = list(range(len(parquet_files)))
    if indices == expected:
        print(f"  Episode numbering: ✓ consecutive 0–{indices[-1]}")
    else:
        missing = set(expected) - set(indices)
        print(f"  Episode numbering: ✗ gaps found: {missing}")

    # Spot-check first and last parquet
    for label, f in [("First", parquet_files[0]), ("Last", parquet_files[-1])]:
        df = pd.read_parquet(f)
        ep_idx = int(re.search(r"episode_(\d+)", f.name).group(1))
        print(f"  {label} episode ({ep_idx}): "
              f"episode_index={df['episode_index'].iloc[0]}, "
              f"index range={df['index'].iloc[0]}–{df['index'].iloc[-1]}, "
              f"task_index={df['task_index'].iloc[0]}")

    # Check meta files exist
    for fname in ["info.json", "episodes.jsonl", "tasks.jsonl",
                  "episodes_stats.jsonl", "modality.json"]:
        exists = (output_dir / "meta" / fname).exists()
        print(f"  meta/{fname}: {'✓' if exists else '✗ MISSING'}")


def main():
    print("=" * 60)
    print("  LeRobot v2.1 Batch Merger")
    print("=" * 60)

    # Validate input directories
    for batch_dir in BATCH_DIRS:
        if not batch_dir.exists():
            print(f"ERROR: batch folder not found: {batch_dir}")
            print("       Update BATCH_DIRS in the CONFIG block.")
            return

    total_eps = sum(len(get_parquet_files(d)) for d in BATCH_DIRS)
    print(f"\nBatches to merge:")
    for batch_dir in BATCH_DIRS:
        n = len(get_parquet_files(batch_dir))
        print(f"  {batch_dir.name}: {n} episodes")
    print(f"  Total: {total_eps} episodes")
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Unified task: \"{MERGED_TASK}\"")
    print()

    if OUTPUT_DIR.exists():
        ans = input(f"Output directory already exists. Overwrite? (y/n): ")
        if ans.strip().lower() != "y":
            print("Aborted.")
            return
        shutil.rmtree(OUTPUT_DIR)

    # Step 1: Parquet files
    print("\n── Step 1/3: Merging parquet files ─────────────────────")
    total_episodes, total_frames = merge_data(OUTPUT_DIR)
    print(f"\n  Done: {total_episodes} episodes, {total_frames} frames")

    # Step 2: Video files
    print("\n── Step 2/3: Copying video files ───────────────────────")
    merge_videos(OUTPUT_DIR)
    print(f"\n  Done")

    # Step 3: Meta files
    print("\n── Step 3/3: Writing meta files ────────────────────────")
    write_meta(OUTPUT_DIR, total_episodes, total_frames)

    # Verify
    verify_output(OUTPUT_DIR)

    print(f"\n✅  Merge complete!")
    print(f"    Output at: {OUTPUT_DIR}")
    print(f"    {total_episodes} episodes | {total_frames} frames | {len(CAMERA_KEYS)} cameras")


if __name__ == "__main__":
    main()