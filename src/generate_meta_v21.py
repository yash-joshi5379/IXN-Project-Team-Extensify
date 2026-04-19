"""
Generates the complete LeRobot meta/ folder in v2.1 format.

Use with lerobot pinned to v0.3.3:
  cd lerobot && git checkout v0.3.3 && pip install -e ".[smolvla]"

USAGE:
  python generate_meta_v21.py --data_dir /path/to/your/dataset

Expects:
  <data_dir>/data/chunk-000/episode_000000.parquet  ... (all episodes)
  <data_dir>/videos/chunk-000/observation.images.static_cam/episode_000000.mp4
  <data_dir>/videos/chunk-000/observation.images.wrist_cam/episode_000000.mp4

Produces all 5 required meta files:
  <data_dir>/meta/info.json
  <data_dir>/meta/episodes.jsonl
  <data_dir>/meta/tasks.jsonl
  <data_dir>/meta/stats.json
  <data_dir>/meta/episodes_stats.jsonl
"""

import glob, json, re, argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_TYPE       = "xarm7"
FPS              = 30
CODEBASE_VERSION = "v2.1"   # must match lerobot v0.3.x

TASK_DESCRIPTION = (
    "Grasp the red cylinder and place it vertically "
    "into the circular slot within the white box."
)

# These must exactly match the folder names inside videos/chunk-000/
# i.e. videos/chunk-000/observation.images.static_cam/episode_000000.mp4
CAMERA_KEYS = ["static_cam", "wrist_cam"]

IMAGE_HEIGHT   = 256
IMAGE_WIDTH    = 256
IMAGE_CHANNELS = 3

MOTOR_NAMES = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7","gripper"]
FT_NAMES    = ["Fx","Fy","Fz","Tx","Ty","Tz"]

PROPRIOCEPTIVE_COLS = [
    "observation.state",
    "action",
    "observation.effort",
    "observation.qvel",
    "observation.force_torque",
]

DIMS = {
    "observation.state":        8,
    "action":                   8,
    "observation.effort":       8,
    "observation.qvel":         8,
    "observation.force_torque": 6,
}

# ImageNet normalisation constants for camera features.
# Shape is (3,1,1) — channels-first with H and W as size-1 broadcast dims.
# This is the exact shape lerobot's _assert_type_and_shape requires for images.
IMAGENET_STATS = {
    "count": [1],
    "mean":  [[[0.485]], [[0.456]], [[0.406]]],   # shape (3,1,1)
    "std":   [[[0.229]], [[0.224]], [[0.225]]],   # shape (3,1,1)
    "min":   [[[0.0  ]], [[0.0  ]], [[0.0  ]]],   # shape (3,1,1)
    "max":   [[[1.0  ]], [[1.0  ]], [[1.0  ]]],   # shape (3,1,1)
}
# ─────────────────────────────────────────────────────────────────────────────


def parse_vec(s):
    """Parse a numpy-printed array string into a Python list of floats."""
    return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(s))]


def load_all_episodes(data_dir: Path):
    pattern = str(data_dir / "data" / "**" / "episode_*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found at: {pattern}\n"
            "Make sure your dataset is at <data_dir>/data/chunk-000/episode_*.parquet"
        )
    print(f"  Found {len(files)} parquet episode files.")
    return [pd.read_parquet(f) for f in files]


def write_episodes_jsonl(dfs, out_path: Path):
    """
    meta/episodes.jsonl — one JSON object per line per episode.
    Fields: episode_index, length, tasks (list of task indices).
    """
    with open(out_path, "w") as f:
        for df in dfs:
            record = {
                "episode_index": int(df["episode_index"].iloc[0]),
                "length":        len(df),
                "tasks":         [0],
            }
            f.write(json.dumps(record) + "\n")
    print(f"  ✓ episodes.jsonl  ({len(dfs)} episodes)")


def write_tasks_jsonl(out_path: Path):
    """
    meta/tasks.jsonl — one JSON object per unique task.
    The 'task' string is the natural-language instruction SmolVLA is
    conditioned on. Use exactly the same string at inference time.
    """
    with open(out_path, "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_DESCRIPTION}) + "\n")
    print(f"  ✓ tasks.jsonl  (\"{TASK_DESCRIPTION}\")")


def write_info_json(dfs, out_path: Path):
    """
    meta/info.json — dataset-level metadata.

    Critical fields:
      codebase_version : must be "v2.1" for lerobot v0.3.x
      data_path        : uses {episode_chunk} and {episode_index} — NOT {chunk_index}
                         (lerobot v0.3.3 calls get_data_file_path with episode_chunk)
      video_path       : same — uses {episode_chunk}, {video_key}, {episode_index}
      features         : camera keys must be "observation.images.<cam_key>" and
                         must exactly match the folder names in videos/chunk-000/
    """
    total_frames   = sum(len(df) for df in dfs)
    total_episodes = len(dfs)

    features = {
        "observation.state": {
            "dtype": "float32", "shape": [DIMS["observation.state"]],
            "names": {"motors": MOTOR_NAMES},
        },
        "observation.effort": {
            "dtype": "float32", "shape": [DIMS["observation.effort"]],
            "names": {"motors": MOTOR_NAMES},
        },
        "observation.qvel": {
            "dtype": "float32", "shape": [DIMS["observation.qvel"]],
            "names": {"motors": MOTOR_NAMES},
        },
        "observation.force_torque": {
            "dtype": "float32", "shape": [DIMS["observation.force_torque"]],
            "names": {"axes": FT_NAMES},
        },
        "action": {
            "dtype": "float32", "shape": [DIMS["action"]],
            "names": {"motors": MOTOR_NAMES},
        },
    }

    # Camera features — key name becomes the video subfolder name,
    # e.g. "observation.images.static_cam" →
    #      videos/chunk-000/observation.images.static_cam/episode_000000.mp4
    for cam in CAMERA_KEYS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps":          float(FPS),
                "video.codec":        "av1",
                "video.pix_fmt":      "yuv420p",
                "video.is_depth_map": False,
                "has_audio":          False,
            },
        }

    for name, dtype in [
        ("timestamp",     "float32"),
        ("frame_index",   "int64"),
        ("episode_index", "int64"),
        ("index",         "int64"),
        ("task_index",    "int64"),
    ]:
        features[name] = {"dtype": dtype, "shape": [1], "names": None}

    features["next.done"] = {"dtype": "bool", "shape": [1], "names": None}

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type":       ROBOT_TYPE,
        "total_episodes":   total_episodes,
        "total_frames":     total_frames,
        "total_tasks":      1,
        "total_videos":     total_episodes * len(CAMERA_KEYS),
        "total_chunks":     1,
        "chunks_size":      1000,
        "fps":              FPS,
        "splits":           {"train": f"0:{total_episodes}"},
        # IMPORTANT: lerobot v0.3.3 uses {episode_chunk} not {chunk_index}
        "data_path":  "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features":   features,
    }

    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  ✓ info.json  (codebase_version={CODEBASE_VERSION}, "
          f"{total_episodes} eps, {total_frames} frames)")


def write_stats_json(dfs, out_path: Path):
    """
    meta/stats.json — global per-feature statistics across all frames.
    SmolVLA uses mean/std to normalise state and action to ~N(0,1).
    Camera stats use ImageNet constants in (3,1,1) shape as required by lerobot.
    """
    def stack(col):
        return np.concatenate(
            [np.stack(df[col].apply(parse_vec).tolist()).astype(np.float32)
             for df in dfs],
            axis=0,
        )

    def feat_stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std":  (arr.std(axis=0) + 1e-8).tolist(),
            "min":  arr.min(axis=0).tolist(),
            "max":  arr.max(axis=0).tolist(),
        }

    print("  Computing global statistics ...")
    stats = {}
    for col in PROPRIOCEPTIVE_COLS:
        arr = stack(col)
        stats[col] = feat_stats(arr)
        print(f"    {col}  mean={[f'{v:.3f}' for v in stats[col]['mean']]}")

    for cam in CAMERA_KEYS:
        stats[f"observation.images.{cam}"] = IMAGENET_STATS

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ stats.json")


def write_episodes_stats_jsonl(dfs, out_path: Path):
    """
    meta/episodes_stats.jsonl — per-episode statistics (required by lerobot v0.3.3).

    One JSON object per line per episode. Each feature entry needs:
      count : [n_frames]  — 1-element list (must be ndim=1 when loaded as numpy)
      mean  : list of floats
      std   : list of floats  (lerobot squares this internally to get variance)
      min   : list of floats
      max   : list of floats

    Camera entries use ImageNet constants in (3,1,1) shape.

    This file was NOT generated by the original script and had to be added
    manually after hitting these errors during training:
      - FileNotFoundError: episodes_stats.jsonl
      - KeyError: 'count'
      - ValueError: Shape of 'mean' must be (3,1,1)
      - KeyError: 'observation.images.static_cam'
    """
    print("  Computing per-episode statistics ...")
    records = []
    for df in dfs:
        ep_idx   = int(df["episode_index"].iloc[0])
        n_frames = len(df)
        stats    = {"episode_index": ep_idx, "stats": {}}

        for col in PROPRIOCEPTIVE_COLS:
            arr = np.stack(df[col].apply(parse_vec).tolist()).astype(np.float32)
            stats["stats"][col] = {
                "count": [n_frames],              # list not scalar — must be ndim=1
                "mean":  arr.mean(axis=0).tolist(),
                "std":   arr.std(axis=0).tolist(), # lerobot does std**2 internally
                "min":   arr.min(axis=0).tolist(),
                "max":   arr.max(axis=0).tolist(),
            }

        # Camera stats: ImageNet constants, shape (3,1,1) required
        for cam in CAMERA_KEYS:
            stats["stats"][f"observation.images.{cam}"] = IMAGENET_STATS

        records.append(stats)
        print(f"    episode {ep_idx:03d} — {n_frames} frames")

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  ✓ episodes_stats.jsonl  ({len(records)} episodes)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate all 5 LeRobot v2.1 meta files for SmolVLA training"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Root of your dataset (folder containing data/ and videos/)",
    )
    args   = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    meta_dir = data_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading episodes from {data_dir} ...")
    dfs = load_all_episodes(data_dir)

    print(f"\nWriting meta/ (format: {CODEBASE_VERSION}) ...")
    write_episodes_jsonl(dfs,       meta_dir / "episodes.jsonl")
    write_tasks_jsonl(              meta_dir / "tasks.jsonl")
    write_info_json(dfs,            meta_dir / "info.json")
    write_stats_json(dfs,           meta_dir / "stats.json")
    write_episodes_stats_jsonl(dfs, meta_dir / "episodes_stats.jsonl")

    print(f"\n✅  All 5 meta files written to: {meta_dir}")
    print(f"    episodes.jsonl")
    print(f"    tasks.jsonl")
    print(f"    info.json")
    print(f"    stats.json")
    print(f"    episodes_stats.jsonl")
    print(f"\n⚠️  Use lerobot pinned to v0.3.3:")
    print(f"     cd lerobot && git checkout v0.3.3 && pip install -e \".[smolvla]\"")


if __name__ == "__main__":
    main()
