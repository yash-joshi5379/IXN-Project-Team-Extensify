"""
Generates the LeRobot meta/ folder in v2.1 format.

Use this with: lerobot code pinned to v0.3.3
  git checkout v0.3.3
  pip install -e ".[smolvla]"

USAGE:
  python generate_meta_v21.py --data_dir /path/to/your/dataset

Expects:
  <data_dir>/data/chunk-000/episode_000000.parquet  ... (all 66 episodes)
  <data_dir>/videos/chunk-000/cam_high/episode_000000.mp4
  <data_dir>/videos/chunk-000/cam_wrist/episode_000000.mp4

Produces:
  <data_dir>/meta/info.json
  <data_dir>/meta/episodes.jsonl
  <data_dir>/meta/tasks.jsonl
  <data_dir>/meta/stats.json
"""

import glob, json, re, argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — confirmed from your parquet schema (episode_24.csv)
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_TYPE       = "xarm7"
FPS              = 30
CODEBASE_VERSION = "v2.1"     # must match lerobot v0.3.x

TASK_DESCRIPTION = (
    "Grasp the red cylinder and place it vertically "
    "into the circular slot within the white box."
)

CAMERA_KEYS    = ["observation.images.static_cam", "observation.images.static_cam"]
IMAGE_HEIGHT   = 256
IMAGE_WIDTH    = 256
IMAGE_CHANNELS = 3

MOTOR_NAMES = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7","gripper"]
FT_NAMES    = ["Fx","Fy","Fz","Tx","Ty","Tz"]

DIMS = {
    "observation.state":        8,
    "action":                   8,
    "observation.effort":       8,
    "observation.qvel":         8,
    "observation.force_torque": 6,
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
    meta/episodes.jsonl  (v2.1 format)
    One JSON object per line, one line per episode.
    Fields: episode_index, length, tasks (list of task indices)
    """
    with open(out_path, "w") as f:
        for df in dfs:
            record = {
                "episode_index": int(df["episode_index"].iloc[0]),
                "length":        len(df),
                "tasks":         [0],   # single-task dataset → always task 0
            }
            f.write(json.dumps(record) + "\n")
    print(f"  ✓ episodes.jsonl  ({len(dfs)} episodes)")


def write_tasks_jsonl(out_path: Path):
    """
    meta/tasks.jsonl  (v2.1 format)
    One JSON object per line per unique task.
    Fields: task_index, task (the natural-language instruction string)

    This string IS a model input at training time — SmolVLA is conditioned
    on it. Keep it exactly as you want to pass it at inference time too.
    """
    with open(out_path, "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_DESCRIPTION}) + "\n")
    print(f"  ✓ tasks.jsonl  (\"{TASK_DESCRIPTION}\")")


def write_info_json(dfs, out_path: Path):
    """
    meta/info.json
    Dataset-level metadata. The most critical field is codebase_version,
    which must match the lerobot software version you train with:
      v2.1 ↔ lerobot v0.3.x
      v3.0 ↔ lerobot v0.4.x / v0.5.x
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
        "data_path":        "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
        "video_path":       "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features":         features,
    }

    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  ✓ info.json  (codebase_version={CODEBASE_VERSION}, {total_episodes} eps, {total_frames} frames)")


def write_stats_json(dfs, out_path: Path):
    """
    meta/stats.json
    Per-feature mean/std/min/max computed across all frames.
    SmolVLA normalises state and action tensors to ~N(0,1) using these.
    A small epsilon (+1e-8) is added to std to avoid division by zero.
    """
    def stack(col):
        return np.concatenate(
            [np.stack(df[col].apply(parse_vec).tolist()).astype(np.float32) for df in dfs],
            axis=0,
        )

    def feat_stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std":  (arr.std(axis=0) + 1e-8).tolist(),
            "min":  arr.min(axis=0).tolist(),
            "max":  arr.max(axis=0).tolist(),
        }

    print("  Computing statistics across all episodes ...")
    stats = {}
    for col in ["observation.state", "action", "observation.effort",
                "observation.qvel", "observation.force_torque"]:
        arr = stack(col)
        stats[col] = feat_stats(arr)
        print(f"    {col}  mean={[f'{v:.3f}' for v in stats[col]['mean']]}")

    # ImageNet constants — used by SmolVLA's vision encoder internally
    imagenet = {
        "mean": [[[0.485, 0.456, 0.406]]],
        "std":  [[[0.229, 0.224, 0.225]]],
        "min":  [[[0.0,   0.0,   0.0  ]]],
        "max":  [[[1.0,   1.0,   1.0  ]]],
    }
    for cam in CAMERA_KEYS:
        stats[f"observation.images.{cam}"] = imagenet

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ stats.json")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LeRobot v2.1 meta/ folder for SmolVLA training"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Root of your dataset (the folder containing data/ and videos/)",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    meta_dir = data_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading episodes from {data_dir} ...")
    dfs = load_all_episodes(data_dir)

    print(f"\nWriting meta/ (format: {CODEBASE_VERSION}) ...")
    write_episodes_jsonl(dfs, meta_dir / "episodes.jsonl")
    write_tasks_jsonl(meta_dir / "tasks.jsonl")
    write_info_json(dfs, meta_dir / "info.json")
    write_stats_json(dfs, meta_dir / "stats.json")

    print(f"\n✅  Done. meta/ is ready at: {meta_dir}")
    print(f"\n⚠️  Remember: use lerobot pinned to v0.3.3 with this v2.1 dataset:")
    print(f"     cd lerobot && git checkout v0.3.3 && pip install -e \".[smolvla]\"")


if __name__ == "__main__":
    main()
