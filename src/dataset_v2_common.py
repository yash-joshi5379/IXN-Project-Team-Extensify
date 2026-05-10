from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESKTOP_SOURCE = Path.home() / "Desktop" / "dataset-v2"
RAW_ROOT = PROJECT_ROOT / "dataset-v2"
PROCESSED_ROOT = PROJECT_ROOT / "dataset-v2-processed"

CAMERA_KEYS = [
    "observation.images.0_femtobolt",
    "observation.images.1_gemini330",
]

TASK_DESCRIPTION = (
    "Grasp the red cylinder and place it vertically into the circular slot within the white box."
)

FPS = 30

VECTOR_COLUMNS = [
    "action",
    "observation.effort",
    "observation.state",
    "observation.qvel",
]

SCALAR_COLUMNS = [
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "next.done",
]

MOTOR_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "gripper",
]


def episode_name(episode: int) -> str:
    return f"episode_{episode:06d}"


def data_path(dataset_root: Path, episode: int) -> Path:
    return dataset_root / "data" / "chunk-000" / f"{episode_name(episode)}.parquet"


def video_path(dataset_root: Path, camera_key: str, episode: int) -> Path:
    return dataset_root / "videos" / "chunk-000" / camera_key / f"{episode_name(episode)}.mp4"


def require_episode_files(dataset_root: Path, episode: int) -> None:
    missing = [data_path(dataset_root, episode)]
    missing += [video_path(dataset_root, cam, episode) for cam in CAMERA_KEYS]
    missing = [path for path in missing if not path.exists()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Episode {episode} is incomplete under {dataset_root}:\n{rendered}")


def ensure_dataset_dirs(dataset_root: Path) -> None:
    (dataset_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for camera_key in CAMERA_KEYS:
        (dataset_root / "videos" / "chunk-000" / camera_key).mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)


def parse_vector(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape {arr.shape}")
    return arr


def stack_column(dfs: list[pd.DataFrame], column: str) -> np.ndarray:
    values = []
    for df in dfs:
        if column in df.columns:
            values.extend(parse_vector(value) for value in df[column].tolist())
    if not values:
        raise ValueError(f"No values found for column {column}")
    return np.stack(values).astype(np.float32)


def feature_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def image_stats() -> dict:
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.485]], [[0.456]], [[0.406]]],
        "std": [[[0.229]], [[0.224]], [[0.225]]],
        "count": [1],
    }


def ffprobe_video(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,pix_fmt,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream.get("codec_name", "unknown"),
        "pix_fmt": stream.get("pix_fmt", "unknown"),
        "fps": _parse_rate(stream.get("avg_frame_rate", "30/1")),
        "frames": _parse_optional_int(stream.get("nb_frames")),
        "duration": _parse_optional_float(stream.get("duration")),
    }


def _parse_rate(value: str) -> float:
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else float(FPS)
    return float(value)


def _parse_optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sorted_episode_indices(dataset_root: Path) -> list[int]:
    files = sorted((dataset_root / "data" / "chunk-000").glob("episode_*.parquet"))
    return [int(path.stem.replace("episode_", "")) for path in files]


def normalize_parquet_indices(dataset_root: Path) -> int:
    global_index = 0
    total_frames = 0
    for episode in sorted_episode_indices(dataset_root):
        path = data_path(dataset_root, episode)
        df = pd.read_parquet(path)
        n_frames = len(df)
        if n_frames == 0:
            raise ValueError(f"{path} has no rows")
        df["episode_index"] = episode
        df["frame_index"] = np.arange(n_frames, dtype=np.int64)
        df["index"] = np.arange(global_index, global_index + n_frames, dtype=np.int64)
        if "task_index" not in df.columns:
            df["task_index"] = 0
        if "next.done" in df.columns:
            df["next.done"] = False
            df.loc[df.index[-1], "next.done"] = True
        df.to_parquet(path, index=False)
        global_index += n_frames
        total_frames += n_frames
    return total_frames


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
