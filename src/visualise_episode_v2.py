from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dataset_v2_common import CAMERA_KEYS, PROCESSED_ROOT, PROJECT_ROOT, data_path, video_path


def rerun_sdk_path() -> Path | None:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    candidate = site_packages / "rerun_sdk"
    return candidate if candidate.exists() else None


def ensure_rerun_import_path() -> None:
    candidate = rerun_sdk_path()
    if candidate is not None and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def ensure_rerun_viewer_path() -> None:
    path_entries = []
    candidate = rerun_sdk_path()
    if candidate is not None:
        os.environ["PYTHONPATH"] = f"{candidate}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
        cli_dir = candidate / "rerun_cli"
        if (cli_dir / "rerun").exists():
            path_entries.append(cli_dir)

    venv_bin = PROJECT_ROOT / ".venv" / "bin"
    if (venv_bin / "rerun").exists():
        path_entries.append(venv_bin)
    path_entries.extend(Path(path) for path in os.environ.get("PATH", "").split(os.pathsep) if path)
    os.environ["PATH"] = os.pathsep.join(str(path) for path in path_entries)


def log_direct_rerun(dataset_root: Path, episode: int) -> None:
    ensure_rerun_import_path()
    ensure_rerun_viewer_path()
    try:
        import rerun as rr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install rerun first: .venv/bin/pip install rerun-sdk==0.22.1") from exc

    parquet_path = data_path(dataset_root, episode)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing processed parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    caps = {}
    for camera_key in CAMERA_KEYS:
        path = video_path(dataset_root, camera_key, episode)
        if not path.exists():
            raise FileNotFoundError(f"Missing processed video: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        caps[camera_key] = cap

    rr.init(f"dataset-v2/episode_{episode:06d}", spawn=True)
    scalar = getattr(rr, "Scalars", None) or getattr(rr, "Scalar")
    print(f"Logging episode_{episode:06d} to Rerun...")

    try:
        for idx, row in df.iterrows():
            frame_index = int(row["frame_index"]) if "frame_index" in row else idx
            timestamp = float(row["timestamp"]) if "timestamp" in row else frame_index / 30.0
            if hasattr(rr, "set_time"):
                rr.set_time("frame_index", sequence=frame_index)
                rr.set_time("timestamp", timestamp=timestamp)
            else:
                rr.set_time_sequence("frame_index", frame_index)
                rr.set_time_seconds("timestamp", timestamp)

            for camera_key, cap in caps.items():
                ok, frame = cap.read()
                if ok:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rr.log(camera_key, rr.Image(rgb))

            if "action" in row:
                for dim, value in enumerate(np.asarray(row["action"], dtype=np.float32)):
                    rr.log(f"action/{dim}", scalar(float(value)))
            if "observation.state" in row:
                for dim, value in enumerate(np.asarray(row["observation.state"], dtype=np.float32)):
                    rr.log(f"state/{dim}", scalar(float(value)))
            if "next.done" in row:
                rr.log("next.done", scalar(float(bool(row["next.done"]))))
    finally:
        for cap in caps.values():
            cap.release()

    print("Rerun logging complete. Use the Rerun window to play the episode.")


def launch_lerobot(dataset_root: Path, episode: int) -> None:
    script = PROJECT_ROOT / "lerobot" / "src" / "lerobot" / "scripts" / "lerobot_dataset_viz.py"
    if not script.exists():
        raise FileNotFoundError(f"LeRobot visualizer script not found: {script}")

    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    command = [
        sys.executable,
        str(script),
        "--repo-id",
        "local/dataset-v2",
        "--root",
        str(dataset_root.resolve()),
        "--episode-index",
        str(episode),
        "--mode",
        "local",
        "--num-workers",
        "0",
        "--tolerance-s",
        "0.05",
    ]
    env = os.environ.copy()
    python_paths = [PROJECT_ROOT / "lerobot" / "src"]
    candidate = rerun_sdk_path()
    if candidate is not None:
        python_paths.append(candidate)
    existing = env.get("PYTHONPATH", "")
    if existing:
        python_paths.extend(Path(path) for path in existing.split(os.pathsep) if path)
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    path_entries = []
    if candidate is not None and (candidate / "rerun_cli" / "rerun").exists():
        path_entries.append(candidate / "rerun_cli")
    venv_bin = PROJECT_ROOT / ".venv" / "bin"
    if (venv_bin / "rerun").exists():
        path_entries.append(venv_bin)
    path_entries.extend(Path(path) for path in env.get("PATH", "").split(os.pathsep) if path)
    env["PATH"] = os.pathsep.join(str(path) for path in path_entries)
    subprocess.run(command, cwd=PROJECT_ROOT / "lerobot", check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise one processed dataset-v2 episode.")
    parser.add_argument("episode", type=int)
    parser.add_argument("--dataset-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument(
        "--backend",
        choices=["direct", "lerobot"],
        default="direct",
        help="direct logs videos/plots to Rerun without depending on LeRobot metadata compatibility.",
    )
    args = parser.parse_args()

    if args.backend == "direct":
        log_direct_rerun(args.dataset_root, args.episode)
    else:
        launch_lerobot(args.dataset_root, args.episode)


if __name__ == "__main__":
    main()
