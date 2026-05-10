from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from dataset_v2_common import CAMERA_KEYS, RAW_ROOT, require_episode_files, video_path

PLAY_DELAY_MS = 33


def load_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from video: {path}")
    return frames


def draw_overlay(frame: np.ndarray, label: str, frame_idx: int, frame_count: int, paused: bool) -> np.ndarray:
    frame = frame.copy()
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 82), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.48, frame, 0.52, 0)
    cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    cv2.putText(
        frame,
        f"Frame: {frame_idx} / {frame_count - 1}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 120),
        2,
    )
    cv2.putText(
        frame,
        "PAUSED" if paused else "PLAYING",
        (10, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 90, 255) if paused else (0, 220, 0),
        2,
    )
    return frame


def resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == target_height:
        return frame
    scale = target_height / h
    return cv2.resize(frame, (int(w * scale), target_height), interpolation=cv2.INTER_AREA)


def visualise(dataset_root: Path, episode: int, start_frame: int, step_size: int) -> int:
    require_episode_files(dataset_root, episode)
    frame_sets = {}
    for camera_key in CAMERA_KEYS:
        path = video_path(dataset_root, camera_key, episode)
        print(f"Loading {path}")
        frame_sets[camera_key] = load_frames(path)
        print(f"  {len(frame_sets[camera_key])} frames")

    frame_count = min(len(frames) for frames in frame_sets.values())
    if len({len(frames) for frames in frame_sets.values()}) != 1:
        print("WARNING: camera videos have different frame counts; using the shorter length.")

    frame_idx = max(0, min(start_frame, frame_count - 1))
    paused = True
    window_name = f"dataset-v2 episode_{episode:06d}"

    print()
    print("Controls:")
    print("  Space : play/pause")
    print(f"  d     : step forward {step_size} frames")
    print(f"  a     : step back {step_size} frames")
    print("  f     : step forward one frame")
    print("  b     : step back one frame")
    print("  q     : quit and print current frame")
    print()

    while True:
        rendered = []
        for camera_key in CAMERA_KEYS:
            label = camera_key.removeprefix("observation.images.")
            rendered.append(draw_overlay(frame_sets[camera_key][frame_idx], label, frame_idx, frame_count, paused))

        target_height = max(frame.shape[0] for frame in rendered)
        combined = np.concatenate([resize_to_height(frame, target_height) for frame in rendered], axis=1)
        cv2.imshow(window_name, combined)

        if frame_idx == frame_count - 1 and not paused:
            paused = True

        key = cv2.waitKey(0 if paused else PLAY_DELAY_MS) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
            if not paused and frame_idx == frame_count - 1:
                frame_idx = 0
            continue
        if key == ord("d"):
            frame_idx = min(frame_count - 1, frame_idx + step_size)
            paused = True
            continue
        if key == ord("a"):
            frame_idx = max(0, frame_idx - step_size)
            paused = True
            continue
        if key == ord("f"):
            frame_idx = min(frame_count - 1, frame_idx + 1)
            paused = True
            continue
        if key == ord("b"):
            frame_idx = max(0, frame_idx - 1)
            paused = True
            continue
        if not paused:
            frame_idx = min(frame_count - 1, frame_idx + 1)

    cv2.destroyAllWindows()
    print()
    print("=" * 52)
    print(f"Episode {episode:06d} end frame index = {frame_idx}")
    print(f"Use this command to process it:")
    print(f"  .venv/bin/python src/process_episode_v2.py {episode} {frame_idx}")
    print("=" * 52)
    return frame_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Open both dataset-v2 camera views to find an end frame.")
    parser.add_argument("episode", type=int)
    parser.add_argument("--dataset-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--step-size", type=int, default=90)
    args = parser.parse_args()

    if args.episode < 0:
        print("Episode must be non-negative.", file=sys.stderr)
        sys.exit(2)

    if args.step_size <= 0:
        print("--step-size must be greater than 0.", file=sys.stderr)
        sys.exit(2)

    visualise(args.dataset_root, args.episode, args.start_frame, args.step_size)


if __name__ == "__main__":
    main()
