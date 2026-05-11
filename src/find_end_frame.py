"""
LeRobot v2.1 Episode Visualiser
=================================
Watch both camera feeds side by side for any episode, with frame-by-frame
controls to find the exact end frame index for trimming.

USAGE:
  python src/visualise_episode_trim.py 5
  python src/visualise_episode_trim.py 5 --dataset merged-LeRobot

Controls:
  [Space] : Play / Pause
  [f]     : Step Forward 1 frame
  [b]     : Step Back 1 frame
  [q]     : Quit and print the current frame index
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "dataset-v2-raw"

# Camera subfolder names inside videos/chunk-000/
CAMERA_KEYS = [
    "observation.images.0_femtobolt",
    "observation.images.1_gemini330",
]

# Playback delay in milliseconds when playing (not paused)
# 33ms ≈ 30fps. Increase to slow down playback.
PLAY_DELAY_MS = 33
# ─────────────────────────────────────────────────────────────────────────────


def find_video_files(dataset_root: Path, episode_idx: int) -> dict[str, Path]:
    """Find the .mp4 files for both cameras for the given episode."""
    video_files = {}
    for cam_key in CAMERA_KEYS:
        path = (
            dataset_root
            / "videos"
            / "chunk-000"
            / cam_key
            / f"episode_{episode_idx:06d}.mp4"
        )
        if not path.exists():
            print(f"ERROR: video not found: {path}")
            sys.exit(1)
        video_files[cam_key] = path
    return video_files


def load_all_frames(video_files: dict[str, Path]) -> dict[str, list[np.ndarray]]:
    """
    Load all frames from both videos into memory.
    Pre-loading avoids seeking issues with OpenCV VideoCapture,
    which can be unreliable when stepping backwards frame by frame.
    """
    all_frames = {}
    for cam_key, path in video_files.items():
        print(f"  Loading {path.name} ...")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"ERROR: could not open video: {path}")
            sys.exit(1)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        all_frames[cam_key] = frames
        print(f"    {len(frames)} frames loaded")

    return all_frames


def draw_overlay(frame: np.ndarray, cam_label: str, t: int,
                 num_frames: int, paused: bool) -> np.ndarray:
    """Draw frame number, camera label, and play/pause indicator onto a frame."""
    frame = frame.copy()
    h, w = frame.shape[:2]

    # Semi-transparent black bar at top for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    # Camera label
    cv2.putText(frame, cam_label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    # Frame counter
    cv2.putText(frame, f"Frame: {t} / {num_frames - 1}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2, cv2.LINE_AA)

    # Play/pause indicator
    status = "PAUSED" if paused else "PLAYING"
    colour = (0, 80, 255) if paused else (0, 220, 0)
    cv2.putText(frame, status, (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

    return frame


def visualise(dataset_root: Path, episode_idx: int):
    print(f"\nLoading episode {episode_idx} from {dataset_root.name} ...")
    video_files = find_video_files(dataset_root, episode_idx)
    all_frames = load_all_frames(video_files)

    # All cameras must have the same number of frames
    frame_counts = {k: len(v) for k, v in all_frames.items()}
    num_frames = min(frame_counts.values())
    if len(set(frame_counts.values())) > 1:
        print(f"  WARNING: cameras have different frame counts: {frame_counts}")
        print(f"  Using minimum: {num_frames} frames")
    else:
        print(f"  {num_frames} frames total ({num_frames / 30:.1f}s at 30fps)")

    print(f"\nControls:")
    print(f"  [Space] : Play / Pause")
    print(f"  [f]     : Step Forward 1 frame  (auto-pauses)")
    print(f"  [b]     : Step Back 1 frame     (auto-pauses)")
    print(f"  [q]     : Quit and print current frame index")
    print(f"\nStarting playback (paused at frame 0) ...")

    cam_labels = [k.split(".")[-1] for k in CAMERA_KEYS]  # e.g. "0_femtobolt"

    t = 0
    paused = True   # start paused so you can see frame 0 immediately
    window_name = f"Episode {episode_idx:06d} — {dataset_root.name}"

    while True:
        # Clamp t to valid range
        t = max(0, min(t, num_frames - 1))

        # Build combined frame from all cameras side by side
        rendered = []
        for cam_key, label in zip(CAMERA_KEYS, cam_labels):
            frame = all_frames[cam_key][t].copy()
            frame = draw_overlay(frame, label, t, num_frames, paused)
            rendered.append(frame)

        # Resize to same height if cameras differ in resolution
        heights = [f.shape[0] for f in rendered]
        if len(set(heights)) > 1:
            target_h = max(heights)
            resized = []
            for f in rendered:
                if f.shape[0] != target_h:
                    scale = target_h / f.shape[0]
                    new_w = int(f.shape[1] * scale)
                    f = cv2.resize(f, (new_w, target_h))
                resized.append(f)
            rendered = resized

        combined = np.concatenate(rendered, axis=1)
        cv2.imshow(window_name, combined)

        # Auto-pause at last frame
        if t == num_frames - 1 and not paused:
            paused = True

        delay = 0 if paused else PLAY_DELAY_MS
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            print(f"\n{'='*50}")
            print(f"  Quit at frame: {t}")
            print(f"  Episode {episode_idx} end frame index = {t}")
            print(f"  (Use this as the trim end index for this episode)")
            print(f"{'='*50}\n")
            break
        elif key == ord(' '):
            paused = not paused
            if not paused and t == num_frames - 1:
                t = 0   # loop back to start if unpausing at end
        elif key == ord('f'):
            t += 1
            paused = True
        elif key == ord('b'):
            t -= 1
            paused = True
        else:
            if not paused:
                t += 1

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot episode visualiser with frame-by-frame controls"
    )
    parser.add_argument(
        "episode", type=int,
        help="Episode index to visualise (e.g. 5)"
    )
    parser.add_argument(
        "--dataset", type=str, default=str(DEFAULT_DATASET),
        help=f"Path to dataset root (default: {DEFAULT_DATASET})"
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        print(f"ERROR: dataset not found at {dataset_root}")
        print(f"       Pass --dataset /path/to/your/dataset")
        sys.exit(1)

    visualise(dataset_root, args.episode)


if __name__ == "__main__":
    main()