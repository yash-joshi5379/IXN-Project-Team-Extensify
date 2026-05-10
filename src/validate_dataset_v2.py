from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from dataset_v2_common import PROJECT_ROOT, PROCESSED_ROOT
from visualise_episode_v2 import launch_lerobot, log_direct_rerun


VALIDATION_HEADER = "## Dataset-v2 Section 2 Validation"
DEFAULT_LEROBOT_ROOT = PROJECT_ROOT / "dataset-v2-processed-lerobot"


def close_rerun_viewer() -> None:
    subprocess.run(
        ["pkill", "-f", "rerun --port=9876"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    time.sleep(0.5)


def load_current_to_original_map(mapping_path: Path) -> dict[int, int]:
    if not mapping_path.exists():
        return {}
    mapping = json.loads(mapping_path.read_text())
    current_to_original = mapping.get("current_to_original", {})
    return {int(current): int(original) for current, original in current_to_original.items()}


def episode_count(dataset_root: Path) -> int:
    info_path = dataset_root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        return int(info["total_episodes"])

    data_dir = dataset_root / "data" / "chunk-000"
    return len(sorted(data_dir.glob("episode_*.parquet")))


def wait_for_decision(episode: int, original_episode: int) -> str:
    window_name = "dataset-v2 validation: q valid, i invalid"
    canvas = np.full((240, 840, 3), 245, dtype=np.uint8)
    lines = [
        f"Current episode {episode:06d}    Original episode {original_episode:06d}",
        "Review the Rerun window, then focus this control window.",
        "Press q = valid",
        "Press i = invalid",
        "Press Esc = stop without recording this episode",
    ]

    y = 45
    for idx, line in enumerate(lines):
        scale = 0.9 if idx == 0 else 0.75
        thickness = 2 if idx in {0, 2, 3} else 1
        color = (20, 20, 20) if idx != 3 else (30, 30, 170)
        cv2.putText(canvas, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += 38

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 840, 240)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    while True:
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(100) & 0xFF
        if key in {ord("q"), ord("Q")}:
            cv2.destroyWindow(window_name)
            return "valid"
        if key in {ord("i"), ord("I")}:
            cv2.destroyWindow(window_name)
            return "invalid"
        if key == 27:
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt


def upsert_note(notes_path: Path, episode: int, original_episode: int, status: str) -> None:
    line_prefix = f"Episode_{episode:06d} "
    line = f"Episode_{episode:06d} - {status}"

    if notes_path.exists():
        lines = notes_path.read_text().splitlines()
    else:
        lines = []

    if VALIDATION_HEADER not in lines:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([VALIDATION_HEADER, line])
        notes_path.write_text("\n".join(lines).rstrip() + "\n")
        return

    start = lines.index(VALIDATION_HEADER)
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break

    for idx in range(start + 1, end):
        if lines[idx].startswith(line_prefix):
            lines[idx] = line
            break
    else:
        lines.insert(end, line)

    notes_path.write_text("\n".join(lines).rstrip() + "\n")


def launch_episode(dataset_root: Path, fallback_root: Path, backend: str, episode: int) -> None:
    if backend == "direct":
        log_direct_rerun(dataset_root, episode)
    else:
        try:
            launch_lerobot(dataset_root, episode)
        except subprocess.CalledProcessError:
            print("LeRobot visualizer failed for this episode; falling back to direct v2 Rerun logging.")
            close_rerun_viewer()
            log_direct_rerun(fallback_root, episode)


def validate_range(
    dataset_root: Path,
    fallback_root: Path,
    backend: str,
    notes_path: Path,
    mapping_path: Path,
    start_episode: int,
    end_episode: int,
) -> None:
    current_to_original = load_current_to_original_map(mapping_path)

    for episode in range(start_episode, end_episode + 1):
        original_episode = current_to_original.get(episode, episode)
        print()
        print(f"Opening episode_{episode:06d} (original episode_{original_episode:06d})")
        close_rerun_viewer()
        launch_episode(dataset_root, fallback_root, backend, episode)

        status = wait_for_decision(episode, original_episode)
        upsert_note(notes_path, episode, original_episode, status)
        print(f"Recorded episode_{episode:06d}: {status}")
        close_rerun_viewer()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset-v2 Section 2 validation with q/i bindings.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--fallback-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--backend", choices=["lerobot", "direct"], default="lerobot")
    parser.add_argument("--notes-path", type=Path, default=PROJECT_ROOT / "notes.txt")
    parser.add_argument("--mapping-path", type=Path, default=PROCESSED_ROOT / "meta" / "original_episode_map.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    fallback_root = args.fallback_root.resolve()
    total_episodes = episode_count(dataset_root)
    end_episode = args.end if args.end is not None else total_episodes - 1
    if args.start < 0 or end_episode >= total_episodes or args.start > end_episode:
        raise ValueError(f"Episode range must be within 0-{total_episodes - 1}; got {args.start}-{end_episode}")

    try:
        validate_range(
            dataset_root=dataset_root,
            fallback_root=fallback_root,
            backend=args.backend,
            notes_path=args.notes_path.resolve(),
            mapping_path=args.mapping_path.resolve(),
            start_episode=args.start,
            end_episode=end_episode,
        )
    except KeyboardInterrupt:
        close_rerun_viewer()
        print("\nStopped validation without recording the current episode.")


if __name__ == "__main__":
    main()
