from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_v2_common import PROJECT_ROOT, episode_name


def record_end_frame(path: Path, episode: int, end_frame: int, note: str | None) -> None:
    if end_frame <= 0:
        raise ValueError("end_frame must be greater than 0")

    data = {}
    if path.exists():
        data = json.loads(path.read_text())

    key = episode_name(episode)
    record = {"end_frame": end_frame}
    if note:
        record["note"] = note
    data[key] = record

    path.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")
    print(f"Recorded {key}: frame {end_frame} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a manually selected dataset-v2 end frame.")
    parser.add_argument("episode", type=int)
    parser.add_argument("end_frame", type=int)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--json-path", type=Path, default=PROJECT_ROOT / "end_frames_dataset_v2_0_40.json")
    args = parser.parse_args()
    record_end_frame(args.json_path, args.episode, args.end_frame, args.note)


if __name__ == "__main__":
    main()
