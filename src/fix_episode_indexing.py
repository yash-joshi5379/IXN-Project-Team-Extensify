import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("/scratch0/yjoshi/IXN-Project-Team-Extensify/dataset-v2-pro/data/chunk-000")

parquet_files = sorted(DATA_DIR.glob("episode_*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

global_frame_index = 0  # running index across all episodes

for new_ep_idx, fpath in enumerate(tqdm(parquet_files)):
    df = pd.read_parquet(fpath)
    old_ep_idx = df["episode_index"].iloc[0]
    n_frames = len(df)

    # Rewrite episode_index to match the new sequential filename
    df["episode_index"] = new_ep_idx

    # Rewrite global frame index (continuous across all episodes)
    df["index"] = list(range(global_frame_index, global_frame_index + n_frames))

    # frame_index is per-episode (0 to N-1) — should already be correct
    # but reset it anyway to be safe
    df["frame_index"] = list(range(n_frames))

    df.to_parquet(fpath, index=False)
    global_frame_index += n_frames

    if old_ep_idx != new_ep_idx:
        print(f"  Fixed {fpath.name}: episode_index {old_ep_idx} → {new_ep_idx}")

print(f"\nDone. Total frames: {global_frame_index}")
print("Now re-run generate_meta.py to regenerate episodes_stats.jsonl with correct indices.")