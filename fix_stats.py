import pandas as pd
import numpy as np
import json
import glob
from pathlib import Path

# --- 1. SETUP ---
DATASET_PATH = Path("/scratch0/yjoshi/lerobot_data/lerobot_home/yjoshi5379/cylinder-pick-place")
PARQUET_FILES = glob.glob(str(DATASET_PATH / "data" / "**" / "*.parquet"), recursive=True)

# --- 2. CRUNCH DATA ---
all_data = {"action": [], "observation.state": []}
total_frames = 0
for f in PARQUET_FILES:
    df = pd.read_parquet(f)
    total_frames += len(df)
    if "action" in df.columns: all_data["action"].append(np.stack(df["action"].values))
    if "observation.state" in df.columns: all_data["observation.state"].append(np.stack(df["observation.state"].values))

final_stats = {}
for feature in all_data:
    if len(all_data[feature]) > 0:
        stacked = np.concatenate(all_data[feature], axis=0)
        final_stats[feature] = {
            "min": stacked.min(axis=0).tolist(), "max": stacked.max(axis=0).tolist(),
            "mean": stacked.mean(axis=0).tolist(), "std": stacked.std(axis=0).tolist(),
            "count": [int(total_frames)]
        }

# --- 3. THE "ALL-IN-ONE" CAMERA MAP ---
# We provide stats for BOTH the original names and the policy names
# This satisfies the Dataset Loader AND the Policy Validator
image_keys = [
    "observation.images.static_cam", 
    "observation.images.wrist_cam",
    "observation.images.camera1", 
    "observation.images.camera2", 
    "observation.images.camera3"
]

for key in image_keys:
    final_stats[key] = {
        "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0],
        "mean": [0.5, 0.5, 0.5], "std": [0.225, 0.225, 0.225],
        "count": [int(total_frames)]
    }

# --- 4. SAVE ---
stats_path = DATASET_PATH / "meta" / "stats.json"
with open(stats_path, "w") as f:
    json.dump(final_stats, f, indent=4)

# Also copy it to your project folder for safety
repo_stats = Path("/scratch0/yjoshi/IXN-Project-Team-Extensify/dataset/processed/meta/stats.json")
with open(repo_stats, "w") as f:
    json.dump(final_stats, f, indent=4)

print(f"[SUCCESS] Master stats.json created with all camera keys.")