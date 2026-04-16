import pandas as pd
import numpy as np
import glob
from pathlib import Path

# --- 1. SETUP ---
DATASET_PATH = Path("/scratch0/yjoshi/lerobot_data/lerobot_home/yjoshi5379/cylinder-pick-place")
# Find the episode metadata file
EPISODE_FILES = glob.glob(str(DATASET_PATH / "meta" / "**" / "data-00000-of-00001.parquet"), recursive=True)

if not EPISODE_FILES:
    print("Could not find episodes.parquet! Check your meta folder.")
    exit(1)

meta_file = EPISODE_FILES[0]
print(f"Updating metadata at: {meta_file}")

# --- 2. UPDATE COLUMNS ---
df = pd.read_parquet(meta_file)

# If 'length' is present, we can calculate the cumulative indices
if 'length' in df.columns:
    # dataset_from_index is the starting frame of each episode
    # It starts at 0 and adds the length of the previous episodes
    df['dataset_from_index'] = df['length'].cumsum().shift(1).fillna(0).astype(int)
    # dataset_to_index is the end frame
    df['dataset_to_index'] = df['length'].cumsum().astype(int)
    
    print("Computed 'dataset_from_index' and 'dataset_to_index' from episode lengths.")
else:
    print("Error: 'length' column not found in episodes.parquet. Cannot calculate indices.")
    exit(1)

# --- 3. SAVE ---
df.to_parquet(meta_file)

# Also copy to your local repo for consistency
repo_meta = Path("/scratch0/yjoshi/IXN-Project-Team-Extensify/dataset/processed/meta/episodes/chunk-000/data-00000-of-00001.parquet")
if repo_meta.parent.exists():
    df.to_parquet(repo_meta)
    print(f"Also updated local repo at: {repo_meta}")

print("\n[SUCCESS] Metadata is now compatible with LeRobot v3.0!")