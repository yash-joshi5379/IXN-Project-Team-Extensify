import json, glob, re
import numpy as np
import pandas as pd
from pathlib import Path

DATASET_ROOT = Path("dataset-v2-pro")

STAT_COLS = [
    "observation.state",
    "action",
    "observation.effort",
    "observation.qvel",
]

CAMERA_KEYS = [
    "observation.images.0_femtobolt",
    "observation.images.1_gemini330",
]

def parse_vec(s):
    return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(s))]

files = sorted(glob.glob(str(DATASET_ROOT / "data" / "**" / "*.parquet"), recursive=True))
print(f"Found {len(files)} parquet files, loading ...")
dfs = [pd.read_parquet(f) for f in files]
print("Loaded.")

def stack(col):
    return np.concatenate(
        [np.stack(df[col].apply(parse_vec).tolist()).astype(np.float32) for df in dfs],
        axis=0,
    )

stats = {}

# Proprioceptive stats — computed exactly from your data
for col in STAT_COLS:
    arr = stack(col)
    stats[col] = {
        "mean": arr.mean(axis=0).tolist(),
        "std":  (arr.std(axis=0) + 1e-8).tolist(),
        "min":  arr.min(axis=0).tolist(),
        "max":  arr.max(axis=0).tolist(),
    }
    print(f"  {col}: mean={[f'{v:.3f}' for v in stats[col]['mean']]}")

# Camera stats — aggregate from episodes_stats.jsonl using weighted mean
print("  Aggregating image stats from episodes_stats.jsonl ...")
ep_stats_path = DATASET_ROOT / "meta" / "episodes_stats.jsonl"
ep_stats = [json.loads(l) for l in ep_stats_path.read_text().splitlines() if l.strip()]

for cam_key in CAMERA_KEYS:
    all_means, all_stds, all_mins, all_maxs, all_counts = [], [], [], [], []
    for rec in ep_stats:
        if cam_key in rec["stats"]:
            s = rec["stats"][cam_key]
            all_means.append(np.array(s["mean"]))
            all_stds.append(np.array(s["std"]))
            all_mins.append(np.array(s["min"]))
            all_maxs.append(np.array(s["max"]))
            all_counts.append(np.array(s["count"]))
    if all_means:
        counts = np.array([c[0] if hasattr(c, '__len__') else c for c in all_counts], dtype=np.float32)
        means  = np.stack(all_means)   # (N_episodes, 3, 1, 1)
        total  = counts.sum()
        weighted_mean = (means * counts[:, None, None, None]).sum(axis=0) / total
        stats[cam_key] = {
            "mean": weighted_mean.tolist(),
            "std":  np.stack(all_stds).mean(axis=0).tolist(),
            "min":  np.stack(all_mins).min(axis=0).tolist(),
            "max":  np.stack(all_maxs).max(axis=0).tolist(),
        }
        print(f"  {cam_key}: aggregated from {len(all_means)} episodes")
    else:
        # Fallback to ImageNet constants in (3,1,1) shape if no image stats found
        print(f"  {cam_key}: no stats in episodes_stats.jsonl, using ImageNet fallback")
        stats[cam_key] = {
            "mean": [[[0.485]], [[0.456]], [[0.406]]],
            "std":  [[[0.229]], [[0.224]], [[0.225]]],
            "min":  [[[0.0  ]], [[0.0  ]], [[0.0  ]]],
            "max":  [[[1.0  ]], [[1.0  ]], [[1.0  ]]],
        }

out_path = DATASET_ROOT / "meta" / "stats.json"
with open(out_path, "w") as f:
    json.dump(stats, f, indent=2)
print(f"\n✅  stats.json written to {out_path}")