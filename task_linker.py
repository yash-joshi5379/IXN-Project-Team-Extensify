import pandas as pd
from pathlib import Path

# 1. Setup paths
DATASET_PATH = Path("/scratch0/yjoshi/lerobot_data/lerobot_home/yjoshi5379/cylinder-pick-place")
EPISODES_PATH = DATASET_PATH / "meta/episodes/chunk-000/data-00000-of-00001.parquet"
TASKS_PATH = DATASET_PATH / "meta/tasks.parquet"

# 2. Check the Task
tasks_df = pd.read_parquet(TASKS_PATH)
print("Detected Tasks:")
print(tasks_df)

# 3. Update Episodes to point to Task 0
episodes_df = pd.read_parquet(EPISODES_PATH)
print(f"\nLinking {len(episodes_df)} episodes to Task 0...")

# Add the missing column
episodes_df['task_index'] = 0

# 4. Save back to scratch and your project repo
episodes_df.to_parquet(EPISODES_PATH)

repo_episodes = Path("/scratch0/yjoshi/IXN-Project-Team-Extensify/dataset/processed/meta/episodes/chunk-000/data-00000-of-00001.parquet")
if repo_episodes.exists():
    episodes_df.to_parquet(repo_episodes)
    print("Updated repo copy as well.")

print("\n[SUCCESS] Task-to-Episode link established!")