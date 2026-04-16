import pandas as pd
import json

# Check tasks.parquet
print('=== tasks.parquet ===')
tasks = pd.read_parquet('/scratch0/yjoshi/lerobot_data/lerobot_home/yjoshi5379/cylinder-pick-place/meta/tasks.parquet')
print(tasks)
print()

# Check episodes parquet
print('=== episodes parquet ===')
eps = pd.read_parquet('/scratch0/yjoshi/lerobot_data/lerobot_home/yjoshi5379/cylinder-pick-place/meta/episodes/chunk-000/data-00000-of-00001.parquet')
print(eps[['episode_index','tasks']].head(3))
print()

# Check a data parquet
print('=== data parquet columns ===')
df = pd.read_parquet('/scratch0/yjoshi/lerobot_data/lerobot_home/yjoshi5379/cylinder-pick-place/data/chunk-000/episode_000000.parquet')
print('Columns:', df.columns.tolist())
print('task_index values:', df['task_index'].unique() if 'task_index' in df.columns else 'MISSING')
print('First row task_index:', df['task_index'].iloc[0] if 'task_index' in df.columns else 'MISSING')
