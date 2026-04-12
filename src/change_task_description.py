import os, json
import pandas as pd

NEW_TASK = 'Grasp the red cylinder and place it vertically into the circular slot within the white box.'
META_DIR = 'dataset/processed/meta'
DATA_DIR = 'dataset/processed/data/chunk-000'

# Update tasks.parquet
tasks_df = pd.DataFrame([{'task_index': 0, 'task': NEW_TASK}])
tasks_df.to_parquet(f'{META_DIR}/tasks.parquet', index=False)

# Regenerate episodes parquet with updated task string
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
episodes_rows = []
for pf in files:
    ep_id = int(pf.replace('episode_', '').replace('.parquet', ''))
    df = pd.read_parquet(os.path.join(DATA_DIR, pf))
    episodes_rows.append({
        'episode_index': ep_id,
        'tasks': [NEW_TASK],
        'length': len(df),
        'data/chunk_index': 0,
        'data/file_index': ep_id,
        'meta/episodes/chunk_index': [0],
        'meta/episodes/file_index': [0],
        'videos/observation.images.wrist_cam/chunk_index': 0,
        'videos/observation.images.wrist_cam/file_index': ep_id,
        'videos/observation.images.wrist_cam/from_timestamp': 0.0,
        'videos/observation.images.static_cam/chunk_index': 0,
        'videos/observation.images.static_cam/file_index': ep_id,
        'videos/observation.images.static_cam/from_timestamp': 0.0,
    })

episodes_df = pd.DataFrame(episodes_rows)
episodes_df.to_parquet(f'{META_DIR}/episodes/chunk-000/data-00000-of-00001.parquet', index=False)

# Update info.json
with open(f'{META_DIR}/info.json') as f:
    info = json.load(f)
info['total_tasks'] = 1
with open(f'{META_DIR}/info.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f'Task updated to: \n{NEW_TASK}')