import os, json, sys
import pandas as pd

# -- CONFIG --
EPISODE_TO_REMOVE = int(sys.argv[1]) if len(sys.argv) > 1 else None

if EPISODE_TO_REMOVE is None:
    print("Usage: python src/remove_episode.py <episode_index>")
    print("Example: python src/remove_episode.py 5")
    sys.exit(1)

DATA_DIR   = 'dataset/processed/data/chunk-000'
WRIST_DIR  = 'dataset/processed/videos/chunk-000/observation.images.wrist_cam'
STATIC_DIR = 'dataset/processed/videos/chunk-000/observation.images.static_cam'
META_DIR   = 'dataset/processed/meta'
TASK       = 'Grasp the red cylinder and place it vertically into the circular slot within the white box.'

# Confirm episode exists
parquet_path = f'{DATA_DIR}/episode_{EPISODE_TO_REMOVE:06d}.parquet'
if not os.path.exists(parquet_path):
    print(f"ERROR: episode_{EPISODE_TO_REMOVE:06d} does not exist.")
    sys.exit(1)

print(f"Removing episode {EPISODE_TO_REMOVE:06d}...")

# Delete the 3 files for this episode
for path in [
    f'{DATA_DIR}/episode_{EPISODE_TO_REMOVE:06d}.parquet',
    f'{WRIST_DIR}/episode_{EPISODE_TO_REMOVE:06d}.mp4',
    f'{STATIC_DIR}/episode_{EPISODE_TO_REMOVE:06d}.mp4',
]:
    if os.path.exists(path):
        os.remove(path)
        print(f'  Deleted: {path}')
    else:
        print(f'  WARNING: not found: {path}')

# Then renumber remaining files to close the gap
for folder in [DATA_DIR, WRIST_DIR, STATIC_DIR]:
    ext   = '.parquet' if folder == DATA_DIR else '.mp4'
    files = sorted([f for f in os.listdir(folder) if f.endswith(ext)])
    for i, f in enumerate(files):
        old_path = os.path.join(folder, f)
        new_path = os.path.join(folder, f'episode_{i:06d}{ext}')
        if old_path != new_path:
            os.rename(old_path, new_path)
    print(f'  Renumbered {len(files)} files in {folder}')

# Fix internal parquet columns 
files      = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
global_idx = 0
for f in files:
    ep_id = int(f.replace('episode_', '').replace('.parquet', ''))
    path  = os.path.join(DATA_DIR, f)
    df    = pd.read_parquet(path)
    n     = len(df)
    df['episode_index'] = ep_id
    df['frame_index']   = range(n)
    df['index']         = range(global_idx, global_idx + n)
    df.to_parquet(path, index=False)
    global_idx += n

print(f'  Fixed parquet internals. Total frames: {global_idx}')

# Regenerate meta files 
files         = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
total_frames  = 0
episodes_rows = []

for pf in files:
    ep_id = int(pf.replace('episode_', '').replace('.parquet', ''))
    df    = pd.read_parquet(os.path.join(DATA_DIR, pf))
    total_frames += len(df)
    episodes_rows.append({
        'episode_index': ep_id,
        'tasks': [TASK],
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
episodes_df.to_parquet(
    f'{META_DIR}/episodes/chunk-000/data-00000-of-00001.parquet',
    index=False
)

with open(f'{META_DIR}/info.json') as f:
    info = json.load(f)
info['total_episodes'] = len(files)
info['total_frames']   = total_frames
with open(f'{META_DIR}/info.json', 'w') as f:
    json.dump(info, f, indent=2)

# Print summary
print()
print(f'Done. Removed episode {EPISODE_TO_REMOVE:06d}.')
print(f'Dataset now has {len(files)} episodes, {total_frames} total frames.')
print(f'Episodes are now numbered 0 to {len(files) - 1}.')
print()
print('NOTE: All episodes after the removed one have been renumbered.')
print(f'      What was episode {EPISODE_TO_REMOVE + 1} is now episode {EPISODE_TO_REMOVE}, and so on.')