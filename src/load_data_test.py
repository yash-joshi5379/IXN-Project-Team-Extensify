import numpy as np
import pandas as pd

df = pd.read_parquet("dataset-v2/data/chunk-000/episode_000124.parquet")
df.to_csv("episode_124.csv")

print(df.columns.tolist())
print(df.dtypes)
print(df.shape)
print(df.head())

# Check the actual shape of each array column
for col in ['action', 'observation.state', 'observation.effort', 'observation.qvel']:
    sample = df[col].iloc[0]
    arr = np.array(sample)
    print(f"{col}: shape={arr.shape}, dtype={arr.dtype}")