import numpy as np
import pandas as pd

df = pd.read_parquet("dataset/raw/data/episode_000000.parquet")
print(df.head())

df.to_csv("episode_00.csv")

print(df.columns.tolist())
print(df.dtypes)
print(df.shape)
print(df.head())

# Check the actual shape of each array column
for col in ['action', 'observation.state', 'observation.effort', 'observation.force_torque', 'observation.qvel']:
    sample = df[col].iloc[0]
    arr = np.array(sample)
    print(f"{col}: shape={arr.shape}, dtype={arr.dtype}")