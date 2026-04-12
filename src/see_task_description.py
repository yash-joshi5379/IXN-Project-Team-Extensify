import pandas as pd
df = pd.read_parquet('dataset/processed/meta/tasks.parquet')
task = f"\n{df.iloc[0].task}"
print(task)