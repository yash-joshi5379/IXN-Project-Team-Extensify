import pandas as pd
df = pd.read_json('dataset-v2/meta/tasks.jsonl')
task = f"\n{df.iloc[0].task}"
print(task)