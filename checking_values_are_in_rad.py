import pandas as pd

df = pd.read_csv("episode_24.csv")

print(df["observation.state"].iloc[0])