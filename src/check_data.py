import pandas as pd
import cv2

# -----------RUN THIS AFTER YOU PROCESS ANY EPISODE (JUST CHANGE EPISODE NUMBER)---------------
EPISODE = "episode_000001"

# test if we can read episode data file
df = pd.read_parquet(f"dataset/processed/data/{EPISODE}.parquet")
print(f"{EPISODE} has {len(df)} rows of data") # should be same number as your end frame index

# test if we can see episode static video file
cap = cv2.VideoCapture(f"dataset/processed/videos/static_camera/{EPISODE}.mp4")
ret, frame = cap.read()     # gets first image frame from video
if ret:
    print(f"Video frame dimensions (height, wdith, channels): {frame.shape}") # should be (256, 256, 3)
cap.release()
