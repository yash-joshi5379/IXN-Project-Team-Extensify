import os
import glob
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURATION ---
INPUT_BASE = "dataset-v2-raw"
OUTPUT_BASE = "dataset-v2-pro"
NOTES_FILE = "notes.txt"

# Input Subdirectories
PARQUET_DIR_IN = os.path.join(INPUT_BASE, "data", "chunk-000")
CAM1_DIR_IN = os.path.join(INPUT_BASE, "videos", "chunk-000", "observation.images.0_femtobolt")
CAM2_DIR_IN = os.path.join(INPUT_BASE, "videos", "chunk-000", "observation.images.1_gemini330")

# Output Subdirectories (mirroring the structure)
PARQUET_DIR_OUT = os.path.join(OUTPUT_BASE, "data", "chunk-000")
CAM1_DIR_OUT = os.path.join(OUTPUT_BASE, "videos", "chunk-000", "observation.images.0_femtobolt")
CAM2_DIR_OUT = os.path.join(OUTPUT_BASE, "videos", "chunk-000", "observation.images.1_gemini330")
# ---------------------

def trim_video(input_path, output_path, end_frame):
    """Reads a video, trims it up to end_frame, and saves it."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # mp4v is standard for .mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        # Stop if we hit the end of the video OR our target end_frame
        if not ret or frame_idx > end_frame:
            break
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

def trim_parquet(input_path, output_path, end_frame):
    """Reads a parquet file, trims the rows, and saves it."""
    df = pd.read_parquet(input_path)
    # Assuming each row is a frame, trim up to end_frame (inclusive)
    df_trimmed = df.iloc[:end_frame + 1] 
    df_trimmed.to_parquet(output_path)

def main():
    # Create all output directories
    os.makedirs(PARQUET_DIR_OUT, exist_ok=True)
    os.makedirs(CAM1_DIR_OUT, exist_ok=True)
    os.makedirs(CAM2_DIR_OUT, exist_ok=True)
    
    # 1. Parse notes.txt
    episode_instructions = {}
    with open(NOTES_FILE, 'r') as f:
        for line in f:
            if '-' in line:
                # Force lowercase 'episode' to match file systems
                ep_name, action = line.strip().split('-')
                ep_name = ep_name.strip().lower() 
                episode_instructions[ep_name] = action.strip()

    new_ep_id = 0
    original_episodes = sorted(list(episode_instructions.keys()))
    
    # 2. Process each episode in sequential order
    for orig_ep in tqdm(original_episodes, desc="Processing Episodes"):
        instruction = episode_instructions[orig_ep]
        
        if instruction == "INVALID":
            print(f"\nSkipping {orig_ep} (Marked INVALID)")
            continue
            
        end_frame = int(instruction)
        new_ep_name = f"episode_{new_ep_id:06d}"
        
        # Build exact file paths based on the LeRobot structure
        orig_parquet = os.path.join(PARQUET_DIR_IN, f"{orig_ep}.parquet")
        orig_cam1 = os.path.join(CAM1_DIR_IN, f"{orig_ep}.mp4")
        orig_cam2 = os.path.join(CAM2_DIR_IN, f"{orig_ep}.mp4")
        
        # Check if the parquet file actually exists before processing
        if not os.path.exists(orig_parquet):
            print(f"\nWarning: Could not find {orig_parquet}, skipping...")
            continue

        # Trim Parquet
        new_parquet = os.path.join(PARQUET_DIR_OUT, f"{new_ep_name}.parquet")
        trim_parquet(orig_parquet, new_parquet, end_frame)
        
        # Trim Video 1 (Femtobolt)
        if os.path.exists(orig_cam1):
            new_cam1 = os.path.join(CAM1_DIR_OUT, f"{new_ep_name}.mp4")
            trim_video(orig_cam1, new_cam1, end_frame)
            
        # Trim Video 2 (Gemini330)
        if os.path.exists(orig_cam2):
            new_cam2 = os.path.join(CAM2_DIR_OUT, f"{new_ep_name}.mp4")
            trim_video(orig_cam2, new_cam2, end_frame)
            
        new_ep_id += 1

    print(f"\n✅ Processing complete. {new_ep_id} valid episodes saved to {OUTPUT_BASE}/.")

if __name__ == "__main__":
    main()