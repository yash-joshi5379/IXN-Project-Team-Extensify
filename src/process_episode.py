import pandas as pd
import subprocess
import os

# --- Configuration (THIS IS ALL YOU NEED TO CHANGE PER EPSISODE) ---
EPISODE = "episode_000045"   # Episode number (ensure the number has 6 characters) 
END_FRAME = 452              # The exact frame the task finishes (obtained from find_end_frame.py)
# -------------------------------------------------------------------

# --- Directory configs (DON'T CHANGE THESE) ---
RAW_DIR = "dataset/raw"         # Where your original files live
PROCESSED_DIR = "dataset/processed" # Where the final padded files will go

# Ensure the output directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ==========================================
# 1. TRIM THE PARQUET DATA
# ==========================================
print(f"[{EPISODE}] Trimming parquet data...")
df = pd.read_parquet(f"{RAW_DIR}/data/{EPISODE}.parquet")

# Keep only the rows from the start up to the END_FRAME
df_trimmed = df.iloc[:END_FRAME] 
df_trimmed.to_parquet(f"{PROCESSED_DIR}/data/{EPISODE}.parquet")
print(f"Parquet file trimmed.")

# ==========================================
# 2. TRIM AND PAD VIDEOS USING FFMPEG
# ==========================================
def process_video(input_path, output_path, end_frame):
    print(f"[{EPISODE}] Running FFmpeg padding on {input_path}...")
    
    # The FFmpeg command constructed for Python's subprocess
    command = [
        "ffmpeg",
        "-y",                        # Overwrite existing files automatically
        "-i", input_path,            # The input video file
        "-frames:v", str(end_frame), # Stop processing exactly at this frame number
        "-vf", "scale=256:256:force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2:black",
        output_path                  # The output video file
    ]
    
    # Run the command invisibly, but catch any errors if it fails
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"  -> Saved perfectly squared video to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"  -> Error processing {input_path}. Ensure FFmpeg is installed.")

# Run for the Static Camera
process_video(
    input_path=f"{RAW_DIR}/videos/static_camera/{EPISODE}.mp4", 
    output_path=f"{PROCESSED_DIR}/videos/static_camera/{EPISODE}.mp4", 
    end_frame=END_FRAME
)

# Run for the Onboard Camera
process_video(
    input_path=f"{RAW_DIR}/videos/onboard_camera/{EPISODE}.mp4", 
    output_path=f"{PROCESSED_DIR}/videos/onboard_camera/{EPISODE}.mp4", 
    end_frame=END_FRAME
)

print(f"\nSuccess! {EPISODE} is processed.")