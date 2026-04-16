import subprocess
import sys
import os
import config
from pathlib import Path

# Ensure all downloads are done on the scratch space
SCRATCH_BASE = "/scratch0/yjoshi/lerobot_data"
HF_LEROBOT_HOME = os.path.join(SCRATCH_BASE, "lerobot_home")
HF_HOME = os.path.join(SCRATCH_BASE, "huggingface_cache")
LOCAL_MODEL_PATH = os.path.join(SCRATCH_BASE, "smolvla_base_weights")

# Create these directories immediately
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(HF_LEROBOT_HOME, exist_ok=True)

# Creating a clean env dictionary for the subprocess
train_env = os.environ.copy()
train_env["HF_HOME"] = HF_HOME
train_env["HF_DATASETS_CACHE"] = HF_HOME
train_env["HF_LEROBOT_HOME"] = HF_LEROBOT_HOME

# Find the path to your conda environment libraries
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    conda_lib_path = os.path.join(conda_prefix, "lib")
    # Add the conda lib path to the front of the library search path
    current_ld = train_env.get("LD_LIBRARY_PATH", "")
    train_env["LD_LIBRARY_PATH"] = f"{conda_lib_path}:{current_ld}"

# !!! IMPORTANT: Replace with your actual token !!!
train_env["HF_TOKEN"] = "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" 

# WandB Offline Mode: Since GPU is firewalled
train_env["WANDB_MODE"] = "offline" 
train_env["WANDB_DIR"] = SCRATCH_BASE

# Final check of important details just before training starts
print("==================================================")
print(f"Launching LeRobot Training: {config.JOB_NAME}")
print("==================================================")
print(f"Dataset     : {config.DATASET_REPO_ID}")
print(f"Model       : {config.POLICY_PATH}")
print(f"Output      : {config.OUTPUT_DIR}")
print(f"Batch       : {config.BATCH_SIZE}")
print(f"Steps       : {config.STEPS}")
print(f"Checkpoints : Every {config.SAVE_FREQ} steps")
print("==================================================\n")

# Maps static cam to both camera1 and camera3, and wrist cam to camera2 to meet SmolVLA requirement
rename_map = '{"observation.images.static_cam": ["observation.images.camera1", "observation.images.camera3"], "observation.images.wrist_cam": "observation.images.camera2"}'

# Construct the command exactly as the Hugging Face website recommends
command = [
    "lerobot-train",
    f"--policy.path={config.POLICY_PATH}",
    f"--policy.repo_id={config.POLICY_REPO_ID}",
    f"--dataset.repo_id={config.DATASET_REPO_ID}",
    f"--batch_size={str(config.BATCH_SIZE)}",
    f"--steps={str(config.STEPS)}",
    f"--save_freq={str(config.SAVE_FREQ)}",
    f"--output_dir={config.OUTPUT_DIR}",
    f"--job_name={config.JOB_NAME}",
    f"--policy.device={config.DEVICE}",
    f"--wandb.enable={str(config.WANDB_ENABLE).lower()}",
    f"--wandb.project={config.WANDB_PROJECT}",
    f"--rename_map", rename_map
]

print(f"\nLAUNCHING OFFLINE TRAINING ON BUMBLEBEE")
print(f"Data is stored in: {HF_LEROBOT_HOME}\n")

try:
    # check=True ensures Python safely catches any crashes from LeRobot
    subprocess.run(command, env=train_env, check=True)
except subprocess.CalledProcessError as e:
    print(f"\n[ERROR] Training interrupted or failed with exit code {e.returncode}.")
    sys.exit(e.returncode)
except KeyboardInterrupt:
    print("\n[INFO] Training manually stopped by user.")
    sys.exit(0)