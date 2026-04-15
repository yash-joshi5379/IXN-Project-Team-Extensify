import subprocess
import sys
import os
import config
from pathlib import Path

# Ensure all downloads are done on the scratch space
SCRATCH_BASE = "/scratch0/yjoshi/lerobot_data"
HF_HOME = os.path.join(SCRATCH_BASE, "huggingface_cache")
HF_LEROBOT_HOME = os.path.join(SCRATCH_BASE, "lerobot_home")

# Create these directories immediately
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(HF_LEROBOT_HOME, exist_ok=True)

print(f"[DEBUG] HF_HOME set to: {HF_HOME}")
print(f"[DEBUG] HF_LEROBOT_HOME set to: {HF_LEROBOT_HOME}")

# Creating a clean env dictionary for the subprocess
train_env = os.environ.copy()
train_env["HF_HOME"] = HF_HOME
train_env["HF_DATASETS_CACHE"] = HF_HOME
train_env["HF_LEROBOT_HOME"] = HF_LEROBOT_HOME

if "LEROBOT_HOME" in train_env:
    del train_env["LEROBOT_HOME"]

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
    f"--wandb.project={config.WANDB_PROJECT}"
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