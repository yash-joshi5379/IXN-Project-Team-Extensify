import subprocess
import sys
import config

# Final check of important details just before training starts
print("==================================================")
print(f"Launching LeRobot Training: {config.JOB_NAME}")
print("==================================================")
print(f"Dataset : {config.DATASET_REPO_ID}")
print(f"Model   : {config.POLICY_PATH}")
print(f"Batch   : {config.BATCH_SIZE}")
print(f"Steps   : {config.STEPS}")
print("==================================================\n")

# Construct the command exactly as the Hugging Face website recommends
command = [
    "lerobot-train",
    f"--policy.path={config.POLICY_PATH}",
    f"--dataset.repo_id={config.DATASET_REPO_ID}",
    f"--batch_size={str(config.BATCH_SIZE)}",
    f"--steps={str(config.STEPS)}",
    f"--output_dir={config.OUTPUT_DIR}",
    f"--job_name={config.JOB_NAME}",
    f"--policy.device={config.DEVICE}",
    f"--wandb.enable={str(config.WANDB_ENABLE).lower()}",
    f"--wandb.project={config.WANDB_PROJECT}"
]

try:
    # check=True ensures Python safely catches any crashes from LeRobot
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"\n[ERROR] Training interrupted or failed with exit code {e.returncode}.")
    sys.exit(e.returncode)
except KeyboardInterrupt:
    print("\n[INFO] Training manually stopped by user.")
    sys.exit(0)