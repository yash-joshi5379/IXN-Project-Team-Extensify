# Core Paths
DATASET_REPO_ID = "yjoshi5379/cylinder-pick-place" # Change to your HF username
POLICY_PATH = "lerobot/smolvla_base"
POLICY_REPO_ID = "yjoshi5379/smolvla-trained-policy"
DEVICE = "cuda"

# Hyperparameters
BATCH_SIZE = 32
STEPS = 20000
SAVE_FREQ = 2000    # saves a new model checkpoint every 2000 steps

# Weights & Biases (WandB) Logging
WANDB_ENABLE = True
WANDB_PROJECT = "smolvla-cylinder-pick-and-place"
JOB_NAME = "training-run-1"

# Output Directory
OUTPUT_DIR = f"outputs/train/{JOB_NAME}"