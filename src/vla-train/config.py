# Core Paths
DATASET_REPO_ID = "yjoshi/cylinder-pick-place" # Change to your HF username
POLICY_PATH = "lerobot/smolvla_base"
DEVICE = "cuda"

# Hyperparameters
BATCH_SIZE = 32
STEPS = 20000

# Weights & Biases (WandB) Logging
WANDB_ENABLE = True
WANDB_PROJECT = "smolvla-cylinder-pick-and-place"
JOB_NAME = "training-run-1"

# Output Directory
OUTPUT_DIR = f"outputs/train/{JOB_NAME}"