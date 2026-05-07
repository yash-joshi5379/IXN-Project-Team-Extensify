"""
ACT Training Script — xArm7 Cylinder Insert Task
==================================================
Wraps act/imitate_episodes.py with a clean CONFIG block and wandb logging.

USAGE (on RunPod, after converting dataset):
  cd /root/IXN-Project-Team-Extensify
  conda activate aloha
  python src/train_act.py

The script also patches imitate_episodes.py to add wandb logging if it
isn't already present, so you get live loss curves at wandb.ai.
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these between runs
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT  = Path("/root/IXN-Project-Team-Extensify")
ACT_DIR       = PROJECT_ROOT / "act"
DATASET_DIR   = PROJECT_ROOT / "act_train_dataset"   # output of convert_to_act_format.py
OUTPUT_DIR    = PROJECT_ROOT / "act_output"

# Dataset
NUM_EPISODES  = 58           # total number of episodes in act_dataset/
CAMERA_NAMES  = ["cam_static", "cam_wrist"]  # must match convert_to_act_format.py CAMERA_RENAME values

# Task name — used as subfolder inside OUTPUT_DIR for checkpoints
TASK_NAME     = "cylinder_insert"

# ACT hyperparameters
POLICY_CLASS      = "ACT"
KL_WEIGHT         = 10       # weight on KL divergence loss (VAE regularisation)
                              # higher = smoother actions, lower = more expressive
CHUNK_SIZE        = 100      # number of actions predicted at once
                              # ~3 seconds at 30fps. Increase for longer motions.
HIDDEN_DIM        = 512      # transformer hidden dimension
DIM_FEEDFORWARD   = 3200     # transformer feedforward dimension
NUM_EPOCHS        = 2000     # full passes through the dataset
                              # 2000 epochs × 58 episodes = ~116K gradient steps
BATCH_SIZE        = 8        # samples per gradient step
                              # keep low (8–16) for stable training with small dataset
LEARNING_RATE     = 1e-5     # ACT uses a lower LR than SmolVLA — 1e-5 is standard
SEED              = 0

# WandB
WANDB_ENABLE  = True
WANDB_PROJECT = "act_aloha_xarm7_insert"

# ═══════════════════════════════════════════════════════════════════════════════


def make_job_name() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"act_{TASK_NAME}_ep{NUM_EPOCHS}_bs{BATCH_SIZE}_lr{LEARNING_RATE:.0e}_{timestamp}"


def inject_wandb_into_act(job_name: str):
    """
    Patch act/imitate_episodes.py to add wandb logging if not already present.
    Injects wandb.init() and wandb.log() calls.
    Only modifies the file in-place if wandb is not already imported.
    """
    script_path = ACT_DIR / "imitate_episodes.py"
    content = script_path.read_text()

    if "import wandb" in content:
        print("  wandb already present in imitate_episodes.py — no patch needed.")
        return

    print("  Injecting wandb into imitate_episodes.py ...")

    # Add wandb import after existing imports
    content = content.replace(
        "import argparse",
        "import argparse\nimport wandb"
    )

    # Add wandb.init() after args are parsed — find the main() function start
    # and inject after the print statements that show config
    wandb_init = f"""
    # WandB initialisation (injected by train_act.py)
    if args.wandb_enable:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    """
    content = content.replace(
        "set_seed(args.seed)",
        "set_seed(args.seed)" + wandb_init
    )

    # Add wandb.log() inside the training loop after each epoch
    content = content.replace(
        "print(f'Train loss: {summary_string}')",
        "print(f'Train loss: {summary_string}')\n"
        "        if args.wandb_enable:\n"
        "            wandb.log({'train_loss': epoch_summary['loss'], 'epoch': epoch})"
    )

    content = content.replace(
        "print(f'Val loss:   {summary_string}')",
        "print(f'Val loss:   {summary_string}')\n"
        "        if args.wandb_enable:\n"
        "            wandb.log({'val_loss': epoch_summary['loss'], 'epoch': epoch})"
    )

    # Add wandb args to argparse
    content = content.replace(
        "parser.add_argument('--seed', action='store', type=int, help='seed', required=True)",
        "parser.add_argument('--seed', action='store', type=int, help='seed', required=True)\n"
        "    parser.add_argument('--wandb_enable', action='store_true', default=False)\n"
        "    parser.add_argument('--wandb_project', action='store', type=str, default='act')\n"
        "    parser.add_argument('--wandb_run_name', action='store', type=str, default='run')"
    )

    script_path.write_text(content)
    print("  ✓ wandb injected into imitate_episodes.py")


def main():
    if not DATASET_DIR.exists():
        print(f"ERROR: act_dataset/ not found at {DATASET_DIR}")
        print("       Run convert_to_act_format.py first.")
        sys.exit(1)

    if not ACT_DIR.exists():
        print(f"ERROR: act/ repo not found at {ACT_DIR}")
        sys.exit(1)

    n_files = len(list(DATASET_DIR.glob("episode_*.hdf5")))
    if n_files != NUM_EPISODES:
        print(f"WARNING: Found {n_files} episodes but NUM_EPISODES={NUM_EPISODES}")
        print(f"         Update NUM_EPISODES in the CONFIG block to match.")

    job_name = make_job_name()
    ckpt_dir = OUTPUT_DIR / job_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ACT Training — xArm7 Cylinder Insert")
    print("=" * 60)
    print(f"  Task name         : {TASK_NAME}")
    print(f"  Dataset           : {DATASET_DIR}  ({n_files} episodes)")
    print(f"  Camera names      : {CAMERA_NAMES}")
    print(f"  Epochs            : {NUM_EPOCHS}")
    print(f"  Batch size        : {BATCH_SIZE}")
    print(f"  Learning rate     : {LEARNING_RATE}")
    print(f"  Chunk size        : {CHUNK_SIZE}")
    print(f"  KL weight         : {KL_WEIGHT}")
    print(f"  Hidden dim        : {HIDDEN_DIM}")
    print(f"  Checkpoint dir    : {ckpt_dir}")
    print(f"  WandB project     : {WANDB_PROJECT if WANDB_ENABLE else 'disabled'}")
    print("=" * 60)

    # Inject wandb into ACT if enabled
    if WANDB_ENABLE:
        inject_wandb_into_act(job_name)

    cmd = [
        sys.executable,
        str(ACT_DIR / "imitate_episodes.py"),
        "--task_name",        TASK_NAME,
        "--ckpt_dir",         str(ckpt_dir),
        "--policy_class",     POLICY_CLASS,
        "--kl_weight",        str(KL_WEIGHT),
        "--chunk_size",       str(CHUNK_SIZE),
        "--hidden_dim",       str(HIDDEN_DIM),
        "--batch_size",       str(BATCH_SIZE),
        "--dim_feedforward",  str(DIM_FEEDFORWARD),
        "--num_epochs",       str(NUM_EPOCHS),
        "--lr",               str(LEARNING_RATE),
        "--seed",             str(SEED),
    ]

    # if WANDB_ENABLE:
    #     cmd += [
    #         "--wandb_enable",
    #         "--wandb_project", WANDB_PROJECT,
    #         "--wandb_run_name", job_name,
    #     ]

    log_file = OUTPUT_DIR / f"{job_name}.log"
    print(f"\nLog file: {log_file}")
    print("Starting training ...\n")

    with open(log_file, "w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ACT_DIR),
        )
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

    rc = process.wait()
    if rc == 0:
        print(f"\n✅  Training complete. Checkpoint at: {ckpt_dir}")
        print(f"    Policy saved as: {ckpt_dir}/policy_best.ckpt")
    else:
        print(f"\n❌  Training failed (exit code {rc}). Check: {log_file}")
        sys.exit(rc)


if __name__ == "__main__":
    main()