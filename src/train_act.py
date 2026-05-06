"""
SmolVLA Fine-tuning Script — xArm7 Pick-and-Insert Task
========================================================
All hyperparameters are defined in the CONFIG block below.
Run with:  python src/train.py

Logs are saved to act_output/<job_name>.log
WandB run is named after JOB_NAME so each run is fully trackable.
"""

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these between runs to track different hyperparameter choices
# ═══════════════════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path("/root/IXN-Project-Team-Extensify")
DATASET_ROOT  = PROJECT_ROOT / "dataset"
LEROBOT_DIR   = PROJECT_ROOT / "lerobot"
OUTPUT_DIR    = PROJECT_ROOT / "act_output"

# ── Dataset ───────────────────────────────────────────────────────────────────
# "local/" prefix tells lerobot-train to load from DATASET_ROOT on disk
# instead of downloading from HuggingFace Hub.
DATASET_REPO_ID = "local/xarm7_cylinder_task"
POLICY_REPO_ID = "local/act-trained-policy"

# ── Training hyperparameters ──────────────────────────────────────────────────
STEPS          = 100000
BATCH_SIZE     = 64      # reduce to 16 if OOM; try 64 if VRAM allows
LEARNING_RATE  = 1e-4
WARMUP_STEPS   = 1000     # LR warmup — ~2.5% of total steps is a safe default
USE_AMP        = True    # mixed precision: faster + less VRAM (recommended)
NUM_WORKERS    = 4       # dataloader workers; reduce if CPU is a bottleneck

# ── Checkpointing ─────────────────────────────────────────────────────────────
SAVE_FREQ      = 10000    # save a checkpoint every N steps

# ── WandB ─────────────────────────────────────────────────────────────────────
WANDB_ENABLE   = True
WANDB_PROJECT  = "act_xarm7_insert"

# ── Cache dirs to clear before each run ───────────────────────────────────────
# Removes stale HuggingFace/lerobot caches that can cause dataset loading
# errors between runs. Does NOT touch your dataset folder or weights.
CACHE_DIRS_TO_CLEAR = [
    Path.home() / ".cache" / "huggingface" / "datasets",
    Path.home() / ".cache" / "lerobot",
]

# ═══════════════════════════════════════════════════════════════════════════════


def clear_caches():
    """Remove stale cache directories before training."""
    print("\n── Clearing caches ──────────────────────────────────────")
    for cache_dir in CACHE_DIRS_TO_CLEAR:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"  Cleared: {cache_dir}")
        else:
            print(f"  Already empty: {cache_dir}")
    print()


def make_job_name() -> str:
    """
    Auto-generate a unique job name that encodes key hyperparameters.
    Example: act_xarm7_s20000_bs32_lr1e-4_2026-04-19_14-32
    This name is used as both the WandB run name and the output subdirectory,
    so every run is fully identifiable from its name alone.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    lr_str = f"{LEARNING_RATE:.0e}".replace("-0", "-")  # e.g. 1e-4 not 1e-04
    return f"act_xarm7_s{STEPS}_bs{BATCH_SIZE}_lr{lr_str}_{timestamp}"


def build_command(job_name: str, run_output_dir: Path) -> list[str]:
    """
    Build the lerobot-train command using the correct flags from the
    actual lerobot v0.3.3 CLI (verified from lerobot-train --help).

    Key flags:
      --policy.path        : HuggingFace repo ID OR local path to load weights from
      --dataset.repo_id    : "local/<n>" for local datasets (no Hub download)
      --dataset.root       : absolute path to the dataset folder on disk
      --policy.optimizer_lr            : learning rate
      --policy.scheduler_warmup_steps  : LR warmup steps
      --policy.use_amp     : mixed precision training
    """
    cmd = [
        "lerobot-train",

        # ── Policy / model ────────────────────────────────────────────────────
        # "--policy.type=smolvla",
        f"--policy.type=act",

        # ── Dataset ───────────────────────────────────────────────────────────
        f"--dataset.repo_id={DATASET_REPO_ID}",  # "local/..." = use local files
        f"--dataset.root={DATASET_ROOT}",         # absolute path to dataset/

        # ── Core training hyperparameters ─────────────────────────────────────
        f"--batch_size={BATCH_SIZE}",
        f"--steps={STEPS}",
        f"--num_workers={NUM_WORKERS}",
        f"--policy.optimizer_lr={LEARNING_RATE}",
        f"--policy.use_amp={'true' if USE_AMP else 'false'}",
        "--policy.device=cuda",

        # ── Output ────────────────────────────────────────────────────────────
        f"--output_dir={run_output_dir}",
        f"--job_name={job_name}",

        # ── Checkpointing ─────────────────────────────────────────────────────
        "--save_checkpoint=true",
        f"--save_freq={SAVE_FREQ}",

        # ── WandB ─────────────────────────────────────────────────────────────
        f"--wandb.enable={'true' if WANDB_ENABLE else 'false'}",
        f"--wandb.project={WANDB_PROJECT}",
        f"--policy.push_to_hub=false"
    ]
    return cmd


def print_config(job_name: str, run_output_dir: Path):
    """Print a human-readable summary of the run config before launching."""
    print("=" * 60)
    print("  SmolVLA Training Run")
    print("=" * 60)
    print(f"  Job name      : {job_name}")
    print(f"  Steps         : {STEPS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  Warmup steps  : {WARMUP_STEPS}")
    print(f"  AMP           : {USE_AMP}")
    print(f"  Num workers   : {NUM_WORKERS}")
    print(f"  Save every    : {SAVE_FREQ} steps")
    print(f"  Dataset       : {DATASET_ROOT}")
    print(f"  Output dir    : {run_output_dir}")
    print(f"  WandB project : {WANDB_PROJECT if WANDB_ENABLE else 'disabled'}")
    print("=" * 60)
    print()


def main():
    # ── Sanity checks ─────────────────────────────────────────────────────────
    for path, label in [
        (DATASET_ROOT, "dataset"),
        (LEROBOT_DIR,  "lerobot repo"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    for fname in ["info.json", "episodes.jsonl", "tasks.jsonl", "stats.json"]:
        fpath = DATASET_ROOT / "meta" / fname
        if not fpath.exists():
            print(f"ERROR: missing meta file: {fpath}")
            sys.exit(1)

    # ── Setup ─────────────────────────────────────────────────────────────────
    job_name       = make_job_name()
    run_output_dir = OUTPUT_DIR / job_name

    # Only create the parent smolvla_output/ directory here.
    # lerobot-train must create the run subdirectory itself —
    # if we pre-create it, lerobot-train will refuse to run.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_config(job_name, run_output_dir)
    clear_caches()

    # ── Build command and save it for reference ────────────────────────────────
    cmd = build_command(job_name, run_output_dir)
    log_file = OUTPUT_DIR / f"{job_name}.log"
    cmd_file = OUTPUT_DIR / f"{job_name}_command.txt"
    cmd_file.write_text(" \\\n  ".join(cmd) + "\n")

    print("── Launching training ───────────────────────────────────")
    print("  Command:\n  " + " \\\n    ".join(cmd))
    print(f"\n  Log file : {log_file}")
    print(f"  Cmd file : {cmd_file}")
    print("  (Ctrl+C to stop — last checkpoint is preserved)")
    print()

    # ── Run lerobot-train ─────────────────────────────────────────────────────
    # stdout and stderr are merged and streamed live to the terminal,
    # while simultaneously being written to the log file.
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(LEROBOT_DIR),
        )
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

    return_code = process.wait()
    if return_code == 0:
        print(f"\n  Training complete. Outputs at: {run_output_dir}")
    else:
        print(f"\n  Training failed (exit code {return_code}). Check: {log_file}")
        sys.exit(return_code)


if __name__ == "__main__":
    main()