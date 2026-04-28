"""
SmolVLA Robot Client — xArm7
==============================
Runs on the machine physically connected to the xArm7 robot.
Connects to the inference server, streams observations, and executes actions.

IMPORTANT — READ BEFORE RUNNING:
  The robot client uses lerobot's built-in robot abstraction layer.
  xArm7 is NOT in SUPPORTED_ROBOTS in constants.py (only so100/so101 are).
  This means the line `if cfg.robot.type not in SUPPORTED_ROBOTS: raise ValueError`
  in robot_client.py WILL block you.

  You have two options:
    Option A (recommended): Add "xarm7" to SUPPORTED_ROBOTS in
             lerobot/src/lerobot/scripts/server/constants.py
             AND implement an xArm7 robot class in lerobot's robot registry.
    Option B: Ask your supervisor — they likely have an existing xArm7
             robot class from the data collection setup, since VR teleoperation
             was already working with this robot.

  This script is ready to run once the xArm7 robot class is registered.

USAGE (on the robot computer, once xArm7 is supported):
  cd /path/to/IXN-Project-Team-Extensify
  python src/run_robot_client.py

  Or to run a specific number of episodes:
  python src/run_robot_client.py --episodes 5
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these to match your setup
# ─────────────────────────────────────────────────────────────────────────────

# Path to the lerobot repo (adjust if running on a different machine)
LEROBOT_DIR = Path(__file__).resolve().parent.parent / "lerobot"
CLIENT_SCRIPT = LEROBOT_DIR / "src" / "lerobot" / "scripts" / "server" / "robot_client.py"

# ── Inference server ──────────────────────────────────────────────────────────
# IP address of the GPU machine running run_inference_server.py
# Replace with the IP printed by the server when it starts
SERVER_IP   = "REPLACE_WITH_SERVER_IP"   # e.g. "192.168.1.42"
SERVER_PORT = 8080
SERVER_ADDRESS = f"{SERVER_IP}:{SERVER_PORT}"

# ── Model ─────────────────────────────────────────────────────────────────────
# Path to the trained model checkpoint.
# If running on a different machine, copy checkpoints/last/pretrained_model/ there first.
PRETRAINED_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent
    / "smolvla_output"
    / "smolvla_xarm7_s20000_bs64_lr1e-4_2026-04-19_02-11"
    / "checkpoints"
    / "last"
    / "pretrained_model"
)



# ── Task instruction ──────────────────────────────────────────────────────────
# Must exactly match the task string used during training
TASK = "Grasp the red cylinder and place it vertically into the circular slot within the white box."

# ── Robot hardware ────────────────────────────────────────────────────────────
# Robot type — must be registered in lerobot's robot factory
# NOTE: "xarm7" needs to be added to SUPPORTED_ROBOTS and the robot registry
ROBOT_TYPE = "xarm7"

# Camera configuration — must match the camera keys used during training:
#   observation.images.static_cam  →  static_cam
#   observation.images.wrist_cam   →  wrist_cam
# Adjust index_or_path, width, height to match your actual camera setup
CAMERAS = (
    "{"
    "static_cam: {type: opencv, index_or_path: 0, width: 256, height: 256, fps: 30}, "
    "wrist_cam:  {type: opencv, index_or_path: 1, width: 256, height: 256, fps: 30}"
    "}"
)

# ── Policy settings ───────────────────────────────────────────────────────────
POLICY_TYPE      = "smolvla"
POLICY_DEVICE    = "cuda"    # "cuda" if running on GPU machine, "cpu" for laptop-only

# Number of actions from each predicted chunk to actually execute before
# re-querying the server. SmolVLA predicts 50 actions at once; 25 means
# the robot asks for new predictions halfway through each chunk.
ACTIONS_PER_CHUNK = 25

# When the action queue drops below this fraction of ACTIONS_PER_CHUNK,
# send a new observation to the server to get the next action chunk.
# 0.5 = request new actions when queue is half empty.
CHUNK_SIZE_THRESHOLD = 0.5

# How action chunks are blended together when a new chunk arrives
# while old actions are still queued. Options:
#   "weighted_average" : 30% old + 70% new  (smooth, recommended)
#   "latest_only"      : discard old, use new immediately  (more reactive)
#   "average"          : 50/50 blend
#   "conservative"     : 70% old + 30% new  (very smooth, slower to react)
AGGREGATE_FN = "weighted_average"

# FPS — must match your training FPS
FPS = 30

# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Run SmolVLA robot client for xArm7")
    parser.add_argument("--server_ip", default=SERVER_IP,
                        help="IP address of the inference server")
    args = parser.parse_args()

    if not CLIENT_SCRIPT.exists():
        print(f"ERROR: robot_client.py not found at {CLIENT_SCRIPT}")
        sys.exit(1)

    if args.server_ip == "REPLACE_WITH_SERVER_IP":
        print("ERROR: You must set SERVER_IP to the IP address of your inference server.")
        print("       Edit the CONFIG block in this script, or pass --server_ip <ip>")
        sys.exit(1)

    server_address = f"{args.server_ip}:{SERVER_PORT}"

    print("=" * 60)
    print("  SmolVLA Robot Client — xArm7")
    print("=" * 60)
    print(f"  Server            : {server_address}")
    print(f"  Model             : {PRETRAINED_MODEL_PATH}")
    print(f"  Task              : {TASK}")
    print(f"  Robot type        : {ROBOT_TYPE}")
    print(f"  Policy device     : {POLICY_DEVICE}")
    print(f"  Actions per chunk : {ACTIONS_PER_CHUNK}")
    print(f"  Chunk threshold   : {CHUNK_SIZE_THRESHOLD}")
    print(f"  FPS               : {FPS}")
    print("=" * 60)
    print()

    # Build the robot_client.py command using draccus-style CLI args
    # (robot_client.py uses @draccus.wrap() so args are passed this way)
    cmd = [
        sys.executable,
        str(CLIENT_SCRIPT),

        # Robot hardware
        f"--robot.type={ROBOT_TYPE}",
        f"--robot.cameras={CAMERAS}",

        # Policy
        f"--policy_type={POLICY_TYPE}",
        f"--pretrained_name_or_path={PRETRAINED_MODEL_PATH}",
        f"--policy_device={POLICY_DEVICE}",

        # Server
        f"--server_address={server_address}",

        # Task
        f"--task={TASK}",

        # Control
        f"--actions_per_chunk={ACTIONS_PER_CHUNK}",
        f"--chunk_size_threshold={CHUNK_SIZE_THRESHOLD}",
        f"--aggregate_fn_name={AGGREGATE_FN}",
        f"--fps={FPS}",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(LEROBOT_DIR))

    if result.returncode != 0:
        print(f"\nClient exited with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
