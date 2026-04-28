"""
SmolVLA Inference Server Launcher
===================================
Starts the gRPC policy server that listens for connections from the robot client.

USAGE (on the GPU machine):
  cd /root/IXN-Project-Team-Extensify
  conda activate smolvla
  python src/run_inference_server.py

The server will print its IP address when it starts. Give that IP to whoever
is running the robot client so they can connect to it.
"""

import socket
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these if needed
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/root/IXN-Project-Team-Extensify")
LEROBOT_DIR  = PROJECT_ROOT / "lerobot"
SERVER_SCRIPT = LEROBOT_DIR / "src" / "lerobot" / "scripts" / "server" / "policy_server.py"

# Network — "0.0.0.0" means accept connections from any machine on the network
HOST = "0.0.0.0"
PORT = 8080

# Timing — match your dataset recording FPS
FPS = 30
INFERENCE_LATENCY = 0.033   # seconds (~1 frame at 30fps)
OBS_QUEUE_TIMEOUT = 2       # seconds to wait for an observation before timing out

# ─────────────────────────────────────────────────────────────────────────────


def get_local_ip() -> str:
    """Get the machine's local network IP address."""
    try:
        # Connect to an external address to determine which local interface is used
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def main():
    if not SERVER_SCRIPT.exists():
        print(f"ERROR: policy_server.py not found at {SERVER_SCRIPT}")
        sys.exit(1)

    local_ip = get_local_ip()

    print("=" * 60)
    print("  SmolVLA gRPC Inference Server")
    print("=" * 60)
    print(f"  Host              : {HOST} (all interfaces)")
    print(f"  Port              : {PORT}")
    print(f"  FPS               : {FPS}")
    print(f"  Inference latency : {INFERENCE_LATENCY}s")
    print(f"  Obs queue timeout : {OBS_QUEUE_TIMEOUT}s")
    print()
    print(f"  Local IP address  : {local_ip}")
    print(f"  Robot client should connect to: {local_ip}:{PORT}")
    print()
    print("  NOTE: The server does NOT load the model at startup.")
    print("  The model is loaded when the robot client connects and")
    print("  sends the model path via SendPolicyInstructions().")
    print()
    print("  Waiting for robot client to connect ...")
    print("=" * 60)
    print()

    cmd = [
        sys.executable,
        str(SERVER_SCRIPT),
        f"--host={HOST}",
        f"--port={PORT}",
        f"--fps={FPS}",
        f"--inference_latency={INFERENCE_LATENCY}",
        f"--obs_queue_timeout={OBS_QUEUE_TIMEOUT}",
    ]

    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(LEROBOT_DIR))
    if result.returncode != 0:
        print(f"\nServer exited with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
