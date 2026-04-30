"""
Check 4: Loads the SmolVLA checkpoint and verifies it can be instantiated.
No hardware or cameras needed.
"""

import os
os.environ["TMPDIR"] = "C:/Users/yashj/projects/IXN-Project-Team-Extensify/tmp"
os.environ["TEMP"] = os.environ["TMPDIR"]
os.environ["TMP"] = os.environ["TMPDIR"]

os.makedirs(os.environ["TMPDIR"], exist_ok=True)

import sys
from pathlib import Path
sys.path.insert(0, "/home/er/IXN-Project-Team-Extensify/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig

DATASET_REPO_ID  = "local/xarm7_pick_place"  # ← change this
DATASET_ROOT     = "/home/er/IXN-Project-Team-Extensify/dataset"  # ← change this
POLICY_PATH      = "/home/er/IXN-Project-Team-Extensify/smolvla-model/last/pretrained_model"

policy_path = Path(POLICY_PATH).expanduser()
assert policy_path.exists(), f"Policy path does not exist: {policy_path}"
assert (policy_path / "model.safetensors").exists(), "model.safetensors not found"
assert (policy_path / "config.json").exists(), "config.json not found"

print(f"✓ Policy path exists: {policy_path}")
print("✓ model.safetensors found")
print("✓ config.json found")

# Load dataset metadata (needed for stats normalisation)
meta = LeRobotDatasetMetadata(repo_id=DATASET_REPO_ID, root=DATASET_ROOT)

# Load the policy config from the checkpoint
policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
policy_cfg.pretrained_path = policy_path


print(f"\n=== Policy Config ===")
print(f"  type   : {type(policy_cfg).__name__}")
print(f"  device : {policy_cfg.device}")

# Load the policy weights
policy = make_policy(cfg=policy_cfg, ds_meta=meta)
print("make policy step done")
policy.eval()
print("eval step done")

print(f"\n✓ Policy loaded successfully")
print(f"✓ Policy type: {type(policy).__name__}")
print(f"✓ Device: {policy_cfg.device}")

# Quick sanity check — reset should not raise
policy.reset()
print("✓ policy.reset() succeeded")

print("\n✓ Check 4 PASSED")