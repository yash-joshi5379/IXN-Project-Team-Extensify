"""
Check 3: Loads dataset metadata and validates feature alignment.
No hardware needed. Requires your local dataset path.
"""
import sys
sys.path.insert(0, "/home/er/IXN-Project-Team-Extensify/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

DATASET_REPO_ID = "local/xarm7_pick_place"  # ← change this
DATASET_ROOT    = "/home/er/IXN-Project-Team-Extensify/dataset"  # ← change this

meta = LeRobotDatasetMetadata(repo_id=DATASET_REPO_ID, root=DATASET_ROOT)

print("=== Dataset Info ===")
print(f"  codebase_version : {meta._version}")
print(f"  fps              : {meta.fps}")
print(f"  total_episodes   : {meta.total_episodes}")
print(f"  robot_type       : {meta.robot_type}")

print("\n=== Features ===")
for k, v in meta.features.items():
    print(f"  {k}: shape={v.get('shape', '?')} dtype={v.get('dtype', '?')}")

# Validate key fields
required_features = [
    "observation.state",
    "observation.images.static_cam",
    "observation.images.wrist_cam",
    "action",
]
for feat in required_features:
    assert feat in meta.features, f"Missing required feature: {feat}"
    print(f"  ✓ Found: {feat}")

# Validate observation.state shape = [8] (7 joints + gripper)
obs_state_shape = meta.features["observation.state"]["shape"]
assert tuple(obs_state_shape) == (8,), \
    f"observation.state shape is {obs_state_shape}, expected (8,)"

# Validate action shape = [8]
action_shape = meta.features["action"]["shape"]
assert tuple(action_shape) == (8,), \
    f"action shape is {action_shape}, expected (8,)"

# Validate image shapes = [256, 256, 3]
for img_key in ["observation.images.static_cam", "observation.images.wrist_cam"]:
    shape = meta.features[img_key]["shape"]
    assert tuple(shape) == (256, 256, 3), \
        f"{img_key} shape is {shape}, expected (256, 256, 3)"

assert meta.fps == 30, f"Dataset fps is {meta.fps}, expected 30"

print("\n✓ All required features present")
print("✓ observation.state shape is (8,) (joints 1-7 + gripper)")
print("✓ action shape is (8,)")
print("✓ Both image features are (256, 256, 3)")
print("✓ fps is 30")
print("\n✓ Check 3 PASSED")