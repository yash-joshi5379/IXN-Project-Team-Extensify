"""
Check 2: Validates UFRobotConfig instantiation and feature key/shape alignment
with info.json. No hardware needed.
"""
import sys
import numpy as np

# --- Adjust this path to your lerobot src ---
sys.path.insert(0, "/home/er/IXN-Project-Team-Extensify/lerobot/src")

from lerobot.robots.ufactory_robot.config_uf_robot import UFRobotConfig
from lerobot.robots.ufactory_robot.uf_robot import UFRobot
from lerobot.cameras.opencv import OpenCVCameraConfig

# Build the same config as your YAML — without connecting
config = UFRobotConfig(
    robot_ip="192.168.1.127",   # value doesn't matter for offline check
    robot_dof=7,
    control_space="joint",
    gripper_control=True,
    gripper_type=2,
    observe_joint_vel=False,
    cameras={
        "static_cam": OpenCVCameraConfig(index_or_path=0, width=256, height=256, fps=30),
        "wrist_cam":  OpenCVCameraConfig(index_or_path=1, width=256, height=256, fps=30),
    },
)

# Instantiate robot (does NOT connect — connect() is called separately)
robot = UFRobot(config)

print("=== Action Features ===")
for k, v in robot.action_features.items():
    print(f"  {k}: {v}")

print("\n=== Observation Features ===")
for k, v in robot.observation_features.items():
    print(f"  {k}: {v}")

# --- Validate against your info.json ---
# Expected action keys for 7-DOF joint space + gripper:
expected_action_keys = {f"J{i}.pos" for i in range(1, 8)} | {"gripper.pos"}
# Expected state observation keys (same as action keys in joint mode):
expected_state_keys = {f"J{i}.pos" for i in range(1, 8)} | {"gripper.pos"}
# Expected camera keys:
expected_cam_keys = {"static_cam", "wrist_cam"}

actual_action_keys = set(robot.action_features.keys())
actual_obs_keys    = set(robot.observation_features.keys())
actual_cam_keys    = {k for k in actual_obs_keys if k in expected_cam_keys}
actual_state_keys  = actual_obs_keys - actual_cam_keys

assert actual_action_keys == expected_action_keys, \
    f"Action key mismatch!\n  Got:      {actual_action_keys}\n  Expected: {expected_action_keys}"
assert actual_state_keys == expected_state_keys, \
    f"State observation key mismatch!\n  Got:      {actual_state_keys}\n  Expected: {expected_state_keys}"
assert actual_cam_keys == expected_cam_keys, \
    f"Camera key mismatch!\n  Got:      {actual_cam_keys}\n  Expected: {expected_cam_keys}"

# Validate camera shapes — must be (256, 256, 3) to match info.json
for cam_key in expected_cam_keys:
    shape = robot.observation_features[cam_key]
    assert shape == (256, 256, 3), \
        f"Camera {cam_key} shape is {shape}, expected (256, 256, 3)"

print("\n✓ Action features match expected keys (J1.pos ... J7.pos + gripper.pos)")
print("✓ State observation features match expected keys")
print("✓ Camera keys match (static_cam, wrist_cam)")
print("✓ Camera shapes are (256, 256, 3)")
print("\n✓ Check 2 PASSED")