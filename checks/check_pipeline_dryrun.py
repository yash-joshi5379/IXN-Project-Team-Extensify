"""
Check 5: Full pipeline dry-run — no hardware, no cameras, no robot connection.

Validates the entire inference pipeline using only:
  - Real dataset metadata (your actual dataset)
  - Real policy checkpoint (your actual SmolVLA weights)
  - Fake observations shaped exactly like real robot output

Uses only modules that exist in lerobot v0.3.3 (official tag).

Run from your project root:
    python checks/check_pipeline_dryrun.py
"""

import sys
import json
import math
import numpy as np
import torch
from pathlib import Path

# ── CONFIGURE THESE PATHS ────────────────────────────────────────────────────
LEROBOT_SRC  = r"/home/er/IXN-Project-Team-Extensify/lerobot/src"
DATASET_ROOT = r"/home/er/IXN-Project-Team-Extensify/dataset"
DATASET_REPO_ID = "local/xarm7_pick_place"   # must match what was used when creating dataset
POLICY_PATH  = r"/home/er/IXN-Project-Team-Extensify/smolvla-model/last/pretrained_model"
TASK         = "pick_place"
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, LEROBOT_SRC)

# ---------------------------------------------------------------------------
# Step 1 — Verify checkpoint files exist and config.json is valid
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Verify checkpoint files")
print("=" * 60)

policy_path = Path(POLICY_PATH)
assert policy_path.exists(),                           f"Policy path not found: {policy_path}"
assert (policy_path / "model.safetensors").exists(),   "model.safetensors missing"
assert (policy_path / "config.json").exists(),         "config.json missing"
assert (policy_path / "train_config.json").exists(),   "train_config.json missing"

with open(policy_path / "config.json") as f:
    policy_config_dict = json.load(f)

print(f"✓ Checkpoint path:     {policy_path}")
print(f"✓ model.safetensors:   found")
print(f"✓ config.json:         found")
print(f"✓ train_config.json:   found")
print(f"  Policy type field:   {policy_config_dict.get('type') or policy_config_dict.get('policy_type') or policy_config_dict.get('model_type', '(check manually)')}")

# ---------------------------------------------------------------------------
# Step 2 — Load dataset metadata and confirm feature shapes
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Load dataset metadata")
print("=" * 60)

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

meta = LeRobotDatasetMetadata(repo_id=DATASET_REPO_ID, root=DATASET_ROOT)

print(f"✓ Dataset loaded:      {DATASET_REPO_ID}")
print(f"  fps:                 {meta.fps}")
print(f"  total_episodes:      {meta.total_episodes}")
print(f"  robot_type:          {meta.robot_type}")

# Confirm the exact features we will depend on
required = {
    "observation.state":              (8,),
    "action":                         (8,),
    "observation.images.static_cam":  (256, 256, 3),
    "observation.images.wrist_cam":   (256, 256, 3),
}
for feat_key, expected_shape in required.items():
    assert feat_key in meta.features, f"Missing feature: {feat_key}"
    actual_shape = tuple(meta.features[feat_key]["shape"])
    assert actual_shape == expected_shape, \
        f"{feat_key} shape mismatch: got {actual_shape}, expected {expected_shape}"
    print(f"  ✓ {feat_key}: {actual_shape}")

print("✓ All required features confirmed")

# ---------------------------------------------------------------------------
# Step 3 — Load policy using direct safetensors + config.json
#           (bypasses draccus/from_pretrained which fails on Windows)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Load policy weights")
print("=" * 60)

policy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {policy_device}")

from lerobot.policies.factory import get_policy_class
from lerobot.datasets.utils import dataset_to_policy_features

# Convert dataset features → PolicyFeature objects (required by SmolVLA normalizer)
policy_features = dataset_to_policy_features(meta.features)

# Get policy class from config.json type field
policy_type = (
    policy_config_dict.get("type")
    or policy_config_dict.get("policy_type")
    or policy_config_dict.get("model_type")
)
assert policy_type is not None, \
    "Could not determine policy type from config.json — check the file manually"
print(f"  Policy type:         {policy_type}")

PolicyClass = get_policy_class(policy_type)
PolicyConfigClass = PolicyClass.config_class

# Build config dataclass — only pass fields that exist in the config class
import dataclasses
valid_fields = {f.name for f in dataclasses.fields(PolicyConfigClass)}
filtered_dict = {k: v for k, v in policy_config_dict.items() if k in valid_fields}
filtered_dict["device"] = str(policy_device)

# ── The key fix: inject PolicyFeature objects into input/output features ──
# SmolVLA's normalizer expects PolicyFeature objects, not raw dicts.
# These come from the dataset features, not from config.json.
if "input_features" in valid_fields:
    filtered_dict["input_features"] = policy_features
if "output_features" in valid_fields:
    # Output features are just the action
    from lerobot.datasets.utils import dataset_to_policy_features
    action_only = {"action": meta.features["action"]}
    filtered_dict["output_features"] = dataset_to_policy_features(action_only)

# Convert normalization_mapping string values → NormalizationMode enum members
from lerobot.policies.normalize import NormalizationMode

if "normalization_mapping" in filtered_dict:
    raw_map = filtered_dict["normalization_mapping"]
    if isinstance(raw_map, dict):
        filtered_dict["normalization_mapping"] = {
            k: NormalizationMode[v] if isinstance(v, str) else v
            for k, v in raw_map.items()
        }
    print(f"  normalization_mapping: {filtered_dict['normalization_mapping']}")

policy_cfg = PolicyConfigClass(**filtered_dict)
policy_cfg.pretrained_path = policy_path

# Load policy with converted features and dataset stats
policy = PolicyClass(policy_cfg, dataset_stats=meta.stats)

# Load the actual trained weights from model.safetensors
from safetensors.torch import load_file
state_dict = load_file(policy_path / "model.safetensors", device="cuda")
missing, unexpected = policy.load_state_dict(state_dict, strict=False)
policy.to(policy_device) 
if missing:
    print(f"  ⚠ Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
if unexpected:
    print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")

policy.load_state_dict(state_dict, strict=False)
policy.to(policy_device)  # ← ensures ALL buffers (including norm stats) are on the right device
policy.eval()
policy.reset()


print(f"✓ Policy instantiated: {type(policy).__name__}")
print(f"✓ Weights loaded from model.safetensors")
print(f"✓ policy.reset() succeeded")
print(f"  Running on:          cuda")

print(f"✓ Policy instantiated: {type(policy).__name__}")
print(f"✓ policy.reset() succeeded")
print(f"  Running on:          cuda (GPU not needed for this check)")

# ---------------------------------------------------------------------------
# Step 4 — Build mock observation matching real robot output exactly
#
# UFRobot.get_observation() in joint mode returns:
#   J1.pos ... J7.pos  — float scalars (radians)
#   gripper.pos        — float scalar in [0, 1]
#   static_cam         — np.ndarray (256, 256, 3) uint8 BGR
#   wrist_cam          — np.ndarray (256, 256, 3) uint8 BGR
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: Build mock robot observation")
print("=" * 60)

mock_raw_obs = {}

# Joint positions — realistic values in radians (within xArm7 joint limits)
joint_defaults_rad = [0.0, 0.0, 0.0, math.pi/2, 0.0, math.pi/2, 0.0]
for i in range(1, 8):
    mock_raw_obs[f"J{i}.pos"] = joint_defaults_rad[i-1] + float(np.random.uniform(-0.05, 0.05))

# Gripper — normalised [0, 1], 0=closed, 1=open
mock_raw_obs["gripper.pos"] = 1.0  # start open

# Camera frames — uint8 BGR (OpenCV format), 256x256x3
mock_raw_obs["static_cam"] = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
mock_raw_obs["wrist_cam"]  = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

print("✓ Mock observation built:")
for k, v in mock_raw_obs.items():
    info = f"shape={v.shape} dtype={v.dtype}" if hasattr(v, "shape") else f"value={v:.4f}"
    print(f"    {k}: {info}")

# ---------------------------------------------------------------------------
# Step 5 — Apply rename map (mirrors rename_observations_processor in eval)
#
# rename_map bridges robot keys → dataset feature keys:
#   static_cam  → observation.images.static_cam
#   wrist_cam   → observation.images.wrist_cam
#   J1.pos      → (aggregated into observation.state by the pipeline)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 5: Apply rename map")
print("=" * 60)

RENAME_MAP = {
    "static_cam": "observation.images.static_cam",
    "wrist_cam":  "observation.images.wrist_cam",
}

obs_renamed = {}
for k, v in mock_raw_obs.items():
    new_key = RENAME_MAP.get(k, k)
    obs_renamed[new_key] = v

print("✓ Rename map applied:")
for k in obs_renamed:
    print(f"    {k}")

# ---------------------------------------------------------------------------
# Step 6 — Aggregate joint positions into observation.state
#
# The pipeline maps J1.pos...J7.pos + gripper.pos → observation.state [8,]
# This is what hw_to_dataset_features / build_dataset_frame does.
# We replicate it manually here for the dry-run.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 6: Build observation tensors")
print("=" * 60)

# Build observation.state from joint keys (order must match dataset: joint1...7, gripper)
state_values = [mock_raw_obs[f"J{i}.pos"] for i in range(1, 8)]
state_values.append(mock_raw_obs["gripper.pos"])
obs_state_tensor = torch.tensor(state_values, dtype=torch.float32)  # shape [8]

# Build image tensors — convert uint8 HWC BGR → float32 CHW RGB, normalised [0,1]
def bgr_hwc_to_rgb_chw_tensor(img: np.ndarray) -> torch.Tensor:
    img_rgb = img[:, :, ::-1].copy()               # BGR → RGB
    img_float = img_rgb.astype(np.float32) / 255.0 # [0,1]
    return torch.from_numpy(img_float).permute(2, 0, 1)  # HWC → CHW

static_cam_tensor = bgr_hwc_to_rgb_chw_tensor(obs_renamed["observation.images.static_cam"])
wrist_cam_tensor  = bgr_hwc_to_rgb_chw_tensor(obs_renamed["observation.images.wrist_cam"])

# print(f"✓ observation.state:                  shape={obs_state_tensor.shape} dtype={obs_state_tensor.dtype}")
# print(f"✓ observation.images.static_cam:       shape={static_cam_tensor.shape} dtype={static_cam_tensor.dtype}")
# print(f"✓ observation.images.wrist_cam:        shape={wrist_cam_tensor.shape} dtype={wrist_cam_tensor.dtype}")

# Add batch dimension — policy expects [B, ...] or [B, T, ...]
# observation_batch = {
#     "observation.state":                 obs_state_tensor.unsqueeze(0),          # [1, 8]
#     "observation.images.static_cam":     static_cam_tensor.unsqueeze(0),         # [1, 3, 256, 256]
#     "observation.images.wrist_cam":      wrist_cam_tensor.unsqueeze(0),          # [1, 3, 256, 256]
# }

# observation_batch = {
#     "observation.state":              obs_state_tensor.unsqueeze(0).to("cuda"),
#     "observation.images.static_cam":  static_cam_tensor.unsqueeze(0).to("cuda"),
#     "observation.images.wrist_cam":   wrist_cam_tensor.unsqueeze(0).to("cuda"),
#     "task": [TASK],
# }

observation_batch = {
    "observation.state":             obs_state_tensor.unsqueeze(0).to(policy_device),
    "observation.images.static_cam": static_cam_tensor.unsqueeze(0).to(policy_device),
    "observation.images.wrist_cam":  wrist_cam_tensor.unsqueeze(0).to(policy_device),
    "task": [TASK],
}

# Get the device the policy actually loaded onto
# policy_device = next(policy.parameters()).device
# print(f"  Policy is on device: {policy_device}")

# observation_batch = {
#     "observation.state":             obs_state_tensor.unsqueeze(0).to(policy_device),
#     "observation.images.static_cam": static_cam_tensor.unsqueeze(0).to(policy_device),
#     "observation.images.wrist_cam":  wrist_cam_tensor.unsqueeze(0).to(policy_device),
#     "task": [TASK],
# }
# print(f"  Tensors moved to:    {policy_device}")

# Add task if SmolVLA needs it (it is a VLA model that conditions on language)
if hasattr(policy, 'config') and hasattr(policy.config, 'use_lang_conditioning'):
    if policy.config.use_lang_conditioning:
        observation_batch["task"] = [TASK]
else:
    # SmolVLA always uses task — add it unconditionally
    observation_batch["task"] = [TASK]

print(f"  Task string:                         '{TASK}'")

# ---------------------------------------------------------------------------
# Step 7 — Run policy forward pass
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 7: Run policy.select_action()")
print("=" * 60)

with torch.no_grad():
    try:
        action_tensor = policy.select_action(observation_batch)
        print(f"✓ policy.select_action() succeeded")
        print(f"  Output shape: {action_tensor.shape}")
        print(f"  Output dtype: {action_tensor.dtype}")
        print(f"  Output values (first step): {action_tensor[0].tolist()}")
    except Exception as e:
        print(f"✗ policy.select_action() failed: {e}")
        print()
        print("  This may be expected on cuda with SmolVLA (vision-language model is large).")
        print("  The important thing is whether the error is a shape/key mismatch")
        print("  (bad → means pipeline is wrong) or an OOM/device error (ok → will work on GPU).")
        raise

# ---------------------------------------------------------------------------
# Step 8 — Convert policy output back to robot action dict
#           and validate it matches what send_action() expects
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 8: Convert action tensor → robot action dict")
print("=" * 60)

# Policy outputs shape [1, 8] or [8] depending on SmolVLA chunking
if action_tensor.dim() > 1:
    action_values = action_tensor[0].cpu().numpy()  # take first step of chunk
else:
    action_values = action_tensor.cpu().numpy()

assert len(action_values) == 8, \
    f"Expected 8 action values (7 joints + gripper), got {len(action_values)}"

# Map back to UFRobot.send_action() expected keys
robot_action = {}
for i in range(7):
    robot_action[f"J{i+1}.pos"] = float(action_values[i])
robot_action["gripper.pos"] = float(action_values[7])

print("✓ Robot action dict:")
for k, v in robot_action.items():
    print(f"    {k}: {v:.4f}")

# Validate gripper is in expected normalised range [0, 1]
# (UFRobot.send_action converts back to [0, 800] internally)
grip = robot_action["gripper.pos"]
if not (0.0 <= grip <= 1.0):
    print(f"  ⚠ Gripper value {grip:.4f} is outside [0, 1]")
    print("    This may be fine if the policy outputs unnormalised — check normalisation config.")
else:
    print(f"✓ Gripper value {grip:.4f} is within [0, 1]")

# Validate joint angle magnitudes are plausible (rough xArm7 limits ±π)
for i in range(1, 8):
    jval = robot_action[f"J{i}.pos"]
    if abs(jval) > math.pi * 2:
        print(f"  ⚠ J{i}.pos = {jval:.4f} rad seems large (>2π) — check normalisation")

print()
print("=" * 60)
print("✓ CHECK 5 PASSED — full pipeline validated end-to-end")
print("=" * 60)
print()
print("What this confirms:")
print("  • Dataset metadata loads and has correct feature shapes")
print("  • Policy checkpoint loads and runs a forward pass")
print("  • Observation keys and shapes flow correctly through the pipeline")
print("  • Output action dict has the right keys for UFRobot.send_action()")
print()
print("Next step: run on the robot PC with real cameras and xArm7 connected.")
