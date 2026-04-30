"""
Check 5: Full pipeline dry-run using a mock robot.
Simulates one inference step: fake obs → preprocessor → policy → postprocessor → action.
No hardware or cameras needed.
"""
import sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, "C:/Users/yashj/projects/IXN-Project-Team-Extensify/lerobot/src")

from lerobot.robots.ufactory_robot.config_uf_robot import UFRobotConfig
from lerobot.robots.ufactory_robot.uf_robot import UFRobot
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
# from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.utils import make_robot_action
from lerobot.policies.factory import make_policy, make_pre_post_processors
# from lerobot.utils.constants import OBS_STR
from lerobot.utils.utils import get_safe_torch_device
from lerobot.processor import make_default_processors
from lerobot.configs.policies import PreTrainedConfig

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_REPO_ID = "local/xarm7_pick_place"  # ← change this
DATASET_ROOT    = "C:/Users/yashj/projects/IXN-Project-Team-Extensify/dataset"  # ← change this
POLICY_PATH      = "C:/Users/yashj/projects/IXN-Project-Team-Extensify/smolvla_output/smolvla_xarm7_s20000_bs64_lr1e-4_2026-04-19_02-11/checkpoints/last/pretrained_model"
TASK            = "pick_place"
# ──────────────────────────────────────────────────────────────────────────────

print("--- Step 1: Build UFRobotConfig (no connection) ---")
config = UFRobotConfig(
    robot_ip="192.168.1.127",
    robot_dof=7,
    control_space="joint",
    gripper_control=True,
    gripper_type=2,
    observe_joint_vel=False,
    cameras={
        "static_cam": OpenCVCameraConfig(index_or_path=0, width=256, height=256, fps=30),
        "wrist_cam":  OpenCVCameraConfig(index_or_path=2, width=256, height=256, fps=30),
    },
)
robot = UFRobot(config)
print("✓ UFRobotConfig and UFRobot instantiated")

print("\n--- Step 2: Load dataset metadata ---")
meta = LeRobotDatasetMetadata(repo_id=DATASET_REPO_ID, root=DATASET_ROOT)
print(f"✓ Dataset metadata loaded (fps={meta.fps})")

print("\n--- Step 3: Build processors ---")
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

dataset_features = combine_feature_dicts(
    aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=robot.action_features),
        use_videos=True,
    ),
    aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=robot.observation_features),
        use_videos=True,
    ),
)
print("✓ Dataset features built from robot interface")

print("\n--- Step 4: Load policy ---")
policy_path = Path(POLICY_PATH).expanduser()
policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
policy_cfg.pretrained_path = policy_path
policy = make_policy(cfg=policy_cfg, ds_meta=meta)
policy.eval()
device = get_safe_torch_device(policy_cfg.device, log=True)
print(f"✓ Policy loaded on device: {device}")

print("\n--- Step 5: Build pre/post processors ---")
preprocessor_overrides = {
    "device_processor": {"device": str(device)},
    "rename_observations_processor": {
        "rename_map": {
            "static_cam": "observation.images.static_cam",
            "wrist_cam":  "observation.images.wrist_cam",
        }
    },
}
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy_cfg,
    pretrained_path=policy_path,
    preprocessor_overrides=preprocessor_overrides,
    dataset_stats=meta.stats,
)
preprocessor.reset()
postprocessor.reset()
print("✓ Pre/post processors built")

print("\n--- Step 6: Build mock observation (fake data, real shapes) ---")
# Mimic exactly what UFRobot.get_observation() returns in joint mode
mock_obs = {}
for i in range(1, 8):
    mock_obs[f"J{i}.pos"] = float(np.random.uniform(-0.5, 0.5))
mock_obs["gripper.pos"] = float(np.random.uniform(0.0, 1.0))
# Cameras: uint8 BGR images from OpenCV (256x256x3)
mock_obs["static_cam"] = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
mock_obs["wrist_cam"]  = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

print("✓ Mock observation built:")
for k, v in mock_obs.items():
    shape = v.shape if hasattr(v, "shape") else "scalar"
    print(f"    {k}: {shape}")

print("\n--- Step 7: Run robot_observation_processor ---")
obs_processed = robot_observation_processor(mock_obs)
print("✓ Observation processed")

print("\n--- Step 8: Build dataset frame ---")
observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
print("✓ Dataset frame built")
print("  Keys in frame:", list(observation_frame.keys())[:5], "...")

print("\n--- Step 9: Run policy predict_action ---")
from lerobot.utils.control_utils import predict_action
action_values = predict_action(
    observation=observation_frame,
    policy=policy,
    device=device,
    preprocessor=preprocessor,
    postprocessor=postprocessor,
    use_amp=policy_cfg.use_amp,
    task=TASK,
    robot_type=robot.robot_type if hasattr(robot, "robot_type") else "xarm7",
)
print(f"✓ predict_action returned: type={type(action_values)}, shape={getattr(action_values, 'shape', 'N/A')}")

print("\n--- Step 10: Convert to robot action dict ---")
act_processed = make_robot_action(action_values, dataset_features)
robot_action  = robot_action_processor((act_processed, mock_obs))
print("✓ Robot action dict produced:")
for k, v in robot_action.items():
    print(f"    {k}: {v:.4f}")

# Validate output keys match what send_action() expects
expected_action_keys = {f"J{i}.pos" for i in range(1, 8)} | {"gripper.pos"}
actual_action_keys   = set(robot_action.keys())
assert actual_action_keys == expected_action_keys, \
    f"Action key mismatch!\n  Got:      {actual_action_keys}\n  Expected: {expected_action_keys}"

# Validate gripper is normalised [0, 1]
grip = robot_action["gripper.pos"]
assert 0.0 <= grip <= 1.0, \
    f"Gripper output {grip:.4f} is outside [0, 1] — check normalisation"

print("\n✓ Output action keys match expected (J1.pos...J7.pos + gripper.pos)")
print(f"✓ Gripper value {grip:.4f} is within [0, 1]")
print("\n✓ Check 5 PASSED — full pipeline works end-to-end with mock data")
print("  You are ready to proceed to hardware testing.")