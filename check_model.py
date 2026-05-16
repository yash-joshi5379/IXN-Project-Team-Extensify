import mujoco
import numpy as np

xml_path = '/extend_robotics_ws/mjcf/scene_gripper.xml'
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("Actuators:")
for i in range(model.na):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  ID {i}: {name}")

print("\nSites:")
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"  ID {i}: {name}")

print("\nJoints:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"  ID {i}: {name} (qposadr: {model.jnt_qposadr[i]}, range: {model.jnt_range[i]})")

# Check keyframe
kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
print(f"\nKeyframe 'scene_home' ID: {kf_id}")
if kf_id != -1:
    print(f"  qpos length: {len(model.key_qpos[kf_id])}")
    print(f"  qpos: {model.key_qpos[kf_id]}")
