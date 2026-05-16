import mujoco
import numpy as np

xml_path = '/extend_robotics_ws/mjcf/scene_gripper.xml'
model = mujoco.MjModel.from_xml_path(xml_path)

print(f"Model: {xml_path}")
print(f"Total geoms: {model.ngeom}")

print("\nGeom details (including body name):")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    body_id = model.geom_bodyid[i]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    contype = model.geom_contype[i]
    conaffinity = model.geom_conaffinity[i]
    group = model.geom_group[i]
    size = model.geom_size[i]
    print(f"  ID {i:2}: {str(name):20} | Body: {str(body_name):20} | Group: {group} | Mask: ({contype},{conaffinity}) | Size: {size}")

print("\nCollision pairs excluded:")
for i in range(model.nexclude):
    b1 = model.exclude_bodyid1[i]
    b2 = model.exclude_bodyid2[i]
    n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
    n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
    print(f"  Exclude: {n1} <-> {n2}")
