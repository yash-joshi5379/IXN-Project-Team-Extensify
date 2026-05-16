import mujoco
import numpy as np

xml_path = '/extend_robotics_ws/mjcf/scene_gripper.xml'
model = mujoco.MjModel.from_xml_path(xml_path)

print("Geoms collision info:")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    name = name if name else f"id_{i}"
    contype = model.geom_contype[i]
    conaffinity = model.geom_conaffinity[i]
    group = model.geom_group[i]
    print(f"  Geom {i}: {name:20} | group: {group} | contype: {contype:4} | conaffinity: {conaffinity:4}")
