import mujoco
import numpy as np

xml_path = '/extend_robotics_ws/mjcf/scene_gripper.xml'
model = mujoco.MjModel.from_xml_path(xml_path)

print(f"Number of actuators (na): {model.na}")
print(f"Number of joints (njnt): {model.njnt}")

for i in range(model.na):
    print(f"Actuator {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")

mujoco.mj_saveLastXML('/tmp/last.xml', model)
with open('/tmp/last.xml', 'r') as f:
    xml_content = f.read()
    print("\nSaved XML length:", len(xml_content))
    if '<actuator>' in xml_content:
        print("Found <actuator> in saved XML")
        # Print a bit of it
        start = xml_content.find('<actuator>')
        print(xml_content[start:start+500])
    else:
        print("MISSING <actuator> in saved XML")
