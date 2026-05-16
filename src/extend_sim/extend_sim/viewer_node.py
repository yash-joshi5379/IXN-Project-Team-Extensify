#!/usr/bin/env python3
"""
Launches an optimized MuJoCo passive viewer that mirrors the sim_node state.
Uses GLFW for GUI. High-frequency updates ensure smooth visualization.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import mujoco
import mujoco.viewer
import threading
import numpy as np
import os

# Force GLFW for the interactive viewer
os.environ['MUJOCO_GL'] = 'glfw'

SCENES = {
    'gripper': '/extend_robotics_ws/mjcf/scene_gripper.xml',
    'xhand':   '/extend_robotics_ws/mjcf/scene_xhand.xml',
}


class ViewerNode(Node):
    def __init__(self):
        super().__init__('extend_viewer_node')

        self.declare_parameter('scene', 'gripper')
        scene_name = self.get_parameter('scene').get_parameter_value().string_value

        xml = SCENES[scene_name]
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data  = mujoco.MjData(self.model)
        
        # Try to load scene_home keyframe
        kf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
        if kf_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.lock = threading.Lock()

        # subscribe to joint states (Optimized for 100Hz)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)

        # launch passive viewer
        self.viewer_thread = threading.Thread(target=self._run_viewer, daemon=True)
        self.viewer_thread.start()

        self.get_logger().info(f'Viewer ready — scene={scene_name} (GLFW)')

    def _joint_cb(self, msg: JointState):
        with self.lock:
            # Map names to qpos addresses
            i = 0
            while i < len(msg.name):
                name = msg.name[i]
                try:
                    if name.endswith('_px'):
                        # This is a freejoint
                        jbase = name[:-3]
                        jid = self.model.joint(jbase).id
                        qadr = self.model.jnt_qposadr[jid]
                        # Freejoints always have 7 components in qpos
                        for k in range(7):
                            self.data.qpos[qadr + k] = msg.position[i + k]
                        i += 7
                        continue
                    else:
                        # Standard 1-DOF joint
                        jid = self.model.joint(name).id
                        qadr = self.model.jnt_qposadr[jid]
                        self.data.qpos[qadr] = msg.position[i]
                except: pass
                i += 1
            mujoco.mj_forward(self.model, self.data)

    def _run_viewer(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as v:
            while rclpy.ok() and v.is_running():
                with self.lock:
                    v.sync()
                # Fast sync for smooth motion
                import time
                time.sleep(1.0/150.0)


def main(args=None):
    rclpy.init(args=args)
    node = ViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except: pass


if __name__ == '__main__':
    main()
