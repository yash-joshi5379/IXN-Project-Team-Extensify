#!/usr/bin/env python3
"""
extend_sim - Headless MuJoCo simulation node for Extend Robotics VLA project
Optimized for 100Hz joint state publication and high-responsiveness IK.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Empty

import mujoco
import numpy as np
import threading
import time
import os

# Force EGL for headless offscreen rendering to avoid GL conflicts with viewer process
os.environ['MUJOCO_GL'] = 'egl'

SCENES = {
    'gripper': {
        'xml':         '/extend_robotics_ws/mjcf/scene_gripper.xml',
        'arm_joints':  [f'joint{i}' for i in range(1, 8)],
        'hand_joints': [
            'left_driver_joint', 'right_driver_joint',
            'left_finger_joint', 'right_finger_joint',
            'left_inner_knuckle_joint', 'right_inner_knuckle_joint'
        ],
        'ee_site':     'link_tcp',
        'hand_type':   'gripper',
        'camera':      'wrist_cam',
    },
    'xhand': {
        'xml':         '/extend_robotics_ws/mjcf/scene_xhand.xml',
        'arm_joints':  [f'joint{i}' for i in range(1, 8)],
        'hand_joints': [
            'right_hand_thumb_bend_joint',
            'right_hand_thumb_rota_joint1',
            'right_hand_thumb_rota_joint2',
            'right_hand_index_bend_joint',
            'right_hand_index_joint1',
            'right_hand_index_joint2',
            'right_hand_mid_joint1',
            'right_hand_mid_joint2',
            'right_hand_ring_joint1',
            'right_hand_ring_joint2',
            'right_hand_pinky_joint1',
            'right_hand_pinky_joint2',
        ],
        'ee_site':   'right_hand_tcp',
        'hand_type': 'xhand',
        'camera':    'wrist_cam',
    },
}

XHAND_OPEN  = np.array([0.0, -0.3, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0])
XHAND_CLOSE = np.array([0.8,  0.5, 1.0,  0.1, 1.5, 1.5,  1.5, 1.5,  1.5, 1.5,  1.5, 1.5])

IMG_W = 224
IMG_H = 224

# Speed limits matching the real xArm7
MAX_EE_SPEED    = 0.5   # m/s  — max Cartesian EE speed
MAX_JOINT_SPEED = np.pi # rad/s — xArm7 rated max per joint (180 °/s)


class SimNode(Node):
    def __init__(self):
        super().__init__('extend_sim_node')

        self.declare_parameter('scene', 'gripper')
        scene_name = self.get_parameter('scene').get_parameter_value().string_value
        if scene_name not in SCENES:
            raise ValueError(f'Unknown scene: {scene_name}')

        self.cfg = SCENES[scene_name]
        self.get_logger().info(f'Loading scene: {scene_name} (Headless)')

        self.model = mujoco.MjModel.from_xml_path(self.cfg['xml'])
        self.data  = mujoco.MjData(self.model)
        
        # Try to load scene_home keyframe
        kf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
        if kf_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Offscreen renderer shared by both cameras (Always uses EGL)
        try:
            self.renderer = mujoco.Renderer(self.model, IMG_H, IMG_W)
            self.cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cfg['camera']
            )
            self.overhead_cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'overhead_cam'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to initialize offscreen renderer: {e}')
            self.renderer = None
            self.overhead_cam_id = -1

        self.arm_joint_ids = [
            self.model.joint(name).id for name in self.cfg['arm_joints']
        ]
        self.hand_joint_ids = [
            self.model.joint(name).id for name in self.cfg['hand_joints']
        ]
        self.arm_actuator_ids = []
        for i in range(7):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'act{i+1}')
            self.arm_actuator_ids.append(aid)

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.cfg['ee_site']
        )

        # EE movement buffer
        self.hand_cmd = 0.0
        self.ee_delta_buf = np.zeros(3)
        self.move_queue   = np.zeros(3)
        self.lock         = threading.Lock()

        # Randomize cylinder before the physics thread starts
        self._randomize_cylinder()

        # physics thread
        self.running = True
        self.physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.physics_thread.start()

        # publishers
        self.joint_pub    = self.create_publisher(JointState, '/joint_states',   10)
        self.image_pub    = self.create_publisher(Image,      '/rgb_image',      10)
        self.overhead_pub = self.create_publisher(Image,      '/overhead_image', 10)

        # subscribers
        self.create_subscription(Twist,   '/ee_pose_cmd', self._ee_cmd_cb,   10)
        self.create_subscription(Float32, '/hand_cmd',    self._hand_cmd_cb, 10)
        self.create_subscription(Empty,   '/reset_sim',   self._reset_cb,    10)

        # timers
        self.create_timer(1.0/150.0, self._publish_states)   # 150Hz publication (matches real robot)
        self.create_timer(1.0/10.0,  self._publish_image)
        self.create_timer(1.0/10.0,  self._publish_overhead)

        self.get_logger().info(
            f'SimNode ready — scene={scene_name}, hand={self.cfg["hand_type"]}'
        )

    def _ee_cmd_cb(self, msg: Twist):
        with self.lock:
            self.ee_delta_buf += np.array([msg.linear.x, msg.linear.y, msg.linear.z])

    def _hand_cmd_cb(self, msg: Float32):
        with self.lock:
            self.hand_cmd = float(np.clip(msg.data, 0.0, 1.0))

    def _reset_cb(self, msg: Empty):
        with self.lock:
            self.get_logger().info('Resetting simulation...')
            kf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
            if kf_id != -1:
                mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)
            else:
                mujoco.mj_resetData(self.model, self.data)
            self.ee_delta_buf.fill(0)
            self.move_queue.fill(0)
            self.hand_cmd = 0.0
            self._randomize_cylinder()
            mujoco.mj_forward(self.model, self.data)

    def _randomize_cylinder(self):
        """Place the cylinder at a random position in the 30×30 cm zone centred at [0.35, 0.35].

        Samples are rejected if outside the arm's reliable reach (0.28 m – 0.62 m from base).
        Caller must hold self.lock (or call before the physics thread starts).
        """
        cyl_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'cylinder_joint')
        if cyl_jid < 0:
            return  # scene has no cylinder

        # 50×50 cm square centred at [0.35, 0.35]: x/y ∈ [0.10, 0.60]
        X_RANGE = (0.10, 0.60)
        Y_RANGE = (0.10, 0.60)
        MIN_REACH = 0.28  # m — closer and the gripper can't orient to grasp
        MAX_REACH = 0.62  # m — reliable workspace boundary

        rng = np.random.default_rng()
        for _ in range(200):
            x = rng.uniform(*X_RANGE)
            y = rng.uniform(*Y_RANGE)
            if MIN_REACH <= np.hypot(x, y) <= MAX_REACH:
                break
        else:
            x, y = 0.35, 0.35  # fallback to zone centre

        qadr = self.model.jnt_qposadr[cyl_jid]
        self.data.qpos[qadr:qadr+3] = [x, y, 0.03]   # position
        self.data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]  # identity quaternion
        dadr = self.model.jnt_dofadr[cyl_jid]
        self.data.qvel[dadr:dadr+6] = 0.0              # zero velocity

        self.get_logger().info(f'Cylinder spawned at ({x:.3f}, {y:.3f})')

    def _physics_loop(self):
        while self.running:
            start_t = time.perf_counter()
            target_dt = self.model.opt.timestep
            
            with self.lock:
                self.move_queue += self.ee_delta_buf
                self.ee_delta_buf.fill(0)

                # Cap queue norm to bound peak EE speed and prevent motion backlog.
                # Derivation: peak_speed ≈ ||move_queue|| * k, so max_queue = MAX_EE_SPEED / k.
                max_queue = MAX_EE_SPEED / 10.0
                queue_norm = np.linalg.norm(self.move_queue)
                if queue_norm > max_queue:
                    self.move_queue *= max_queue / queue_norm

                hand = self.hand_cmd

                # Motion smoothing logic (1st order filter)
                # k=10.0 gives a ~0.1s time constant, making movements smooth and stable.
                k = 10.0
                fraction = 1.0 - np.exp(-k * target_dt)
                delta_step = self.move_queue * fraction
                self.move_queue -= delta_step

            if np.any(np.abs(delta_step) > 1e-7):
                self._apply_ik(delta_step)

            self._apply_hand(hand)
            mujoco.mj_step(self.model, self.data)
            
            while (time.perf_counter() - start_t) < target_dt:
                pass

    def _apply_ik(self, delta: np.ndarray):
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site_id)
        arm_cols = [self.model.joint(jid).dofadr[0] for jid in self.arm_joint_ids]
        J    = jacp[:, arm_cols]
        # Regularization lam=0.05 for rock-solid stability near singularities
        lam  = 0.05
        Jinv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(3))
        dq   = Jinv @ delta

        # Clamp joint speeds to xArm7 rated max (scale whole vector to preserve direction)
        max_dq = MAX_JOINT_SPEED * self.model.opt.timestep
        scale = max_dq / (np.max(np.abs(dq)) + 1e-9)
        if scale < 1.0:
            dq *= scale

        for i, jid in enumerate(self.arm_joint_ids):
            act_id = self.arm_actuator_ids[i]
            if act_id != -1:
                lim = self.model.jnt_range[jid]
                current_target = self.data.ctrl[act_id]
                self.data.ctrl[act_id] = float(np.clip(current_target + dq[i], lim[0], lim[1]))

    def _apply_hand(self, t: float):
        if self.cfg['hand_type'] == 'gripper':
            try:
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'gripper')
                if act_id >= 0:
                    # Position-controlled via split tendon.
                    # 0.0 = open (tendon pos 0), 1.0 = closed (tendon pos 0.85).
                    self.data.ctrl[act_id] = t * 0.85
            except: pass
        elif self.cfg['hand_type'] == 'xhand':
            # Multi-fingered hand actuators
            hand_actuators = [
                'thumb_bend', 'thumb_rota1', 'thumb_rota2',
                'index_bend', 'index_rota1', 'index_rota2',
                'mid_rota1', 'mid_rota2',
                'ring_rota1', 'ring_rota2',
                'pinky_rota1', 'pinky_rota2'
            ]
            targets = XHAND_OPEN + t * (XHAND_CLOSE - XHAND_OPEN)
            for i, aname in enumerate(hand_actuators):
                try:
                    aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
                    if aid >= 0:
                        self.data.ctrl[aid] = float(targets[i])
                except: pass

    def _publish_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        for jid in range(self.model.njnt):
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            jtype = self.model.jnt_type[jid]
            qadr  = self.model.jnt_qposadr[jid]
            
            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                # Freejoints have 7 components (3 pos, 4 quat)
                suffixes = ['_px', '_py', '_pz', '_qx', '_qy', '_qz', '_qw']
                for i, s in enumerate(suffixes):
                    msg.name.append(jname + s)
                    msg.position.append(float(self.data.qpos[qadr + i]))
            else:
                # Standard 1-DOF joints
                msg.name.append(jname)
                msg.position.append(float(self.data.qpos[qadr]))
                
        self.joint_pub.publish(msg)

    def _publish_image(self):
        if self.renderer is None: return
        try:
            self.renderer.update_scene(self.data, camera=self.cam_id)
            img = self.renderer.render().copy()
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.height, msg.width = IMG_H, IMG_W
            msg.encoding = 'rgb8'
            msg.step = IMG_W * 3
            msg.data = img.tobytes()
            self.image_pub.publish(msg)
        except: pass

    def _publish_overhead(self):
        if self.renderer is None or self.overhead_cam_id < 0:
            return
        try:
            self.renderer.update_scene(self.data, camera=self.overhead_cam_id)
            img = self.renderer.render().copy()
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.height, msg.width = IMG_H, IMG_W
            msg.encoding = 'rgb8'
            msg.step = IMG_W * 3
            msg.data = img.tobytes()
            self.overhead_pub.publish(msg)
        except: pass

    def destroy_node(self):
        self.running = False
        if self.renderer: self.renderer.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SimNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        try: rclpy.shutdown()
        except: pass

if __name__ == '__main__':
    main()
