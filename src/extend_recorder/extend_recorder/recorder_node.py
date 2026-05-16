#!/usr/bin/env python3
"""
extend_recorder - Data collection node for Extend Robotics VLA project

Records episodes of:
  - JPEG frames from /rgb_image      (wrist camera)
  - JPEG frames from /overhead_image (static overhead camera)
  - Joint states from /joint_states
  - Hand command from /hand_cmd
  - EE pose commands from /ee_pose_cmd

Output: Parquet file per episode in /extend_robotics_ws/data/
Images are JPEG-encoded (quality 90) and stored as binary columns.
Decoding: np.frombuffer(row, np.uint8) -> cv2.imdecode(..., cv2.IMREAD_COLOR) -> cv2.cvtColor(..., cv2.COLOR_BGR2RGB)

Controls (via /recorder_cmd topic):
  'start'   - begin recording
  'stop'    - stop and save episode
  'discard' - stop and discard episode
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
import os
import threading
import time
from datetime import datetime
from cv_bridge import CvBridge

JPEG_QUALITY = 90


class RecorderNode(Node):
    def __init__(self):
        super().__init__('extend_recorder_node')

        self.declare_parameter('scene',       'gripper')
        self.declare_parameter('data_dir',    '/extend_robotics_ws/data')
        self.declare_parameter('img_width',   224)
        self.declare_parameter('img_height',  224)

        self.scene    = self.get_parameter('scene').get_parameter_value().string_value
        self.data_dir = self.get_parameter('data_dir').get_parameter_value().string_value
        self.img_w    = self.get_parameter('img_width').get_parameter_value().integer_value
        self.img_h    = self.get_parameter('img_height').get_parameter_value().integer_value

        os.makedirs(self.data_dir, exist_ok=True)

        self.bridge    = CvBridge()
        self.lock      = threading.Lock()
        self.recording = False

        self._reset_buffers()

        # latest observations (updated by subscriber callbacks)
        self.latest_joints       = None
        self.latest_wrist_img    = None
        self.latest_overhead_img = None
        self.latest_hand_cmd     = 0.0
        self.latest_ee_cmd       = np.zeros(3, dtype=np.float32)

        self.create_subscription(JointState, '/joint_states',   self._joint_cb,    10)
        self.create_subscription(Image,      '/rgb_image',      self._wrist_cb,    10)
        self.create_subscription(Image,      '/overhead_image', self._overhead_cb, 10)
        self.create_subscription(Float32,    '/hand_cmd',       self._hand_cb,     10)
        self.create_subscription(Twist,      '/ee_pose_cmd',    self._ee_cb,       10)
        self.create_subscription(String,     '/recorder_cmd',   self._cmd_cb,      10)

        self.create_timer(0.1, self._record_step)  # 10 Hz

        self.get_logger().info(f'Recorder ready — scene={self.scene}')
        self.get_logger().info(f'Data dir: {self.data_dir}')
        self.get_logger().info('Publish to /recorder_cmd: "start" | "stop" | "discard"')

    # ── buffer management ────────────────────────────────────────────────────

    def _reset_buffers(self):
        self.buf_frame_index   = []
        self.buf_timestamps    = []
        self.buf_joints        = []
        self.buf_wrist_imgs    = []
        self.buf_overhead_imgs = []
        self.buf_ee_cmd        = []
        self.buf_hand_cmd      = []

    # ── subscribers ──────────────────────────────────────────────────────────

    def _joint_cb(self, msg: JointState):
        with self.lock:
            self.latest_joints = np.array(msg.position, dtype=np.float32)

    def _wrist_cb(self, msg: Image):
        img = self._decode_ros_image(msg)
        if img is not None:
            with self.lock:
                self.latest_wrist_img = img

    def _overhead_cb(self, msg: Image):
        img = self._decode_ros_image(msg)
        if img is not None:
            with self.lock:
                self.latest_overhead_img = img

    def _decode_ros_image(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if img.shape[:2] != (self.img_h, self.img_w):
                img = cv2.resize(img, (self.img_w, self.img_h))
            return img
        except Exception as e:
            self.get_logger().warn(f'Image decode failed: {e}')
            return None

    def _hand_cb(self, msg: Float32):
        with self.lock:
            self.latest_hand_cmd = float(msg.data)

    def _ee_cb(self, msg: Twist):
        with self.lock:
            self.latest_ee_cmd = np.array(
                [msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float32
            )

    def _cmd_cb(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd == 'start':
            self._start_recording()
        elif cmd == 'stop':
            self._stop_recording(save=True)
        elif cmd == 'discard':
            self._stop_recording(save=False)
        else:
            self.get_logger().warn(f'Unknown recorder command: {cmd}')

    # ── recording control ────────────────────────────────────────────────────

    def _start_recording(self):
        with self.lock:
            if self.recording:
                self.get_logger().warn('Already recording!')
                return
            self._reset_buffers()
            self.recording = True
        self.get_logger().info('Recording started.')

    def _stop_recording(self, save: bool):
        with self.lock:
            if not self.recording:
                self.get_logger().warn('Not recording!')
                return
            self.recording = False
        if save and self.buf_joints:
            self._save_episode()
        else:
            self.get_logger().info('Episode discarded.')

    def _record_step(self):
        with self.lock:
            if not self.recording or self.latest_joints is None:
                return

            blank = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            self.buf_frame_index.append(len(self.buf_frame_index))
            self.buf_timestamps.append(time.time())
            self.buf_joints.append(self.latest_joints.copy())
            self.buf_wrist_imgs.append(
                self.latest_wrist_img.copy() if self.latest_wrist_img is not None else blank
            )
            self.buf_overhead_imgs.append(
                self.latest_overhead_img.copy() if self.latest_overhead_img is not None else blank
            )
            self.buf_ee_cmd.append(self.latest_ee_cmd.copy())
            self.buf_hand_cmd.append(self.latest_hand_cmd)

    # ── save ─────────────────────────────────────────────────────────────────

    def _encode_jpeg(self, img: np.ndarray) -> bytes:
        """Encode an RGB image to JPEG bytes."""
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes()

    def _save_episode(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(
            self.data_dir, f'episode_{self.scene}_{timestamp}.parquet'
        )

        self.get_logger().info(f'Encoding {len(self.buf_joints)} frames…')
        wrist_bytes    = [self._encode_jpeg(img) for img in self.buf_wrist_imgs]
        overhead_bytes = [self._encode_jpeg(img) for img in self.buf_overhead_imgs]

        table = pa.table({
            'frame_index':                 pa.array(self.buf_frame_index,                     type=pa.int32()),
            'timestamp':                   pa.array(self.buf_timestamps,                      type=pa.float64()),
            'observation.joint_positions': pa.array([j.tolist() for j in self.buf_joints],   type=pa.list_(pa.float32())),
            'observation.wrist_image':     pa.array(wrist_bytes,                              type=pa.binary()),
            'observation.overhead_image':  pa.array(overhead_bytes,                           type=pa.binary()),
            'action.ee_cmd':               pa.array([e.tolist() for e in self.buf_ee_cmd],   type=pa.list_(pa.float32())),
            'action.hand_cmd':             pa.array(self.buf_hand_cmd,                        type=pa.float32()),
        })

        table = table.replace_schema_metadata({
            b'scene':        self.scene.encode(),
            b'timestamp':    timestamp.encode(),
            b'n_frames':     str(len(self.buf_joints)).encode(),
            b'img_width':    str(self.img_w).encode(),
            b'img_height':   str(self.img_h).encode(),
            b'jpeg_quality': str(JPEG_QUALITY).encode(),
        })

        pq.write_table(table, fname, compression='snappy')

        self.get_logger().info(f'Episode saved: {fname} ({len(self.buf_joints)} steps)')
        self._reset_buffers()


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
