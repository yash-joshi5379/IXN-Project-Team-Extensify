#!/usr/bin/env python3
"""
extend_teleop - Keyboard teleoperation node for Extend Robotics VLA project

Controls:
  W/S     - move EE forward/backward (x)
  A/D     - move EE left/right (y)
  Q/E     - move EE up/down (z)
  SPACE   - toggle hand open/close
  ESC/X   - quit
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Empty

import sys
import tty
import termios
import threading

STEP = 0.02  # metres per keypress

KEY_MAP = {
    'w': ( STEP,  0.0,   0.0),
    's': (-STEP,  0.0,   0.0),
    'a': ( 0.0,   STEP,  0.0),
    'd': ( 0.0,  -STEP,  0.0),
    'q': ( 0.0,   0.0,   STEP),
    'e': ( 0.0,   0.0,  -STEP),
}

def get_key():
    """Read a single keypress from terminal."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


class TeleopNode(Node):
    def __init__(self):
        super().__init__('extend_teleop_node')

        self.ee_pub   = self.create_publisher(Twist,   '/ee_pose_cmd', 10)
        self.hand_pub = self.create_publisher(Float32, '/hand_cmd',    10)
        self.reset_pub = self.create_publisher(Empty,  '/reset_sim',   10)

        self.hand_open = True
        self.running   = True
        self.step_size = STEP

        self.get_logger().info('Teleop ready.')
        self.get_logger().info('W/S=X  A/D=Y  Q/E=Z  SPACE=hand  R=reset  P=precision  X=quit')


        # keyboard thread
        self.kb_thread = threading.Thread(target=self._kb_loop, daemon=True)
        self.kb_thread.start()

    def _kb_loop(self):
        while self.running:
            key = get_key()

            if key in KEY_MAP:
                dx, dy, dz = KEY_MAP[key]
                msg = Twist()
                scale = self.step_size / STEP
                msg.linear.x = float(dx * scale)
                msg.linear.y = float(dy * scale)
                msg.linear.z = float(dz * scale)
                self.ee_pub.publish(msg)

            elif key == ' ':
                self.hand_open = not self.hand_open
                msg = Float32()
                msg.data = 0.0 if self.hand_open else 1.0
                self.hand_pub.publish(msg)
                state = 'OPEN' if self.hand_open else 'CLOSED'
                self.get_logger().info(f'Hand: {state}')

            elif key in ('r', 'R'):
                self.reset_pub.publish(Empty())
                self.hand_open = True
                self.get_logger().info('Simulation reset requested.')

            elif key in ('p', 'P'):
                if self.step_size == STEP:
                    self.step_size = 0.001
                    self.get_logger().info('PRECISION MODE: 1mm steps')
                else:
                    self.step_size = STEP
                    self.get_logger().info(f'NORMAL MODE: {int(STEP*1000)}mm steps')

            elif key in ('x', 'X', '\x1b'):
                self.get_logger().info('Quitting teleop.')
                self.running = False
                rclpy.shutdown()
                break

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
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
