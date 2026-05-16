# Extend Robotics - VLA Simulation Workspace

A MuJoCo-based simulation environment for collecting robotic demonstration data to train Vision-Language-Action (VLA) models. Built on ROS 2 Humble, it provides a full pipeline from teleoperation to structured episode recording, matching the cadence and data format of the real robot system.

![Simulation Overview](docs/images/sim_overview.png)
*The MuJoCo viewer showing the xArm7 with parallel gripper in the gripper scene*

---

## What this is

This workspace simulates an **xArm7 robotic arm** in a tabletop pick-and-place environment. An operator uses the keyboard to move the arm in Cartesian space, open and close the gripper, and record demonstrations. Each recorded episode captures joint states, actions, and synchronized feeds from two cameras - a wrist-mounted camera and a static overhead camera - all saved into a single Parquet file ready for model training.

The simulation is designed to closely mirror the real robot setup: joint states publish at 150 Hz, the physics engine runs at 1000 Hz, and the IK controller uses the same damped least-squares algorithm used on the physical hardware.

---

## Robot configurations

Two end-effector configurations are supported, each with its own MuJoCo scene:

| Scene | End effector | Launch argument |
|-------|-------------|-----------------|
| Gripper | xArm parallel gripper | `scene:=gripper` |
| xHand | xHand1 multi-fingered hand (12 DOF) | `scene:=xhand` |

![Gripper scene](docs/images/scene_gripper.png)
*Left: gripper scene with the orange cylinder task object. Right: xHand scene.*

Each scene includes a small orange cylinder placed on the table as the task object, spawned at a random position within a 30×30 cm zone at 45° from the arm base on each reset. A semi-transparent green disc marks the target placement area.

---

## Requirements

- Docker and Docker Compose
- A display (X11) - the MuJoCo viewer renders to the host desktop via X11 forwarding
- No GPU required - offscreen rendering uses EGL, viewer uses GLFW over X11

---

## Setup

Everything runs inside a Docker container based on `osrf/ros:humble-desktop-full`. The workspace directory is bind-mounted into the container, so any edits on the host are immediately reflected without rebuilding the image.

**First time only - build the image:**
```bash
cd ~/IXN-Project-Team-Extensify/docker
docker compose build
```

This installs MuJoCo, PyArrow, OpenCV, and the other dependencies listed in `docker/Dockerfile`.

**Start the container:**
```bash
docker compose up -d
```

**Open a shell inside it:**
```bash
docker exec -it extend_robotics_ixn bash
source /extend_robotics_ws/install/setup.bash
```

You'll need to run the `source` command in every new terminal you open into the container. If you rebuild the ROS packages, source again afterwards.

---

## Running the simulation

Everything starts from a single launch command. Open three terminals, all inside the container with the workspace sourced.

**Terminal 1 - launch the sim, recorder, and viewer:**
```bash
ros2 launch extend_bringup sim.launch.py scene:=gripper
```

This starts three nodes at once: the physics engine (`sim_node`), the passive 3D viewer (`viewer_node`), and the data recorder (`recorder_node`). The MuJoCo viewer window will appear on your desktop.

To use the xHand instead:
```bash
ros2 launch extend_bringup sim.launch.py scene:=xhand
```

To run headless (no viewer window, useful for scripted collection):
```bash
ros2 launch extend_bringup sim.launch.py use_viewer:=false
```

**Terminal 2 - start teleoperation:**
```bash
ros2 run extend_teleop teleop_node
```

---

## Controlling the arm

The teleop node reads keyboard input and publishes Cartesian delta commands at 20 mm per keypress by default.

| Key | Action |
|-----|--------|
| `W` / `S` | Move end-effector forward / backward |
| `A` / `D` | Move end-effector left / right |
| `Q` / `E` | Move end-effector up / down |
| `Space` | Toggle gripper / hand open and closed |
| `R` | Reset simulation and randomise cylinder position |
| `P` | Toggle precision mode (1 mm steps instead of 20 mm) |
| `X` / `Esc` | Quit teleop |

The arm's Cartesian speed is capped at 0.5 m/s and each joint is limited to 180°/s - matching the real xArm7's rated maximums. Precision mode is useful when you need fine control near the object before grasping.

![Teleop in action](docs/images/teleop_demo.png)
*Operator moving the arm toward the cylinder before closing the gripper*

---

## Camera feeds

The simulation publishes two camera streams, both at 10 Hz and 224×224 RGB:

| Topic | Camera | Description |
|-------|--------|-------------|
| `/rgb_image` | Wrist camera | Mounted on the gripper base, looking downward at the grasp area |
| `/overhead_image` | Overhead camera | Static camera fixed at [0.3, 0, 1.2] m looking straight down |

![Camera views](docs/images/camera_views.png)
*Left: wrist camera view. Right: overhead camera view.*

To view them while the sim is running:
```bash
# Both feeds in separate windows
ros2 run image_view image_view --ros-args -r image:=/rgb_image &
ros2 run image_view image_view --ros-args -r image:=/overhead_image

# Or use rqt to switch between topics in one window
ros2 run rqt_image_view rqt_image_view
```

---

## Recording demonstrations

Recordings are controlled by publishing string commands to `/recorder_cmd`. The recorder captures at 10 Hz and saves both camera feeds alongside joint states and actions.

**Terminal 3 - control recording:**
```bash
# Start a new episode
ros2 topic pub --once /recorder_cmd std_msgs/msg/String "data: 'start'"

# Save the episode
ros2 topic pub --once /recorder_cmd std_msgs/msg/String "data: 'stop'"

# Discard if something went wrong
ros2 topic pub --once /recorder_cmd std_msgs/msg/String "data: 'discard'"
```

Episodes are saved to `/extend_robotics_ws/data/` inside the container, which maps to the `data/` folder in this repository.

---

## Data format

Each episode is saved as a Parquet file named `episode_<scene>_<timestamp>.parquet`. Images are stored as JPEG-compressed binary columns (quality 90), and the file uses Snappy compression overall. This keeps file sizes small while remaining fast to read.

| Column | Type | Description |
|--------|------|-------------|
| `frame_index` | int32 | 0-based frame counter |
| `timestamp` | float64 | Unix wall-clock time |
| `observation.joint_positions` | list\<float32\> | Full joint state vector |
| `observation.wrist_image` | binary | JPEG bytes from wrist camera |
| `observation.overhead_image` | binary | JPEG bytes from overhead camera |
| `action.ee_cmd` | list\<float32\> | Cartesian delta [dx, dy, dz] in metres |
| `action.hand_cmd` | float32 | 0.0 = open, 1.0 = closed |

Reading an episode and decoding a frame:
```python
import pyarrow.parquet as pq
import numpy as np
import cv2

table = pq.read_table('data/episode_gripper_20260516_120000.parquet')
df = table.to_pandas()

# Decode a wrist camera frame
raw = np.frombuffer(df['observation.wrist_image'].iloc[0], np.uint8)
rgb = cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
```

---

## Project structure

```
├── assets/
│   ├── xarm7/          # xArm7 URDF/MJCF and STL meshes
│   └── xhand1/         # xHand1 meshes and config
├── data/               # Recorded episodes (Parquet, git-ignored)
├── docker/             # Dockerfile and docker-compose.yml
├── mjcf/               # MuJoCo scene definitions
│   ├── scene_gripper.xml
│   └── scene_xhand.xml
└── src/
    ├── extend_bringup/  # Launch files
    ├── extend_recorder/ # Episode recording node
    ├── extend_sim/      # Physics engine + viewer nodes
    └── extend_teleop/   # Keyboard teleoperation node
```

---

## Rebuilding after code changes

If you edit any of the ROS packages in `src/`, rebuild from inside the container:

```bash
cd /extend_robotics_ws
colcon build
source install/setup.bash
```

Python source files are copied into the install directory at build time, so a rebuild is needed to pick up changes. The MJCF and asset files in `mjcf/` and `assets/` are read directly from disk at runtime and don't require a rebuild.

---

## Reference

For a detailed breakdown of the physics, IK implementation, topic interface, and data format see [`DESC.md`](DESC.md). For a full list of copy-pasteable shell commands see [`COMMANDS.md`](COMMANDS.md).
