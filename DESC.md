# Extend Robotics VLA Workspace

This workspace is a ROS 2 based environment for the **Extend Robotics VLA (Vision-Language-Action)** project. It provides a simulation-to-data pipeline for collecting robotic demonstration data using MuJoCo.

## 🏗️ Architecture Overview

The project is organized as a standard ROS 2 workspace:

```text
/home/shcooray/extend_robotics_ws/
├── assets/             # Robot meshes (STL) and MuJoCo XMLs
│   ├── xarm7/          # xArm7 assets and configuration
│   └── xhand1/         # xHand multi-fingered hand assets
├── data/               # Recorded demonstration episodes (HDF5)
├── docker/             # Containerization files (Dockerfile, Compose)
├── mjcf/               # MuJoCo scene definition files
└── src/                # ROS 2 package source code
    ├── extend_bringup/ # Launch files and global configurations
    ├── extend_recorder/# Data collection and HDF5 serialization
    ├── extend_sim/     # MuJoCo physics and rendering nodes
    └── extend_teleop/  # Keyboard-based control interface
```

The project is divided into several ROS 2 packages:

- **`extend_sim`**: Core simulation package using **MuJoCo**.
  - `sim_node.py`: Manages the physics engine, robot state, and wrist camera rendering.
  - `viewer_node.py`: Provides a live 3D visualization of the simulation.
- **`extend_teleop`**: Teleoperation interface.
  - `teleop_node.py`: Maps keyboard inputs (W/A/S/D/Q/E/SPACE) to end-effector velocity commands and gripper actions.
- **`extend_recorder`**: Data collection utility.
  - `recorder_node.py`: Captures observations and actions at 10Hz and saves them as HDF5 files for training VLA models.
- **`extend_bringup`**: Launch configurations.
  - `sim.launch.py`: Unified entry point to start simulation and recording nodes.

## ⚙️ Simulation Mechanics

### Physics & Control
- **Engine**: MuJoCo physics running at **1000Hz** (1ms timestep).
- **Control Strategy**: **Differential Inverse Kinematics (IK)**. The simulation accepts Cartesian delta commands for the end-effector (EE) and maps them to joint velocities/positions.
- **IK Algorithm**: Uses the **Jacobian pseudo-inverse** method with Levenberg-Marquardt regularization (`λ = 0.05`) for singularity robustness and stability.
- **Actuation**:
  - **Arm Joints**: 7-DOF position-controlled via actuator targets updated by the IK loop.
  - **Gripper**: Position-controlled via the `split` tendon (PD on tendon position). Command `0.0` = open (tendon pos 0 rad), `1.0` = closed (tendon pos 0.85 rad).
  - **xHand**: Postural interpolation. Commands are mapped to a linear blend between predefined `OPEN` and `CLOSED` joint configurations.

### Deep Dive: IK Physics

The simulation translates Cartesian commands into joint positions using **Differential Inverse Kinematics**. This allows the operator to control the end-effector (EE) in 3D space without managing individual joint angles.

#### 1. The Jacobian Matrix ($J$)
The Jacobian represents the linear mapping between joint velocities ($\dot{q}$) and EE velocities ($\dot{x}$). In `sim_node.py`, MuJoCo computes a $3 \times 7$ matrix (for X, Y, Z translation) using `mj_jacSite`:
$$\dot{x} = J(q) \dot{q}$$

#### 2. Damped Least Squares (Levenberg-Marquardt)
To solve for $\dot{q}$ while maintaining stability near singularities (e.g., when the arm is fully extended), the simulation employs **Levenberg-Marquardt regularization**. Instead of a simple pseudo-inverse, it uses a damping factor $\lambda = 0.05$:
$$J_{inv} = J^T (J J^T + \lambda^2 I)^{-1}$$
This "virtual damping" prevents the matrix from becoming ill-conditioned, ensuring the robot doesn't "explode" or jitter when it reaches the edges of its workspace.

#### 3. Motion Profile & Smoothing
To ensure physics stability at the 1000Hz simulation rate, commands are not applied instantly. The system uses an exponential filter ($k=10.0$, giving a ~0.1s time constant) to smoothly interpolate the movement:
$$fraction = 1 - e^{-k \cdot dt}$$
$$\Delta step = move\_queue \times fraction$$
This creates a natural acceleration/deceleration curve, preventing high-frequency oscillations in the MuJoCo solver.

#### 4. Control Loop Execution
1. **Accumulate**: Teleop commands are buffered in `ee_delta_buf`.
2. **Compute**: The Jacobian is sampled at the current state.
3. **Solve**: The regularized inverse maps the filtered $\Delta step$ to joint deltas $\Delta q$.
4. **Step**: Actuator targets are updated: $ctrl_{new} = ctrl_{old} + \Delta q$.
5. **Simulate**: MuJoCo steps the physics engine to solve for forces and contacts.

### Rendering & Visualization
- **SimNode**: Headless execution using **EGL** for high-performance offscreen rendering of the wrist camera (224x224 RGB).
- **ViewerNode**: Interactive visualization using **GLFW**. It operates as a passive observer, mirroring the state of the physics engine by subscribing to `/joint_states`.

## 🤖 Supported Robots & Scenes

The workspace currently supports the **xArm7** robotic arm with two end-effector configurations:
1.  **`gripper`**: Standard parallel gripper.
2.  **`xhand`**: A more complex multi-fingered hand (`xhand1`).

Assets for these are stored in `assets/` (STL/XML) and `mjcf/` (MuJoCo scene definitions).

## 📡 Network & Topics

| Topic | Type | Freq | Description |
| :--- | :--- | :--- | :--- |
| `/joint_states` | `sensor_msgs/JointState` | 150Hz | Complete robot and object state (qpos). Matches real robot cadence. |
| `/rgb_image` | `sensor_msgs/Image` | 10Hz | 224×224 RGB wrist camera feed (mounted on gripper base). |
| `/overhead_image` | `sensor_msgs/Image` | 10Hz | 224×224 RGB static overhead camera at [0.3, 0, 1.2] m. |
| `/ee_pose_cmd` | `geometry_msgs/Twist` | N/A | Cartesian delta commands (linear x, y, z). |
| `/hand_cmd` | `std_msgs/Float32` | N/A | Hand state (0=Open, 1=Closed). |
| `/reset_sim` | `std_msgs/Empty` | N/A | Resets sim to `scene_home` keyframe and re-randomizes cylinder. |
| `/recorder_cmd` | `std_msgs/String` | N/A | Control recording: `start`, `stop`, `discard`. |

## 🕹️ Teleop Key Bindings

| Key | Action |
| :--- | :--- |
| `W` / `S` | EE forward / backward (X) |
| `A` / `D` | EE left / right (Y) |
| `Q` / `E` | EE up / down (Z) |
| `SPACE` | Toggle hand open / closed |
| `R` | Reset simulation to `scene_home` keyframe |
| `P` | Toggle precision mode (1 mm steps vs. 20 mm default) |
| `X` / `ESC` | Quit teleop |

Default step size is **20 mm** (`STEP = 0.02` m). Precision mode drops to **1 mm** — useful for fine placement near grasp targets.

## 📊 Data Format (Parquet)

Recorded episodes are stored in `data/` as `episode_<scene>_<YYYYMMDD_HHMMSS>.parquet`. Compression: Snappy. Images are JPEG-encoded (quality 90) and stored as binary columns.

### Schema

| Column | Type | Description |
| :--- | :--- | :--- |
| `frame_index` | `int32` | 0-based frame counter |
| `timestamp` | `float64` | Unix wall-clock time |
| `observation.joint_positions` | `list<float32>` | Full MuJoCo `qpos` snapshot (arm + hand + freejoint objects) |
| `observation.wrist_image` | `binary` | JPEG bytes — wrist camera (224×224 RGB) |
| `observation.overhead_image` | `binary` | JPEG bytes — static overhead camera (224×224 RGB) |
| `action.ee_cmd` | `list<float32>` | Cartesian delta `[dx, dy, dz]` in metres |
| `action.hand_cmd` | `float32` | `0.0` = open, `1.0` = closed |

File-level metadata keys: `scene`, `timestamp`, `n_frames`, `img_width`, `img_height`, `jpeg_quality`.

### Decoding images

```python
import numpy as np, cv2, pyarrow.parquet as pq

table = pq.read_table('episode_gripper_....parquet')
df = table.to_pandas()

# Decode one frame
raw = np.frombuffer(df['observation.wrist_image'].iloc[0], np.uint8)
bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # shape: (224, 224, 3)
```

> Note: `joint_positions` is the **full** `qpos` vector — arm joints + hand/gripper joints + freejoint components of floating objects (7 values each: 3 pos + 4 quat). Ordering matches joint declaration order in the MJCF.

## 🐳 Docker Setup

The workspace runs inside a Docker container based on `osrf/ros:humble-desktop-full`.

Key Python dependencies (pinned versions matter):
- `mujoco` — physics engine
- `numpy<2` — NumPy 2.x breaks MuJoCo C bindings
- `opencv-python-headless<4.10`
- `pyarrow`, `pandas` — episode storage and inspection
- `scipy`, `pynput`

The `docker-compose.yml` mounts the entire workspace as a bind-volume (`../:/extend_robotics_ws`) so edits on the host are reflected instantly inside the container without rebuilding. X11 forwarding is handled via `.Xauthority` and `/tmp/.X11-unix`, enabling the GLFW viewer window to display on the host desktop. The environment forces `MUJOCO_GL=egl` at the compose level to keep the headless sim node from touching the display.

## 🗺️ Scenes & Task Objects

### `scene_gripper`
- Includes `xarm7.xml` (arm + parallel gripper).
- **Task object**: a small orange cylinder (`0.025m radius × 0.06m tall`, 0.3 kg, friction-rich) placed at `[0.5, 0.0, 0.03]`.
- **Target zone**: a semi-transparent green disc at `[0.5, 0.2]` (collision disabled, visual only).
- **Home keyframe** (`scene_home`): arm in a safe upright pose, cylinder in starting position.

### `scene_xhand`
- Includes `xhand1_right.xml` mounted on the xArm7.
- 12-DOF hand controlled via postural interpolation between `XHAND_OPEN` and `XHAND_CLOSE` configs.
- TCP site: `right_hand_tcp`.

## 🔧 Utility / Debug Scripts

These scripts run **inside the container** (not as ROS nodes) and are useful for inspecting the model:

| Script | Purpose |
| :--- | :--- |
| `check_model.py` | Print all actuators, sites, joints, joint ranges, and keyframe qpos |
| `check_model_2.py` | Extended model inspection |
| `check_collisions.py` | Print geom collision groups/types for the gripper scene |
| `check_collisions_v2.py` | Extended collision debugging |

Run them with `python3 /extend_robotics_ws/<script>.py` from inside the container.

## 🚀 Getting Started

1.  **Environment**: Uses Docker (see `docker/`).
2.  **Build the image** (first time): `cd ~/extend_robotics_ws/docker && docker compose build`
3.  **Start container**: `docker compose up -d`
4.  **Enter container**: `docker exec -it extend_robotics bash`
5.  **Source workspace**: `source /extend_robotics_ws/install/setup.bash`
6.  **Launch**: `ros2 launch extend_bringup sim.launch.py scene:=gripper`
7.  **Control**: `ros2 run extend_teleop teleop_node`
8.  **Record**: Publish `'start'`/`'stop'` to `/recorder_cmd`.

Refer to `COMMANDS.md` for a detailed list of shell commands and topic references.

