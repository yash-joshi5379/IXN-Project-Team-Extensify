# Extend Robotics VLA Workspace

This workspace is a ROS 2 based environment for the **Extend Robotics VLA (Vision-Language-Action)** project. It provides a simulation-to-data pipeline for collecting robotic demonstration data using MuJoCo.

## Architecture Overview

```text
IXN-Project-Team-Extensify/
├── assets/             # Robot meshes (STL) and MuJoCo XMLs
│   ├── xarm7/          # xArm7 arm + gripper assets
│   └── xhand1/         # xHand1 multi-fingered hand meshes
├── data/               # Recorded demonstration episodes (Parquet)
├── docker/             # Dockerfile and docker-compose.yml
├── mjcf/               # MuJoCo scene definition files
└── src/
    ├── extend_bringup/ # Launch files and global configurations
    ├── extend_recorder/# Data collection and Parquet serialization
    ├── extend_sim/     # MuJoCo physics, rendering, and viewer nodes
    └── extend_teleop/  # Keyboard-based teleoperation interface
```

**ROS 2 packages:**

- **`extend_sim`** — Core simulation. `sim_node.py` runs the physics engine, IK controller, and both camera renderers. `viewer_node.py` provides a live passive 3D viewer.
- **`extend_teleop`** — Keyboard teleoperation. Maps W/A/S/D/Q/E/Space/P/R to Cartesian EE delta commands and gripper/hand control.
- **`extend_recorder`** — Episode recording. Captures observations and actions at 10 Hz and serialises them to Parquet.
- **`extend_bringup`** — Launch entry point. `sim.launch.py` starts the sim, recorder, and viewer together.

---

## Simulation Mechanics

### Physics & Control

- **Engine**: MuJoCo at **1000 Hz** (1 ms timestep), `integrator=implicitfast`.
- **Control strategy**: Differential Inverse Kinematics. Cartesian delta commands from teleop are mapped to joint positions via the Jacobian pseudo-inverse.
- **IK algorithm**: Damped least-squares (Levenberg-Marquardt), λ = 0.05.
- **Actuation**:
  - **Arm joints**: 7-DOF position-controlled (`biastype="affine"` PD actuators, `act1`–`act7`).
  - **Gripper**: Position-controlled via the `split` tendon. `gainprm=150`, `biastype="affine"`, `biasprm="0 -150 -15"`, `ctrlrange="0 0.85"`. In `sim_node.py`, `hand_cmd` (0.0–1.0) maps to `ctrl = hand_cmd × 0.85`. Setting ctrl=0 actively drives the fingers open; ctrl=0.85 drives them closed.
  - **xHand**: Postural interpolation across 12 DOF. `hand_cmd` linearly blends `XHAND_OPEN` and `XHAND_CLOSE` joint configurations; individual finger actuators are position-controlled.

### IK Physics

#### 1. Jacobian Matrix
MuJoCo computes a 3×7 translational Jacobian at the EE site using `mj_jacSite`:

$$\dot{x} = J(q)\,\dot{q}$$

#### 2. Damped Least Squares
To avoid instability near singularities the damped pseudo-inverse is used:

$$J_{\text{inv}} = J^T \left(J J^T + \lambda^2 I\right)^{-1}, \quad \lambda = 0.05$$

#### 3. Motion Smoothing
Commands accumulate in `move_queue` and are drained by a first-order exponential filter (k = 10.0, time constant ≈ 0.1 s):

$$\text{fraction} = 1 - e^{-k \cdot dt}, \qquad \Delta_{\text{step}} = \text{move\_queue} \times \text{fraction}$$

The queue is capped at `MAX_EE_SPEED / k = 0.05 m` to prevent backlog from key-repeat.

#### 4. Speed Limits
Applied after IK in every physics step:

| Limit | Value | Where enforced |
|-------|-------|----------------|
| Max EE Cartesian speed | 0.5 m/s | `move_queue` cap in `_physics_loop` |
| Max joint speed | π rad/s (180 °/s) | `dq` scaling in `_apply_ik` |

Both limits match the xArm7's rated maximums.

#### 5. Control Loop
1. **Accumulate** — teleop deltas buffered in `ee_delta_buf`.
2. **Cap** — `move_queue` norm clamped to 0.05 m.
3. **Filter** — exponential smoothing produces `delta_step`.
4. **IK** — `delta_step` → `dq` via damped Jacobian.
5. **Clamp** — `dq` scaled so no joint exceeds π rad/s.
6. **Apply** — `ctrl[arm] += dq`, clamped to joint limits.
7. **Step** — `mj_step` advances physics by 1 ms.

### Rendering & Cameras

Two cameras are rendered offscreen using EGL (headless) via a shared `mujoco.Renderer`:

| Camera | Name in MJCF | Mounting | FOV | Published topic |
|--------|-------------|----------|-----|-----------------|
| Wrist | `wrist_cam` | On `xarm_gripper_base_link` (gripper) / `right_hand_link` (xhand), `pos="0 0 0.15"`, pointing downward | 60° | `/rgb_image` |
| Overhead | `overhead_cam` | World-fixed at `pos="0.3 0 1.2"`, pointing straight down | default | `/overhead_image` |

Both publish 224×224 RGB at 10 Hz. The `ViewerNode` uses GLFW for interactive 3D visualisation; press `Tab` in the viewer to cycle cameras.

---

## Supported Robots & Scenes

### `scene_gripper` (`mjcf/scene_gripper.xml`)
- Includes `assets/xarm7/xarm7.xml` (arm + parallel gripper).
- **Task object**: orange cylinder, radius 0.025 m, height 0.06 m, mass 0.3 kg.
- **Spawn zone**: 50×50 cm square, x ∈ [0.10, 0.60] m, y ∈ [0.10, 0.60] m. Visualised by a yellow border (4 non-collidable box geoms). Cylinder position is randomised within the zone on every launch and every reset, subject to reach constraints (0.28–0.62 m from base).
- **Target zone**: semi-transparent green disc at [0.35, 0.35] — the exact centre of the spawn square. Collision disabled, visual only.
- **Home keyframe** (`scene_home`): arm in upright over-table pose, gripper open, cylinder at a random valid position.

### `scene_xhand` (`mjcf/scene_xhand.xml`)
- Includes `assets/xarm7/xarm7_nohand.xml` (arm + xHand1 right hand).
- Same task setup as the gripper scene.
- 12-DOF hand; TCP site `right_hand_tcp` at `pos="0 0 0.13"` on `right_hand_link`.
- **Home keyframe**: joint7 = 0 (hand pointing downward), arm position identical to gripper scene. Arm ctrl values match qpos so actuators hold the pose on reset rather than fighting it.
- Arm actuators `act1`–`act7` are defined in `xarm7_nohand.xml`; the scene file adds only the 12 hand actuators.

---

## Network & Topics

| Topic | Type | Freq | Description |
| :--- | :--- | :--- | :--- |
| `/joint_states` | `sensor_msgs/JointState` | 150 Hz | Full `qpos` snapshot — arm + hand + freejoint objects. Matches real robot cadence. |
| `/rgb_image` | `sensor_msgs/Image` | 10 Hz | 224×224 RGB wrist camera feed. |
| `/overhead_image` | `sensor_msgs/Image` | 10 Hz | 224×224 RGB static overhead camera. |
| `/ee_pose_cmd` | `geometry_msgs/Twist` | — | Cartesian delta commands (linear x, y, z only). |
| `/hand_cmd` | `std_msgs/Float32` | — | Hand state: 0.0 = open, 1.0 = closed. |
| `/reset_sim` | `std_msgs/Empty` | — | Resets to `scene_home` keyframe and re-randomises cylinder position. |
| `/recorder_cmd` | `std_msgs/String` | — | `start` / `stop` / `discard` episode. |

---

## Teleop Key Bindings

| Key | Action |
| :--- | :--- |
| `W` / `S` | EE forward / backward (X axis) |
| `A` / `D` | EE left / right (Y axis) |
| `Q` / `E` | EE up / down (Z axis) |
| `Space` | Toggle gripper / hand open–closed |
| `R` | Reset simulation and re-randomise cylinder |
| `P` | Toggle precision mode (1 mm steps vs. 20 mm default) |
| `X` / `Esc` | Quit teleop |

Default step size is **20 mm**. Precision mode (1 mm) is useful for fine placement. Both modes are subject to the 0.5 m/s EE speed cap.

---

## Data Format (Parquet)

Episodes are saved to `data/` as `episode_<scene>_<YYYYMMDD_HHMMSS>.parquet` (Snappy compression). Images are JPEG-encoded at quality 90 and stored as binary columns.

| Column | Type | Description |
| :--- | :--- | :--- |
| `frame_index` | `int32` | 0-based frame counter |
| `timestamp` | `float64` | Unix wall-clock time |
| `observation.joint_positions` | `list<float32>` | Full `qpos` vector (arm + hand + object freejoint) |
| `observation.wrist_image` | `binary` | JPEG bytes — wrist camera |
| `observation.overhead_image` | `binary` | JPEG bytes — overhead camera |
| `action.ee_cmd` | `list<float32>` | Cartesian delta [dx, dy, dz] in metres |
| `action.hand_cmd` | `float32` | 0.0 = open, 1.0 = closed |

File-level metadata: `scene`, `timestamp`, `n_frames`, `img_width`, `img_height`, `jpeg_quality`.

**Decoding images:**
```python
import pyarrow.parquet as pq, numpy as np, cv2

table = pq.read_table('data/episode_gripper_....parquet')
df = table.to_pandas()

raw = np.frombuffer(df['observation.wrist_image'].iloc[0], np.uint8)
rgb = cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
```

> `joint_positions` is the full MuJoCo `qpos` vector — arm joints, hand/gripper joints, and 7 values per floating object (3 pos + 4 quat). Order matches joint declaration in the MJCF.

---

## Docker Setup

Base image: `osrf/ros:humble-desktop-full`.
Project name: `extend_robotics_ixn` · Container name: `extend_robotics_ixn`.

The workspace is bind-mounted as `../:/extend_robotics_ws`, so host edits are immediately visible inside the container. Python source changes still require `colcon build` because files are copied (not symlinked) into `install/`.

**Key Python dependencies (pinned versions matter):**

| Package | Reason |
|---------|--------|
| `mujoco` | Physics engine |
| `numpy<2` | NumPy 2.x breaks MuJoCo C bindings |
| `opencv-python-headless<4.10` | Image encoding |
| `pyarrow` | Parquet read/write |
| `pandas` | Episode inspection |
| `pynput`, `scipy` | Teleop and utilities |

X11 forwarding is handled via `.Xauthority` and `/tmp/.X11-unix`. `MUJOCO_GL=egl` is set in the compose environment so the sim node never touches the display; the viewer node sets `MUJOCO_GL=glfw` in process to claim a GLFW window.

---

## Utility / Debug Scripts

Run from inside the container with `python3 /extend_robotics_ws/<script>.py`:

| Script | Purpose |
| :--- | :--- |
| `check_model.py` | Print all actuators, sites, joints, ranges, and keyframe qpos |
| `check_model_2.py` | Extended model inspection |
| `check_collisions.py` | Print geom collision groups/types for the gripper scene |
| `check_collisions_v2.py` | Extended collision debugging |

---

## Getting Started

```bash
# [host] build image (first time only)
cd ~/IXN-Project-Team-Extensify/docker && docker compose build

# [host] start container
docker compose up -d

# [host] open a shell (repeat for each terminal)
docker exec -it extend_robotics_ixn bash

# [container] source workspace (run in every terminal)
source /extend_robotics_ws/install/setup.bash

# [container] launch simulation
ros2 launch extend_bringup sim.launch.py scene:=gripper   # or scene:=xhand

# [container] start teleoperation (separate terminal)
ros2 run extend_teleop teleop_node
```

After any code change in `src/`, rebuild:
```bash
cd /extend_robotics_ws && colcon build
source install/setup.bash
```

MJCF and asset files under `mjcf/` and `assets/` are read at runtime and do not require a rebuild.

Refer to `COMMANDS.md` for a full command reference including recording and topic monitoring.
