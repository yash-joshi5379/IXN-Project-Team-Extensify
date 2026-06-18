# Extend Robotics — Command Reference

## Legend
- `[host]` — run on your Linux machine
- `[docker]` — run inside the container

---

## 1 — Environment setup

```bash
# [host] start container (X11 is handled automatically via .Xauthority)
cd ~/IXN-Project-Team-Extensify/docker && docker compose up -d

# [host] enter container (repeat for each new terminal)
docker exec -it extend_robotics_ixn bash

# [docker] FIRST TIME ONLY — build the workspace (creates install/).
# Required on a fresh checkout; the repo ships source only, no install/ dir.
cd /extend_robotics_ws && colcon build

# [docker] source workspace (run in every terminal you open)
source /extend_robotics_ws/install/setup.bash
```

> If `source .../install/setup.bash` reports "No such file or directory" or
> `ros2 launch` says `Package 'extend_bringup' not found`, the workspace has
> not been built yet — run the `colcon build` step above first.

---

## 2 — Launch Sim + Recorder + Viewer (Terminal 1)

This command starts the MuJoCo physics engine, the data recorder, and the integrated 3D viewer.

```bash
# xArm7 + parallel gripper (Viewer opens automatically)
ros2 launch extend_bringup sim.launch.py scene:=gripper

# xArm7 + xHand1
ros2 launch extend_bringup sim.launch.py scene:=xhand

# To run headless (physics only, no viewer window):
ros2 launch extend_bringup sim.launch.py use_viewer:=false
```

---

## 3 — Launch Teleop (Terminal 2)

```bash
# keyboard control (works for both scenes)
ros2 run extend_teleop teleop_node
```

---

## 4 — Teleop controls

| Key | Action |
|-----|--------|
| `W` / `S` | move EE forward / backward (X axis) |
| `A` / `D` | move EE left / right (Y axis) |
| `Q` / `E` | move EE up / down (Z axis) |
| `SPACE` | toggle hand open / closed |
| `R` | reset simulation (robot + objects) |
| `P` | toggle precision mode (1 mm steps vs. 20 mm default) |
| `X` or `ESC` | quit teleop |

---

## 5 — Data recording (Terminal 3)

```bash
# begin recording an episode
ros2 topic pub --once /recorder_cmd std_msgs/msg/String "data: 'start'"

# save the episode
ros2 topic pub --once /recorder_cmd std_msgs/msg/String "data: 'stop'"

# discard the episode
ros2 topic pub --once /recorder_cmd std_msgs/msg/String "data: 'discard'"

# list saved episodes
ls -lh /extend_robotics_ws/data/*.parquet
```

---

## 6 — Inspect a saved episode

```bash
python3 -c "
import pyarrow.parquet as pq, numpy as np, cv2

table = pq.read_table('/extend_robotics_ws/data/FILENAME.parquet')
meta  = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
df    = table.to_pandas()

print('Scene:   ', meta['scene'])
print('Frames:  ', meta['n_frames'])
print('Columns: ', df.columns.tolist())
print(df[['frame_index','timestamp','action.hand_cmd']].head())

# Decode a wrist camera frame
raw = np.frombuffer(df['observation.wrist_image'].iloc[0], np.uint8)
rgb = cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
print('Wrist frame shape:', rgb.shape)
"
```

---

## 7 — Topic monitoring

```bash
# list all active topics
ros2 topic list

# check camera publish rate (10Hz)
ros2 topic hz /rgb_image

# check joint state publish rate (150Hz)
ros2 topic hz /joint_states
```

---

## 8 — Rebuild packages

```bash
# rebuild all packages
cd /extend_robotics_ws && colcon build

# re-source after rebuild
source /extend_robotics_ws/install/setup.bash
```

---

## 9 — Stop container

```bash
# [host]
cd ~/IXN-Project-Team-Extensify/docker && docker compose down
```

---

## Topic reference

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | sim → all | 7 DOF arm joint positions (150Hz) |
| `/rgb_image` | `sensor_msgs/Image` | sim → all | 224×224 RGB wrist camera (10Hz) |
| `/overhead_image` | `sensor_msgs/Image` | sim → all | 224×224 RGB static overhead camera (10Hz) |
| `/ee_pose_cmd` | `geometry_msgs/Twist` | teleop → sim | end effector delta (linear only) |
| `/hand_cmd` | `std_msgs/Float32` | teleop → sim | hand 0.0=open 1.0=closed |
| `/recorder_cmd` | `std_msgs/String` | user → recorder | start / stop / discard |
