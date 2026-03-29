# IXN-Project-Team-Extensify

Repository to store all simulation and hardware-based code.

Goal: create and train a SmolVLA model to automate a robot arm (XArm 7) with a 2-finger gripper to grasp a cylinder and place it into a slot.

## Installation and Setup
1. Clone this repository and go into the project directory
```
git clone https://github.com/yash-joshi5379/IXN-Project-Team-Extensify.git
cd IXN-Project-Team-Extensify/
```

2. Create a virtual environment, activate it, and install all requirements
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Check the ```load_data_test.py``` file runs without errors. It should print the following output.
```
python src\load_data_test.py

                                              action  ... task_index
0  [0.35293713, 0.68267363, -0.07424462, 2.556133...  ...          0
1  [0.35295725, 0.6825197, -0.07424697, 2.5558503...  ...          0
2  [0.35297787, 0.6823274, -0.074249804, 2.55549,...  ...          0
3  [0.35299087, 0.6822012, -0.07425172, 2.5552552...  ...          0
4  [0.35283133, 0.6822953, -0.07299664, 2.5529456...  ...          0

[5 rows x 10 columns]
['action', 'observation.effort', 'observation.force_torque', 'observation.state', 'observation.qvel', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
action                       object
observation.effort           object
observation.force_torque     object
observation.state            object
observation.qvel             object
timestamp                   float32
frame_index                   int64
episode_index                 int64
index                         int64
task_index                    int64
dtype: object
(600, 10)
                                              action  ... task_index
0  [0.35293713, 0.68267363, -0.07424462, 2.556133...  ...          0
1  [0.35295725, 0.6825197, -0.07424697, 2.5558503...  ...          0

[2 rows x 10 columns]
action: shape=(8,), dtype=float32
observation.state: shape=(8,), dtype=float32
observation.effort: shape=(8,), dtype=float32
observation.force_torque: shape=(6,), dtype=float32
observation.qvel: shape=(8,), dtype=float32
```