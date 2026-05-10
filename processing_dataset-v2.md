# IXN-Project-Team-Extensify

Repository to store all simulation and hardware-based code.

Goal: create and train a SmolVLA model to automate a robot arm (XArm 7) with a 2-finger gripper to grasp a cylinder and place it into a slot.

## 1. Installation and Setup
1. Clone this repository and go into the project directory
```
git clone https://github.com/yash-joshi5379/IXN-Project-Team-Extensify.git
cd IXN-Project-Team-Extensify/
git checkout second-dataset
```

2. Create a virtual environment, activate it, and install all requirements
```
python -m venv .venv
.venv\Scripts\activate      # This is for Windows, on Linux/macOS try: source .venv/bin/activate
pip install -r requirements.txt
pip install datasets
```

3. Download the new dataset and place it at the project root directory. The structure should be ```IXN-Project-Team-Extensify/dataset-v2```.

4. Check the ```load_data_test.py``` file runs without errors by running ```python src/load_data_test.py```. It should print the following output.
```
['action', 'observation.effort', 'observation.state', 'observation.qvel', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
action                 object
observation.effort     object
observation.state      object
observation.qvel       object
timestamp             float32
frame_index             int64
episode_index           int64
index                   int64
task_index              int64
dtype: object
(600, 9)
                                              action  ... task_index
0  [-0.77471477, -0.42089885, 0.45596159, 1.47330...  ...          0
1  [-0.774749, -0.42090473, 0.4559829, 1.4733104,...  ...          0
2  [-0.77476674, -0.42090854, 0.45599383, 1.47331...  ...          0
3  [-0.7748592, -0.42055917, 0.45600408, 1.474035...  ...          0
4  [-0.7747894, -0.4212474, 0.45586726, 1.4746351...  ...          0

[5 rows x 9 columns]
action: shape=(8,), dtype=float32
observation.state: shape=(8,), dtype=float32
observation.effort: shape=(8,), dtype=float32
observation.qvel: shape=(8,), dtype=float32
```

5. While the venv is active, clone the LeRobot repository
```
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
git checkout v0.3.3
cd ..
pip install rerun-sdk==0.22.1 datasets==3.6.0
```
Note: The -e means "editable," so if you change anything in the LeRobot folder, it updates automatically in your project.

Note: The ```pip install -e .``` command took around 10 mins on my laptop.

6. Uninstall the "headless" version of OpenCV installed from when we installed lerobot, and install the GUI-enabled version:
```
(.venv) ...\IXN-Project-Team-Extensify>pip uninstall opencv-python-headless -y
(.venv) ...\IXN-Project-Team-Extensify>pip install opencv-python
```

7. For your episode *i*, run this command from the project root directory: ```python src/find_end_frame.py i ```. You should see a window with both camera views. Use **SPACE** to start/stop the videos, **f** to step forward 1 frame and **b** to step back 1 frame. Once you have found the end frame index, press **q**.

8. Open ```notes.txt``` and add your episode number *i*, with either the end frame index, or INVALID. 
**An episode is invalid if:**
- the video ends but the cylinder is not yet in the slot
- someone walks through the video
- the arm pushes down on the box edges
- anything else weird

9. Repeat this for all your episodes, Yash will take care of the actual trimming of the videos and training
