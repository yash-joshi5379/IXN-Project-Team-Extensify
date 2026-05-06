# Guide for Preprocessing Aloha Dataset Episodes

Goal: find end frame indices for all *valid* episodes to remove useless info before training an ACT model

## 1. Installation and Setup
1. Clone this repository and go into the project directory
```
git clone https://github.com/yash-joshi5379/IXN-Project-Team-Extensify.git
cd IXN-Project-Team-Extensify/
git checkout aloha-dataset
```

2. Create a virtual environment, activate it, and install all requirements
```
python -m venv .venv
.venv/Scripts/activate      # This is for Windows, on Linux/macOS try: source .venv/bin/activate
pip install -r requirements.txt
pip install h5py opencv-python
```

3. Download the ALOHA dataset and place it into the project directory (structure should be .../IXN-Project-Team-Extensify/aloha_dataset).

4. Check the ```load_aloha_data_test.py``` file runs without errors by running ```python src/load_aloha_data_test.py```. It should print something like the following output:
```
Inspecting: C:/Users/yashj/projects/IXN-Project-Team-Extensify/aloha_dataset/episode_05.hdf5
Dataset: /action | Shape: (600, 8) | Type: float32
Dataset: /action_headers | Shape: (2,) | Type: |S42
Dataset: /camera_names | Shape: (2,) | Type: |S11
Dataset: /compress_len | Shape: (2, 600) | Type: float32
Dataset: /contact_force_headers | Shape: (1,) | Type: |S49
Dataset: /effort_headers | Shape: (2,) | Type: |S42
Dataset: /force_torque_headers | Shape: (1,) | Type: |S52
Group: /modality_index
Dataset: /modality_index/robokit | Shape: (2,) | Type: object
Dataset: /modality_index/sensekit | Shape: (2,) | Type: object
Group: /observations
Dataset: /observations/contact_force | Shape: (600, 0) | Type: float32
Dataset: /observations/effort | Shape: (600, 8) | Type: float32
Dataset: /observations/force_torque | Shape: (600, 6) | Type: float32
Group: /observations/images
Dataset: /observations/images/0_femtobolt | Shape: (600, 64016) | Type: uint8
Dataset: /observations/images/1_gemini330 | Shape: (600, 55969) | Type: uint8
Dataset: /observations/qpos | Shape: (600, 8) | Type: float32
Dataset: /observations/qvel | Shape: (600, 8) | Type: float32
Dataset: /observations_headers | Shape: (7,) | Type: |S32
Dataset: /qpos_headers | Shape: (2,) | Type: |S40
Dataset: /qvel_headers | Shape: (2,) | Type: |S40
```

5. Open the script ```find_end_frame_aloha.py``` and scroll to the bottom of this script. Replace the episode number with your allocated episode number, do not change anything else. 

6. Run the script using ```python src/find_end_frame_aloha.py``` in the terminal within your venv. A window should pop up with both videos playing simultaneously. Use **SPACE** to start/stop the videos, **f** to move forward 1 frame, **b** to move backward 1 frame or **q** to terminate the videos. 

7. Using these controls, find the end frame index of the videos by stopping the video once the cylinder has clearly fallen into the slot and the cylinder stops moving. Note the end frame index by looking at the green text over the videos.

8. Open up ```end_frame_indexes.txt```, and add your episode number and your observed end frame index for this episode into this text file. 

**IMPORTANT: We only want perfect episodes for this ACT policy. Hence if there any jumpy movements or if the robot fails to pick up the cylinder in the first try, record the end frame index as ZERO in ```end_frame_indexes.txt```. Yash will remove these episodes from the dataset later.**